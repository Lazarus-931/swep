#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <numeric>
#include <string>

static id<MTLDevice> dev;
static id<MTLCommandQueue> q;
static id<MTLLibrary> lib;

static id<MTLComputePipelineState> pso(NSString* name) {
    NSError* e; auto fn = [lib newFunctionWithName:name];
    if (!fn) { fprintf(stderr, "missing: %s\n", name.UTF8String); exit(1); }
    auto p = [dev newComputePipelineStateWithFunction:fn error:&e];
    if (!p) { fprintf(stderr, "pso: %s\n", e.localizedDescription.UTF8String); exit(1); }
    return p;
}

static id<MTLBuffer> buf(size_t n) { return [dev newBufferWithLength:n options:MTLResourceStorageModeShared]; }
static id<MTLBuffer> cbuf(const void* d, size_t n) { return [dev newBufferWithBytes:d length:n options:MTLResourceStorageModeShared]; }

static double run(id<MTLComputePipelineState> p, void(^enc)(id<MTLComputeCommandEncoder>), MTLSize g, MTLSize tg) {
    auto cb = [q commandBuffer]; auto e = [cb computeCommandEncoder];
    [e setComputePipelineState:p]; enc(e);
    [e dispatchThreads:g threadsPerThreadgroup:tg];
    [e endEncoding]; [cb commit]; [cb waitUntilCompleted];
    return cb.GPUEndTime - cb.GPUStartTime;
}

static double median(id<MTLComputePipelineState> p, void(^enc)(id<MTLComputeCommandEncoder>), MTLSize g, MTLSize tg, int w, int n) {
    for (int i = 0; i < w; i++) run(p, enc, g, tg);
    std::vector<double> t; for (int i = 0; i < n; i++) t.push_back(run(p, enc, g, tg));
    std::sort(t.begin(), t.end()); return t[n/2];
}

static std::vector<uint32_t> buildChase(uint32_t n) {
    std::vector<uint32_t> ord(n); std::iota(ord.begin(), ord.end(), 0);
    for (uint32_t i = n-1; i > 0; i--) { uint32_t j = arc4random_uniform(i+1); std::swap(ord[i], ord[j]); }
    std::vector<uint32_t> c(n);
    for (uint32_t i = 0; i < n-1; i++) c[ord[i]] = ord[i+1];
    c[ord[n-1]] = ord[0]; return c;
}

static FILE* csv;

int main() {
    @autoreleasepool {
        dev = MTLCreateSystemDefaultDevice();
        if (!dev) { fprintf(stderr, "No Metal\n"); return 1; }
        q = [dev newCommandQueue];
        NSError* e;
        NSString* cwd = [[NSFileManager defaultManager] currentDirectoryPath];
        auto src = [NSString stringWithContentsOfFile:[cwd stringByAppendingPathComponent:@"kernels.metal"] encoding:NSUTF8StringEncoding error:&e];
        if (!src) { fprintf(stderr, "Can't read kernels.metal\n"); return 1; }
        auto opts = [[MTLCompileOptions alloc] init]; opts.languageVersion = MTLLanguageVersion3_0;
        lib = [dev newLibraryWithSource:src options:opts error:&e];
        if (!lib) { fprintf(stderr, "Compile: %s\n", e.localizedDescription.UTF8String); return 1; }

        csv = fopen("results.csv", "w");
        fprintf(csv, "test,param,value,unit\n");

        printf("M4 Memory Deep-Dive — %s\n", dev.name.UTF8String);
        printf("====================================\n");

        size_t dramSize = 128 * 1024 * 1024;
        auto srcBuf = buf(dramSize);
        auto dstBuf4 = buf(dramSize);
        float* fp = (float*)srcBuf.contents;
        for (size_t i = 0; i < dramSize/4; i++) fp[i] = 0.001f;

        // =====================================================================
        // 1. Bandwidth saturation: how many threads to hit peak?
        // M4 has 10 cores. Sweep from 32 to 65536 threads.
        // =====================================================================
        printf("\n[1] Bandwidth Saturation Point\n");
        printf("  %8s  %8s\n", "Threads", "GB/s");
        {
            uint32_t iters = 2000;
            uint32_t mask = (uint32_t)(dramSize / 16 - 1);
            auto maskB = cbuf(&mask, 4); auto iterB = cbuf(&iters, 4);
            uint32_t counts[] = {32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536};
            for (auto nt : counts) {
                auto dst = buf(nt * 4);
                auto p = pso(@"read_only");
                double t = median(p, ^(id<MTLComputeCommandEncoder> e) {
                    [e setBuffer:srcBuf offset:0 atIndex:0]; [e setBuffer:dst offset:0 atIndex:1];
                    [e setBuffer:maskB offset:0 atIndex:2]; [e setBuffer:iterB offset:0 atIndex:3];
                }, MTLSizeMake(nt,1,1), MTLSizeMake(256,1,1), 2, 5);
                double gb = (double)nt * iters * 16 / t / 1e9;
                printf("  %8u  %8.1f\n", nt, gb);
                fprintf(csv, "bw_saturation,%u,%.2f,GB/s\n", nt, gb);
            }
        }

        // =====================================================================
        // 2. Latency under increasing bandwidth load
        // Pointer chase in L1 (16KB) and DRAM (32MB) while co-running bandwidth threads
        // =====================================================================
        printf("\n[2] Latency Under Load\n");
        printf("  %8s  %8s  %12s  %12s\n", "WS", "BWthds", "Lat(ns)", "BW(GB/s)");
        {
            uint32_t hops = 30000;
            auto hopB = cbuf(&hops, 4);
            uint32_t streamMask = (uint32_t)(dramSize / 16 - 1);
            auto smB = cbuf(&streamMask, 4);

            uint32_t wsSizes[] = {16, 32768};
            const char* wsNames[] = {"L1(16K)", "DRAM(32M)"};
            uint32_t bwThreadCounts[] = {0, 256, 1024, 4096, 16384};

            for (int w = 0; w < 2; w++) {
                uint32_t wsKB = wsSizes[w];
                uint32_t n = wsKB * 1024 / 4;
                auto chain = buildChase(n);
                auto chainB = cbuf(chain.data(), n * 4);

                for (auto bwt : bwThreadCounts) {
                    if (bwt == 0) {
                        auto outB = buf(4);
                        auto p = pso(@"latency_chase");
                        double t = median(p, ^(id<MTLComputeCommandEncoder> e) {
                            [e setBuffer:chainB offset:0 atIndex:0]; [e setBuffer:outB offset:0 atIndex:1];
                            [e setBuffer:hopB offset:0 atIndex:2];
                        }, MTLSizeMake(1,1,1), MTLSizeMake(1,1,1), 3, 7);
                        double lat = t / hops * 1e9;
                        printf("  %8s  %8s  %12.1f  %12s\n", wsNames[w], "none", lat, "-");
                        fprintf(csv, "lat_under_load_%s,0,%.2f,ns\n", wsNames[w], lat);
                    } else {
                        NSUInteger total = bwt;
                        NSUInteger tg = 256;
                        auto outB = buf(total * 4);
                        auto p = pso(@"latency_plus_bandwidth");
                        double t = median(p, ^(id<MTLComputeCommandEncoder> e) {
                            [e setBuffer:chainB offset:0 atIndex:0]; [e setBuffer:outB offset:0 atIndex:1];
                            [e setBuffer:hopB offset:0 atIndex:2]; [e setBuffer:srcBuf offset:0 atIndex:3];
                            [e setBuffer:smB offset:0 atIndex:4];
                        }, MTLSizeMake(total,1,1), MTLSizeMake(tg,1,1), 2, 5);
                        double lat = t / hops * 1e9;
                        uint32_t bwThreadsActual = (uint32_t)(total - (total / tg) * 8);
                        double gb = (double)bwThreadsActual * hops * 16 / t / 1e9;
                        printf("  %8s  %8u  %12.1f  %12.1f\n", wsNames[w], bwt, lat, gb);
                        fprintf(csv, "lat_under_load_%s,%u,%.2f,ns\n", wsNames[w], bwt, lat);
                    }
                }
            }
        }

        // =====================================================================
        // 3. Effective cache size under bandwidth pressure
        // Sweep working set with high thread count — see where BW drops
        // =====================================================================
        printf("\n[3] Cache Boundaries Under Pressure\n");
        printf("  %10s  %8s\n", "WS(KB)", "GB/s");
        {
            uint32_t iters = 2000;
            auto iterB = cbuf(&iters, 4);
            NSUInteger threads = 16384;
            auto dst = buf(threads * 4);
            uint32_t wsSweep[] = {8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072};
            for (auto wsKB : wsSweep) {
                uint32_t wsElems = wsKB * 1024 / 4;
                uint32_t po2 = 1; while (po2 < wsElems) po2 <<= 1;
                uint32_t mask = po2 / 4 - 1;
                auto maskB = cbuf(&mask, 4);
                auto p = pso(@"read_only");
                double t = median(p, ^(id<MTLComputeCommandEncoder> e) {
                    [e setBuffer:srcBuf offset:0 atIndex:0]; [e setBuffer:dst offset:0 atIndex:1];
                    [e setBuffer:maskB offset:0 atIndex:2]; [e setBuffer:iterB offset:0 atIndex:3];
                }, MTLSizeMake(threads,1,1), MTLSizeMake(256,1,1), 2, 5);
                double gb = (double)threads * iters * 16 / t / 1e9;
                printf("  %10u  %8.1f\n", wsKB, gb);
                fprintf(csv, "cache_under_pressure,%u,%.2f,GB/s\n", wsKB, gb);
            }
        }

        // =====================================================================
        // 4. Read vs Write vs Read+Write asymmetry
        // =====================================================================
        printf("\n[4] Read / Write / Copy Bandwidth\n");
        {
            uint32_t iters = 2000;
            uint32_t mask = (uint32_t)(dramSize / 16 - 1);
            auto maskB = cbuf(&mask, 4); auto iterB = cbuf(&iters, 4);
            NSUInteger threads = 16384;

            auto dst = buf(threads * 4);
            auto p1 = pso(@"read_only");
            double tR = median(p1, ^(id<MTLComputeCommandEncoder> e) {
                [e setBuffer:srcBuf offset:0 atIndex:0]; [e setBuffer:dst offset:0 atIndex:1];
                [e setBuffer:maskB offset:0 atIndex:2]; [e setBuffer:iterB offset:0 atIndex:3];
            }, MTLSizeMake(threads,1,1), MTLSizeMake(256,1,1), 2, 5);
            double gbR = (double)threads * iters * 16 / tR / 1e9;

            auto p2 = pso(@"write_only");
            double tW = median(p2, ^(id<MTLComputeCommandEncoder> e) {
                [e setBuffer:dstBuf4 offset:0 atIndex:0];
                [e setBuffer:maskB offset:0 atIndex:1]; [e setBuffer:iterB offset:0 atIndex:2];
            }, MTLSizeMake(threads,1,1), MTLSizeMake(256,1,1), 2, 5);
            double gbW = (double)threads * iters * 16 / tW / 1e9;

            auto p3 = pso(@"read_write");
            double tRW = median(p3, ^(id<MTLComputeCommandEncoder> e) {
                [e setBuffer:srcBuf offset:0 atIndex:0]; [e setBuffer:dstBuf4 offset:0 atIndex:1];
                [e setBuffer:maskB offset:0 atIndex:2]; [e setBuffer:iterB offset:0 atIndex:3];
            }, MTLSizeMake(threads,1,1), MTLSizeMake(256,1,1), 2, 5);
            double gbRW = (double)threads * iters * 32 / tRW / 1e9;

            printf("  Read:       %8.1f GB/s\n", gbR);
            printf("  Write:      %8.1f GB/s\n", gbW);
            printf("  Copy (R+W): %8.1f GB/s\n", gbRW);
            fprintf(csv, "rw_asymmetry,read,%.2f,GB/s\n", gbR);
            fprintf(csv, "rw_asymmetry,write,%.2f,GB/s\n", gbW);
            fprintf(csv, "rw_asymmetry,copy,%.2f,GB/s\n", gbRW);
        }

        // =====================================================================
        // 5. Shared memory + DRAM interaction
        // Does staging through threadgroup memory compete with DRAM reads?
        // =====================================================================
        printf("\n[5] Shared Memory + DRAM Interaction\n");
        {
            uint32_t iters = 2000;
            uint32_t mask = (uint32_t)(dramSize / 16 - 1);
            auto maskB = cbuf(&mask, 4); auto iterB = cbuf(&iters, 4);
            NSUInteger threads = 16384;

            auto dst = buf(threads * 4);
            auto p1 = pso(@"read_only");
            double tDirect = median(p1, ^(id<MTLComputeCommandEncoder> e) {
                [e setBuffer:srcBuf offset:0 atIndex:0]; [e setBuffer:dst offset:0 atIndex:1];
                [e setBuffer:maskB offset:0 atIndex:2]; [e setBuffer:iterB offset:0 atIndex:3];
            }, MTLSizeMake(threads,1,1), MTLSizeMake(256,1,1), 2, 5);
            double gbDirect = (double)threads * iters * 16 / tDirect / 1e9;

            auto p2 = pso(@"tgmem_plus_dram");
            double tStaged = median(p2, ^(id<MTLComputeCommandEncoder> e) {
                [e setBuffer:srcBuf offset:0 atIndex:0]; [e setBuffer:dst offset:0 atIndex:1];
                [e setBuffer:maskB offset:0 atIndex:2]; [e setBuffer:iterB offset:0 atIndex:3];
            }, MTLSizeMake(threads,1,1), MTLSizeMake(256,1,1), 2, 5);
            double gbStaged = (double)threads * iters * 16 / tStaged / 1e9;

            printf("  Direct DRAM read:           %8.1f GB/s\n", gbDirect);
            printf("  DRAM -> shared mem -> read: %8.1f GB/s\n", gbStaged);
            double overhead = (1.0 - gbStaged / gbDirect) * 100;
            printf("  Staging overhead:           %8.1f%%\n", overhead);
            fprintf(csv, "tgmem_interaction,direct,%.2f,GB/s\n", gbDirect);
            fprintf(csv, "tgmem_interaction,staged,%.2f,GB/s\n", gbStaged);
        }

        // =====================================================================
        // 6. Atomics at realistic contention (10-32 counters = cross-core reduction)
        // =====================================================================
        printf("\n[6] Atomic Throughput (Realistic Contention)\n");
        printf("  %10s  %12s\n", "Counters", "Gops/s");
        {
            uint32_t iters = 50000;
            auto iterB = cbuf(&iters, 4);
            NSUInteger threads = 4096;
            uint32_t counts[] = {1, 2, 4, 8, 10, 16, 20, 32, 64};
            for (auto nc : counts) {
                auto ctr = buf(nc * 4); memset(ctr.contents, 0, nc * 4);
                auto ncB = cbuf(&nc, 4);
                auto p = pso(@"atomic_contention");
                double t = median(p, ^(id<MTLComputeCommandEncoder> e) {
                    [e setBuffer:ctr offset:0 atIndex:0]; [e setBuffer:iterB offset:0 atIndex:1];
                    [e setBuffer:ncB offset:0 atIndex:2];
                }, MTLSizeMake(threads,1,1), MTLSizeMake(256,1,1), 2, 5);
                double gops = (double)threads * iters / t / 1e9;
                printf("  %10u  %12.2f\n", nc, gops);
                fprintf(csv, "atomic_contention,%u,%.2f,Gops/s\n", nc, gops);
            }
        }

        // =====================================================================
        // 7. Per-core bandwidth scaling
        // 1 threadgroup = 1 core. Sweep 1-20 threadgroups.
        // =====================================================================
        printf("\n[7] Per-Core Bandwidth Scaling\n");
        printf("  %8s  %8s\n", "TGroups", "GB/s");
        {
            uint32_t iters = 3000;
            uint32_t mask = (uint32_t)(dramSize / 16 - 1);
            auto maskB = cbuf(&mask, 4); auto iterB = cbuf(&iters, 4);
            NSUInteger tg = 256;
            for (int ng = 1; ng <= 20; ng++) {
                NSUInteger total = ng * tg;
                auto dst = buf(total * 4);
                auto p = pso(@"read_only");
                double t = median(p, ^(id<MTLComputeCommandEncoder> e) {
                    [e setBuffer:srcBuf offset:0 atIndex:0]; [e setBuffer:dst offset:0 atIndex:1];
                    [e setBuffer:maskB offset:0 atIndex:2]; [e setBuffer:iterB offset:0 atIndex:3];
                }, MTLSizeMake(total,1,1), MTLSizeMake(tg,1,1), 2, 5);
                double gb = (double)total * iters * 16 / t / 1e9;
                printf("  %8d  %8.1f\n", ng, gb);
                fprintf(csv, "core_scaling,%d,%.2f,GB/s\n", ng, gb);
            }
        }

        fclose(csv);
        printf("\nResults written to results.csv\n");
        printf("Done.\n");
    }
    return 0;
}
