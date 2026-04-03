#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <QuartzCore/QuartzCore.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>
#include <numeric>

// ============================================================================
// Helpers
// ============================================================================

static id<MTLDevice> gDevice;
static id<MTLCommandQueue> gQueue;
static id<MTLLibrary> gLibrary;

static id<MTLComputePipelineState> makePSO(NSString* name) {
    NSError* err = nil;
    id<MTLFunction> fn = [gLibrary newFunctionWithName:name];
    if (!fn) {
        fprintf(stderr, "ERROR: function '%s' not found\n", name.UTF8String);
        exit(1);
    }
    id<MTLComputePipelineState> pso = [gDevice newComputePipelineStateWithFunction:fn error:&err];
    if (!pso) {
        fprintf(stderr, "ERROR: PSO creation failed: %s\n", err.localizedDescription.UTF8String);
        exit(1);
    }
    return pso;
}

static id<MTLBuffer> makeBuffer(size_t bytes, MTLResourceOptions opts = MTLResourceStorageModeShared) {
    return [gDevice newBufferWithLength:bytes options:opts];
}

static id<MTLBuffer> makeBufferWithData(const void* data, size_t bytes) {
    return [gDevice newBufferWithBytes:data length:bytes options:MTLResourceStorageModeShared];
}

// Dispatch and return GPU time in seconds
static double dispatchAndTime(id<MTLComputePipelineState> pso,
                              void (^encode)(id<MTLComputeCommandEncoder>),
                              MTLSize grid, MTLSize tgSize)
{
    id<MTLCommandBuffer> cb = [gQueue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:pso];
    encode(enc);
    [enc dispatchThreads:grid threadsPerThreadgroup:tgSize];
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    return cb.GPUEndTime - cb.GPUStartTime;
}

// Multiple dispatches, return median time
static double timedDispatch(id<MTLComputePipelineState> pso,
                            void (^encode)(id<MTLComputeCommandEncoder>),
                            MTLSize grid, MTLSize tgSize,
                            int warmup, int trials)
{
    for (int i = 0; i < warmup; i++)
        dispatchAndTime(pso, encode, grid, tgSize);

    std::vector<double> times;
    for (int i = 0; i < trials; i++)
        times.push_back(dispatchAndTime(pso, encode, grid, tgSize));

    std::sort(times.begin(), times.end());
    return times[trials / 2];
}

// ============================================================================
// Probe: GPU Timer Resolution
// ============================================================================
static void probeTimerResolution() {
    printf("\n=== GPU Timer Resolution ===\n");

    id<MTLComputePipelineState> pso = makePSO(@"probe_dram_bandwidth");
    size_t bytes = 256; // tiny
    id<MTLBuffer> src = makeBuffer(bytes);
    id<MTLBuffer> dst = makeBuffer(bytes);

    std::vector<double> deltas;
    for (int i = 0; i < 200; i++) {
        double t = dispatchAndTime(pso, ^(id<MTLComputeCommandEncoder> enc) {
            [enc setBuffer:src offset:0 atIndex:0];
            [enc setBuffer:dst offset:0 atIndex:1];
        }, MTLSizeMake(1, 1, 1), MTLSizeMake(1, 1, 1));
        if (t > 0) deltas.push_back(t);
    }

    std::sort(deltas.begin(), deltas.end());

    // Find smallest non-zero delta
    double minDelta = deltas.empty() ? 0 : deltas[0];
    // Find GCD-like resolution by looking at differences between sorted times
    std::vector<double> diffs;
    for (size_t i = 1; i < deltas.size(); i++) {
        double d = deltas[i] - deltas[i-1];
        if (d > 1e-12) diffs.push_back(d);
    }
    std::sort(diffs.begin(), diffs.end());
    double resolution = diffs.empty() ? minDelta : diffs[0];

    printf("  Min observed dispatch time: %.3f µs\n", minDelta * 1e6);
    printf("  Estimated timer resolution: %.3f µs\n", resolution * 1e6);
    printf("  Recommendation: ensure kernel runs > %.0f µs for <1%% error\n", resolution * 1e6 * 100);
}

// ============================================================================
// Probe: Direct Queries
// ============================================================================
static void probeDirectQueries() {
    printf("\n=== Direct Device Queries ===\n");
    printf("  Device name:                %s\n", gDevice.name.UTF8String);
    printf("  Max threadgroup memory:     %lu bytes (%lu KB)\n",
           (unsigned long)gDevice.maxThreadgroupMemoryLength,
           (unsigned long)gDevice.maxThreadgroupMemoryLength / 1024);
    printf("  Max threads per threadgroup: %lu\n",
           (unsigned long)gDevice.maxThreadsPerThreadgroup.width);
    printf("  Max buffer length:          %lu MB\n",
           (unsigned long)(gDevice.maxBufferLength / (1024*1024)));
    printf("  Recommended working set:    %llu MB\n",
           gDevice.recommendedMaxWorkingSetSize / (1024*1024));
    printf("  Has unified memory:         %s\n", gDevice.hasUnifiedMemory ? "YES" : "NO");

    // SIMD width from a PSO
    id<MTLComputePipelineState> pso = makePSO(@"probe_dram_bandwidth");
    printf("  SIMD width (threadExecutionWidth): %lu\n",
           (unsigned long)pso.threadExecutionWidth);
    printf("  Max total threads per TG (PSO):    %lu\n",
           (unsigned long)pso.maxTotalThreadsPerThreadgroup);
}

// ============================================================================
// Probe: DRAM Bandwidth
// ============================================================================
static void probeDRAMBandwidth() {
    printf("\n=== DRAM Bandwidth ===\n");
    id<MTLComputePipelineState> pso = makePSO(@"probe_dram_bandwidth");

    // 256 MB copy — well beyond any on-chip cache
    size_t bytes = 256 * 1024 * 1024;
    size_t numFloat4 = bytes / sizeof(float) / 4;
    id<MTLBuffer> src = makeBuffer(bytes);
    id<MTLBuffer> dst = makeBuffer(bytes);

    // Fill src
    float* p = (float*)src.contents;
    for (size_t i = 0; i < bytes / sizeof(float); i++) p[i] = 1.0f;

    NSUInteger tgw = pso.threadExecutionWidth;
    double t = timedDispatch(pso, ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:src offset:0 atIndex:0];
        [enc setBuffer:dst offset:0 atIndex:1];
    }, MTLSizeMake(numFloat4, 1, 1), MTLSizeMake(tgw * 4, 1, 1), 3, 10);

    double totalBytes = 2.0 * bytes; // read + write
    double gbps = totalBytes / t / 1e9;
    printf("  Buffer size:    %zu MB\n", bytes / (1024*1024));
    printf("  GPU time:       %.3f ms\n", t * 1000);
    printf("  DRAM bandwidth: %.1f GB/s\n", gbps);
}

// ============================================================================
// Probe: Threadgroup Memory Bandwidth
// ============================================================================
static void probeTGMemBandwidth() {
    printf("\n=== Threadgroup Memory Bandwidth ===\n");
    id<MTLComputePipelineState> pso = makePSO(@"probe_tgmem_bandwidth");

    NSUInteger tgSize = 256;
    NSUInteger numGroups = 1024;
    id<MTLBuffer> out = makeBuffer(4);

    double t = timedDispatch(pso, ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:out offset:0 atIndex:0];
    }, MTLSizeMake(numGroups * tgSize, 1, 1), MTLSizeMake(tgSize, 1, 1), 3, 10);

    // Each thread does 256 writes + 256 reads of float4 (16 bytes)
    double bytesPerThread = 256.0 * 16 * 2; // read + write
    double totalBytes = bytesPerThread * numGroups * tgSize;
    double gbps = totalBytes / t / 1e9;
    printf("  Threadgroups: %lu, threads/group: %lu\n", (unsigned long)numGroups, (unsigned long)tgSize);
    printf("  GPU time:     %.3f ms\n", t * 1000);
    printf("  TG mem bandwidth: %.1f GB/s (aggregate)\n", gbps);
}

// ============================================================================
// Probe: Cache Size Sweep (L1 / SLC)
// ============================================================================
static void probeCacheSizes() {
    printf("\n=== Cache Size Sweep (L1 / SLC boundaries) ===\n");
    id<MTLComputePipelineState> pso = makePSO(@"probe_cache_sweep");

    // Allocate largest buffer we'll need: 64 MB
    size_t maxBytes = 64 * 1024 * 1024;
    id<MTLBuffer> data = makeBuffer(maxBytes);
    float* p = (float*)data.contents;
    for (size_t i = 0; i < maxBytes / sizeof(float); i++) p[i] = 1.0f;

    id<MTLBuffer> out = makeBuffer(1024 * sizeof(float));

    NSUInteger threads = 256;
    NSUInteger iters = 1024 * 64;

    // Sweep working set sizes: 4KB to 48MB
    uint32_t sizes_kb[] = {4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096,
                           6144, 8192, 12288, 16384, 24576, 32768, 49152};
    int nSizes = sizeof(sizes_kb) / sizeof(sizes_kb[0]);

    printf("  %8s  %10s  %10s\n", "WS (KB)", "Time (µs)", "BW (GB/s)");
    printf("  %8s  %10s  %10s\n", "--------", "----------", "----------");

    for (int s = 0; s < nSizes; s++) {
        uint32_t wsBytes = sizes_kb[s] * 1024;
        uint32_t wsElements = wsBytes / sizeof(float);
        // Ensure power of 2
        uint32_t po2 = 1;
        while (po2 < wsElements) po2 <<= 1;
        wsElements = po2;
        uint32_t itersVal = (uint32_t)iters;

        id<MTLBuffer> wsBuf = makeBufferWithData(&wsElements, sizeof(uint32_t));
        id<MTLBuffer> iterBuf = makeBufferWithData(&itersVal, sizeof(uint32_t));

        double t = timedDispatch(pso, ^(id<MTLComputeCommandEncoder> enc) {
            [enc setBuffer:data offset:0 atIndex:0];
            [enc setBuffer:out offset:0 atIndex:1];
            [enc setBuffer:wsBuf offset:0 atIndex:2];
            [enc setBuffer:iterBuf offset:0 atIndex:3];
        }, MTLSizeMake(threads, 1, 1), MTLSizeMake(threads, 1, 1), 2, 5);

        double totalBytes = (double)threads * itersVal * sizeof(float);
        double gbps = totalBytes / t / 1e9;
        printf("  %8u  %10.1f  %10.1f\n", sizes_kb[s], t * 1e6, gbps);
    }
    printf("  -> Look for throughput cliffs: first cliff = L1 size, second = SLC size\n");
}

// ============================================================================
// Probe: Cache Latency (pointer chase)
// ============================================================================
static void probeCacheLatency() {
    printf("\n=== Cache Hit Latency (pointer chase) ===\n");
    id<MTLComputePipelineState> pso = makePSO(@"probe_cache_latency");

    uint32_t sizes_kb[] = {4, 16, 64, 256, 1024, 4096, 16384, 32768};
    int nSizes = sizeof(sizes_kb) / sizeof(sizes_kb[0]);
    uint32_t hops = 100000;

    printf("  %8s  %12s  %10s\n", "WS (KB)", "Time (µs)", "Lat (ns)");
    printf("  %8s  %12s  %10s\n", "--------", "----------", "--------");

    for (int s = 0; s < nSizes; s++) {
        uint32_t wsBytes = sizes_kb[s] * 1024;
        uint32_t n = wsBytes / sizeof(uint32_t);

        // Build a random pointer chase within the working set
        std::vector<uint32_t> chain(n);
        std::iota(chain.begin(), chain.end(), 0);
        // Fisher-Yates shuffle to create a single cycle
        for (uint32_t i = n - 1; i > 0; i--) {
            uint32_t j = arc4random_uniform(i + 1);
            std::swap(chain[i], chain[j]);
        }
        // Convert permutation to next-pointer cycle
        std::vector<uint32_t> buf(n);
        for (uint32_t i = 0; i < n - 1; i++)
            buf[chain[i]] = chain[i + 1];
        buf[chain[n - 1]] = chain[0];

        id<MTLBuffer> chainBuf = makeBufferWithData(buf.data(), n * sizeof(uint32_t));
        id<MTLBuffer> outBuf = makeBuffer(sizeof(uint32_t));
        id<MTLBuffer> hopsBuf = makeBufferWithData(&hops, sizeof(uint32_t));

        // Single thread for true latency measurement
        double t = timedDispatch(pso, ^(id<MTLComputeCommandEncoder> enc) {
            [enc setBuffer:chainBuf offset:0 atIndex:0];
            [enc setBuffer:outBuf offset:0 atIndex:1];
            [enc setBuffer:hopsBuf offset:0 atIndex:2];
        }, MTLSizeMake(1, 1, 1), MTLSizeMake(1, 1, 1), 3, 10);

        double latNs = (t / hops) * 1e9;
        printf("  %8u  %12.1f  %10.1f\n", sizes_kb[s], t * 1e6, latNs);
    }
    printf("  -> Latency jumps indicate cache level transitions\n");
}

// ============================================================================
// Probe: Register Spill Boundary
// ============================================================================
static void probeRegisterSpill() {
    printf("\n=== Register Spill Boundary (simdgroup_float8x8 accumulators) ===\n");

    NSString* names[] = {
        @"probe_regpressure_4",
        @"probe_regpressure_8",
        @"probe_regpressure_16",
        @"probe_regpressure_24",
        @"probe_regpressure_32",
        @"probe_regpressure_48"
    };
    int counts[] = {4, 8, 16, 24, 32, 48};
    int nVariants = 6;

    uint32_t K = 512;
    size_t matBytes = K * 8 * sizeof(float);
    id<MTLBuffer> A = makeBuffer(matBytes);
    id<MTLBuffer> B = makeBuffer(matBytes);
    id<MTLBuffer> C = makeBuffer(8 * 8 * sizeof(float));

    float* pa = (float*)A.contents;
    float* pb = (float*)B.contents;
    for (size_t i = 0; i < matBytes / sizeof(float); i++) {
        pa[i] = 0.01f;
        pb[i] = 0.01f;
    }

    id<MTLBuffer> kBuf = makeBufferWithData(&K, sizeof(uint32_t));

    printf("  %6s  %10s  %10s\n", "Accums", "Time (µs)", "TFLOPS");
    printf("  %6s  %10s  %10s\n", "------", "----------", "--------");

    // Launch enough SIMDs to saturate
    NSUInteger simdWidth = 32;
    NSUInteger numSimds = 1024;

    for (int v = 0; v < nVariants; v++) {
        id<MTLComputePipelineState> pso = makePSO(names[v]);

        double t = timedDispatch(pso, ^(id<MTLComputeCommandEncoder> enc) {
            [enc setBuffer:A offset:0 atIndex:0];
            [enc setBuffer:B offset:0 atIndex:1];
            [enc setBuffer:C offset:0 atIndex:2];
            [enc setBuffer:kBuf offset:0 atIndex:3];
        }, MTLSizeMake(numSimds * simdWidth, 1, 1), MTLSizeMake(simdWidth, 1, 1), 3, 10);

        // Each SIMD does counts[v] MMA ops per K iteration, each MMA = 8*8*2 FLOPS
        double flops = (double)numSimds * counts[v] * (K / 8.0) * 8 * 8 * 2;
        double tflops = flops / t / 1e12;
        printf("  %6d  %10.1f  %10.3f\n", counts[v], t * 1e6, tflops);
    }
    printf("  -> TFLOPS drop indicates register spill (compiler ran out of registers)\n");
}

// ============================================================================
// Probe: Shader Core Count
// ============================================================================
static void probeShaderCoreCount() {
    printf("\n=== Shader Core Count (saturation sweep) ===\n");
    id<MTLComputePipelineState> pso = makePSO(@"probe_core_saturation");

    uint32_t fmaIters = 100000;
    id<MTLBuffer> iterBuf = makeBufferWithData(&fmaIters, sizeof(uint32_t));
    NSUInteger tgSize = 256;

    printf("  %8s  %10s  %10s\n", "TGroups", "Time (ms)", "GFLOPS");
    printf("  %8s  %10s  %10s\n", "--------", "----------", "--------");

    double prevTflops = 0;
    int plateauCount = 0;
    int inferredCores = 0;

    for (int nGroups = 1; nGroups <= 128; nGroups++) {
        NSUInteger totalThreads = nGroups * tgSize;
        id<MTLBuffer> out = makeBuffer(totalThreads * sizeof(float));

        double t = timedDispatch(pso, ^(id<MTLComputeCommandEncoder> enc) {
            [enc setBuffer:out offset:0 atIndex:0];
            [enc setBuffer:iterBuf offset:0 atIndex:1];
        }, MTLSizeMake(totalThreads, 1, 1), MTLSizeMake(tgSize, 1, 1), 2, 5);

        // 4 FMAs per iteration, each = 2 FLOPS, per thread
        double flops = (double)totalThreads * fmaIters * 4 * 2;
        double gflops = flops / t / 1e9;

        if (nGroups <= 20 || nGroups % 4 == 0)
            printf("  %8d  %10.3f  %10.1f\n", nGroups, t * 1000, gflops);

        double tflops = gflops / 1000.0;
        if (prevTflops > 0 && tflops < prevTflops * 1.02) {
            plateauCount++;
            if (plateauCount == 3 && inferredCores == 0) {
                inferredCores = nGroups - 2; // plateau started ~2 groups ago
            }
        } else {
            plateauCount = 0;
        }
        prevTflops = tflops;
    }

    if (inferredCores > 0)
        printf("  -> Throughput plateaus around %d threadgroups => ~%d shader cores\n",
               inferredCores, inferredCores);
    else
        printf("  -> No clear plateau detected; try larger sweep range\n");
}

// ============================================================================
// Probe: Threadgroup Memory Bank Conflicts
// ============================================================================
static void probeBankConflicts() {
    printf("\n=== Threadgroup Memory Bank Layout ===\n");
    id<MTLComputePipelineState> pso = makePSO(@"probe_bank_conflicts");

    NSUInteger tgSize = 32; // one SIMD
    NSUInteger numGroups = 256;
    NSUInteger totalThreads = numGroups * tgSize;
    uint32_t iterations = 100000;

    id<MTLBuffer> out = makeBuffer(totalThreads * sizeof(float));
    id<MTLBuffer> iterBuf = makeBufferWithData(&iterations, sizeof(uint32_t));

    printf("  %6s  %10s  %10s\n", "Stride", "Time (µs)", "BW (GB/s)");
    printf("  %6s  %10s  %10s\n", "------", "----------", "----------");

    for (uint32_t stride = 1; stride <= 64; stride++) {
        id<MTLBuffer> strideBuf = makeBufferWithData(&stride, sizeof(uint32_t));

        double t = timedDispatch(pso, ^(id<MTLComputeCommandEncoder> enc) {
            [enc setBuffer:out offset:0 atIndex:0];
            [enc setBuffer:strideBuf offset:0 atIndex:1];
            [enc setBuffer:iterBuf offset:0 atIndex:2];
        }, MTLSizeMake(totalThreads, 1, 1), MTLSizeMake(tgSize, 1, 1), 2, 5);

        double bytes = (double)totalThreads * iterations * sizeof(float) * 2; // r+w
        double gbps = bytes / t / 1e9;
        printf("  %6u  %10.1f  %10.1f\n", stride, t * 1e6, gbps);
    }
    printf("  -> Periodic bandwidth drops reveal bank count (period of drops = bank count)\n");
}

// ============================================================================
// Probe: SLC Associativity
// ============================================================================
static void probeSLCAssociativity() {
    printf("\n=== SLC Associativity (conflict miss probe) ===\n");
    id<MTLComputePipelineState> pso = makePSO(@"probe_slc_assoc");

    // Estimate SLC as ~32 MB (will be refined by cache sweep probe)
    // Stride = SLC size / sizeof(float)
    uint32_t slcSizeMB = 32;
    uint32_t strideFloats = slcSizeMB * 1024 * 1024 / sizeof(float);
    uint32_t iterations = 10000;

    // Need buffer large enough for max_ways * stride
    uint32_t maxWays = 24;
    size_t bufBytes = (size_t)maxWays * strideFloats * sizeof(float);
    if (bufBytes > gDevice.maxBufferLength) {
        printf("  SKIP: needs %zu MB buffer, max is %lu MB\n",
               bufBytes / (1024*1024), (unsigned long)(gDevice.maxBufferLength / (1024*1024)));
        return;
    }

    id<MTLBuffer> data = makeBuffer(bufBytes);
    float* p = (float*)data.contents;
    for (size_t i = 0; i < bufBytes / sizeof(float); i++) p[i] = 1.0f;

    NSUInteger threads = 64;
    id<MTLBuffer> out = makeBuffer(threads * sizeof(float));
    id<MTLBuffer> strideBuf = makeBufferWithData(&strideFloats, sizeof(uint32_t));
    id<MTLBuffer> iterBuf = makeBufferWithData(&iterations, sizeof(uint32_t));

    printf("  SLC stride assumption: %u MB\n", slcSizeMB);
    printf("  %6s  %10s  %10s\n", "Ways", "Time (µs)", "BW (GB/s)");
    printf("  %6s  %10s  %10s\n", "------", "----------", "----------");

    for (uint32_t ways = 1; ways <= maxWays; ways++) {
        id<MTLBuffer> waysBuf = makeBufferWithData(&ways, sizeof(uint32_t));

        double t = timedDispatch(pso, ^(id<MTLComputeCommandEncoder> enc) {
            [enc setBuffer:data offset:0 atIndex:0];
            [enc setBuffer:out offset:0 atIndex:1];
            [enc setBuffer:waysBuf offset:0 atIndex:2];
            [enc setBuffer:strideBuf offset:0 atIndex:3];
            [enc setBuffer:iterBuf offset:0 atIndex:4];
        }, MTLSizeMake(threads, 1, 1), MTLSizeMake(threads, 1, 1), 2, 5);

        double bytes = (double)threads * iterations * ways * sizeof(float);
        double gbps = bytes / t / 1e9;
        printf("  %6u  %10.1f  %10.1f\n", ways, t * 1e6, gbps);
    }
    printf("  -> Sudden bandwidth drop at N ways => associativity is N-1\n");
}

// ============================================================================
// Probe: Pipeline Depth
// ============================================================================
static void probePipelineDepth() {
    printf("\n=== Pipeline Depth / FMA Latency ===\n");

    NSString* names[] = {
        @"probe_pipeline_dep1",
        @"probe_pipeline_dep2",
        @"probe_pipeline_dep4",
        @"probe_pipeline_dep8",
        @"probe_pipeline_indep"
    };
    const char* labels[] = {"dep-1", "dep-2", "dep-4", "dep-8", "indep-8"};
    int fmaPerIter[] = {4, 2, 4, 8, 8}; // total FMAs executed per iteration per thread
    int nVariants = 5;

    uint32_t iterations = 500000;
    NSUInteger threads = 1024;
    NSUInteger tgSize = 256;

    id<MTLBuffer> iterBuf = makeBufferWithData(&iterations, sizeof(uint32_t));

    printf("  %10s  %10s  %10s  %10s\n", "Variant", "Time (ms)", "ns/FMA", "GFLOPS");
    printf("  %10s  %10s  %10s  %10s\n", "----------", "----------", "--------", "--------");

    for (int v = 0; v < nVariants; v++) {
        id<MTLComputePipelineState> pso = makePSO(names[v]);
        id<MTLBuffer> out = makeBuffer(threads * sizeof(float));

        double t = timedDispatch(pso, ^(id<MTLComputeCommandEncoder> enc) {
            [enc setBuffer:out offset:0 atIndex:0];
            [enc setBuffer:iterBuf offset:0 atIndex:1];
        }, MTLSizeMake(threads, 1, 1), MTLSizeMake(tgSize, 1, 1), 3, 10);

        double totalFMAs = (double)threads * iterations * fmaPerIter[v];
        double nsPerFMA = (t * 1e9) / totalFMAs;
        double gflops = totalFMAs * 2.0 / t / 1e9; // each FMA = 2 FLOPS
        printf("  %10s  %10.3f  %10.2f  %10.1f\n", labels[v], t * 1000, nsPerFMA, gflops);
    }
    printf("  -> dep-1 latency / indep throughput = pipeline depth needed to saturate\n");
}

// ============================================================================
// Probe: Occupancy per Core
// ============================================================================
static void probeOccupancy() {
    printf("\n=== Occupancy vs Threadgroup Memory ===\n");
    id<MTLComputePipelineState> pso = makePSO(@"probe_occupancy");

    uint32_t iterations = 100000;
    id<MTLBuffer> iterBuf = makeBufferWithData(&iterations, sizeof(uint32_t));

    NSUInteger tgSize = 256;
    NSUInteger numGroups = 1024;
    NSUInteger totalThreads = numGroups * tgSize;

    id<MTLBuffer> out = makeBuffer(totalThreads * sizeof(float));

    printf("  %10s  %10s  %10s\n", "TGMem(KB)", "Time (ms)", "BW (GB/s)");
    printf("  %10s  %10s  %10s\n", "----------", "----------", "----------");

    // Sweep threadgroup memory usage
    uint32_t tgmem_sizes_kb[] = {1, 2, 4, 8, 12, 16, 20, 24, 28, 32};
    int nSizes = sizeof(tgmem_sizes_kb) / sizeof(tgmem_sizes_kb[0]);

    for (int s = 0; s < nSizes; s++) {
        uint32_t tgmemFloats = tgmem_sizes_kb[s] * 1024 / sizeof(float);
        // Clamp to max
        if (tgmem_sizes_kb[s] * 1024 > gDevice.maxThreadgroupMemoryLength)
            continue;

        id<MTLBuffer> tgmemBuf = makeBufferWithData(&tgmemFloats, sizeof(uint32_t));

        double t = timedDispatch(pso, ^(id<MTLComputeCommandEncoder> enc) {
            [enc setBuffer:out offset:0 atIndex:0];
            [enc setBuffer:tgmemBuf offset:0 atIndex:1];
            [enc setBuffer:iterBuf offset:0 atIndex:2];
        }, MTLSizeMake(totalThreads, 1, 1), MTLSizeMake(tgSize, 1, 1), 2, 5);

        double bytes = (double)totalThreads * iterations * sizeof(float) * 2;
        double gbps = bytes / t / 1e9;
        printf("  %10u  %10.3f  %10.1f\n", tgmem_sizes_kb[s], t * 1000, gbps);
    }
    printf("  -> Step drops in throughput reveal occupancy reduction as TG mem grows\n");
}

// ============================================================================
// Probe: Atomic Throughput
// ============================================================================
static void probeAtomicThroughput() {
    printf("\n=== Atomic Operation Throughput ===\n");
    id<MTLComputePipelineState> pso = makePSO(@"probe_atomic_throughput");

    uint32_t iterations = 100000;
    NSUInteger threads = 1024;
    NSUInteger tgSize = 256;
    id<MTLBuffer> iterBuf = makeBufferWithData(&iterations, sizeof(uint32_t));

    printf("  %10s  %10s  %12s\n", "Counters", "Time (ms)", "Gatomics/s");
    printf("  %10s  %10s  %12s\n", "----------", "----------", "----------");

    uint32_t counterCounts[] = {1, 4, 16, 64, 256, 1024};
    for (int c = 0; c < 6; c++) {
        uint32_t nc = counterCounts[c];
        id<MTLBuffer> counters = makeBuffer(nc * sizeof(uint32_t));
        memset(counters.contents, 0, nc * sizeof(uint32_t));
        id<MTLBuffer> ncBuf = makeBufferWithData(&nc, sizeof(uint32_t));

        double t = timedDispatch(pso, ^(id<MTLComputeCommandEncoder> enc) {
            [enc setBuffer:counters offset:0 atIndex:0];
            [enc setBuffer:iterBuf offset:0 atIndex:1];
            [enc setBuffer:ncBuf offset:0 atIndex:2];
        }, MTLSizeMake(threads, 1, 1), MTLSizeMake(tgSize, 1, 1), 2, 5);

        double totalOps = (double)threads * iterations;
        double gops = totalOps / t / 1e9;
        printf("  %10u  %10.3f  %12.2f\n", nc, t * 1000, gops);
    }
    printf("  -> Low counter count = high contention. Throughput scaling shows atomic unit width.\n");
}

// ============================================================================
// Probe: SIMD Shuffle Bandwidth
// ============================================================================
static void probeSimdShuffle() {
    printf("\n=== SIMD Shuffle Bandwidth ===\n");
    id<MTLComputePipelineState> pso = makePSO(@"probe_simd_shuffle");

    uint32_t iterations = 500000;
    NSUInteger threads = 4096;
    NSUInteger tgSize = 256;
    id<MTLBuffer> out = makeBuffer(threads * sizeof(float));
    id<MTLBuffer> iterBuf = makeBufferWithData(&iterations, sizeof(uint32_t));

    double t = timedDispatch(pso, ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:out offset:0 atIndex:0];
        [enc setBuffer:iterBuf offset:0 atIndex:1];
    }, MTLSizeMake(threads, 1, 1), MTLSizeMake(tgSize, 1, 1), 3, 10);

    // 5 shuffles per iteration, each moves 4 bytes
    double totalShuffles = (double)threads * iterations * 5;
    double gshuffles = totalShuffles / t / 1e9;
    double totalBytes = totalShuffles * sizeof(float);
    double gbps = totalBytes / t / 1e9;
    printf("  Threads: %lu, iterations: %u\n", (unsigned long)threads, iterations);
    printf("  GPU time:       %.3f ms\n", t * 1000);
    printf("  Shuffle rate:   %.1f Gshuffles/s\n", gshuffles);
    printf("  Effective BW:   %.1f GB/s\n", gbps);
}

// ============================================================================
// Probe: SIMD Reduction
// ============================================================================
static void probeSimdReduce() {
    printf("\n=== SIMD Reduction (simd_sum) Throughput ===\n");
    id<MTLComputePipelineState> pso = makePSO(@"probe_simd_reduce");

    uint32_t iterations = 500000;
    NSUInteger threads = 4096;
    NSUInteger tgSize = 256;
    id<MTLBuffer> out = makeBuffer(threads * sizeof(float));
    id<MTLBuffer> iterBuf = makeBufferWithData(&iterations, sizeof(uint32_t));

    double t = timedDispatch(pso, ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:out offset:0 atIndex:0];
        [enc setBuffer:iterBuf offset:0 atIndex:1];
    }, MTLSizeMake(threads, 1, 1), MTLSizeMake(tgSize, 1, 1), 3, 10);

    double totalReductions = (double)threads * iterations;
    double greductions = totalReductions / t / 1e9;
    printf("  GPU time:          %.3f ms\n", t * 1000);
    printf("  simd_sum rate:     %.1f Greductions/s\n", greductions);
    printf("  ns per simd_sum:   %.2f\n", t * 1e9 / totalReductions);
}

// ============================================================================
// Probe: Integer vs Float ALU Throughput
// ============================================================================
static void probeIntVsFloat() {
    printf("\n=== Integer vs Float ALU Throughput ===\n");

    // Integer
    {
        id<MTLComputePipelineState> pso = makePSO(@"probe_int_throughput");
        uint32_t iterations = 500000;
        NSUInteger threads = 4096;
        NSUInteger tgSize = 256;
        id<MTLBuffer> out = makeBuffer(threads * sizeof(uint32_t));
        id<MTLBuffer> iterBuf = makeBufferWithData(&iterations, sizeof(uint32_t));

        double t = timedDispatch(pso, ^(id<MTLComputeCommandEncoder> enc) {
            [enc setBuffer:out offset:0 atIndex:0];
            [enc setBuffer:iterBuf offset:0 atIndex:1];
        }, MTLSizeMake(threads, 1, 1), MTLSizeMake(tgSize, 1, 1), 3, 10);

        // 4 int MADs per iteration (multiply + add = 2 IOPs each)
        double totalIOPs = (double)threads * iterations * 4 * 2;
        double giops = totalIOPs / t / 1e9;
        printf("  INT32:  %.3f ms, %.1f GIOPS\n", t * 1000, giops);
    }

    // Float (reuse probe_float_throughput)
    {
        id<MTLComputePipelineState> pso = makePSO(@"probe_float_throughput");
        uint32_t iterations = 500000;
        NSUInteger threads = 4096;
        NSUInteger tgSize = 256;
        id<MTLBuffer> out = makeBuffer(threads * sizeof(float));
        id<MTLBuffer> iterBuf = makeBufferWithData(&iterations, sizeof(uint32_t));

        double t = timedDispatch(pso, ^(id<MTLComputeCommandEncoder> enc) {
            [enc setBuffer:out offset:0 atIndex:0];
            [enc setBuffer:iterBuf offset:0 atIndex:1];
        }, MTLSizeMake(threads, 1, 1), MTLSizeMake(tgSize, 1, 1), 3, 10);

        // 4 float4 FMAs per iteration, each FMA = 2 FLOPS * 4 lanes = 8 FLOPS
        double totalFLOPs = (double)threads * iterations * 4 * 8;
        double gflops = totalFLOPs / t / 1e9;
        printf("  FP32:   %.3f ms, %.1f GFLOPS\n", t * 1000, gflops);
    }
    printf("  -> Ratio reveals whether int and float share the same ALU or have separate pipes.\n");
}

// ============================================================================
// Probe: Half vs Float Throughput Ratio
// ============================================================================
static void probeHalfPrecision() {
    printf("\n=== Half-Precision (FP16) vs FP32 Throughput ===\n");

    // FP16
    {
        id<MTLComputePipelineState> pso = makePSO(@"probe_half_throughput");
        uint32_t iterations = 500000;
        NSUInteger threads = 4096;
        NSUInteger tgSize = 256;
        id<MTLBuffer> out = makeBuffer(threads * sizeof(uint16_t));
        id<MTLBuffer> iterBuf = makeBufferWithData(&iterations, sizeof(uint32_t));

        double t = timedDispatch(pso, ^(id<MTLComputeCommandEncoder> enc) {
            [enc setBuffer:out offset:0 atIndex:0];
            [enc setBuffer:iterBuf offset:0 atIndex:1];
        }, MTLSizeMake(threads, 1, 1), MTLSizeMake(tgSize, 1, 1), 3, 10);

        // 4 half4 FMAs per iteration, each = 2*4 = 8 FLOPS
        double totalFLOPs = (double)threads * iterations * 4 * 8;
        double gflops = totalFLOPs / t / 1e9;
        printf("  FP16: %.3f ms, %.1f GFLOPS\n", t * 1000, gflops);
    }

    // FP32
    {
        id<MTLComputePipelineState> pso = makePSO(@"probe_float_throughput");
        uint32_t iterations = 500000;
        NSUInteger threads = 4096;
        NSUInteger tgSize = 256;
        id<MTLBuffer> out = makeBuffer(threads * sizeof(float));
        id<MTLBuffer> iterBuf = makeBufferWithData(&iterations, sizeof(uint32_t));

        double t = timedDispatch(pso, ^(id<MTLComputeCommandEncoder> enc) {
            [enc setBuffer:out offset:0 atIndex:0];
            [enc setBuffer:iterBuf offset:0 atIndex:1];
        }, MTLSizeMake(threads, 1, 1), MTLSizeMake(tgSize, 1, 1), 3, 10);

        double totalFLOPs = (double)threads * iterations * 4 * 8;
        double gflops = totalFLOPs / t / 1e9;
        printf("  FP32: %.3f ms, %.1f GFLOPS\n", t * 1000, gflops);
    }
    printf("  -> 2:1 ratio means fp16 runs at double rate (packed). 1:1 means same pipe.\n");
}

// ============================================================================
// Probe: Cache Line Size
// ============================================================================
static void probeCacheLineSize() {
    printf("\n=== Cache Line Size ===\n");
    id<MTLComputePipelineState> pso = makePSO(@"probe_cacheline");

    // Small working set that fits in L1 (~4 KB)
    // Build pointer chase with increasing byte strides
    // When stride >= cache line size, each hop is a new cache line fetch
    // so latency per hop jumps

    uint32_t hops = 100000;

    printf("  %10s  %10s  %10s\n", "Stride(B)", "Time (µs)", "Lat (ns)");
    printf("  %10s  %10s  %10s\n", "----------", "----------", "--------");

    uint32_t strides[] = {4, 8, 16, 32, 64, 128, 256, 512};
    for (int s = 0; s < 8; s++) {
        uint32_t strideBytes = strides[s];
        uint32_t strideElems = strideBytes / sizeof(uint32_t); // in uint32_t units
        // Keep working set at 4 KB to stay in L1
        uint32_t wsBytes = 4096;
        uint32_t n = wsBytes / sizeof(uint32_t);

        // Build sequential chain with given stride
        std::vector<uint32_t> buf(n, 0);
        uint32_t pos = 0;
        for (uint32_t i = 0; i < n; i++) {
            uint32_t next = (pos + strideElems) % n;
            buf[pos] = next;
            pos = next;
            if (pos == 0) break; // completed the cycle
        }

        id<MTLBuffer> chainBuf = makeBufferWithData(buf.data(), n * sizeof(uint32_t));
        id<MTLBuffer> outBuf = makeBuffer(sizeof(uint32_t));
        id<MTLBuffer> hopsBuf = makeBufferWithData(&hops, sizeof(uint32_t));

        double t = timedDispatch(pso, ^(id<MTLComputeCommandEncoder> enc) {
            [enc setBuffer:chainBuf offset:0 atIndex:0];
            [enc setBuffer:outBuf offset:0 atIndex:1];
            [enc setBuffer:hopsBuf offset:0 atIndex:2];
        }, MTLSizeMake(1, 1, 1), MTLSizeMake(1, 1, 1), 3, 10);

        double latNs = (t / hops) * 1e9;
        printf("  %10u  %10.1f  %10.1f\n", strideBytes, t * 1e6, latNs);
    }
    printf("  -> Latency jump when stride crosses cache line boundary reveals line size.\n");
}

// ============================================================================
// Probe: Memory Coalescing Penalty
// ============================================================================
static void probeCoalescing() {
    printf("\n=== Memory Coalescing (stride penalty) ===\n");
    id<MTLComputePipelineState> pso = makePSO(@"probe_coalescing");

    uint32_t iterations = 50000;
    NSUInteger threads = 4096;
    NSUInteger tgSize = 256;

    size_t bufBytes = threads * 512 * sizeof(float); // large enough for max stride
    id<MTLBuffer> data = makeBuffer(bufBytes);
    float* p = (float*)data.contents;
    for (size_t i = 0; i < bufBytes / sizeof(float); i++) p[i] = 1.0f;

    id<MTLBuffer> out = makeBuffer(threads * sizeof(float));
    id<MTLBuffer> iterBuf = makeBufferWithData(&iterations, sizeof(uint32_t));

    printf("  %8s  %10s  %10s\n", "Stride", "Time (ms)", "BW (GB/s)");
    printf("  %8s  %10s  %10s\n", "--------", "----------", "----------");

    uint32_t stridesArr[] = {1, 2, 4, 8, 16, 32, 64, 128, 256};
    for (int s = 0; s < 9; s++) {
        uint32_t stride = stridesArr[s];
        id<MTLBuffer> strideBuf = makeBufferWithData(&stride, sizeof(uint32_t));

        double t = timedDispatch(pso, ^(id<MTLComputeCommandEncoder> enc) {
            [enc setBuffer:data offset:0 atIndex:0];
            [enc setBuffer:out offset:0 atIndex:1];
            [enc setBuffer:strideBuf offset:0 atIndex:2];
            [enc setBuffer:iterBuf offset:0 atIndex:3];
        }, MTLSizeMake(threads, 1, 1), MTLSizeMake(tgSize, 1, 1), 2, 5);

        double totalBytes = (double)threads * iterations * sizeof(float);
        double gbps = totalBytes / t / 1e9;
        printf("  %8u  %10.3f  %10.1f\n", stride, t * 1000, gbps);
    }
    printf("  -> Bandwidth degrades with stride. Ratio of stride-1 to stride-N = coalescing factor.\n");
}

// ============================================================================
// Probe: Threadgroup Barrier Cost
// ============================================================================
static void probeBarrierCost() {
    printf("\n=== Threadgroup Barrier Cost ===\n");
    id<MTLComputePipelineState> pso = makePSO(@"probe_barrier_cost");

    uint32_t iterations = 500000;
    id<MTLBuffer> iterBuf = makeBufferWithData(&iterations, sizeof(uint32_t));

    printf("  %8s  %10s  %10s\n", "TGSize", "Time (ms)", "ns/barrier");
    printf("  %8s  %10s  %10s\n", "--------", "----------", "----------");

    NSUInteger tgSizes[] = {32, 64, 128, 256, 512, 1024};
    for (int s = 0; s < 6; s++) {
        NSUInteger tgSize = tgSizes[s];
        NSUInteger totalThreads = 4096;
        if (totalThreads < tgSize) totalThreads = tgSize;
        id<MTLBuffer> out = makeBuffer(totalThreads * sizeof(float));

        double t = timedDispatch(pso, ^(id<MTLComputeCommandEncoder> enc) {
            [enc setBuffer:out offset:0 atIndex:0];
            [enc setBuffer:iterBuf offset:0 atIndex:1];
        }, MTLSizeMake(totalThreads, 1, 1), MTLSizeMake(tgSize, 1, 1), 2, 5);

        double nsPerBarrier = (t * 1e9) / iterations;
        printf("  %8lu  %10.3f  %10.2f\n", (unsigned long)tgSize, t * 1000, nsPerBarrier);
    }
    printf("  -> Cost scaling with TG size reveals sync hardware architecture.\n");
}

// ============================================================================
// Probe: Dispatch Overhead
// ============================================================================
static void probeDispatchOverhead() {
    printf("\n=== Dispatch Overhead (empty kernel) ===\n");
    id<MTLComputePipelineState> pso = makePSO(@"probe_dispatch_empty");

    printf("  %10s  %10s\n", "Threads", "Time (µs)");
    printf("  %10s  %10s\n", "----------", "----------");

    NSUInteger sizes[] = {1, 32, 256, 1024, 4096, 32768};
    for (int s = 0; s < 6; s++) {
        NSUInteger threads = sizes[s];
        NSUInteger tgSize = threads < 256 ? threads : 256;
        id<MTLBuffer> out = makeBuffer(threads * sizeof(float));

        double t = timedDispatch(pso, ^(id<MTLComputeCommandEncoder> enc) {
            [enc setBuffer:out offset:0 atIndex:0];
        }, MTLSizeMake(threads, 1, 1), MTLSizeMake(tgSize, 1, 1), 5, 20);

        printf("  %10lu  %10.2f\n", (unsigned long)threads, t * 1e6);
    }
    printf("  -> Base overhead = cost at 1 thread. Growth = per-threadgroup scheduling cost.\n");
}

// ============================================================================
// Probe: Texture Read Bandwidth
// ============================================================================
static void probeTextureBandwidth() {
    printf("\n=== Texture Read Bandwidth vs Buffer Read ===\n");

    // Create a 2048x2048 RGBA float texture
    uint32_t w = 2048, h = 2048;
    MTLTextureDescriptor* desc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA32Float
                                                                                    width:w height:h mipmapped:NO];
    desc.usage = MTLTextureUsageShaderRead;
    desc.storageMode = MTLStorageModeShared;
    id<MTLTexture> tex = [gDevice newTextureWithDescriptor:desc];

    // Fill with data
    size_t bytesPerRow = w * 4 * sizeof(float);
    std::vector<float> texData(w * h * 4, 1.0f);
    [tex replaceRegion:MTLRegionMake2D(0, 0, w, h) mipmapLevel:0
           withBytes:texData.data() bytesPerRow:bytesPerRow];

    id<MTLComputePipelineState> pso = makePSO(@"probe_texture_read");
    uint32_t iterations = 1000;
    id<MTLBuffer> out = makeBuffer(w * h * sizeof(float));
    id<MTLBuffer> iterBuf = makeBufferWithData(&iterations, sizeof(uint32_t));

    double t = timedDispatch(pso, ^(id<MTLComputeCommandEncoder> enc) {
        [enc setTexture:tex atIndex:0];
        [enc setBuffer:out offset:0 atIndex:0];
        [enc setBuffer:iterBuf offset:0 atIndex:1];
    }, MTLSizeMake(w, h, 1), MTLSizeMake(16, 16, 1), 2, 5);

    // Each thread reads iterations * 16 bytes (RGBA float)
    double totalBytes = (double)w * h * iterations * 16;
    double gbps = totalBytes / t / 1e9;
    printf("  Texture: %ux%u RGBA32Float\n", w, h);
    printf("  GPU time:   %.3f ms\n", t * 1000);
    printf("  Texture BW: %.1f GB/s\n", gbps);

    // Compare with buffer read of equivalent size
    size_t bufBytes = (size_t)w * h * 16;
    id<MTLBuffer> bufData = makeBuffer(bufBytes);
    float* bp = (float*)bufData.contents;
    for (size_t i = 0; i < bufBytes / sizeof(float); i++) bp[i] = 1.0f;

    id<MTLComputePipelineState> pso2 = makePSO(@"probe_cache_sweep");
    uint32_t wsElements = (uint32_t)(bufBytes / sizeof(float));
    // Round to power of 2
    uint32_t po2 = 1;
    while (po2 < wsElements) po2 <<= 1;
    wsElements = po2;
    uint32_t sweepIters = iterations;
    id<MTLBuffer> wsBuf = makeBufferWithData(&wsElements, sizeof(uint32_t));
    id<MTLBuffer> siIterBuf = makeBufferWithData(&sweepIters, sizeof(uint32_t));
    id<MTLBuffer> bufOut = makeBuffer(w * h * sizeof(float));

    double t2 = timedDispatch(pso2, ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:bufData offset:0 atIndex:0];
        [enc setBuffer:bufOut offset:0 atIndex:1];
        [enc setBuffer:wsBuf offset:0 atIndex:2];
        [enc setBuffer:siIterBuf offset:0 atIndex:3];
    }, MTLSizeMake(w * h, 1, 1), MTLSizeMake(256, 1, 1), 2, 5);

    double totalBytes2 = (double)w * h * sweepIters * sizeof(float);
    double gbps2 = totalBytes2 / t2 / 1e9;
    printf("  Buffer BW:  %.1f GB/s (equivalent working set)\n", gbps2);
    printf("  -> Texture/Buffer ratio shows texture cache path overhead or benefit.\n");
}

// ============================================================================
// Binary Inspection Hint (register file via metal-objdump)
// ============================================================================
static void printDisassemblyHint() {
    printf("\n=== Register File / Spill Verification via Disassembly ===\n");
    printf("  To inspect compiler output for spill instructions:\n");
    printf("    1. Build the metallib:\n");
    printf("       xcrun -sdk macosx metal -o Shaders.air Shaders.metal\n");
    printf("       xcrun -sdk macosx metallib -o Shaders.metallib Shaders.air\n");
    printf("    2. Disassemble with metal-objdump (Xcode 15+):\n");
    printf("       xcrun metal-objdump -d Shaders.metallib | grep -E 'spill|SPILL|stack'\n");
    printf("    3. Compare spill instructions across regpressure variants\n");
    printf("       to confirm which accumulator count triggers register exhaustion.\n");
}

// ============================================================================
// Main
// ============================================================================
int main(int argc, const char* argv[]) {
    @autoreleasepool {
        gDevice = MTLCreateSystemDefaultDevice();
        if (!gDevice) {
            fprintf(stderr, "ERROR: No Metal device found\n");
            return 1;
        }
        gQueue = [gDevice newCommandQueue];

        // Compile shaders from source
        NSString* shaderPath = [[NSBundle mainBundle] pathForResource:@"Shaders" ofType:@"metallib"];
        NSError* err = nil;

        if (shaderPath) {
            NSURL* url = [NSURL fileURLWithPath:shaderPath];
            gLibrary = [gDevice newLibraryWithURL:url error:&err];
        }

        if (!gLibrary) {
            // Try loading from metallib in current directory
            NSString* cwd = [[NSFileManager defaultManager] currentDirectoryPath];
            NSString* libPath = [cwd stringByAppendingPathComponent:@"Shaders.metallib"];
            NSURL* url = [NSURL fileURLWithPath:libPath];
            gLibrary = [gDevice newLibraryWithURL:url error:&err];
        }

        if (!gLibrary) {
            // Compile from source at runtime
            NSString* cwd = [[NSFileManager defaultManager] currentDirectoryPath];
            NSString* srcPath = [cwd stringByAppendingPathComponent:@"Shaders.metal"];
            NSString* src = [NSString stringWithContentsOfFile:srcPath encoding:NSUTF8StringEncoding error:&err];
            if (!src) {
                fprintf(stderr, "ERROR: Cannot read Shaders.metal: %s\n", err.localizedDescription.UTF8String);
                return 1;
            }
            MTLCompileOptions* opts = [[MTLCompileOptions alloc] init];
            opts.languageVersion = MTLLanguageVersion3_0;
            gLibrary = [gDevice newLibraryWithSource:src options:opts error:&err];
            if (!gLibrary) {
                fprintf(stderr, "ERROR: Shader compilation failed: %s\n", err.localizedDescription.UTF8String);
                return 1;
            }
        }

        printf("M4 GPU Probe Suite\n");
        printf("==================\n");

        probeDirectQueries();
        probeTimerResolution();
        probeDRAMBandwidth();
        probeTGMemBandwidth();
        probeCacheSizes();
        probeCacheLatency();
        probeRegisterSpill();
        probeShaderCoreCount();
        probeBankConflicts();
        probeSLCAssociativity();
        probePipelineDepth();
        probeOccupancy();

        // New probes
        probeAtomicThroughput();
        probeSimdShuffle();
        probeSimdReduce();
        probeIntVsFloat();
        probeHalfPrecision();
        probeCacheLineSize();
        probeCoalescing();
        probeBarrierCost();
        probeDispatchOverhead();
        probeTextureBandwidth();

        printDisassemblyHint();

        printf("\n=== All probes complete ===\n");
    }
    return 0;
}
