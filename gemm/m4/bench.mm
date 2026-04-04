#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>

static id<MTLDevice> dev;
static id<MTLCommandQueue> q;

static id<MTLComputePipelineState> compile(const char* file, const char* func) {
    NSError* e;
    NSString* cwd = [[NSFileManager defaultManager] currentDirectoryPath];
    auto src = [NSString stringWithContentsOfFile:[cwd stringByAppendingPathComponent:[NSString stringWithUTF8String:file]] encoding:NSUTF8StringEncoding error:&e];
    if (!src) { fprintf(stderr, "Can't read %s\n", file); exit(1); }
    auto opts = [[MTLCompileOptions alloc] init]; opts.languageVersion = MTLLanguageVersion3_0;
    auto lib = [dev newLibraryWithSource:src options:opts error:&e];
    if (!lib) { fprintf(stderr, "Compile %s: %s\n", file, e.localizedDescription.UTF8String); exit(1); }
    auto fn = [lib newFunctionWithName:[NSString stringWithUTF8String:func]];
    auto p = [dev newComputePipelineStateWithFunction:fn error:&e];
    return p;
}

static double run(id<MTLComputePipelineState> p, void(^enc)(id<MTLComputeCommandEncoder>), MTLSize g, MTLSize tg, bool tgroups) {
    auto cb = [q commandBuffer]; auto e = [cb computeCommandEncoder];
    [e setComputePipelineState:p]; enc(e);
    if (tgroups) [e dispatchThreadgroups:g threadsPerThreadgroup:tg];
    else [e dispatchThreads:g threadsPerThreadgroup:tg];
    [e endEncoding]; [cb commit]; [cb waitUntilCompleted];
    return cb.GPUEndTime - cb.GPUStartTime;
}

static double bench(id<MTLComputePipelineState> p, void(^enc)(id<MTLComputeCommandEncoder>), MTLSize g, MTLSize tg, bool tgroups) {
    for (int i = 0; i < 3; i++) run(p, enc, g, tg, tgroups);
    std::vector<double> t;
    for (int i = 0; i < 7; i++) t.push_back(run(p, enc, g, tg, tgroups));
    std::sort(t.begin(), t.end()); return t[3];
}

int main() {
    @autoreleasepool {
        dev = MTLCreateSystemDefaultDevice();
        if (!dev) { fprintf(stderr, "No Metal\n"); return 1; }
        q = [dev newCommandQueue];

        auto pNaive = compile("naive.metal", "gemm_naive");
        auto pBest = compile("best.metal", "gemm_m4");

        struct Shape { uint32_t M, N, K; };
        Shape shapes[] = {
            {256, 256, 64},   {256, 256, 128},  {256, 256, 256},
            {256, 256, 512},  {256, 256, 1024}, {256, 512, 128},
            {256, 512, 256},  {256, 512, 512},  {256, 512, 1024},
            {384, 384, 128},  {384, 384, 256},  {384, 384, 512},
            {512, 256, 128},  {512, 256, 256},  {512, 256, 512},
            {512, 512, 128},  {512, 512, 256},  {512, 512, 512},
            {512, 512, 1024}, {768, 768, 256},  {768, 768, 512},
            {1024, 512, 256}, {1024, 512, 512}, {1024, 1024, 128},
            {1024, 1024, 256},{1024, 1024, 512},{1024, 1024, 1024},
            {256, 1024, 256}, {512, 1024, 512}, {384, 768, 512},
        };
        int n = sizeof(shapes) / sizeof(shapes[0]);

        size_t maxBytes = 1024 * 1024 * sizeof(float);
        auto A = [dev newBufferWithLength:maxBytes options:MTLResourceStorageModeShared];
        auto B = [dev newBufferWithLength:maxBytes options:MTLResourceStorageModeShared];
        auto C = [dev newBufferWithLength:maxBytes options:MTLResourceStorageModeShared];
        float* pa = (float*)A.contents; float* pb = (float*)B.contents;
        for (size_t i = 0; i < maxBytes/4; i++) { pa[i] = 0.01f; pb[i] = 0.01f; }

        FILE* csv = fopen("results.csv", "w");
        fprintf(csv, "M,N,K,naive_ms,naive_gflops,best_ms,best_gflops,speedup\n");

        printf("best vs naive — %s\n", dev.name.UTF8String);
        printf("===========================================\n");
        printf("  %5s %5s %5s  %10s %10s  %10s %10s  %7s\n",
               "M", "N", "K", "naive ms", "naive GF", "best ms", "best GF", "speedup");

        for (int i = 0; i < n; i++) {
            auto& s = shapes[i];
            auto mB = [dev newBufferWithBytes:&s.M length:4 options:MTLResourceStorageModeShared];
            auto nB = [dev newBufferWithBytes:&s.N length:4 options:MTLResourceStorageModeShared];
            auto kB = [dev newBufferWithBytes:&s.K length:4 options:MTLResourceStorageModeShared];

            auto enc = ^(id<MTLComputeCommandEncoder> e) {
                [e setBuffer:A offset:0 atIndex:0]; [e setBuffer:B offset:0 atIndex:1];
                [e setBuffer:C offset:0 atIndex:2]; [e setBuffer:mB offset:0 atIndex:3];
                [e setBuffer:nB offset:0 atIndex:4]; [e setBuffer:kB offset:0 atIndex:5];
            };

            double tN = bench(pNaive, enc,
                MTLSizeMake(s.N, s.M, 1), MTLSizeMake(16, 16, 1), false);

            uint32_t gx = (s.N + 127) / 128;
            uint32_t gy = (s.M + 15) / 16;
            double tB = bench(pBest, enc,
                MTLSizeMake(gx, gy, 1), MTLSizeMake(128, 1, 1), true);

            double flops = 2.0 * s.M * s.N * s.K;
            double gfN = flops / tN / 1e9;
            double gfB = flops / tB / 1e9;
            double speedup = tN / tB;

            printf("  %5u %5u %5u  %10.3f %10.1f  %10.3f %10.1f  %6.1fx\n",
                   s.M, s.N, s.K, tN*1000, gfN, tB*1000, gfB, speedup);
            fprintf(csv, "%u,%u,%u,%.4f,%.2f,%.4f,%.2f,%.1f\n",
                    s.M, s.N, s.K, tN*1000, gfN, tB*1000, gfB, speedup);
        }

        fclose(csv);
        printf("\nResults written to results.csv\n");
    }
    return 0;
}
