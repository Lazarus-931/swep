#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>

static id<MTLDevice> dev;
static id<MTLCommandQueue> q;

struct Kernel { const char* name; const char* file; const char* func; int tileM, tileN; };

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

static double run(id<MTLComputePipelineState> p, void(^enc)(id<MTLComputeCommandEncoder>), MTLSize g, MTLSize tg) {
    auto cb = [q commandBuffer]; auto e = [cb computeCommandEncoder];
    [e setComputePipelineState:p]; enc(e);
    [e dispatchThreadgroups:g threadsPerThreadgroup:tg];
    [e endEncoding]; [cb commit]; [cb waitUntilCompleted];
    return cb.GPUEndTime - cb.GPUStartTime;
}

static double bench(id<MTLComputePipelineState> p, void(^enc)(id<MTLComputeCommandEncoder>), MTLSize g, MTLSize tg) {
    for (int i = 0; i < 3; i++) run(p, enc, g, tg);
    std::vector<double> t;
    for (int i = 0; i < 7; i++) t.push_back(run(p, enc, g, tg));
    std::sort(t.begin(), t.end()); return t[3];
}

int main() {
    @autoreleasepool {
        dev = MTLCreateSystemDefaultDevice();
        if (!dev) { fprintf(stderr, "No Metal\n"); return 1; }
        q = [dev newCommandQueue];

        Kernel kernels[] = {
            {"k16",  "config_a.metal", "gemm_k16",  32, 128},
            {"k32",  "config_b.metal", "gemm_k32",  32, 128},
            {"k48",  "config_c.metal", "gemm_k48",  32, 128},
        };

        struct Shape { uint32_t M, N, K; };
        Shape shapes[] = {
            {256, 256, 128}, {256, 256, 512}, {256, 256, 1024},
            {256, 512, 256}, {256, 512, 1024},
            {512, 512, 128}, {512, 512, 256}, {512, 512, 512},
            {384, 384, 256}, {384, 384, 512},
        };
        int nShapes = sizeof(shapes) / sizeof(shapes[0]);

        size_t maxBytes = 1024 * 1024 * sizeof(float);
        auto A = [dev newBufferWithLength:maxBytes options:MTLResourceStorageModeShared];
        auto B = [dev newBufferWithLength:maxBytes options:MTLResourceStorageModeShared];
        auto C = [dev newBufferWithLength:maxBytes options:MTLResourceStorageModeShared];
        float* pa = (float*)A.contents; float* pb = (float*)B.contents;
        for (size_t i = 0; i < maxBytes/4; i++) { pa[i] = 0.01f; pb[i] = 0.01f; }

        FILE* csv = fopen("results.csv", "w");
        fprintf(csv, "config,M,N,K,time_ms,gflops,pct_peak\n");

        printf("gemm_2: K-block depth sweep (32x128 wide tile) — %s\n", dev.name.UTF8String);
        printf("=============================================\n\n");

        for (auto& k : kernels) {
            auto p = compile(k.file, k.func);
            printf("--- %s ---\n", k.name);
            printf("  %5s %5s %5s  %8s  %8s  %6s\n", "M", "N", "K", "ms", "GFLOPS", "peak%");

            for (int s = 0; s < nShapes; s++) {
                auto& sh = shapes[s];
                auto mB = [dev newBufferWithBytes:&sh.M length:4 options:MTLResourceStorageModeShared];
                auto nB = [dev newBufferWithBytes:&sh.N length:4 options:MTLResourceStorageModeShared];
                auto kB = [dev newBufferWithBytes:&sh.K length:4 options:MTLResourceStorageModeShared];

                uint32_t gx = (sh.N + k.tileN - 1) / k.tileN;
                uint32_t gy = (sh.M + k.tileM - 1) / k.tileM;

                double t = bench(p, ^(id<MTLComputeCommandEncoder> e) {
                    [e setBuffer:A offset:0 atIndex:0]; [e setBuffer:B offset:0 atIndex:1];
                    [e setBuffer:C offset:0 atIndex:2]; [e setBuffer:mB offset:0 atIndex:3];
                    [e setBuffer:nB offset:0 atIndex:4]; [e setBuffer:kB offset:0 atIndex:5];
                }, MTLSizeMake(gx, gy, 1), MTLSizeMake(256, 1, 1));

                double flops = 2.0 * sh.M * sh.N * sh.K;
                double gflops = flops / t / 1e9;
                double pct = gflops / 5170.0 * 100;
                printf("  %5u %5u %5u  %8.3f  %8.1f  %5.1f%%\n", sh.M, sh.N, sh.K, t*1000, gflops, pct);
                fprintf(csv, "%s,%u,%u,%u,%.4f,%.2f,%.1f\n", k.name, sh.M, sh.N, sh.K, t*1000, gflops, pct);
            }
            printf("\n");
        }

        fclose(csv);
        printf("Results written to results.csv\n");
    }
    return 0;
}
