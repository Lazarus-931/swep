#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>

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
    [e dispatchThreadgroups:g threadsPerThreadgroup:tg];
    [e endEncoding]; [cb commit]; [cb waitUntilCompleted];
    return cb.GPUEndTime - cb.GPUStartTime;
}

static double median_run(id<MTLComputePipelineState> p, void(^enc)(id<MTLComputeCommandEncoder>), MTLSize g, MTLSize tg, int w, int n) {
    for (int i = 0; i < w; i++) run(p, enc, g, tg);
    std::vector<double> t; for (int i = 0; i < n; i++) t.push_back(run(p, enc, g, tg));
    std::sort(t.begin(), t.end()); return t[n/2];
}

struct Config {
    const char* name;
    NSString* kernel;
    MTLSize tgSize;
    int tileM, tileN;
};

static FILE* csv;

static void sweep(Config cfg, id<MTLBuffer> mBuf, id<MTLBuffer> nBuf, id<MTLBuffer> kBuf,
                  uint32_t M, uint32_t N, uint32_t K,
                  id<MTLBuffer> A, id<MTLBuffer> B, id<MTLBuffer> C) {
    auto p = pso(cfg.kernel);

    uint32_t gridX = (N + cfg.tileN - 1) / cfg.tileN;
    uint32_t gridY = (M + cfg.tileM - 1) / cfg.tileM;

    double t = median_run(p, ^(id<MTLComputeCommandEncoder> e) {
        [e setBuffer:A offset:0 atIndex:0]; [e setBuffer:B offset:0 atIndex:1];
        [e setBuffer:C offset:0 atIndex:2]; [e setBuffer:mBuf offset:0 atIndex:3];
        [e setBuffer:nBuf offset:0 atIndex:4]; [e setBuffer:kBuf offset:0 atIndex:5];
    }, MTLSizeMake(gridX, gridY, 1), cfg.tgSize, 2, 5);

    double flops = 2.0 * M * N * K;
    double gflops = flops / t / 1e9;
    double totalBytes = (double)(M*K + K*N + M*N) * sizeof(float);
    double bw = totalBytes / t / 1e9;
    double ai = flops / totalBytes;
    double peakGflops = 5170.0;
    double pct = gflops / peakGflops * 100;

    printf("  %-14s  %5u %5u %5u  %8.2f  %8.1f  %6.1f  %5.1f%%\n",
           cfg.name, M, N, K, t*1000, gflops, ai, pct);
    fprintf(csv, "%s,%u,%u,%u,%.4f,%.2f,%.2f,%.1f,%.1f\n",
            cfg.name, M, N, K, t*1000, gflops, bw, ai, pct);
}

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

        csv = fopen("gemm_results.csv", "w");
        fprintf(csv, "kernel,M,N,K,time_ms,gflops,bw_gbps,arith_intensity,pct_peak\n");

        printf("GEMM Sweep — %s\n", dev.name.UTF8String);
        printf("============================================\n");

        Config configs[] = {
            {"naive",     @"gemm_naive",      MTLSizeMake(16, 16, 1),  16, 16},
            {"simd",      @"gemm_simd",       MTLSizeMake(32, 1, 1),    8, 16},
            {"tiled_16a", @"gemm_tiled",      MTLSizeMake(256, 1, 1),  64, 64},
            {"tiled_24a", @"gemm_tiled_24acc", MTLSizeMake(256, 1, 1), 48, 64},
        };
        int nConfigs = 4;

        struct Shape { uint32_t M, N, K; const char* label; };
        Shape shapes[] = {
            // Square
            {128,   128,   128,   "small-sq"},
            {256,   256,   256,   "med-sq"},
            {512,   512,   512,   "mid-sq"},
            {1024,  1024,  1024,  "large-sq"},
            {2048,  2048,  2048,  "xl-sq"},
            {4096,  4096,  4096,  "xxl-sq"},
            // Tall-skinny (matvec, decode)
            {4096,  1,     4096,  "matvec"},
            {4096,  4,     4096,  "mv-batch4"},
            {4096,  32,    4096,  "mv-batch32"},
            // Fat K (attention-style)
            {64,    64,    4096,  "attn-small"},
            {128,   128,   4096,  "attn-med"},
            {256,   256,   4096,  "attn-large"},
            // Wide N (batched output)
            {1,     4096,  4096,  "wide-N"},
            {32,    4096,  4096,  "wide-N-32"},
            // LLM shapes (Llama-ish)
            {4096,  4096,  4096,  "llama-qkv"},
            {4096,  14336, 4096,  "llama-ffn-up"},
            {14336, 4096,  4096,  "llama-ffn-down"},
            {4096,  4096,  128,   "llama-attn-short"},
            // Odd sizes (non-power-of-2)
            {1000,  1000,  1000,  "odd-1k"},
            {3000,  3000,  3000,  "odd-3k"},
            {768,   768,   768,   "bert-base"},
            {1024,  1024,  4096,  "long-k"},
        };
        int nShapes = sizeof(shapes) / sizeof(shapes[0]);

        size_t maxDim = 14336;
        size_t maxBytes = maxDim * maxDim * sizeof(float);
        auto A = buf(maxBytes);
        auto B = buf(maxBytes);
        auto C = buf(maxBytes);
        float* pa = (float*)A.contents;
        float* pb = (float*)B.contents;
        for (size_t i = 0; i < maxBytes/4; i++) { pa[i] = 0.01f; pb[i] = 0.01f; }

        for (int c = 0; c < nConfigs; c++) {
            printf("\n--- %s ---\n", configs[c].name);
            printf("  %-14s  %5s %5s %5s  %8s  %8s  %6s  %6s\n",
                   "kernel", "M", "N", "K", "ms", "GFLOPS", "AI", "peak%");

            for (int s = 0; s < nShapes; s++) {
                auto& sh = shapes[s];
                if ((size_t)sh.M * sh.K > maxDim * maxDim) continue;
                if ((size_t)sh.K * sh.N > maxDim * maxDim) continue;
                if ((size_t)sh.M * sh.N > maxDim * maxDim) continue;

                auto mB = cbuf(&sh.M, 4);
                auto nB = cbuf(&sh.N, 4);
                auto kB = cbuf(&sh.K, 4);
                sweep(configs[c], mB, nB, kB, sh.M, sh.N, sh.K, A, B, C);
            }
        }

        fclose(csv);
        printf("\nResults written to gemm_results.csv\n");
    }
    return 0;
}
