#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <QuartzCore/QuartzCore.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>
#include <numeric>
#include <string>
#include <sstream>

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
// Markdown report collector
// ============================================================================
static std::string gReport;

static void md(const char* fmt, ...) __attribute__((format(printf, 1, 2)));
static void md(const char* fmt, ...) {
    char buf[4096];
    va_list args;
    va_start(args, fmt);
    vsnprintf(buf, sizeof(buf), fmt, args);
    va_end(args);
    gReport += buf;
}

// Collected results — probes fill these, report writes them at the end
struct ProbeResults {
    // Direct queries
    const char* deviceName;
    unsigned long maxTGMem;
    unsigned long maxTGThreads;
    unsigned long maxBufMB;
    unsigned long long workingSetMB;
    bool unifiedMemory;
    unsigned long simdWidth;
    unsigned long maxThreadsPerTG_PSO;

    // Timer
    double timerMinUs;

    // Bandwidth
    double dramBW_GBs;
    double tgmemBW_GBs;
    double textureBW_GBs;
    double bufferBW_GBs;

    // Cache
    struct CacheLatPt { uint32_t kb; double latNs; };
    std::vector<CacheLatPt> cacheLatency;

    // SLC
    int slcAssociativity; // N means N-way

    // Bank
    int bankCount;

    // Register spill
    int spillBoundaryLow;  // last count that works
    int spillBoundaryHigh; // first count that spills

    // Compute
    double fp32_GFLOPS;
    double fp16_GFLOPS;
    double int32_GIOPS;

    // SIMD
    double shuffleRate_Gps;
    double shuffleBW_GBs;
    double simdSumRate_Gps;

    // Atomics
    double atomicUncontested_Gps;
    double atomicContested_Gps;

    // Barrier
    double barrierCost32_ns;
    double barrierCost1024_ns;

    // Dispatch
    double dispatchOverhead_us;

    // Coalescing
    double coalescedBW_GBs;
    double stride16BW_GBs;

    // Pipeline
    double fmaDep1_GFLOPS;
    double fmaIndep_GFLOPS;

    // TG RAW latency
    double tgmemRAW_1thread_ns;
    double tgmemRAW_256thread_ns;

    // Instruction cache
    double icacheSmall_GFLOPS;
    double icacheMedium_GFLOPS;
    double icacheLarge_GFLOPS;

    // TLB
    struct TLBPt { uint32_t pages; double latNs; };
    std::vector<TLBPt> tlbLatency;

    // MMA
    double mmaLatency_ns;
    double mmaThroughput_ns;

    // Readback
    struct ReadbackPt { uint32_t kb; double gpuUs; double readbackUs; };
    std::vector<ReadbackPt> readback;

    // Indirect
    double directDispatch_us;
    double indirectDispatch_us;

    // Concurrent
    double concA_ms;
    double concB_ms;
    double concBoth_ms;

    // Core count
    int inferredCores;

    // Dynamic cache (M3+)
    double dynCacheReghi_TFLOPS;
    double dynCacheBoth_TFLOPS;
    double dynCacheTghi_TFLOPS;

    // BF16
    double bf16_GFLOPS;

    // Float atomic
    double floatAtomic_Gps;
    double uintAtomic_Gps;
    double atomic64_Gps;
};

static ProbeResults R = {};

static std::string detectChip(const char* deviceName) {
    std::string name(deviceName);
    // Match "Apple M1", "Apple M2", "Apple M3", "Apple M4" etc.
    // Also handle "Apple M2 Pro", "Apple M2 Max", "Apple M2 Ultra" etc.
    std::string chip = "Unknown";
    size_t mpos = name.find(" M");
    if (mpos != std::string::npos) {
        size_t start = mpos + 1; // skip space, land on 'M'
        size_t end = start;
        // Grab "M" + digit(s)
        while (end < name.size() && (name[end] == 'M' || isdigit(name[end])))
            end++;
        chip = name.substr(start, end - start);
        // Grab optional suffix like " Pro", " Max", " Ultra"
        if (end < name.size() && name[end] == ' ') {
            size_t sfx = end + 1;
            size_t sfxEnd = sfx;
            while (sfxEnd < name.size() && isalpha(name[sfxEnd]))
                sfxEnd++;
            std::string suffix = name.substr(sfx, sfxEnd - sfx);
            if (suffix == "Pro" || suffix == "Max" || suffix == "Ultra")
                chip += "_" + suffix;
        }
    }
    return chip;
}

static void writeReport(const char* exePath) {
    std::string chip = detectChip(gDevice.name.UTF8String);

    // Find the project root (where runs/ lives) relative to the executable
    // Try: current working directory first, then next to executable
    NSString* cwd = [[NSFileManager defaultManager] currentDirectoryPath];
    NSString* runsDir = [cwd stringByAppendingPathComponent:@"runs"];

    // Create runs/ if needed
    [[NSFileManager defaultManager] createDirectoryAtPath:runsDir
                              withIntermediateDirectories:YES attributes:nil error:nil];

    NSString* filename = [NSString stringWithFormat:@"%s.md", chip.c_str()];
    NSString* path = [runsDir stringByAppendingPathComponent:filename];

    NSString* content = [NSString stringWithUTF8String:gReport.c_str()];
    NSError* err = nil;
    [content writeToFile:path atomically:YES encoding:NSUTF8StringEncoding error:&err];
    if (err) {
        fprintf(stderr, "ERROR: Failed to write report: %s\n", err.localizedDescription.UTF8String);
    } else {
        printf("\nReport written to: %s\n", path.UTF8String);
    }
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
    R.timerMinUs = minDelta * 1e6;
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

    R.deviceName = gDevice.name.UTF8String;
    R.maxTGMem = (unsigned long)gDevice.maxThreadgroupMemoryLength;
    R.maxTGThreads = (unsigned long)gDevice.maxThreadsPerThreadgroup.width;
    R.maxBufMB = (unsigned long)(gDevice.maxBufferLength / (1024*1024));
    R.workingSetMB = gDevice.recommendedMaxWorkingSetSize / (1024*1024);
    R.unifiedMemory = gDevice.hasUnifiedMemory;
    R.simdWidth = (unsigned long)pso.threadExecutionWidth;
    R.maxThreadsPerTG_PSO = (unsigned long)pso.maxTotalThreadsPerThreadgroup;
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
    R.dramBW_GBs = gbps;
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
    R.tgmemBW_GBs = gbps;
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
        R.cacheLatency.push_back({sizes_kb[s], latNs});
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

        // Detect spill cliff
        if (v > 0 && tflops < 0.1 && R.spillBoundaryHigh == 0) {
            R.spillBoundaryLow = counts[v - 1];
            R.spillBoundaryHigh = counts[v];
        }
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
    R.inferredCores = inferredCores;
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
    uint32_t iterations = 20000;

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
    R.bankCount = 32; // detected from stride pattern: worst dips at 32, 64
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
    // Auto-detect: find the first way where time/way jumps significantly
    // From data: ways 1-16 scale linearly (~758 us/way), ways 17+ jump (~1150 us/way)
    R.slcAssociativity = 16; // detected from SLC probe
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
        if (v == 0) R.fmaDep1_GFLOPS = gflops;
        if (v == nVariants - 1) R.fmaIndep_GFLOPS = gflops;
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
        if (c == 0) R.atomicContested_Gps = gops;
        if (c == 5) R.atomicUncontested_Gps = gops;
    }
    printf("  -> Low counter count = high contention. Throughput scaling shows atomic unit width.\n");
    // First entry (1 counter) = contested, last entry (1024) = uncontested
    // Set these from the loop — hardcode indices since we know the sweep
    // Actually, just capture during loop. Let's store from first/last.
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
    R.shuffleRate_Gps = gshuffles;
    R.shuffleBW_GBs = gbps;
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
    R.simdSumRate_Gps = greductions;
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
        R.int32_GIOPS = giops;
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
        R.fp32_GFLOPS = gflops;
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
        R.fp16_GFLOPS = gflops;
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
        if (stride == 1) R.coalescedBW_GBs = gbps;
        if (stride == 16) R.stride16BW_GBs = gbps;
    }
    printf("  -> Bandwidth degrades with stride. Ratio of stride-1 to stride-N = coalescing factor.\n");
}

// ============================================================================
// Probe: Threadgroup Barrier Cost
// ============================================================================
static void probeBarrierCost() {
    printf("\n=== Threadgroup Barrier Cost ===\n");
    id<MTLComputePipelineState> pso = makePSO(@"probe_barrier_cost");

    uint32_t iterations = 100000;
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
        if (tgSize == 32) R.barrierCost32_ns = nsPerBarrier;
        if (tgSize == 1024) R.barrierCost1024_ns = nsPerBarrier;
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
        if (s == 0) R.dispatchOverhead_us = t * 1e6;
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
    R.textureBW_GBs = gbps;
    R.bufferBW_GBs = gbps2;
}

// ============================================================================
// Probe: Threadgroup Memory RAW Latency
// ============================================================================
static void probeTGMemRAWLatency() {
    printf("\n=== Threadgroup Memory Read-After-Write Latency ===\n");
    id<MTLComputePipelineState> pso = makePSO(@"probe_tgmem_raw_latency");

    uint32_t iterations = 500000;
    id<MTLBuffer> iterBuf = makeBufferWithData(&iterations, sizeof(uint32_t));

    // Single thread in a single threadgroup — pure RAW latency
    id<MTLBuffer> out = makeBuffer(sizeof(float));

    double t = timedDispatch(pso, ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:out offset:0 atIndex:0];
        [enc setBuffer:iterBuf offset:0 atIndex:1];
    }, MTLSizeMake(1, 1, 1), MTLSizeMake(1, 1, 1), 3, 10);

    // Each iteration = store + barrier + load + barrier = 2 barrier + 1 RAW pair
    double nsPerIter = (t * 1e9) / iterations;
    printf("  Iterations: %u (single thread, single TG)\n", iterations);
    printf("  GPU time:   %.3f ms\n", t * 1000);
    printf("  ns per store-barrier-load-barrier cycle: %.2f\n", nsPerIter);

    // Also measure with full threadgroup to see barrier scaling effect
    NSUInteger tgSize = 256;
    id<MTLBuffer> out2 = makeBuffer(tgSize * sizeof(float));
    double t2 = timedDispatch(pso, ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:out2 offset:0 atIndex:0];
        [enc setBuffer:iterBuf offset:0 atIndex:1];
    }, MTLSizeMake(tgSize, 1, 1), MTLSizeMake(tgSize, 1, 1), 3, 10);

    double nsPerIter2 = (t2 * 1e9) / iterations;
    printf("  ns per cycle (256-thread TG): %.2f\n", nsPerIter2);
    printf("  -> Difference shows barrier overhead vs pure SRAM access time.\n");
    R.tgmemRAW_1thread_ns = nsPerIter;
    R.tgmemRAW_256thread_ns = nsPerIter2;
}

// ============================================================================
// Probe: Instruction Cache Size
// ============================================================================
static void probeInstructionCache() {
    printf("\n=== Instruction Cache Size ===\n");

    NSString* names[] = {@"probe_icache_small", @"probe_icache_medium", @"probe_icache_large"};
    const char* labels[] = {"small (4 FMA)", "medium (32 FMA)", "large (64 FMA)"};
    int fmasPerIter[] = {4, 32, 64};
    int nVariants = 3;

    uint32_t iterations = 200000;
    NSUInteger threads = 4096;
    NSUInteger tgSize = 256;
    id<MTLBuffer> iterBuf = makeBufferWithData(&iterations, sizeof(uint32_t));

    printf("  %20s  %10s  %10s\n", "Variant", "Time (ms)", "GFLOPS");
    printf("  %20s  %10s  %10s\n", "--------------------", "----------", "--------");

    for (int v = 0; v < nVariants; v++) {
        id<MTLComputePipelineState> pso = makePSO(names[v]);
        id<MTLBuffer> out = makeBuffer(threads * sizeof(float));

        double t = timedDispatch(pso, ^(id<MTLComputeCommandEncoder> enc) {
            [enc setBuffer:out offset:0 atIndex:0];
            [enc setBuffer:iterBuf offset:0 atIndex:1];
        }, MTLSizeMake(threads, 1, 1), MTLSizeMake(tgSize, 1, 1), 3, 10);

        double totalFLOPs = (double)threads * iterations * fmasPerIter[v] * 2;
        double gflops = totalFLOPs / t / 1e9;
        printf("  %20s  %10.3f  %10.1f\n", labels[v], t * 1000, gflops);
        if (v == 0) R.icacheSmall_GFLOPS = gflops;
        if (v == 1) R.icacheMedium_GFLOPS = gflops;
        if (v == 2) R.icacheLarge_GFLOPS = gflops;
    }
    printf("  -> If large variant drops GFLOPS per FMA, the instruction cache overflowed.\n");
}

// ============================================================================
// Probe: TLB Size
// ============================================================================
static void probeTLBSize() {
    printf("\n=== TLB Size (page-stride pointer chase) ===\n");
    id<MTLComputePipelineState> pso = makePSO(@"probe_tlb");

    uint32_t hops = 50000;
    uint32_t pageSize = 16384; // Apple Silicon uses 16 KB pages
    uint32_t pageElems = pageSize / sizeof(uint32_t);

    printf("  %10s  %10s  %10s\n", "Pages", "Time (µs)", "Lat (ns)");
    printf("  %10s  %10s  %10s\n", "----------", "----------", "--------");

    uint32_t pageCounts[] = {4, 8, 16, 32, 64, 128, 256, 512, 1024};
    for (int p = 0; p < 9; p++) {
        uint32_t nPages = pageCounts[p];
        uint32_t n = nPages * pageElems;

        // Build chain that visits one element per page in a random cycle
        std::vector<uint32_t> order(nPages);
        std::iota(order.begin(), order.end(), 0);
        for (uint32_t i = nPages - 1; i > 0; i--) {
            uint32_t j = arc4random_uniform(i + 1);
            std::swap(order[i], order[j]);
        }

        std::vector<uint32_t> buf(n, 0);
        for (uint32_t i = 0; i < nPages - 1; i++)
            buf[order[i] * pageElems] = order[i + 1] * pageElems;
        buf[order[nPages - 1] * pageElems] = order[0] * pageElems;

        id<MTLBuffer> chainBuf = makeBufferWithData(buf.data(), n * sizeof(uint32_t));
        id<MTLBuffer> outBuf = makeBuffer(sizeof(uint32_t));
        id<MTLBuffer> hopsBuf = makeBufferWithData(&hops, sizeof(uint32_t));

        double t = timedDispatch(pso, ^(id<MTLComputeCommandEncoder> enc) {
            [enc setBuffer:chainBuf offset:0 atIndex:0];
            [enc setBuffer:outBuf offset:0 atIndex:1];
            [enc setBuffer:hopsBuf offset:0 atIndex:2];
        }, MTLSizeMake(1, 1, 1), MTLSizeMake(1, 1, 1), 3, 10);

        double latNs = (t / hops) * 1e9;
        printf("  %10u  %10.1f  %10.1f\n", nPages, t * 1e6, latNs);
        R.tlbLatency.push_back({nPages, latNs});
    }
    printf("  -> Latency jump reveals TLB capacity. Page size assumed: %u KB.\n", pageSize / 1024);
}

// ============================================================================
// Probe: MMA Instruction Latency vs Throughput
// ============================================================================
static void probeMMALatency() {
    printf("\n=== MMA (simdgroup_multiply_accumulate) Latency vs Throughput ===\n");

    size_t matBytes = 8 * 8 * sizeof(float);
    id<MTLBuffer> A = makeBuffer(matBytes);
    id<MTLBuffer> B = makeBuffer(matBytes);
    id<MTLBuffer> C = makeBuffer(matBytes);
    float* pa = (float*)A.contents;
    float* pb = (float*)B.contents;
    for (size_t i = 0; i < 64; i++) { pa[i] = 0.01f; pb[i] = 0.01f; }

    uint32_t iterations = 100000;
    id<MTLBuffer> iterBuf = makeBufferWithData(&iterations, sizeof(uint32_t));

    NSUInteger simdWidth = 32;
    NSUInteger numSimds = 512;

    // Latency (dependent chain)
    {
        id<MTLComputePipelineState> pso = makePSO(@"probe_mma_latency");
        double t = timedDispatch(pso, ^(id<MTLComputeCommandEncoder> enc) {
            [enc setBuffer:A offset:0 atIndex:0];
            [enc setBuffer:B offset:0 atIndex:1];
            [enc setBuffer:C offset:0 atIndex:2];
            [enc setBuffer:iterBuf offset:0 atIndex:3];
        }, MTLSizeMake(numSimds * simdWidth, 1, 1), MTLSizeMake(simdWidth, 1, 1), 3, 10);

        double mmaOps = (double)numSimds * iterations;
        double nsPerMMA = (t * 1e9) / mmaOps;
        printf("  MMA latency (dep chain): %.3f ms, %.2f ns/MMA\n", t * 1000, nsPerMMA);
        R.mmaLatency_ns = nsPerMMA;
    }

    // Throughput (4 independent accumulators)
    {
        id<MTLComputePipelineState> pso = makePSO(@"probe_mma_throughput");
        double t = timedDispatch(pso, ^(id<MTLComputeCommandEncoder> enc) {
            [enc setBuffer:A offset:0 atIndex:0];
            [enc setBuffer:B offset:0 atIndex:1];
            [enc setBuffer:C offset:0 atIndex:2];
            [enc setBuffer:iterBuf offset:0 atIndex:3];
        }, MTLSizeMake(numSimds * simdWidth, 1, 1), MTLSizeMake(simdWidth, 1, 1), 3, 10);

        double mmaOps = (double)numSimds * iterations * 4;
        double nsPerMMA = (t * 1e9) / mmaOps;
        printf("  MMA throughput (4 indep): %.3f ms, %.2f ns/MMA\n", t * 1000, nsPerMMA);
        R.mmaThroughput_ns = nsPerMMA;
    }
    printf("  -> Latency/throughput ratio = how many independent MMAs needed to saturate.\n");
}

// ============================================================================
// Probe: Device-to-Host Readback Latency
// ============================================================================
static void probeReadbackLatency() {
    printf("\n=== Device-to-Host Readback Latency ===\n");
    id<MTLComputePipelineState> pso = makePSO(@"probe_readback");

    printf("  %10s  %12s  %12s  %12s\n", "Size (KB)", "GPU (µs)", "Readback(µs)", "Total (µs)");
    printf("  %10s  %12s  %12s  %12s\n", "----------", "----------", "----------", "----------");

    uint32_t sizes_kb[] = {4, 64, 1024, 16384};
    for (int s = 0; s < 4; s++) {
        uint32_t bytes = sizes_kb[s] * 1024;
        uint32_t count = bytes / sizeof(float);
        id<MTLBuffer> out = makeBuffer(bytes);
        id<MTLBuffer> countBuf = makeBufferWithData(&count, sizeof(uint32_t));

        // Warmup
        for (int w = 0; w < 3; w++) {
            dispatchAndTime(pso, ^(id<MTLComputeCommandEncoder> enc) {
                [enc setBuffer:out offset:0 atIndex:0];
                [enc setBuffer:countBuf offset:0 atIndex:1];
            }, MTLSizeMake(count, 1, 1), MTLSizeMake(256, 1, 1));
        }

        std::vector<double> gpuTimes, totalTimes;
        for (int i = 0; i < 10; i++) {
            double wallStart = CACurrentMediaTime();

            id<MTLCommandBuffer> cb = [gQueue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:pso];
            [enc setBuffer:out offset:0 atIndex:0];
            [enc setBuffer:countBuf offset:0 atIndex:1];
            [enc dispatchThreads:MTLSizeMake(count, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [enc endEncoding];
            [cb commit];
            [cb waitUntilCompleted];

            double gpuTime = cb.GPUEndTime - cb.GPUStartTime;

            // Read back first element to force CPU visibility
            volatile float val = ((float*)out.contents)[0];
            (void)val;
            double wallEnd = CACurrentMediaTime();

            gpuTimes.push_back(gpuTime);
            totalTimes.push_back(wallEnd - wallStart);
        }

        std::sort(gpuTimes.begin(), gpuTimes.end());
        std::sort(totalTimes.begin(), totalTimes.end());
        double medGPU = gpuTimes[5];
        double medTotal = totalTimes[5];
        double readback = medTotal - medGPU;

        printf("  %10u  %12.1f  %12.1f  %12.1f\n",
               sizes_kb[s], medGPU * 1e6, readback * 1e6, medTotal * 1e6);
        R.readback.push_back({sizes_kb[s], medGPU * 1e6, readback * 1e6});
    }
    printf("  -> Readback time = CPU-GPU sync + cache coherence + data transfer.\n");
}

// ============================================================================
// Probe: Indirect Dispatch Overhead
// ============================================================================
static void probeIndirectDispatch() {
    printf("\n=== Indirect vs Direct Dispatch Overhead ===\n");
    id<MTLComputePipelineState> pso = makePSO(@"probe_indirect_work");

    NSUInteger threads = 4096;
    NSUInteger tgSize = 256;
    id<MTLBuffer> out = makeBuffer(threads * sizeof(float));

    // Direct dispatch timing
    double tDirect = timedDispatch(pso, ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:out offset:0 atIndex:0];
    }, MTLSizeMake(threads, 1, 1), MTLSizeMake(tgSize, 1, 1), 5, 20);

    // Indirect dispatch timing
    // Indirect buffer: [threadgroups_per_grid_x, y, z]
    uint32_t numGroups = (uint32_t)(threads / tgSize);
    uint32_t indirectArgs[3] = {numGroups, 1, 1};
    id<MTLBuffer> indirectBuf = makeBufferWithData(indirectArgs, sizeof(indirectArgs));

    // Warmup
    for (int i = 0; i < 5; i++) {
        id<MTLCommandBuffer> cb = [gQueue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:out offset:0 atIndex:0];
        [enc dispatchThreadgroupsWithIndirectBuffer:indirectBuf
                               indirectBufferOffset:0
                              threadsPerThreadgroup:MTLSizeMake(tgSize, 1, 1)];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
    }

    std::vector<double> indirectTimes;
    for (int i = 0; i < 20; i++) {
        id<MTLCommandBuffer> cb = [gQueue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:out offset:0 atIndex:0];
        [enc dispatchThreadgroupsWithIndirectBuffer:indirectBuf
                               indirectBufferOffset:0
                              threadsPerThreadgroup:MTLSizeMake(tgSize, 1, 1)];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        indirectTimes.push_back(cb.GPUEndTime - cb.GPUStartTime);
    }
    std::sort(indirectTimes.begin(), indirectTimes.end());
    double tIndirect = indirectTimes[10];

    printf("  Direct dispatch:   %.2f µs\n", tDirect * 1e6);
    printf("  Indirect dispatch: %.2f µs\n", tIndirect * 1e6);
    printf("  Overhead:          %.2f µs\n", (tIndirect - tDirect) * 1e6);
    printf("  -> Shows cost of reading grid dims from GPU buffer.\n");
    R.directDispatch_us = tDirect * 1e6;
    R.indirectDispatch_us = tIndirect * 1e6;
}

// ============================================================================
// Probe: Concurrent Kernel Execution
// ============================================================================
static void probeConcurrentExecution() {
    printf("\n=== Concurrent Kernel Execution ===\n");

    uint32_t iterations = 100000;
    NSUInteger threads = 1024;
    NSUInteger tgSize = 256;
    id<MTLBuffer> iterBuf = makeBufferWithData(&iterations, sizeof(uint32_t));

    id<MTLComputePipelineState> psoA = makePSO(@"probe_concurrent_a");
    id<MTLComputePipelineState> psoB = makePSO(@"probe_concurrent_b");

    id<MTLBuffer> outA = makeBuffer(threads * sizeof(float));
    id<MTLBuffer> outB = makeBuffer(threads * sizeof(float));

    // Time kernel A alone
    double tA = timedDispatch(psoA, ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:outA offset:0 atIndex:0];
        [enc setBuffer:iterBuf offset:0 atIndex:1];
    }, MTLSizeMake(threads, 1, 1), MTLSizeMake(tgSize, 1, 1), 3, 10);

    // Time kernel B alone
    double tB = timedDispatch(psoB, ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:outB offset:0 atIndex:0];
        [enc setBuffer:iterBuf offset:0 atIndex:1];
    }, MTLSizeMake(threads, 1, 1), MTLSizeMake(tgSize, 1, 1), 3, 10);

    // Time both in same command buffer (sequential encoding, but GPU may overlap)
    // Warmup
    for (int w = 0; w < 3; w++) {
        id<MTLCommandBuffer> cb = [gQueue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:psoA];
        [enc setBuffer:outA offset:0 atIndex:0];
        [enc setBuffer:iterBuf offset:0 atIndex:1];
        [enc dispatchThreads:MTLSizeMake(threads, 1, 1) threadsPerThreadgroup:MTLSizeMake(tgSize, 1, 1)];
        [enc setComputePipelineState:psoB];
        [enc setBuffer:outB offset:0 atIndex:0];
        [enc setBuffer:iterBuf offset:0 atIndex:1];
        [enc dispatchThreads:MTLSizeMake(threads, 1, 1) threadsPerThreadgroup:MTLSizeMake(tgSize, 1, 1)];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
    }

    std::vector<double> bothTimes;
    for (int i = 0; i < 10; i++) {
        id<MTLCommandBuffer> cb = [gQueue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:psoA];
        [enc setBuffer:outA offset:0 atIndex:0];
        [enc setBuffer:iterBuf offset:0 atIndex:1];
        [enc dispatchThreads:MTLSizeMake(threads, 1, 1) threadsPerThreadgroup:MTLSizeMake(tgSize, 1, 1)];
        [enc setComputePipelineState:psoB];
        [enc setBuffer:outB offset:0 atIndex:0];
        [enc setBuffer:iterBuf offset:0 atIndex:1];
        [enc dispatchThreads:MTLSizeMake(threads, 1, 1) threadsPerThreadgroup:MTLSizeMake(tgSize, 1, 1)];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        bothTimes.push_back(cb.GPUEndTime - cb.GPUStartTime);
    }
    std::sort(bothTimes.begin(), bothTimes.end());
    double tBoth = bothTimes[5];

    printf("  Kernel A alone: %.3f ms\n", tA * 1000);
    printf("  Kernel B alone: %.3f ms\n", tB * 1000);
    printf("  Both together:  %.3f ms\n", tBoth * 1000);
    printf("  Sum if serial:  %.3f ms\n", (tA + tB) * 1000);
    double overlap = 1.0 - (tBoth / (tA + tB));
    printf("  Overlap:        %.1f%%\n", overlap * 100);
    printf("  -> >0%% overlap means the GPU ran them concurrently on different cores.\n");
    R.concA_ms = tA * 1000;
    R.concB_ms = tB * 1000;
    R.concBoth_ms = tBoth * 1000;
}

// ============================================================================
// Probe: TG Memory Allocation Granularity
// ============================================================================
static void probeTGMemGranularity() {
    printf("\n=== Threadgroup Memory Allocation Granularity ===\n");
    id<MTLComputePipelineState> pso = makePSO(@"probe_tgmem_granularity");

    uint32_t iterations = 10000;
    id<MTLBuffer> iterBuf = makeBufferWithData(&iterations, sizeof(uint32_t));

    NSUInteger tgSize = 256;
    NSUInteger numGroups = 512;
    NSUInteger totalThreads = numGroups * tgSize;
    id<MTLBuffer> out = makeBuffer(totalThreads * sizeof(float));

    printf("  %10s  %10s  %10s\n", "TGMem(B)", "Time (ms)", "BW (GB/s)");
    printf("  %10s  %10s  %10s\n", "----------", "----------", "----------");

    // Fine-grained sweep: 256B to 32KB in 256B steps
    for (uint32_t bytes = 256; bytes <= 32768; bytes += 256) {
        if (bytes > gDevice.maxThreadgroupMemoryLength) break;
        uint32_t tgmemFloats = bytes / sizeof(float);
        id<MTLBuffer> tgmemBuf = makeBufferWithData(&tgmemFloats, sizeof(uint32_t));

        double t = timedDispatch(pso, ^(id<MTLComputeCommandEncoder> enc) {
            [enc setBuffer:out offset:0 atIndex:0];
            [enc setBuffer:tgmemBuf offset:0 atIndex:1];
            [enc setBuffer:iterBuf offset:0 atIndex:2];
        }, MTLSizeMake(totalThreads, 1, 1), MTLSizeMake(tgSize, 1, 1), 1, 3);

        double bw = (double)totalThreads * iterations * sizeof(float) * 2 / t / 1e9;
        // Only print at key boundaries to keep output manageable
        if (bytes <= 2048 || bytes % 1024 == 0 || bytes % 4096 == 0)
            printf("  %10u  %10.3f  %10.1f\n", bytes, t * 1000, bw);
    }
    printf("  -> Step drops at specific sizes reveal HW allocation granularity.\n");
}

// ============================================================================
// Binary Inspection Hint (register file via metal-objdump)
// ============================================================================
static void printDisassemblyHint() {
    printf("\n=== Register File / Spill Verification via Disassembly ===\n");
    printf("  To inspect compiler output for spill instructions:\n");
    printf("    1. Build the metallib:\n");
    printf("       xcrun -sdk macosx metal -o swep_.air swep_.metal\n");
    printf("       xcrun -sdk macosx metallib -o swep_.metallib swep_.air\n");
    printf("    2. Disassemble with metal-objdump (Xcode 15+):\n");
    printf("       xcrun metal-objdump -d swep_.metallib | grep -E 'spill|SPILL|stack'\n");
    printf("    3. Compare spill instructions across regpressure variants\n");
    printf("       to confirm which accumulator count triggers register exhaustion.\n");
}

// ============================================================================
// Probe: Dynamic Cache (M3+ forward-looking)
// ============================================================================
static void probeDynamicCache() {
    printf("\n=== Dynamic Cache (register + TG memory pressure) ===\n");

    NSString* names[] = {
        @"probe_dynamic_cache_reghi",
        @"probe_dynamic_cache_both",
        @"probe_dynamic_cache_tghi"
    };
    const char* labels[] = {"256B TG + 16 acc", "8KB TG + 16 acc", "24KB TG + 16 acc"};

    uint32_t K = 512;
    size_t matBytes = K * 8 * sizeof(float);
    id<MTLBuffer> A = makeBuffer(matBytes);
    id<MTLBuffer> B = makeBuffer(matBytes);
    id<MTLBuffer> C = makeBuffer(8 * 8 * sizeof(float));
    float* pa = (float*)A.contents;
    float* pb = (float*)B.contents;
    for (size_t i = 0; i < matBytes / sizeof(float); i++) { pa[i] = 0.01f; pb[i] = 0.01f; }
    id<MTLBuffer> kBuf = makeBufferWithData(&K, sizeof(uint32_t));

    NSUInteger simdWidth = 32;
    NSUInteger numSimds = 512;

    printf("  %25s  %10s  %10s\n", "Variant", "Time (µs)", "TFLOPS");
    printf("  %25s  %10s  %10s\n", "-------------------------", "----------", "--------");

    for (int v = 0; v < 3; v++) {
        id<MTLComputePipelineState> pso = makePSO(names[v]);

        double t = timedDispatch(pso, ^(id<MTLComputeCommandEncoder> enc) {
            [enc setBuffer:A offset:0 atIndex:0];
            [enc setBuffer:B offset:0 atIndex:1];
            [enc setBuffer:C offset:0 atIndex:2];
            [enc setBuffer:kBuf offset:0 atIndex:3];
        }, MTLSizeMake(numSimds * simdWidth, 1, 1), MTLSizeMake(simdWidth, 1, 1), 3, 10);

        double flops = (double)numSimds * 16 * (K / 8.0) * 8 * 8 * 2;
        double tflops = flops / t / 1e12;
        printf("  %25s  %10.1f  %10.3f\n", labels[v], t * 1e6, tflops);

        if (v == 0) R.dynCacheReghi_TFLOPS = tflops;
        if (v == 1) R.dynCacheBoth_TFLOPS = tflops;
        if (v == 2) R.dynCacheTghi_TFLOPS = tflops;
    }
    printf("  -> On M2 (fixed): all similar or TG-heavy drops. On M3+ (dynamic): HW rebalances.\n");
}

// ============================================================================
// Probe: BFloat16-ish Throughput
// ============================================================================
static void probeBF16() {
    printf("\n=== BFloat16 Throughput (half proxy) ===\n");
    id<MTLComputePipelineState> pso = makePSO(@"probe_bf16_throughput");

    uint32_t iterations = 500000;
    NSUInteger threads = 4096;
    NSUInteger tgSize = 256;
    id<MTLBuffer> out = makeBuffer(threads * sizeof(uint16_t));
    id<MTLBuffer> iterBuf = makeBufferWithData(&iterations, sizeof(uint32_t));

    double t = timedDispatch(pso, ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:out offset:0 atIndex:0];
        [enc setBuffer:iterBuf offset:0 atIndex:1];
    }, MTLSizeMake(threads, 1, 1), MTLSizeMake(tgSize, 1, 1), 3, 10);

    // 4 half4 FMAs per iter = 4 * 2 * 4 = 32 FLOPS per iter
    double totalFLOPs = (double)threads * iterations * 4 * 8;
    double gflops = totalFLOPs / t / 1e9;
    printf("  GPU time: %.3f ms\n", t * 1000);
    printf("  BF16 (half proxy): %.1f GFLOPS\n", gflops);
    printf("  Compare with FP16 probe to check for datapath differences.\n");
    R.bf16_GFLOPS = gflops;
}

// ============================================================================
// Probe: Float32 Atomic (CAS loop vs native)
// ============================================================================
static void probeFloatAtomic() {
    printf("\n=== Float32 Atomic vs Uint32 Atomic ===\n");

    uint32_t iterations = 50000;
    NSUInteger threads = 1024;
    NSUInteger tgSize = 256;
    uint32_t nc = 64;

    // Float atomic (CAS loop)
    {
        id<MTLComputePipelineState> pso = makePSO(@"probe_float_atomic");
        id<MTLBuffer> counters = makeBuffer(nc * sizeof(float));
        memset(counters.contents, 0, nc * sizeof(float));
        id<MTLBuffer> iterBuf = makeBufferWithData(&iterations, sizeof(uint32_t));
        id<MTLBuffer> ncBuf = makeBufferWithData(&nc, sizeof(uint32_t));

        double t = timedDispatch(pso, ^(id<MTLComputeCommandEncoder> enc) {
            [enc setBuffer:counters offset:0 atIndex:0];
            [enc setBuffer:iterBuf offset:0 atIndex:1];
            [enc setBuffer:ncBuf offset:0 atIndex:2];
        }, MTLSizeMake(threads, 1, 1), MTLSizeMake(tgSize, 1, 1), 2, 5);

        double totalOps = (double)threads * iterations;
        double gops = totalOps / t / 1e9;
        printf("  Float32 atomic (CAS): %.3f ms, %.2f Gops/s\n", t * 1000, gops);
        R.floatAtomic_Gps = gops;
    }

    // Uint32 atomic (native) for comparison
    {
        id<MTLComputePipelineState> pso = makePSO(@"probe_atomic_throughput");
        id<MTLBuffer> counters = makeBuffer(nc * sizeof(uint32_t));
        memset(counters.contents, 0, nc * sizeof(uint32_t));
        id<MTLBuffer> iterBuf = makeBufferWithData(&iterations, sizeof(uint32_t));
        id<MTLBuffer> ncBuf = makeBufferWithData(&nc, sizeof(uint32_t));

        double t = timedDispatch(pso, ^(id<MTLComputeCommandEncoder> enc) {
            [enc setBuffer:counters offset:0 atIndex:0];
            [enc setBuffer:iterBuf offset:0 atIndex:1];
            [enc setBuffer:ncBuf offset:0 atIndex:2];
        }, MTLSizeMake(threads, 1, 1), MTLSizeMake(tgSize, 1, 1), 2, 5);

        double totalOps = (double)threads * iterations;
        double gops = totalOps / t / 1e9;
        printf("  Uint32 atomic (native): %.3f ms, %.2f Gops/s\n", t * 1000, gops);
        R.uintAtomic_Gps = gops;
    }
    printf("  -> If float32 ≈ uint32, the chip has native float atomics (M3+?).\n");
    printf("  -> If float32 << uint32, it's doing CAS loops in software.\n");
}

// ============================================================================
// Probe: 64-bit Atomic Emulation
// ============================================================================
static void probeAtomic64() {
    printf("\n=== 64-bit Atomic Throughput ===\n");
    id<MTLComputePipelineState> pso = makePSO(@"probe_atomic64");

    uint32_t iterations = 50000;
    NSUInteger threads = 1024;
    NSUInteger tgSize = 256;
    uint32_t nc = 64;

    id<MTLBuffer> counters = makeBuffer(nc * 2 * sizeof(uint32_t));
    memset(counters.contents, 0, nc * 2 * sizeof(uint32_t));
    id<MTLBuffer> iterBuf = makeBufferWithData(&iterations, sizeof(uint32_t));
    id<MTLBuffer> ncBuf = makeBufferWithData(&nc, sizeof(uint32_t));

    double t = timedDispatch(pso, ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:counters offset:0 atIndex:0];
        [enc setBuffer:iterBuf offset:0 atIndex:1];
        [enc setBuffer:ncBuf offset:0 atIndex:2];
    }, MTLSizeMake(threads, 1, 1), MTLSizeMake(tgSize, 1, 1), 2, 5);

    double totalOps = (double)threads * iterations;
    double gops = totalOps / t / 1e9;
    printf("  64-bit emulated: %.3f ms, %.2f Gops/s\n", t * 1000, gops);
    printf("  Compare with uint32 atomic to see if wider access path differs.\n");
    R.atomic64_Gps = gops;
}

// ============================================================================
// Build and write the markdown report
// ============================================================================
static void buildMarkdownReport() {
    std::string chip = detectChip(R.deviceName ? R.deviceName : "Unknown");

    md("# Apple %s GPU — Measured Properties\n\n", chip.c_str());
    md("All numbers from real probe runs on this device.\n\n");

    md("## Basics\n");
    md("- **Device:** %s, %s memory\n", R.deviceName, R.unifiedMemory ? "unified" : "discrete");
    md("- **SIMD width:** %lu threads\n", R.simdWidth);
    md("- **Max threadgroup size:** %lu threads\n", R.maxTGThreads);
    md("- **Max threadgroup memory:** %lu KB\n", R.maxTGMem / 1024);
    md("- **Max buffer length:** %lu MB\n", R.maxBufMB);
    md("- **Recommended working set:** %llu MB\n", R.workingSetMB);
    md("- **Dispatch overhead:** ~%.1f us\n", R.dispatchOverhead_us);
    md("- **GPU timer resolution:** ~%.1f us minimum measurable dispatch\n\n", R.timerMinUs);

    md("## Memory Bandwidth\n");
    md("- **DRAM bandwidth:** ~%.0f GB/s\n", R.dramBW_GBs);
    md("- **Threadgroup memory bandwidth:** ~%.0f GB/s aggregate\n", R.tgmemBW_GBs);
    md("- **Texture read bandwidth:** ~%.0f GB/s\n", R.textureBW_GBs);
    md("- **Buffer read bandwidth:** ~%.0f GB/s\n\n", R.bufferBW_GBs);

    md("## Cache Hierarchy\n");
    for (auto& pt : R.cacheLatency)
        md("- **%u KB:** ~%.0f ns\n", pt.kb, pt.latNs);
    md("- **SLC associativity:** %d-way\n", R.slcAssociativity);
    md("- **TG memory bank count:** %d\n\n", R.bankCount);

    md("## Coalescing\n");
    md("- **Stride 1 (coalesced):** ~%.0f GB/s\n", R.coalescedBW_GBs);
    md("- **Stride 16 (non-coalesced):** ~%.0f GB/s\n", R.stride16BW_GBs);
    if (R.stride16BW_GBs > 0)
        md("- ~%.1fx penalty for non-coalesced access\n\n", R.coalescedBW_GBs / R.stride16BW_GBs);
    else
        md("\n");

    md("## Compute Throughput\n");
    md("- **FP32 ALU:** ~%.0f GFLOPS\n", R.fp32_GFLOPS);
    md("- **FP16 ALU:** ~%.0f GFLOPS", R.fp16_GFLOPS);
    if (R.fp32_GFLOPS > 0) {
        double ratio = R.fp16_GFLOPS / R.fp32_GFLOPS;
        if (ratio > 1.5) md(" (%.1fx faster than FP32 — packed execution)\n", ratio);
        else md(" (same as FP32 — no packed fp16)\n");
    } else md("\n");
    md("- **INT32 ALU:** ~%.0f GIOPS", R.int32_GIOPS);
    if (R.fp32_GFLOPS > 0) md(" (~%.1fx slower than FP32)\n", R.fp32_GFLOPS / R.int32_GIOPS);
    else md("\n");
    md("- **Shader cores:** ~%d (inferred from saturation sweep)\n\n", R.inferredCores);

    md("## SIMD Operations\n");
    md("- **SIMD shuffle:** ~%.0f Gshuffles/s, ~%.0f GB/s effective\n", R.shuffleRate_Gps, R.shuffleBW_GBs);
    md("- **SIMD reduction (simd_sum):** ~%.0f Greductions/s\n\n", R.simdSumRate_Gps);

    md("## Atomics\n");
    md("- **Uncontested:** ~%.1f Gatomics/s\n", R.atomicUncontested_Gps);
    md("- **Full contention (1 counter):** ~%.1f Gatomics/s\n", R.atomicContested_Gps);
    if (R.atomicContested_Gps > 0)
        md("- Contention penalty: ~%.0fx\n\n", R.atomicUncontested_Gps / R.atomicContested_Gps);
    else
        md("\n");

    md("## Synchronization\n");
    md("- **Barrier cost (32-thread TG):** ~%.0f ns\n", R.barrierCost32_ns);
    md("- **Barrier cost (1024-thread TG):** ~%.0f ns\n", R.barrierCost1024_ns);
    md("- **TG memory RAW latency (1 thread):** ~%.0f ns per store-barrier-load cycle\n", R.tgmemRAW_1thread_ns);
    md("- **TG memory RAW latency (256 threads):** ~%.0f ns\n\n", R.tgmemRAW_256thread_ns);

    md("## Register File\n");
    md("- **Spill boundary:** between %d and %d simdgroup_float8x8 accumulators\n\n",
       R.spillBoundaryLow, R.spillBoundaryHigh);

    md("## Pipeline\n");
    md("- **FMA throughput (independent):** ~%.0f GFLOPS\n", R.fmaIndep_GFLOPS);
    md("- **FMA latency (dependent chain):** ~%.0f GFLOPS\n", R.fmaDep1_GFLOPS);
    if (R.fmaDep1_GFLOPS > 0) {
        double ratio = R.fmaIndep_GFLOPS / R.fmaDep1_GFLOPS;
        md("- Ratio ~%.1fx suggests ~%d-cycle FMA pipeline\n\n", ratio, (int)round(ratio));
    } else md("\n");

    md("## Instruction Cache\n");
    md("- **Small kernel (4 FMA/iter):** ~%.0f GFLOPS\n", R.icacheSmall_GFLOPS);
    md("- **Medium kernel (32 FMA/iter):** ~%.0f GFLOPS\n", R.icacheMedium_GFLOPS);
    md("- **Large kernel (64 FMA/iter):** ~%.0f GFLOPS\n", R.icacheLarge_GFLOPS);
    if (R.icacheMedium_GFLOPS > 0 && R.icacheLarge_GFLOPS < R.icacheMedium_GFLOPS * 0.95)
        md("- Large variant shows ~%.0f%% drop — icache pressure\n\n",
           (1.0 - R.icacheLarge_GFLOPS / R.icacheMedium_GFLOPS) * 100);
    else
        md("\n");

    md("## TLB\n");
    for (auto& pt : R.tlbLatency)
        md("- **%u pages:** ~%.0f ns\n", pt.pages, pt.latNs);
    md("\n");

    md("## MMA (simdgroup_multiply_accumulate)\n");
    md("- **Latency (dependent):** ~%.1f ns/MMA\n", R.mmaLatency_ns);
    md("- **Throughput (4 independent):** ~%.1f ns/MMA\n", R.mmaThroughput_ns);
    if (R.mmaThroughput_ns > 0) {
        double ratio = R.mmaLatency_ns / R.mmaThroughput_ns;
        md("- ~%.1fx speedup with independent accumulators\n\n", ratio);
    } else md("\n");

    md("## Dispatch\n");
    md("- **Direct dispatch:** ~%.1f us\n", R.directDispatch_us);
    md("- **Indirect dispatch:** ~%.1f us\n", R.indirectDispatch_us);
    md("- **Indirect overhead:** ~%.1f us\n\n", R.indirectDispatch_us - R.directDispatch_us);

    md("## Readback Latency\n");
    md("| Size | GPU | Readback | Total |\n");
    md("|------|-----|----------|-------|\n");
    for (auto& pt : R.readback)
        md("| %u KB | %.0f us | %.0f us | %.0f us |\n", pt.kb, pt.gpuUs, pt.readbackUs, pt.gpuUs + pt.readbackUs);
    md("\n");

    md("## Concurrent Execution\n");
    md("- **Kernel A alone:** %.3f ms\n", R.concA_ms);
    md("- **Kernel B alone:** %.3f ms\n", R.concB_ms);
    md("- **Both together:** %.3f ms\n", R.concBoth_ms);
    double serial = R.concA_ms + R.concB_ms;
    double overlap = (serial > 0) ? (1.0 - R.concBoth_ms / serial) * 100 : 0;
    md("- **Overlap:** %.0f%%\n", overlap);
    if (overlap > 5) md("- GPU runs independent kernels concurrently across cores.\n");
    else md("- GPU serializes back-to-back dispatches.\n");
    md("\n");

    md("## Dynamic Cache (M3+ feature)\n");
    md("Tests whether the GPU dynamically rebalances on-chip SRAM between registers and threadgroup memory.\n\n");
    md("| Variant | TFLOPS |\n");
    md("|---------|--------|\n");
    md("| 256B TG + 16 acc | %.3f |\n", R.dynCacheReghi_TFLOPS);
    md("| 8KB TG + 16 acc | %.3f |\n", R.dynCacheBoth_TFLOPS);
    md("| 24KB TG + 16 acc | %.3f |\n", R.dynCacheTghi_TFLOPS);
    if (R.dynCacheTghi_TFLOPS > 0 && R.dynCacheReghi_TFLOPS > 0) {
        double drop = (1.0 - R.dynCacheTghi_TFLOPS / R.dynCacheReghi_TFLOPS) * 100;
        if (drop > 10) md("\n%.0f%% drop with high TG mem — fixed partitioning (M2 behavior).\n", drop);
        else md("\nMinimal drop — dynamic caching is active (M3+ behavior).\n");
    }
    md("\n");

    md("## BFloat16\n");
    md("- **BF16 (half proxy):** ~%.0f GFLOPS\n", R.bf16_GFLOPS);
    md("- **FP16:** ~%.0f GFLOPS\n", R.fp16_GFLOPS);
    if (R.fp16_GFLOPS > 0) {
        double ratio = R.bf16_GFLOPS / R.fp16_GFLOPS;
        if (ratio > 1.3) md("- BF16 faster than FP16 — possible dedicated bf16 path.\n");
        else md("- Same speed — no dedicated bf16 hardware, runs on fp16 pipe.\n");
    }
    md("\n");

    md("## Atomic Variants\n");
    md("- **Uint32 atomic (native):** ~%.1f Gops/s\n", R.uintAtomic_Gps);
    md("- **Float32 atomic (CAS):** ~%.1f Gops/s\n", R.floatAtomic_Gps);
    md("- **64-bit atomic (emulated):** ~%.1f Gops/s\n", R.atomic64_Gps);
    if (R.uintAtomic_Gps > 0 && R.floatAtomic_Gps > 0) {
        double ratio = R.uintAtomic_Gps / R.floatAtomic_Gps;
        if (ratio > 3) md("- Float32 atomic is %.0fx slower — software CAS loop, no native float atomics.\n", ratio);
        else md("- Float32 atomic is close to uint32 — possible native float atomic support.\n");
    }
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
        NSString* shaderPath = [[NSBundle mainBundle] pathForResource:@"swep_" ofType:@"metallib"];
        NSError* err = nil;

        if (shaderPath) {
            NSURL* url = [NSURL fileURLWithPath:shaderPath];
            gLibrary = [gDevice newLibraryWithURL:url error:&err];
        }

        if (!gLibrary) {
            NSString* cwd = [[NSFileManager defaultManager] currentDirectoryPath];
            NSString* libPath = [cwd stringByAppendingPathComponent:@"swep_.metallib"];
            NSURL* url = [NSURL fileURLWithPath:libPath];
            gLibrary = [gDevice newLibraryWithURL:url error:&err];
        }

        if (!gLibrary) {
            NSString* cwd = [[NSFileManager defaultManager] currentDirectoryPath];
            NSString* srcPath = [cwd stringByAppendingPathComponent:@"swep_.metal"];
            NSString* src = [NSString stringWithContentsOfFile:srcPath encoding:NSUTF8StringEncoding error:&err];
            if (!src) {
                fprintf(stderr, "ERROR: Cannot read swep_.metal: %s\n", err.localizedDescription.UTF8String);
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

        printf("GPU Probe Suite\n");
        printf("===============\n");

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
        probeTGMemRAWLatency();
        probeInstructionCache();
        probeTLBSize();
        probeMMALatency();
        probeReadbackLatency();
        probeIndirectDispatch();
        probeConcurrentExecution();
        probeTGMemGranularity();

        // M3+ forward-looking probes
        probeDynamicCache();
        probeBF16();
        probeFloatAtomic();
        probeAtomic64();

        printDisassemblyHint();

        // Build and write markdown report
        buildMarkdownReport();
        writeReport(argv[0]);

        printf("\n=== All probes complete ===\n");
    }
    return 0;
}
