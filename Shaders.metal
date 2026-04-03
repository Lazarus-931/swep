#include <metal_stdlib>
using namespace metal;

// =============================================================================
// DRAM Bandwidth — pure buffer copy, large enough to blow all caches
// =============================================================================
kernel void probe_dram_bandwidth(
    device const float4* src [[buffer(0)]],
    device float4* dst       [[buffer(1)]],
    uint tid [[thread_position_in_grid]])
{
    dst[tid] = src[tid];
}

// =============================================================================
// Threadgroup Memory Bandwidth — tight read/write loop in shared SRAM
// =============================================================================
kernel void probe_tgmem_bandwidth(
    device atomic_uint* out [[buffer(0)]],
    uint lid [[thread_index_in_threadgroup]],
    uint tid [[thread_position_in_grid]])
{
    threadgroup float4 shmem[1024]; // 16 KB

    // Write pattern
    for (int i = 0; i < 256; i++) {
        shmem[lid] = float4(float(i));
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Read pattern
    float4 acc = 0.0;
    for (int i = 0; i < 256; i++) {
        acc += shmem[lid];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Prevent dead-code elimination
    if (acc.x == -999.0f) {
        atomic_fetch_add_explicit(out, 1, memory_order_relaxed);
    }
}

// =============================================================================
// Cache Size Sweep — stride-1 reads over increasing working set sizes
// Throughput cliffs reveal L1 and SLC boundaries
// =============================================================================
kernel void probe_cache_sweep(
    device const float* data [[buffer(0)]],
    device float* out        [[buffer(1)]],
    constant uint& workingSetElements [[buffer(2)]],
    constant uint& iterations         [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    float acc = 0.0f;
    uint mask = workingSetElements - 1; // power-of-2 wrap
    uint idx = tid;
    for (uint i = 0; i < iterations; i++) {
        acc += data[idx & mask];
        idx += 1;
    }
    out[tid] = acc;
}

// =============================================================================
// Cache Latency — pointer-chase (dependent load chain)
// Each element stores the index of the next element to load
// =============================================================================
kernel void probe_cache_latency(
    device const uint* chain [[buffer(0)]],
    device uint* out         [[buffer(1)]],
    constant uint& hops      [[buffer(2)]],
    uint tid [[thread_position_in_grid]])
{
    uint idx = tid;
    for (uint i = 0; i < hops; i++) {
        idx = chain[idx];
    }
    out[tid] = idx;
}

// =============================================================================
// Register Spill Boundary — increasing simdgroup_float8x8 accumulator count
// Variant 0: 4 accumulators (baseline)
// =============================================================================
kernel void probe_regpressure_4(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C       [[buffer(2)]],
    constant uint& K      [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    simdgroup_float8x8 acc0, acc1, acc2, acc3;
    simdgroup_float8x8 a, b;

    acc0 = simdgroup_float8x8(0); acc1 = simdgroup_float8x8(0);
    acc2 = simdgroup_float8x8(0); acc3 = simdgroup_float8x8(0);

    for (uint k = 0; k < K; k += 8) {
        simdgroup_load(a, A + k * 8, 8);
        simdgroup_load(b, B + k * 8, 8);
        simdgroup_multiply_accumulate(acc0, a, b, acc0);
        simdgroup_multiply_accumulate(acc1, a, b, acc1);
        simdgroup_multiply_accumulate(acc2, a, b, acc2);
        simdgroup_multiply_accumulate(acc3, a, b, acc3);
    }
    simdgroup_store(acc0, C, 8);
}

// Variant 1: 8 accumulators
kernel void probe_regpressure_8(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C       [[buffer(2)]],
    constant uint& K      [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    simdgroup_float8x8 acc[8];
    simdgroup_float8x8 a, b;
    for (int i = 0; i < 8; i++) acc[i] = simdgroup_float8x8(0);

    for (uint k = 0; k < K; k += 8) {
        simdgroup_load(a, A + k * 8, 8);
        simdgroup_load(b, B + k * 8, 8);
        for (int i = 0; i < 8; i++)
            simdgroup_multiply_accumulate(acc[i], a, b, acc[i]);
    }
    simdgroup_store(acc[0], C, 8);
}

// Variant 2: 16 accumulators
kernel void probe_regpressure_16(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C       [[buffer(2)]],
    constant uint& K      [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    simdgroup_float8x8 acc[16];
    simdgroup_float8x8 a, b;
    for (int i = 0; i < 16; i++) acc[i] = simdgroup_float8x8(0);

    for (uint k = 0; k < K; k += 8) {
        simdgroup_load(a, A + k * 8, 8);
        simdgroup_load(b, B + k * 8, 8);
        for (int i = 0; i < 16; i++)
            simdgroup_multiply_accumulate(acc[i], a, b, acc[i]);
    }
    simdgroup_store(acc[0], C, 8);
}

// Variant 3: 24 accumulators
kernel void probe_regpressure_24(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C       [[buffer(2)]],
    constant uint& K      [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    simdgroup_float8x8 acc[24];
    simdgroup_float8x8 a, b;
    for (int i = 0; i < 24; i++) acc[i] = simdgroup_float8x8(0);

    for (uint k = 0; k < K; k += 8) {
        simdgroup_load(a, A + k * 8, 8);
        simdgroup_load(b, B + k * 8, 8);
        for (int i = 0; i < 24; i++)
            simdgroup_multiply_accumulate(acc[i], a, b, acc[i]);
    }
    simdgroup_store(acc[0], C, 8);
}

// Variant 4: 32 accumulators
kernel void probe_regpressure_32(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C       [[buffer(2)]],
    constant uint& K      [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    simdgroup_float8x8 acc[32];
    simdgroup_float8x8 a, b;
    for (int i = 0; i < 32; i++) acc[i] = simdgroup_float8x8(0);

    for (uint k = 0; k < K; k += 8) {
        simdgroup_load(a, A + k * 8, 8);
        simdgroup_load(b, B + k * 8, 8);
        for (int i = 0; i < 32; i++)
            simdgroup_multiply_accumulate(acc[i], a, b, acc[i]);
    }
    simdgroup_store(acc[0], C, 8);
}

// Variant 5: 48 accumulators (likely spills)
kernel void probe_regpressure_48(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C       [[buffer(2)]],
    constant uint& K      [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    simdgroup_float8x8 acc[48];
    simdgroup_float8x8 a, b;
    for (int i = 0; i < 48; i++) acc[i] = simdgroup_float8x8(0);

    for (uint k = 0; k < K; k += 8) {
        simdgroup_load(a, A + k * 8, 8);
        simdgroup_load(b, B + k * 8, 8);
        for (int i = 0; i < 48; i++)
            simdgroup_multiply_accumulate(acc[i], a, b, acc[i]);
    }
    simdgroup_store(acc[0], C, 8);
}

// =============================================================================
// Shader Core Count — compute-bound FMA saturation kernel
// Each thread does a known number of FMAs, total FLOPS / time = peak
// =============================================================================
kernel void probe_core_saturation(
    device float* out [[buffer(0)]],
    constant uint& fma_iters [[buffer(1)]],
    uint tid [[thread_position_in_grid]])
{
    float a = float(tid) * 0.001f;
    float b = 1.0001f;
    float c = 0.0f;

    for (uint i = 0; i < fma_iters; i++) {
        c = fma(a, b, c);  // 2 FLOPS
        a = fma(c, b, a);
        c = fma(a, b, c);
        a = fma(c, b, a);
    }

    out[tid] = c + a;
}

// =============================================================================
// Threadgroup Bank Conflict Probe — stride sweep to find bank count
// =============================================================================
kernel void probe_bank_conflicts(
    device float* out          [[buffer(0)]],
    constant uint& stride      [[buffer(1)]],
    constant uint& iterations  [[buffer(2)]],
    uint lid [[thread_index_in_threadgroup]],
    uint tid [[thread_position_in_grid]])
{
    threadgroup float shmem[4096]; // 16 KB

    float acc = 0.0f;
    uint base = lid * stride;

    for (uint i = 0; i < iterations; i++) {
        shmem[base & 4095] = float(i);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        acc += shmem[base & 4095];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    out[tid] = acc;
}

// =============================================================================
// SLC Associativity — conflict miss probe with controlled offsets
// =============================================================================
kernel void probe_slc_assoc(
    device const float* data [[buffer(0)]],
    device float* out        [[buffer(1)]],
    constant uint& num_ways  [[buffer(2)]],
    constant uint& stride    [[buffer(3)]],  // = SLC size in floats
    constant uint& iterations [[buffer(4)]],
    uint tid [[thread_position_in_grid]])
{
    float acc = 0.0f;
    for (uint i = 0; i < iterations; i++) {
        for (uint w = 0; w < num_ways; w++) {
            acc += data[w * stride + (tid & 63)];
        }
    }
    out[tid] = acc;
}

// =============================================================================
// Pipeline Depth — dependency chain of FMAs with varying chain lengths
// =============================================================================
kernel void probe_pipeline_dep1(
    device float* out [[buffer(0)]],
    constant uint& iterations [[buffer(1)]],
    uint tid [[thread_position_in_grid]])
{
    float a = float(tid) * 0.001f;
    for (uint i = 0; i < iterations; i++) {
        a = fma(a, 1.0001f, 0.0f); // chain length 1
    }
    out[tid] = a;
}

kernel void probe_pipeline_dep2(
    device float* out [[buffer(0)]],
    constant uint& iterations [[buffer(1)]],
    uint tid [[thread_position_in_grid]])
{
    float a = float(tid) * 0.001f;
    float b = a + 0.001f;
    for (uint i = 0; i < iterations; i++) {
        a = fma(a, 1.0001f, b);
        b = fma(b, 1.0001f, a);
    }
    out[tid] = a + b;
}

kernel void probe_pipeline_dep4(
    device float* out [[buffer(0)]],
    constant uint& iterations [[buffer(1)]],
    uint tid [[thread_position_in_grid]])
{
    float a = float(tid) * 0.001f;
    float b = a + 0.001f;
    float c = a + 0.002f;
    float d = a + 0.003f;
    for (uint i = 0; i < iterations; i++) {
        a = fma(a, 1.0001f, b);
        b = fma(b, 1.0001f, c);
        c = fma(c, 1.0001f, d);
        d = fma(d, 1.0001f, a);
    }
    out[tid] = a + b + c + d;
}

kernel void probe_pipeline_dep8(
    device float* out [[buffer(0)]],
    constant uint& iterations [[buffer(1)]],
    uint tid [[thread_position_in_grid]])
{
    float v[8];
    for (int j = 0; j < 8; j++) v[j] = float(tid) * 0.001f + float(j) * 0.001f;
    for (uint i = 0; i < iterations; i++) {
        for (int j = 0; j < 8; j++)
            v[j] = fma(v[j], 1.0001f, v[(j+1)&7]);
    }
    float acc = 0;
    for (int j = 0; j < 8; j++) acc += v[j];
    out[tid] = acc;
}

kernel void probe_pipeline_indep(
    device float* out [[buffer(0)]],
    constant uint& iterations [[buffer(1)]],
    uint tid [[thread_position_in_grid]])
{
    // 8 fully independent chains — measures peak throughput (no dependency)
    float v[8];
    for (int j = 0; j < 8; j++) v[j] = float(tid) * 0.001f + float(j) * 0.001f;
    for (uint i = 0; i < iterations; i++) {
        for (int j = 0; j < 8; j++)
            v[j] = fma(v[j], 1.0001f, 0.0f);
    }
    float acc = 0;
    for (int j = 0; j < 8; j++) acc += v[j];
    out[tid] = acc;
}

// =============================================================================
// Atomic throughput — measures atomic_fetch_add rate to device memory
// Reveals contention model and atomic unit throughput
// =============================================================================
kernel void probe_atomic_throughput(
    device atomic_uint* counters [[buffer(0)]],
    constant uint& iterations    [[buffer(1)]],
    constant uint& num_counters  [[buffer(2)]],
    uint tid [[thread_position_in_grid]])
{
    for (uint i = 0; i < iterations; i++) {
        atomic_fetch_add_explicit(&counters[tid % num_counters], 1u, memory_order_relaxed);
    }
}

// =============================================================================
// SIMD shuffle bandwidth — how fast can threads exchange data within a SIMD?
// =============================================================================
kernel void probe_simd_shuffle(
    device float* out          [[buffer(0)]],
    constant uint& iterations  [[buffer(1)]],
    uint tid [[thread_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]])
{
    float val = float(tid);
    for (uint i = 0; i < iterations; i++) {
        val = simd_shuffle_xor(val, 1u);
        val = simd_shuffle_xor(val, 2u);
        val = simd_shuffle_xor(val, 4u);
        val = simd_shuffle_xor(val, 8u);
        val = simd_shuffle_xor(val, 16u);
    }
    out[tid] = val;
}

// =============================================================================
// SIMD reduction — measures simd_sum throughput
// =============================================================================
kernel void probe_simd_reduce(
    device float* out          [[buffer(0)]],
    constant uint& iterations  [[buffer(1)]],
    uint tid [[thread_position_in_grid]])
{
    float val = float(tid) * 0.001f;
    float acc = 0.0f;
    for (uint i = 0; i < iterations; i++) {
        acc += simd_sum(val);
        val = acc * 0.001f;
    }
    out[tid] = acc;
}

// =============================================================================
// Integer ALU throughput — compare int vs float execution rates
// =============================================================================
kernel void probe_int_throughput(
    device uint* out           [[buffer(0)]],
    constant uint& iterations  [[buffer(1)]],
    uint tid [[thread_position_in_grid]])
{
    uint a = tid;
    uint b = tid + 1;
    uint c = tid + 2;
    uint d = tid + 3;
    for (uint i = 0; i < iterations; i++) {
        a = a * 3 + b;
        b = b * 3 + c;
        c = c * 3 + d;
        d = d * 3 + a;
    }
    out[tid] = a + b + c + d;
}

// =============================================================================
// Half-precision (float16) throughput — find the fp16:fp32 ratio
// =============================================================================
kernel void probe_half_throughput(
    device half* out           [[buffer(0)]],
    constant uint& iterations  [[buffer(1)]],
    uint tid [[thread_position_in_grid]])
{
    half4 a = half4(half(tid) * 0.001h);
    half4 b = half4(1.001h);
    half4 c = half4(0.0h);
    for (uint i = 0; i < iterations; i++) {
        c = fma(a, b, c);
        a = fma(c, b, a);
        c = fma(a, b, c);
        a = fma(c, b, a);
    }
    out[tid] = c.x + a.x;
}

// float32 throughput for direct comparison
kernel void probe_float_throughput(
    device float* out          [[buffer(0)]],
    constant uint& iterations  [[buffer(1)]],
    uint tid [[thread_position_in_grid]])
{
    float4 a = float4(float(tid) * 0.001f);
    float4 b = float4(1.001f);
    float4 c = float4(0.0f);
    for (uint i = 0; i < iterations; i++) {
        c = fma(a, b, c);
        a = fma(c, b, a);
        c = fma(a, b, c);
        a = fma(c, b, a);
    }
    out[tid] = c.x + a.x;
}

// =============================================================================
// Cache line size — stride sweep measuring latency to find fetch granularity
// =============================================================================
kernel void probe_cacheline(
    device const uint* chain [[buffer(0)]],
    device uint* out         [[buffer(1)]],
    constant uint& hops      [[buffer(2)]],
    uint tid [[thread_position_in_grid]])
{
    // Same as cache_latency — host builds chains with different strides
    uint idx = 0;
    for (uint i = 0; i < hops; i++) {
        idx = chain[idx];
    }
    out[tid] = idx;
}

// =============================================================================
// Memory coalescing — strided vs contiguous access pattern comparison
// =============================================================================
kernel void probe_coalescing(
    device const float* data [[buffer(0)]],
    device float* out        [[buffer(1)]],
    constant uint& stride    [[buffer(2)]],
    constant uint& iterations [[buffer(3)]],
    uint tid [[thread_position_in_grid]],
    uint tcount [[threads_per_grid]])
{
    float acc = 0.0f;
    uint base = tid * stride;
    for (uint i = 0; i < iterations; i++) {
        acc += data[base % (tcount * stride)];
        base += stride;
    }
    out[tid] = acc;
}

// =============================================================================
// Threadgroup barrier cost — measures synchronization overhead
// =============================================================================
kernel void probe_barrier_cost(
    device float* out          [[buffer(0)]],
    constant uint& iterations  [[buffer(1)]],
    uint lid [[thread_index_in_threadgroup]],
    uint tid [[thread_position_in_grid]])
{
    threadgroup float dummy[1];
    float acc = float(lid);
    for (uint i = 0; i < iterations; i++) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        acc += 1.0f;
    }
    // One write to prevent elimination of the barrier loop
    if (lid == 0) dummy[0] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    out[tid] = acc + dummy[0];
}

// =============================================================================
// Dispatch overhead — empty kernel, measures pure launch cost
// =============================================================================
kernel void probe_dispatch_empty(
    device float* out [[buffer(0)]],
    uint tid [[thread_position_in_grid]])
{
    out[tid] = 0.0f;
}

// =============================================================================
// Texture read bandwidth — compare texture path vs buffer path
// =============================================================================
kernel void probe_texture_read(
    texture2d<float, access::read> tex [[texture(0)]],
    device float* out                  [[buffer(0)]],
    constant uint& iterations          [[buffer(1)]],
    uint2 gid [[thread_position_in_grid]])
{
    float4 acc = 0.0f;
    uint2 pos = gid;
    uint w = tex.get_width();
    uint h = tex.get_height();
    for (uint i = 0; i < iterations; i++) {
        acc += tex.read(uint2(pos.x % w, pos.y % h));
        pos.x += 1;
    }
    out[gid.y * w + gid.x] = acc.x + acc.y + acc.z + acc.w;
}

// =============================================================================
// Occupancy Sweep — kernel that claims variable threadgroup memory
// =============================================================================
kernel void probe_occupancy(
    device float* out            [[buffer(0)]],
    constant uint& tgmem_floats [[buffer(1)]],
    constant uint& iterations   [[buffer(2)]],
    uint lid [[thread_index_in_threadgroup]],
    uint tid [[thread_position_in_grid]])
{
    // Dynamically use threadgroup memory by writing to claimed region
    threadgroup float shmem[8192]; // 32 KB max — actual usage controlled by tgmem_floats

    float acc = 0.0f;
    for (uint i = 0; i < iterations; i++) {
        uint idx = lid % tgmem_floats;
        shmem[idx] = float(i);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        acc += shmem[idx];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    out[tid] = acc;
}
