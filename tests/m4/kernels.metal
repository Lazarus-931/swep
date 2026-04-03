#include <metal_stdlib>
using namespace metal;

kernel void read_only(
    device const float4* src [[buffer(0)]],
    device float* dst        [[buffer(1)]],
    constant uint& mask      [[buffer(2)]],
    constant uint& iters     [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    float4 acc = 0;
    uint idx = tid;
    for (uint i = 0; i < iters; i++) {
        acc += src[idx & mask];
        idx += 1024;
    }
    dst[tid] = acc.x + acc.y + acc.z + acc.w;
}

kernel void write_only(
    device float4* dst       [[buffer(0)]],
    constant uint& mask      [[buffer(1)]],
    constant uint& iters     [[buffer(2)]],
    uint tid [[thread_position_in_grid]])
{
    float4 val = float4(float(tid) * 0.001f);
    uint idx = tid;
    for (uint i = 0; i < iters; i++) {
        dst[idx & mask] = val;
        idx += 1024;
        val += 0.001f;
    }
}

kernel void read_write(
    device const float4* src [[buffer(0)]],
    device float4* dst       [[buffer(1)]],
    constant uint& mask      [[buffer(2)]],
    constant uint& iters     [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    uint idx = tid;
    for (uint i = 0; i < iters; i++) {
        dst[idx & mask] = src[idx & mask];
        idx += 1024;
    }
}

kernel void latency_chase(
    device const uint* chain [[buffer(0)]],
    device uint* out         [[buffer(1)]],
    constant uint& hops      [[buffer(2)]],
    uint tid [[thread_position_in_grid]])
{
    uint idx = tid & 255;
    for (uint i = 0; i < hops; i++)
        idx = chain[idx];
    out[tid] = idx;
}

kernel void latency_plus_bandwidth(
    device const uint* chain    [[buffer(0)]],
    device uint* out            [[buffer(1)]],
    constant uint& hops         [[buffer(2)]],
    device const float4* stream [[buffer(3)]],
    constant uint& streamMask   [[buffer(4)]],
    uint tid [[thread_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]])
{
    if (lid < 8) {
        uint idx = tid & 255;
        for (uint i = 0; i < hops; i++)
            idx = chain[idx];
        out[tid] = idx;
    } else {
        float4 acc = 0;
        uint base = tid;
        for (uint i = 0; i < hops; i++) {
            acc += stream[base & streamMask];
            base += 256;
        }
        out[tid] = as_type<uint>(acc.x);
    }
}

kernel void tgmem_plus_dram(
    device const float4* src [[buffer(0)]],
    device float* dst        [[buffer(1)]],
    constant uint& mask      [[buffer(2)]],
    constant uint& iters     [[buffer(3)]],
    uint tid [[thread_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]])
{
    threadgroup float4 tile[256];
    float4 acc = 0;
    uint idx = tid;
    for (uint i = 0; i < iters; i++) {
        tile[lid] = src[idx & mask];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        acc += tile[lid] + tile[(lid + 32) & 255];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        idx += 1024;
    }
    dst[tid] = acc.x + acc.y + acc.z + acc.w;
}

kernel void atomic_contention(
    device atomic_uint* counters [[buffer(0)]],
    constant uint& iters         [[buffer(1)]],
    constant uint& num_counters  [[buffer(2)]],
    uint tid [[thread_position_in_grid]])
{
    for (uint i = 0; i < iters; i++)
        atomic_fetch_add_explicit(&counters[tid % num_counters], 1u, memory_order_relaxed);
}
