#include <metal_stdlib>
using namespace metal;

kernel void gemm_naive(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C       [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    constant uint& K [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= N || gid.y >= M) return;
    float acc = 0;
    for (uint k = 0; k < K; k++)
        acc = fma(A[gid.y * K + k], B[k * N + gid.x], acc);
    C[gid.y * N + gid.x] = acc;
}
