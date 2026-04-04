// Best M4 GEMM for M>=256, N>=256, K>64
// 16x128 tile, K=16, 128 threads (4 SIMDs), 8 accumulators per SIMD
// Shared mem: 9.3 KB — leaves 22+ KB for dynamic caching

#include <metal_stdlib>
using namespace metal;

kernel void gemm_m4(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C       [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    constant uint& K [[buffer(5)]],
    uint2 gid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]])
{
    threadgroup float As[16][17];
    threadgroup float Bs[16][129];

    uint tileRow = gid.y * 16;
    uint tileCol = gid.x * 128;

    simdgroup_float8x8 acc[8];
    for (int i = 0; i < 8; i++) acc[i] = simdgroup_float8x8(0);

    uint sr = (simd_id / 4) * 8;
    uint sc = (simd_id % 4) * 32;

    for (uint kb = 0; kb < K; kb += 16) {
        for (uint i = lid; i < 16 * 16; i += 128) {
            uint r = i / 16, c = i % 16;
            uint gr = tileRow + r, gc = kb + c;
            As[r][c] = (gr < M && gc < K) ? A[gr * K + gc] : 0;
        }
        for (uint i = lid; i < 16 * 128; i += 128) {
            uint r = i / 128, c = i % 128;
            uint gr = kb + r, gc = tileCol + c;
            Bs[r][c] = (gr < K && gc < N) ? B[gr * N + gc] : 0;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint kk = 0; kk < 16; kk += 8) {
            simdgroup_float8x8 a0, b0, b1, b2, b3;
            simdgroup_load(a0, &As[sr][kk], 17);
            simdgroup_load(b0, &Bs[kk][sc], 129);
            simdgroup_load(b1, &Bs[kk][sc + 8], 129);
            simdgroup_load(b2, &Bs[kk][sc + 16], 129);
            simdgroup_load(b3, &Bs[kk][sc + 24], 129);

            simdgroup_multiply_accumulate(acc[0], a0, b0, acc[0]);
            simdgroup_multiply_accumulate(acc[1], a0, b1, acc[1]);
            simdgroup_multiply_accumulate(acc[2], a0, b2, acc[2]);
            simdgroup_multiply_accumulate(acc[3], a0, b3, acc[3]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (int ac = 0; ac < 4; ac++) {
        uint r = tileRow + sr;
        uint c = tileCol + sc + ac * 8;
        if (r < M && c < N)
            simdgroup_store(acc[ac], C + r * N + c, N);
    }
}
