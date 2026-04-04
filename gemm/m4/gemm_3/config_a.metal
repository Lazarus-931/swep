// Core saturation sweep — 32x128 tile, K=16
// Config A: 128 threads (4 SIMDs) per threadgroup — more threadgroups, better distribution

#include <metal_stdlib>
using namespace metal;

kernel void gemm_tg128(
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
    threadgroup float As[32][17];
    threadgroup float Bs[16][129];

    uint tileRow = gid.y * 32;
    uint tileCol = gid.x * 128;

    simdgroup_float8x8 acc[16];
    for (int i = 0; i < 16; i++) acc[i] = simdgroup_float8x8(0);

    uint sr = (simd_id / 2) * 16;
    uint sc = (simd_id % 2) * 64;

    for (uint kb = 0; kb < K; kb += 16) {
        for (uint i = lid; i < 32 * 16; i += 128) {
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
            simdgroup_float8x8 a0, a1, b0, b1, b2, b3, b4, b5, b6, b7;
            simdgroup_load(a0, &As[sr][kk], 17);
            simdgroup_load(a1, &As[sr + 8][kk], 17);
            simdgroup_load(b0, &Bs[kk][sc], 129);
            simdgroup_load(b1, &Bs[kk][sc + 8], 129);
            simdgroup_load(b2, &Bs[kk][sc + 16], 129);
            simdgroup_load(b3, &Bs[kk][sc + 24], 129);
            simdgroup_load(b4, &Bs[kk][sc + 32], 129);
            simdgroup_load(b5, &Bs[kk][sc + 40], 129);
            simdgroup_load(b6, &Bs[kk][sc + 48], 129);
            simdgroup_load(b7, &Bs[kk][sc + 56], 129);

            simdgroup_multiply_accumulate(acc[0],  a0, b0, acc[0]);
            simdgroup_multiply_accumulate(acc[1],  a0, b1, acc[1]);
            simdgroup_multiply_accumulate(acc[2],  a0, b2, acc[2]);
            simdgroup_multiply_accumulate(acc[3],  a0, b3, acc[3]);
            simdgroup_multiply_accumulate(acc[4],  a0, b4, acc[4]);
            simdgroup_multiply_accumulate(acc[5],  a0, b5, acc[5]);
            simdgroup_multiply_accumulate(acc[6],  a0, b6, acc[6]);
            simdgroup_multiply_accumulate(acc[7],  a0, b7, acc[7]);
            simdgroup_multiply_accumulate(acc[8],  a1, b0, acc[8]);
            simdgroup_multiply_accumulate(acc[9],  a1, b1, acc[9]);
            simdgroup_multiply_accumulate(acc[10], a1, b2, acc[10]);
            simdgroup_multiply_accumulate(acc[11], a1, b3, acc[11]);
            simdgroup_multiply_accumulate(acc[12], a1, b4, acc[12]);
            simdgroup_multiply_accumulate(acc[13], a1, b5, acc[13]);
            simdgroup_multiply_accumulate(acc[14], a1, b6, acc[14]);
            simdgroup_multiply_accumulate(acc[15], a1, b7, acc[15]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (int ar = 0; ar < 2; ar++)
        for (int ac = 0; ac < 8; ac++) {
            uint r = tileRow + sr + ar * 8;
            uint c = tileCol + sc + ac * 8;
            if (r < M && c < N)
                simdgroup_store(acc[ar * 8 + ac], C + r * N + c, N);
        }
}
