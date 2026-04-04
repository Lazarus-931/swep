// For M>=256, N>=256, K>64 on M4
// Config A: 48x64 tile, K-block 64, 24 accumulators, single-buffered

#include <metal_stdlib>
using namespace metal;

kernel void gemm_config_a(
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
    threadgroup float As[48][33];
    threadgroup float Bs[64][65];

    uint tileRow = gid.y * 48;
    uint tileCol = gid.x * 64;

    simdgroup_float8x8 acc[24];
    for (int i = 0; i < 24; i++) acc[i] = simdgroup_float8x8(0);

    uint sr = (simd_id / 2) * 24;
    uint sc = (simd_id % 2) * 32;

    for (uint kb = 0; kb < K; kb += 64) {
        for (uint i = lid; i < 48 * 64; i += 256) {
            uint r = i / 64, c = i % 64;
            uint gr = tileRow + r, gc = kb + c;
            As[r][c] = (gr < M && gc < K) ? A[gr * K + gc] : 0;
        }
        for (uint i = lid; i < 64 * 64; i += 256) {
            uint r = i / 64, c = i % 64;
            uint gr = kb + r, gc = tileCol + c;
            Bs[r][c] = (gr < K && gc < N) ? B[gr * N + gc] : 0;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint kk = 0; kk < 64; kk += 8) {
            simdgroup_float8x8 a0, a1, a2, b0, b1, b2, b3;
            simdgroup_load(a0, &As[sr][kk], 33);
            simdgroup_load(a1, &As[sr + 8][kk], 33);
            simdgroup_load(a2, &As[sr + 16][kk], 33);
            simdgroup_load(b0, &Bs[kk][sc], 65);
            simdgroup_load(b1, &Bs[kk][sc + 8], 65);
            simdgroup_load(b2, &Bs[kk][sc + 16], 65);
            simdgroup_load(b3, &Bs[kk][sc + 24], 65);

            simdgroup_multiply_accumulate(acc[0],  a0, b0, acc[0]);
            simdgroup_multiply_accumulate(acc[1],  a0, b1, acc[1]);
            simdgroup_multiply_accumulate(acc[2],  a0, b2, acc[2]);
            simdgroup_multiply_accumulate(acc[3],  a0, b3, acc[3]);
            simdgroup_multiply_accumulate(acc[4],  a1, b0, acc[4]);
            simdgroup_multiply_accumulate(acc[5],  a1, b1, acc[5]);
            simdgroup_multiply_accumulate(acc[6],  a1, b2, acc[6]);
            simdgroup_multiply_accumulate(acc[7],  a1, b3, acc[7]);
            simdgroup_multiply_accumulate(acc[8],  a2, b0, acc[8]);
            simdgroup_multiply_accumulate(acc[9],  a2, b1, acc[9]);
            simdgroup_multiply_accumulate(acc[10], a2, b2, acc[10]);
            simdgroup_multiply_accumulate(acc[11], a2, b3, acc[11]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (int ar = 0; ar < 3; ar++)
        for (int ac = 0; ac < 4; ac++) {
            uint r = tileRow + sr + ar * 8;
            uint c = tileCol + sc + ac * 8;
            if (r < M && c < N)
                simdgroup_store(acc[ar * 4 + ac], C + r * N + c, N);
        }
}
