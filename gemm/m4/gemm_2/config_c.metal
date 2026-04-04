// K-block depth sweep on M4 — using winning 32x128 wide tile
// Config C: K-block 48 (28.9 KB total — near the 32 KB ceiling)

#include <metal_stdlib>
using namespace metal;

kernel void gemm_k48(
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
    threadgroup float As[32][49];
    threadgroup float Bs[48][129];

    uint tileRow = gid.y * 32;
    uint tileCol = gid.x * 128;

    simdgroup_float8x8 acc[16];
    for (int i = 0; i < 16; i++) acc[i] = simdgroup_float8x8(0);

    uint sr = (simd_id / 4) * 16;
    uint sc = (simd_id % 4) * 32;

    for (uint kb = 0; kb < K; kb += 48) {
        uint kblock = min(48u, K - kb);
        for (uint i = lid; i < 32 * 48; i += 256) {
            uint r = i / 48, c = i % 48;
            uint gr = tileRow + r, gc = kb + c;
            As[r][c] = (gr < M && gc < K && c < kblock) ? A[gr * K + gc] : 0;
        }
        for (uint i = lid; i < 48 * 128; i += 256) {
            uint r = i / 128, c = i % 128;
            uint gr = kb + r, gc = tileCol + c;
            Bs[r][c] = (gr < K && gc < N && r < kblock) ? B[gr * N + gc] : 0;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint kk = 0; kk < kblock; kk += 8) {
            simdgroup_float8x8 a0, a1, b0, b1, b2, b3;
            simdgroup_load(a0, &As[sr][kk], 49);
            simdgroup_load(a1, &As[sr + 8][kk], 49);
            simdgroup_load(b0, &Bs[kk][sc], 129);
            simdgroup_load(b1, &Bs[kk][sc + 8], 129);
            simdgroup_load(b2, &Bs[kk][sc + 16], 129);
            simdgroup_load(b3, &Bs[kk][sc + 24], 129);

            simdgroup_multiply_accumulate(acc[0],  a0, b0, acc[0]);
            simdgroup_multiply_accumulate(acc[1],  a0, b1, acc[1]);
            simdgroup_multiply_accumulate(acc[2],  a0, b2, acc[2]);
            simdgroup_multiply_accumulate(acc[3],  a0, b3, acc[3]);
            simdgroup_multiply_accumulate(acc[4],  a1, b0, acc[4]);
            simdgroup_multiply_accumulate(acc[5],  a1, b1, acc[5]);
            simdgroup_multiply_accumulate(acc[6],  a1, b2, acc[6]);
            simdgroup_multiply_accumulate(acc[7],  a1, b3, acc[7]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (int ar = 0; ar < 2; ar++)
        for (int ac = 0; ac < 4; ac++) {
            uint r = tileRow + sr + ar * 8;
            uint c = tileCol + sc + ac * 8;
            if (r < M && c < N)
                simdgroup_store(acc[ar * 4 + ac], C + r * N + c, N);
        }
}
