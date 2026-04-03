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

kernel void gemm_simd(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C       [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    constant uint& K [[buffer(5)]],
    uint2 gid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]])
{
    simdgroup_float8x8 acc[4];
    for (int i = 0; i < 4; i++) acc[i] = simdgroup_float8x8(0);
    simdgroup_float8x8 a, b0, b1;

    uint row = gid.y * 8;
    uint col = gid.x * 16;
    if (row >= M || col >= N) return;

    for (uint k = 0; k < K; k += 8) {
        simdgroup_load(a, A + row * K + k, K);
        simdgroup_load(b0, B + k * N + col, N);
        simdgroup_load(b1, B + k * N + col + 8, N);
        simdgroup_multiply_accumulate(acc[0], a, b0, acc[0]);
        simdgroup_multiply_accumulate(acc[1], a, b1, acc[1]);
    }
    simdgroup_store(acc[0], C + row * N + col, N);
    if (col + 8 < N) simdgroup_store(acc[1], C + row * N + col + 8, N);
}

kernel void gemm_tiled(
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
    threadgroup float As[64 * 32];
    threadgroup float Bs[32 * 64];

    uint tileRow = gid.y * 64;
    uint tileCol = gid.x * 64;

    simdgroup_float8x8 acc[16];
    for (int i = 0; i < 16; i++) acc[i] = simdgroup_float8x8(0);

    uint simd_row = (simd_id / 2) * 16;
    uint simd_col = (simd_id % 2) * 32;

    for (uint kb = 0; kb < K; kb += 32) {
        for (uint i = lid; i < 64 * 32; i += 256) {
            uint r = i / 32, c = i % 32;
            uint gr = tileRow + r, gc = kb + c;
            As[r * 32 + c] = (gr < M && gc < K) ? A[gr * K + gc] : 0;
        }
        for (uint i = lid; i < 32 * 64; i += 256) {
            uint r = i / 64, c = i % 64;
            uint gr = kb + r, gc = tileCol + c;
            Bs[r * 64 + c] = (gr < K && gc < N) ? B[gr * N + gc] : 0;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint kk = 0; kk < 32; kk += 8) {
            simdgroup_float8x8 a0, a1, b0, b1, b2, b3;
            simdgroup_load(a0, As + (simd_row) * 32 + kk, 32);
            simdgroup_load(a1, As + (simd_row + 8) * 32 + kk, 32);
            simdgroup_load(b0, Bs + kk * 64 + simd_col, 64);
            simdgroup_load(b1, Bs + kk * 64 + simd_col + 8, 64);
            simdgroup_load(b2, Bs + kk * 64 + simd_col + 16, 64);
            simdgroup_load(b3, Bs + kk * 64 + simd_col + 24, 64);

            simdgroup_multiply_accumulate(acc[0], a0, b0, acc[0]);
            simdgroup_multiply_accumulate(acc[1], a0, b1, acc[1]);
            simdgroup_multiply_accumulate(acc[2], a0, b2, acc[2]);
            simdgroup_multiply_accumulate(acc[3], a0, b3, acc[3]);
            simdgroup_multiply_accumulate(acc[4], a1, b0, acc[4]);
            simdgroup_multiply_accumulate(acc[5], a1, b1, acc[5]);
            simdgroup_multiply_accumulate(acc[6], a1, b2, acc[6]);
            simdgroup_multiply_accumulate(acc[7], a1, b3, acc[7]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (int ar = 0; ar < 2; ar++) {
        for (int ac = 0; ac < 4; ac++) {
            uint r = tileRow + simd_row + ar * 8;
            uint c = tileCol + simd_col + ac * 8;
            if (r < M && c < N)
                simdgroup_store(acc[ar * 4 + ac], C + r * N + c, N);
        }
    }
}

kernel void gemm_tiled_24acc(
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
    threadgroup float As[48 * 32];
    threadgroup float Bs[32 * 64];

    uint tileRow = gid.y * 48;
    uint tileCol = gid.x * 64;

    simdgroup_float8x8 acc[24];
    for (int i = 0; i < 24; i++) acc[i] = simdgroup_float8x8(0);

    uint simd_row = (simd_id / 2) * 24;
    uint simd_col = (simd_id % 2) * 32;

    for (uint kb = 0; kb < K; kb += 32) {
        for (uint i = lid; i < 48 * 32; i += 256) {
            uint r = i / 32, c = i % 32;
            uint gr = tileRow + r, gc = kb + c;
            As[r * 32 + c] = (gr < M && gc < K) ? A[gr * K + gc] : 0;
        }
        for (uint i = lid; i < 32 * 64; i += 256) {
            uint r = i / 64, c = i % 64;
            uint gr = kb + r, gc = tileCol + c;
            Bs[r * 64 + c] = (gr < K && gc < N) ? B[gr * N + gc] : 0;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint kk = 0; kk < 32; kk += 8) {
            simdgroup_float8x8 a0, a1, a2, b0, b1, b2, b3;
            simdgroup_load(a0, As + (simd_row) * 32 + kk, 32);
            simdgroup_load(a1, As + (simd_row + 8) * 32 + kk, 32);
            simdgroup_load(a2, As + (simd_row + 16) * 32 + kk, 32);
            simdgroup_load(b0, Bs + kk * 64 + simd_col, 64);
            simdgroup_load(b1, Bs + kk * 64 + simd_col + 8, 64);
            simdgroup_load(b2, Bs + kk * 64 + simd_col + 16, 64);
            simdgroup_load(b3, Bs + kk * 64 + simd_col + 24, 64);

            simdgroup_multiply_accumulate(acc[0], a0, b0, acc[0]);
            simdgroup_multiply_accumulate(acc[1], a0, b1, acc[1]);
            simdgroup_multiply_accumulate(acc[2], a0, b2, acc[2]);
            simdgroup_multiply_accumulate(acc[3], a0, b3, acc[3]);
            simdgroup_multiply_accumulate(acc[4], a1, b0, acc[4]);
            simdgroup_multiply_accumulate(acc[5], a1, b1, acc[5]);
            simdgroup_multiply_accumulate(acc[6], a1, b2, acc[6]);
            simdgroup_multiply_accumulate(acc[7], a1, b3, acc[7]);
            simdgroup_multiply_accumulate(acc[8], a2, b0, acc[8]);
            simdgroup_multiply_accumulate(acc[9], a2, b1, acc[9]);
            simdgroup_multiply_accumulate(acc[10], a2, b2, acc[10]);
            simdgroup_multiply_accumulate(acc[11], a2, b3, acc[11]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (int ar = 0; ar < 3; ar++) {
        for (int ac = 0; ac < 4; ac++) {
            uint r = tileRow + simd_row + ar * 8;
            uint c = tileCol + simd_col + ac * 8;
            if (r < M && c < N)
                simdgroup_store(acc[ar * 4 + ac], C + r * N + c, N);
        }
    }
}
