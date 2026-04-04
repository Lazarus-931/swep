# swep

Utilizing the most out of our gpu starts with understanding it.

Swep makes a internal script that runs and reports everthing about your chip, albiet it m1 to m5 across thread size, memory and speed.


Note, for this to work, make sure you:

- have macOS 14+ with Command Line Tools installed (`xcode-select --install`)
- any Apple Silicon Mac (M1, M2...)


To get started, simply

1. clone
2. build
3. run


```bash
git clone https://github.com/Lazarus-931/swep.git && cd swep
make run
```

Results print to stdout and a report is saved to `runs/<chip>.md` based on your device.

## gemm/

GEMM kernels tuned per chip. Each `gemm_N/` folder targets a specific matrix size range with 3+ configs.

```
gemm/m4/gemm_1/    M>=256, N>=256, K>64 — tile shape (max reuse, double-buffered, write-optimized)
gemm/m4/gemm_2/    K-block depth sweep — how deep to hide M4's 300ns loaded latency
gemm/m4/gemm_3/    Core saturation — threadgroup size and tile shape vs core utilization
```

Run any of them:
```bash
cd gemm/m4/gemm_1 && make run
```

## Findings

M4 vs M2, side by side from real runs.

| Property | M2 | M4 | Difference |
|---|---|---|---|
| DRAM bandwidth | 81 GB/s | 102 GB/s | +26% |
| Shared memory bandwidth | 1,571 GB/s | 3,215 GB/s | **2x** |
| L1 cache latency | 57 ns | 43 ns | -25% |
| Barrier cost (32 threads) | 75 ns | 19 ns | **4x cheaper** |
| Kernel launch overhead | 2.7 us | 1.5 us | -44% |
| Texture read bandwidth | 77 GB/s | 479 GB/s | **6x** |
| Buffer read bandwidth | 136 GB/s | 764 GB/s | **5.6x** |
| SIMD shuffle rate | 377 B/s | 833 B/s | **2.2x** |
| FP32 compute | 2,806 GFLOPS | 5,170 GFLOPS | **1.8x** |
| FP16 compute | 2,856 GFLOPS | 5,169 GFLOPS | 1.8x |
| INT32 compute | 619 GIOPS | 971 GIOPS | 1.6x |
| Matrix multiply latency | 1.0 ns | 0.3 ns | **3x faster** |
| SIMD reduction rate | 65 B/s | 115 B/s | 1.8x |
| Atomic (uncontested) | 19.7 B ops/s | 26.2 B ops/s | +33% |
| Register spill cliff | 24→32 acc | 24→32 acc | same |
| TLB capacity | ~128 pages | 1024+ pages | **8x+** |
| Dynamic caching | no (fixed) | **yes** | M3+ feature |
| Concurrent kernels | — | no overlap | serialized |
| Native float atomics | no | no | CAS loop both |
