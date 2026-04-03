# Methodology

How each GPU property is measured, what kernel does the work, and what to look for in the output.

All timing uses `commandBuffer.GPUEndTime - GPUStartTime`. Every probe runs warmup dispatches first, then takes the median of multiple trials to reject outliers.

---

## Measured Properties (existing probes)

### GPU Timer Resolution
**Host:** `probeTimerResolution()` | **Kernel:** reuses `probe_dram_bandwidth` with a tiny 256-byte buffer

Fires 200 near-empty dispatches and collects the GPU timestamps. Sorts them and looks at the smallest gaps between consecutive measurements. The smallest non-zero gap is the timer's quantization step. This tells you how long a kernel needs to run before you can trust the timing.

### Direct Device Queries
**Host:** `probeDirectQueries()` | **Kernel:** none (API queries only)

Reads `device.maxThreadgroupMemoryLength`, `device.maxThreadsPerThreadgroup`, `device.maxBufferLength`, `device.recommendedMaxWorkingSetSize`, `device.hasUnifiedMemory`, and `pso.threadExecutionWidth`. No measurement needed — these are the numbers Apple actually exposes.

### DRAM Bandwidth
**Host:** `probeDRAMBandwidth()` | **Kernel:** `probe_dram_bandwidth`

Copies a 256 MB buffer from src to dst on the GPU. That's way bigger than any on-chip cache, so every byte has to come from DRAM. Total bytes moved (read + write = 512 MB) divided by GPU time gives you the raw memory bandwidth.

### Threadgroup Memory Bandwidth
**Host:** `probeTGMemBandwidth()` | **Kernel:** `probe_tgmem_bandwidth`

Each thread writes then reads a float4 from threadgroup memory 256 times in a tight loop, with barriers between iterations. 1024 threadgroups of 256 threads each run in parallel. Total bytes transferred divided by time gives the aggregate shared memory bandwidth.

### L1 / SLC Cache Size
**Host:** `probeCacheSizes()` | **Kernel:** `probe_cache_sweep`

Reads through a buffer in a tight loop, sweeping the working set size from 4 KB to 48 MB. When the working set fits in L1, you get fast throughput. When it spills past L1, throughput drops — that's your L1 size. When it spills past SLC into DRAM, throughput drops again — that's your SLC size. You're looking for the two cliffs.

### Cache Hit Latency
**Host:** `probeCacheLatency()` | **Kernel:** `probe_cache_latency`

Builds a random pointer chase inside a buffer — each element stores the index of the next one to load. A single thread walks the chain for 100K hops. Because each load depends on the previous one, the GPU can't hide the latency. Total time / hops = latency per load. Run this at different working set sizes and you see latency jump at each cache level boundary.

### Register Spill Boundary
**Host:** `probeRegisterSpill()` | **Kernels:** `probe_regpressure_{4,8,16,24,32,48}`

Six kernel variants that each run a simdgroup MMA loop, but with different numbers of `simdgroup_float8x8` accumulators held live simultaneously. At some count, the compiler runs out of registers and starts spilling to threadgroup memory. That shows up as a massive TFLOPS cliff. On this M2, it's between 24 and 32 accumulators.

### Shader Core Count
**Host:** `probeShaderCoreCount()` | **Kernel:** `probe_core_saturation`

Each thread runs a long chain of FMAs with a known FLOP count. Sweep the number of threadgroups from 1 to 128. When you're below the core count, adding threadgroups adds proportional throughput. Once every core is busy, throughput plateaus. The knee in the curve is your core count.

### Threadgroup Memory Bank Count
**Host:** `probeBankConflicts()` | **Kernel:** `probe_bank_conflicts`

One SIMD's worth of threads (32) each access threadgroup memory at a controlled stride. Sweep the stride from 1 to 64. When the stride equals the bank count, all threads hit the same bank and serialize — you see a bandwidth dip. The period of those dips is the bank count. On this hardware: 32 banks.

### SLC Associativity
**Host:** `probeSLCAssociativity()` | **Kernel:** `probe_slc_assoc`

Allocates addresses spaced exactly one SLC-size apart, then accesses 1 through 24 of them in a loop. When the number of conflicting addresses exceeds the associativity, the cache can't hold them all and you get conflict misses. The bandwidth drop at N ways means the associativity is N-1. On this M2: 16-way.

### Pipeline Depth / FMA Latency
**Host:** `probePipelineDepth()` | **Kernels:** `probe_pipeline_dep{1,2,4,8}`, `probe_pipeline_indep`

Five kernel variants with dependency chains of length 1, 2, 4, 8, and a fully independent version. The dependent chain measures true instruction latency (since the GPU can't overlap them). The independent version measures peak throughput. Latency / throughput = how many operations the pipeline needs in flight to stay full.

### Occupancy vs Threadgroup Memory
**Host:** `probeOccupancy()` | **Kernel:** `probe_occupancy`

Launches a compute kernel that claims a variable amount of threadgroup memory, sweeping from 1 KB to 32 KB. More memory per threadgroup means fewer threadgroups can fit on a core simultaneously. Step drops in throughput reveal the occupancy boundaries and tell you how the on-chip SRAM is partitioned.

---

## New Probes (added in this round)

### Atomic Operation Throughput
**Host:** `probeAtomicThroughput()` | **Kernel:** `probe_atomic_throughput`

Each thread does 100K `atomic_fetch_add` operations to device memory. We sweep the number of target counters from 1 (max contention — every thread fights over the same word) to 1024 (zero contention). The throughput curve shows how the atomic unit handles contention and reveals its peak uncontested rate. On this M2: ~20 Gatomics/s when uncontested, drops to 0.7 when fully serialized on one address.

### SIMD Shuffle Bandwidth
**Host:** `probeSimdShuffle()` | **Kernel:** `probe_simd_shuffle`

Each thread does 500K iterations of 5 `simd_shuffle_xor` operations (XOR masks 1, 2, 4, 8, 16 — a butterfly pattern). This measures how fast threads within a SIMD can exchange register values without going through memory. On this M2: ~377 Gshuffles/s, about 1.5 TB/s effective bandwidth.

### SIMD Reduction Throughput
**Host:** `probeSimdReduce()` | **Kernel:** `probe_simd_reduce`

Each thread calls `simd_sum` in a tight loop. `simd_sum` reduces a value across all 32 lanes in a SIMD. This tells you the hardware cost of a full warp reduction — whether it's a single-cycle intrinsic or decomposed into shuffles. On this M2: ~66 Greductions/s.

### Integer vs Float ALU Throughput
**Host:** `probeIntVsFloat()` | **Kernels:** `probe_int_throughput`, `probe_float_throughput`

Two kernels running structurally identical workloads — one doing integer multiply-add, one doing float4 FMA. By comparing their throughput, you can tell whether integer and float operations share an ALU or have separate execution pipes. On this M2: FP32 is about 5.5x faster than INT32, meaning int probably shares with float but has lower throughput (not a separate dedicated unit).

### Half-Precision (FP16) vs FP32 Throughput
**Host:** `probeHalfPrecision()` | **Kernels:** `probe_half_throughput`, `probe_float_throughput`

Same structure as above, but comparing half4 FMA to float4 FMA. If fp16 runs at exactly 2x the fp32 rate, the hardware packs two fp16 ops into one fp32 slot. If they're roughly equal, fp16 just runs on the same datapath with no rate advantage. On this M2: 1:1 ratio — fp16 is the same speed as fp32. No packed fp16 execution.

### Cache Line Size
**Host:** `probeCacheLineSize()` | **Kernel:** `probe_cacheline`

Builds pointer-chase chains inside a 4 KB buffer (guaranteed L1-resident) with increasing byte strides. When the stride is smaller than a cache line, consecutive hops hit the same line — prefetched for free. When stride exceeds the line size, every hop fetches a new line and latency should jump. On this M2: flat at ~57 ns across all strides (meaning 4 KB fits so fully in L1 that line boundaries don't matter at this scale).

### Memory Coalescing Penalty
**Host:** `probeCoalescing()` | **Kernel:** `probe_coalescing`

Each thread reads from device memory at a controlled stride (1 = contiguous/coalesced, 256 = maximally scattered). Bandwidth at stride-1 vs stride-N tells you how aggressively the memory controller coalesces requests from threads in the same SIMD. On this M2: stride-1 gets ~69 GB/s, stride-16 drops to ~23 GB/s — about 3x penalty for non-coalesced access.

### Threadgroup Barrier Cost
**Host:** `probeBarrierCost()` | **Kernel:** `probe_barrier_cost`

Runs 500K `threadgroup_barrier` calls in a loop, sweeping the threadgroup size from 32 to 1024. The per-barrier cost reveals the synchronization hardware. On this M2: ~49 ns per barrier for groups up to 256 threads, rising to ~66 ns at 1024. The jump suggests the sync hardware works at SIMD granularity and needs extra cycles when multiple SIMDs must coordinate.

### Dispatch Overhead
**Host:** `probeDispatchOverhead()` | **Kernel:** `probe_dispatch_empty`

An empty kernel — just writes zero to one output float. Sweep from 1 to 32K threads. The time at 1 thread is pure dispatch overhead (encoding, scheduling, command buffer round-trip). The growth shows per-threadgroup scheduling cost. On this M2: ~2.1 µs base overhead, barely grows to ~2.7 µs at 32K threads. Very cheap dispatch.

### Texture Read vs Buffer Read Bandwidth
**Host:** `probeTextureBandwidth()` | **Kernel:** `probe_texture_read`, reuses `probe_cache_sweep` for comparison

Reads from a 2048x2048 RGBA32Float texture in a tight loop, then compares against reading the same amount of data from a plain buffer. The texture path goes through the texture cache and addressing hardware, which can be faster (hardware filtering, spatial locality) or slower (overhead) depending on access pattern. On this M2: texture path gets ~208 GB/s, buffer path gets ~453 GB/s — texture path is about 2x slower for raw sequential reads with no filtering.

---

## 10 More Properties You Could Probe Next

1. **Threadgroup memory read-after-write latency** — Instead of throughput, measure the latency of a single threadgroup memory load that depends on a prior store. Write-barrier-read chain timed at single-thread level. Tells you the actual SRAM access time independent of bandwidth.

2. **Instruction cache size** — Build kernels with increasing code footprints (unroll a loop to different degrees) and measure FLOPS. When the instruction cache overflows, you'll see a throughput cliff as the GPU starts fetching instructions from memory.

3. **Warp scheduler fairness / round-robin period** — Launch multiple threadgroups on the same core, each recording timestamps via atomic operations into a shared buffer. The interleaving pattern of timestamps reveals the scheduling policy (round-robin, oldest-first, etc.) and the scheduling quantum.

4. **TLB size and miss penalty** — Pointer-chase across buffers with page-aligned strides (e.g., 16 KB apart on Apple). Sweep the number of touched pages. When you exceed TLB capacity, latency jumps reveal the TLB size and the page table walk cost.

5. **Memory controller channel count** — Allocate multiple large buffers and read them concurrently from different threadgroups. Bandwidth scaling from 1 to N concurrent streams saturates at the channel count. Or use strided patterns that alias to the same DRAM bank and measure degradation.

6. **Simdgroup matrix instruction (MMA) latency** — Run a single `simdgroup_multiply_accumulate` in a dependency chain (output feeds back as input). Measure the true latency of one matrix multiply instruction vs the throughput-optimized case where multiple independent MMAs overlap.

7. **Device-to-host readback latency** — Time the gap between `commandBuffer.GPUEndTime` and `commandBuffer.GPUStartTime` of a subsequent command that reads the output. Reveals the cost of the GPU writing results and the CPU seeing them through the unified memory fabric.

8. **Indirect dispatch overhead** — Compare the cost of `dispatchThreads:` vs `dispatchThreadgroups:` with an indirect buffer. The indirect path reads grid dimensions from a GPU buffer, which adds overhead. The difference tells you the cost of the indirection.

9. **Concurrent kernel execution** — Encode two independent kernels into the same command buffer (compute + compute). Measure whether the GPU overlaps them or serializes. If overlap is possible, throughput should exceed a single kernel's peak, revealing whether the GPU supports concurrent kernel execution on different cores.

10. **Threadgroup memory allocation granularity** — The hardware likely allocates threadgroup memory in fixed-size blocks (e.g., 256 bytes or 1 KB). Sweep the threadgroup memory size in small increments and measure occupancy. Jumps in occupancy at specific sizes reveal the allocation granularity.
