# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Metal GPU probe suite that empirically measures Apple Silicon GPU memory hierarchy and microarchitectural properties through timing side channels, occupancy manipulation, and compiler output inspection. Results feed into GEMMConfig tile sizing and dispatch logic.

## Build & Run

```bash
# Build (Release)
xcodebuild -project GPUProbe.xcodeproj -scheme GPUProbe -configuration Release build

# Run (from DerivedData)
$(xcodebuild -project GPUProbe.xcodeproj -scheme GPUProbe -configuration Release -showBuildSettings | grep -m1 BUILT_PRODUCTS_DIR | awk '{print $3}')/GPUProbe

# Compile shaders standalone (for disassembly)
xcrun -sdk macosx metal -o Shaders.air Shaders.metal
xcrun -sdk macosx metallib -o Shaders.metallib Shaders.air
xcrun metal-objdump -d Shaders.metallib | grep -E 'spill|SPILL|stack'
```

## Architecture

- **Shaders.metal** — All probe compute kernels. Each kernel isolates one GPU property by making only that property the bottleneck. Kernels: `probe_dram_bandwidth`, `probe_tgmem_bandwidth`, `probe_cache_sweep`, `probe_cache_latency`, `probe_regpressure_{4,8,16,24,32,48}`, `probe_core_saturation`, `probe_bank_conflicts`, `probe_slc_assoc`, `probe_pipeline_dep{1,2,4,8}`, `probe_pipeline_indep`, `probe_occupancy`.

- **main.mm** — Host-side Objective-C++ that configures Metal, dispatches each probe kernel with controlled parameters, times via `commandBuffer.GPUEndTime - GPUStartTime`, and prints results. Runtime-compiles Shaders.metal if no metallib is found.

## Measurement approach

All timing uses `cb.GPUEndTime - cb.GPUStartTime` with warmup dispatches and median of multiple trials. Probes that infer hidden properties (core count, bank layout, SLC associativity, pipeline depth) work by sweeping a parameter and looking for throughput cliffs or periodicity in the results.
