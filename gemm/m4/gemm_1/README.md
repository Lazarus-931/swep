# gemm_1 — M>=256, N>=256, K>64

Three configs tuned for medium matrices on M4.

| Config | File | Tile | Accumulators | K-block | Strategy |
|--------|------|------|-------------|---------|----------|
| A | config_a.metal | 48x64 | 24 | 64 | Max compute reuse, single-buffered |
| B | config_b.metal | 32x64 | 16 | 32 | Double-buffered, hides DRAM latency |
| C | config_c.metal | 32x128 | 16 | 32 | Wide output, write-optimized |

```bash
make run
```
