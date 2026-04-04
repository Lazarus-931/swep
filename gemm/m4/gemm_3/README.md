# gemm_3 — Core saturation

gemm_1 found wide tiles win, gemm_2 found K=16 wins. But are we feeding all 10 cores?

A 512x512 matrix with 32x128 tiles = 64 threadgroups. That's 6 per core — should be enough. But maybe smaller tiles with more threadgroups distribute better.

| Config | File | Tile | TG size | Threadgroups at 512x512 |
|--------|------|------|---------|------------------------|
| A | config_a.metal | 32x128 | 128 | 64 (same tile, fewer threads per group) |
| B | config_b.metal | 32x128 | 256 | 64 (baseline from gemm_1) |
| C | config_c.metal | 16x64 | 128 | 256 (4x more groups, 4x less work each) |

Also tests 1024x1024 to see if larger matrices close the gap to peak.

```bash
make run
```
