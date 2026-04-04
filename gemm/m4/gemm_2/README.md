# gemm_2 — K-block depth sweep

gemm_1 showed the 32x128 wide tile wins on M4. Now: how deep should the K-block be?

M4 has 300ns+ latency under load. Deeper K-blocks mean more compute per staging load, hiding that latency. But deeper K-blocks use more threadgroup memory, leaving less for dynamic caching.

| Config | File | K-block | Shared mem used | Tradeoff |
|--------|------|---------|-----------------|----------|
| A | config_a.metal | 16 | 2.6 KB | Shallow, many barriers, lots of cache room |
| B | config_b.metal | 64 | 10.1 KB | Deep, fewer barriers, moderate cache |
| C | config_c.metal | 128 | 20.1 KB | Very deep, minimal barriers, cache starved |

```bash
make run
```
