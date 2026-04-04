# gemm_2 — K-block depth sweep

gemm_1 showed the 32x128 wide tile wins on M4. Now: how deep should the K-block be?

Deeper K = more compute per staging load, but more threadgroup memory used.

| Config | File | K-block | Shared mem | Tradeoff |
|--------|------|---------|------------|----------|
| A | config_a.metal | 16 | 10.4 KB | Shallow, lots of cache room |
| B | config_b.metal | 32 | 20.6 KB | Middle ground |
| C | config_c.metal | 48 | 30.9 KB | Near the 32 KB ceiling, cache starved |

```bash
make run
```
