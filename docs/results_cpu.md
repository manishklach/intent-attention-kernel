# CPU Benchmark Results

## Warning

CPU timing is **not representative** of GPU kernel performance. PyTorch's
CPU attention path is unoptimised (no FlashAttention, no fused kernels).
These numbers measure Python overhead and naive matmul on CPU only.

## What to expect

- `dense_attention` grows quadratically with KV length.
- `semantic_block_attention` grows with the *selected* KV length.
- The **speedup** on CPU shows the theoretical reduction in matmul size —
  real GPU speedups will be higher due to HBM bandwidth savings from
  skipping page loads.

## How to Run

```bash
python benchmarks/bench_cpu_reference.py
```

## Interpreting Output

```
Tokens  Dense (s)   Semantic (s)   Speedup
  512      0.1234        0.0890      1.39x
 1024      0.4567        0.2345      1.95x
 2048      1.2345        0.4567      2.70x
 4096      4.5678        0.8901      5.13x
```

The speedup grows with context length because more tokens are classified
as SKIP.  For GPU, multiply the speedup by additional HBM-bandwidth gains
from not loading skipped pages.

For analytical (tensor-free) estimates, use:

```bash
python benchmarks/bench_cost_model.py
```
