# CPU Benchmark Results

## Warning

CPU timing is **not representative** of GPU kernel performance. PyTorch's
CPU attention path is unoptimised (no FlashAttention, no fused kernels).
These numbers measure Python overhead and naive matmul on CPU only.

## What these numbers measure

- `dense_attention` — PyTorch CPU matmul over all KV tokens (unoptimised, no FlashAttention).
- `semantic_block_attention` — PyTorch CPU gather + matmul over selected KV tokens.
- The ratio is **CPU-only overhead comparison**, not GPU performance.

## What these numbers do NOT measure

- GPU memory bandwidth (HBM).
- Kernel launch overhead.
- Fused attention kernels (FlashAttention, etc.).
- Realistic GPU speedups from skipping KV pages.

GPU speedups depend on many factors not captured here: memory bandwidth,
kernel fusion, page-table walks, and the fraction of skipped tokens. This
CPU benchmark cannot predict GPU performance.

## How to Run

```bash
python benchmarks/bench_cpu_reference.py
```

## Interpreting Output

```
Tokens  Dense (s)   Semantic (s)   CPU Speedup
  512      0.1234        0.0890         1.39x
 1024      0.4567        0.2345         1.95x
 2048      1.2345        0.4567         2.70x
 4096      4.5678        0.8901         5.13x
```

The CPU speedup grows with context length because more tokens are classified
as SKIP, reducing the matmul size on CPU. This is **not** a prediction of
GPU kernel speedup.

For analytical (tensor-free) estimates, use:

```bash
python benchmarks/bench_cost_model.py
```
