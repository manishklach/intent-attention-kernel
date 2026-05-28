# CPU Benchmark Results

## Warning

CPU timing is **not representative** of GPU kernel performance.  PyTorch's
CPU attention path is unoptimised (no FlashAttention, no fused kernels).
These numbers measure Python overhead and naive matmul on CPU only.

## What These Numbers Measure

- `dense_attention` — PyTorch CPU matmul over all KV tokens (unoptimised,
  no FlashAttention).
- `semantic_block_attention` — PyTorch CPU gather + matmul over selected
  KV tokens.
- The ratio is **CPU-only overhead comparison**, not GPU performance.

## What These Numbers Do NOT Measure

- GPU memory bandwidth (HBM).
- Kernel launch overhead.
- Fused attention kernels (FlashAttention, etc.).
- Realistic GPU speedups from skipping KV pages.

GPU speedups depend on many factors not captured here: memory bandwidth,
kernel fusion, page-table walks, and the fraction of skipped tokens.  This
CPU benchmark cannot predict GPU performance.

## Sample Output

```
Tokens    Dense (s)   Semantic (s)   CPU Ratio
     512       0.0032         0.0030       1.08x
    1024       0.0054         0.0058       0.94x
    2048       0.0089         0.0100       0.89x
    4096       0.0163         0.0140       1.16x
```

## Interpreting CPU Ratio

- A ratio **above 1.0** means dense attention took longer on CPU (the
  selected-block path was faster).
- A ratio **below 1.0** can happen for small cases due to gather overhead,
  cache behaviour, PyTorch dispatch, or small-tensor matmul effects.
- The trend across increasing KV lengths is more informative than any
  single number.
- **CPU Ratio is not a GPU speedup prediction.**

## How to Run

```bash
python benchmarks/bench_cpu_reference.py
```

## Analytical Estimates

For tensor-free analytical estimates (FLOPs and KV bytes saved), use:

```bash
python benchmarks/bench_cost_model.py
```
