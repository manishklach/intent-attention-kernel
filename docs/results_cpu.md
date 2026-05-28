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

## Fused Selected-Quant Decode Benchmark

The fused selected-quant decode benchmark measures the CPU reference path for the
fused kernel (which simulates the Triton kernel's algorithm on CPU). This
benchmark is primarily for correctness validation and algorithmic analysis.

### Sample Output (CPU reference)

```
Benchmark: B=1 H=8 D=64 NP=64 PS=16 selected=16/64 (25%)
Device: cpu
Iterations: 50  Warmup: 10

SDPA (full KV):        0.807 ms
Fused selected-quant:   19.454 ms
Ratio (SDPA / Fused): 0.04x

Note: ratio > 1 means fused kernel is faster than full SDPA.
This is an untuned prototype. No GPU speedup is claimed.
Results depend on hardware, precision distribution,
and selected-frac.
```

### Interpreting Fused Selected-Quant Results

- **SDPA (full KV)**: PyTorch SDPA attention over all KV tokens (baseline)
- **Fused selected-quant**: CPU reference implementation of the fused kernel
   algorithm (simulating what the Triton kernel would do)
- **Ratio (SDPA / Fused)**: 
   - > 1.0 means the fused kernel algorithm would be faster than full SDPA
   - < 1.0 means the fused kernel algorithm would be slower than full SDPA
- **Important**: This CPU reference **does not measure GPU performance**. 
   The actual Triton kernel would have different performance characteristics
   due to GPU parallelism, memory bandwidth, and fused execution.

## Adaptive Format KV Attention Benchmark

The adaptive format KV attention benchmark compares the adaptive format reference
implementation against dense and semantic block attention references. This
benchmark demonstrates the computational overhead of handling multiple storage
formats (FP16 dense, INT8 dense+scale, sparse top-k) on CPU.

### Sample Output (CPU reference)

```
Benchmark: B=1 H=4 D=64 KV_pages=64 PS=16 selected=16/64 (25%)
Device: cpu
Iterations: 50  Warmup: 10

Dense attention:         0.686 ms
Semantic block attn:     0.438 ms
Adaptive format attn:    2.775 ms
Ratio (Dense/Adaptive): 0.25x
Ratio (Semantic/Adaptive): 0.16x

Note: ratio > 1 means adaptive format is faster than baseline.
This is a CPU reference. No GPU speedup is claimed.
Results depend on format distribution, sparsity, and hardware.
```

### Interpreting Adaptive Format Results

- **Dense attention**: Standard dense attention over all KV tokens (baseline)
- **Semantic block attn**: Selected-block attention over chosen KV tokens
- **Adaptive format attn**: Attention with mixed storage formats per page
- **Ratio (Dense/Adaptive)**: 
   - > 1.0 means adaptive format is faster than dense attention
   - < 1.0 means adaptive format is slower than dense attention
- **Ratio (Semantic/Adaptive)**: 
   - > 1.0 means adaptive format is faster than semantic block attention
   - < 1.0 means adaptive format is slower than semantic block attention
- **Important**: This CPU reference **does not measure GPU performance** or claim
   any speedup. The adaptive format logic would need to be implemented in a
   GPU kernel to realize potential benefits from reduced memory bandwidth.
- Results depend on the distribution of storage formats (FP16/INT8/SPARSE) and
   sparsity level (K).

## Analytical Estimates

For tensor-free analytical estimates (FLOPs and KV bytes saved), use:

```bash
python benchmarks/bench_cost_model.py
```
