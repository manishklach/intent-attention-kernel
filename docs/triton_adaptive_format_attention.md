# Triton Adaptive-Format Decode Attention Kernel

## Motivation

In the repo's arc a KV cache comprises many physical pages, each
containing `page_size × head_dim` values.  When a page is loaded into
on-chip SRAM during decode, the *storage format* of that page determines:

- Total memory traffic (bytes moved)
- Dequantization cost (if any)
- Whether the page can participate in the attention dot product at all

The *precision* per page is already handled by `IntentQuant` (FP16 vs
INT8).  This prototype extends the idea to *storage format* per page,
adding **sparse** and **skip** as explicit page-level tags.

## Kernel Thesis

- **FP16 pages**: direct `float16` loads → no dequant, full fidelity.
- **INT8 pages**: `int8` loads + per-page `float16` scale → 2× reduction
  in memory traffic vs FP16.
- **SPARSE pages**: the KV is stored as index-value pairs (e.g. top-k
  reconstruction).  A sparse dot product queries only the stored
  indices.  This is the hardest path — attention fundamentally requires
  full `QK` scores, and sparsifying the KV dimension changes the
  dot-product structure.  The Triton path is interface-first: the API
  contract accepts sparse inputs but falls back to CPU reference.
- **SKIP pages**: no memory traffic, no contribution.  Enforced at the
  page-selection level.

## Relationship to the Repo

| Component | Role |
|-----------|------|
| `adaptive_format_attention.py` | CPU reference (v0.4) |
| `triton_adaptive_format_attention.py` | Optional Triton GPU kernel (v0.5) |
| `triton_intent_quant_attention.py` | Existing FP16/INT8 per-page precision kernel |
| `fused_selected_quant_decode.py` | Existing fused selected-quant decode approach |

The adaptive-format kernel reuses the per-page dispatch idea from
`triton_intent_quant_attention.py` but *generalises it*: instead of
precision only, the dispatch key is *storage format*.

## Input Contract

```
q             [B, H, D]                    float16, query tokens
page_ids      [B, H, max_selected_pages]   int32,   selected physical page IDs
page_formats  [num_pages]                  int32,   format tag per page
fp16_k_pages  [num_pages, page_size, D]    float16, FP16 K pages
fp16_v_pages  [num_pages, page_size, D]    float16, FP16 V pages
int8_k_pages  [num_pages, page_size, D]    int8,    INT8 K pages     (optional)
int8_v_pages  [num_pages, page_size, D]    int8,    INT8 V pages     (optional)
int8_k_scales [num_pages]                  float32, INT8 K scales    (optional)
int8_v_scales [num_pages]                  float32, INT8 V scales    (optional)
sparse_...    [...]                        ---      SPARSE buffers   (optional)
```

A `page_id` of -1 means "no page at this slot" (skipped).

A page with `page_formats[p] == 3` (`SKIP`) is skipped during
attention — no memory traffic, no flash.

## Execution Flow

1. **Format dispatch**: For each selected page the kernel checks the
   `page_formats[p]` tag.

2. **FP16 path**: Load `page_size` tokens of `float16` KV → direct
   contribution to online-softmax accumulator.

3. **INT8 path**: Load `int8` tile, convert to `float32`, multiply by
   per-page scale → contribution.

4. **SPARSE path**: (Interface-first.)  The kernel accepts sparse
   indices/values but the GPU tile remains zero when sparse pages are
   encountered.  The real sparse dot product is deferred to the CPU
   reference.

5. **Online softmax**: Standard numeric-stable accumulator, matching
   other kernels in the repo.

## Warp Divergence Challenge

Different heads within the same block may have different:

- Number of selected pages (`page_counts`)
- Format mix (FP16-heavy heads vs INT8-heavy heads)

This causes warp divergence on GPU: all warps in a block wait for the
slowest head.  Possible future strategies:

- **Format bucketing**: group heads by format mix into separate kernel
  launches.
- **Separate kernels per format group**: dedicated kernel variants for
  "FP16-only", "mixed", etc.
- **Persistent scheduling**: dynamic warp assignment to heads.

## Limitations

- **No GPU speedup claims** without measurement.
- **No accuracy/perplexity claims** without evaluation.
- **SPARSE Triton path** is interface-first only — real sparse GPU
  execution is deferred.
- **CUDA/Triton required** for the GPU path.  CPU fallback via
  `adaptive_format_attention_reference` is always available.
- **Not a production kernel**.  This is a research prototype exploring
  per-page storage-format dispatch.

## Usage

```python
from intent_attention.triton_adaptive_format_attention import (
    adaptive_format_decode_attention_triton,
    adaptive_format_decode_attention_reference_dispatch,
    AdaptiveFormatKernelConfig,
    make_adaptive_page_tables,
)

cfg = AdaptiveFormatKernelConfig(page_size=16, head_dim=64, max_selected_pages=32)

# GPU path (requires CUDA + Triton)
out = adaptive_format_decode_attention_triton(q, page_ids, page_formats,
                                              fp16_k, fp16_v, ..., config=cfg)

# CPU reference dispatch (always available)
out_cpu = adaptive_format_decode_attention_reference_dispatch(q, page_ids, page_formats,
                                                              fp16_k, fp16_v, ..., config=cfg)
```

## Benchmark

```
pytest benchmarks/bench_triton_adaptive_format_attention.py -v
```

Or, for a dry-run (validates imports only):

```
python benchmarks/bench_triton_adaptive_format_attention.py --dry-run
```

Scenarios: `fp16_only`, `int8_only`, `mixed` (FP16+INT8), `mixed_with_skip`.
