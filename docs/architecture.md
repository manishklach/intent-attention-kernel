# Architecture

Intent-Aware KV Execution is organised as a Python package with a
simulator-first design. Each module has a single responsibility.

## Module Map

```
src/intent_attention/
    __init__.py              # Public API
    _enum.py                 # StrEnum base for BlockPolicy
    block_metadata.py        # BlockPolicy, SemanticBlock, BlockLayout
    block_scorer.py          # Dynamic block scoring (cosine similarity)
    block_table.py           # Physical page mapping for paged KV cache
    cost_model.py            # Analytical FLOPs / memory savings
    hf_patch.py              # HuggingFace Transformers integration
    kv_quant.py              # INT8 KV cache quantisation modeling
    prefetch.py              # Speculative KV block prefetch simulation
    reference.py             # Dense and selected-block attention (PyTorch)
    synthetic_traces.py      # Generate realistic agentic layouts
    triton_kernel.py         # Triton stub (falls back to reference on CPU)
    triton_kernel_quant.py   # INT8 quantised Triton kernel stub
    vllm_bridge.py           # vLLM-style paged-attention bridge
```

## Data Flow

1. The **runtime** constructs a `BlockLayout` by tagging known semantic
   regions with policies and optional scores.
2. A `BlockLayout` is **validated** against total token count.
3. Selected blocks are converted to a flat list of token indices.
4. Optional **dynamic scoring** re-ranks ambiguous ATTEND blocks using
   query-to-block cosine similarity.
5. Optional **block table** maps logical ranges to physical KV pages.
6. Optional **prefetch** predicts likely next-step pages.
7. The **reference attention** gathers K/V at selected indices and runs
   standard dense attention on the sub-tensor.
8. The **cost model** computes analytical FLOPs/bytes saved without
   running any tensors.

## Design Principles

- **CPU-first**: everything must run without CUDA or Triton.
- **Deterministic**: layout generators accept an optional `seed`.
- **Verifiable**: `semantic_block_attention` output equals
  `dense_attention` on the gathered K/V sub-tensor.
- **Honest**: no GPU speedups are claimed or implied.

## Current Limitations

- This is a **simulator**, not a GPU kernel.
- The reference attention path gathers K/V then computes dense attention;
  a real GPU kernel would avoid loading skipped pages in the first place.
- Causal masking uses a `triu(-inf)` mask on the selected sub-tensor.
- Quantization is prototype-level with no accuracy or throughput validation.
- Prefetch is a simulation with no hardware latency measurements.
