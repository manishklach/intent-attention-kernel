# Architecture

Intent Attention Simulator is organised as a Python package with a
simulator-first design.  Each module has a single responsibility.

## Module Map

```
src/intent_attention/
    __init__.py          # Public API
    _enum.py             # StrEnum base for BlockPolicy
    block_metadata.py    # BlockPolicy, SemanticBlock, BlockLayout
    reference.py         # Dense and selected-block attention (PyTorch)
    cost_model.py        # Analytical FLOPs / memory savings
    synthetic_traces.py  # Generate realistic agentic layouts
    triton_kernel.py     # Triton stub (falls back to reference on CPU)
    block_table.py       # Physical page mapping for paged KV cache
```

## Data Flow

1. The **runtime** constructs a `BlockLayout` by tagging known semantic regions.
2. A `BlockLayout` is **validated** against total token count.
3. Selected blocks are converted to a flat list of token indices.
4. The **reference attention** gathers K/V at those indices and runs
   standard dense attention on the sub-tensor.
5. The **cost model** computes the analytical FLOPs/bytes saved without
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
- Causal masking is not implemented.
