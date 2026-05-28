# GPU Kernel Plan

## Goal

Implement a Triton kernel that iterates only over physical KV pages
corresponding to selected semantic blocks, skipping unused pages
entirely.

## Proposed Signature

```python
def semantic_attention_triton(
    q, k, v,
    block_table,       # physical page IDs
    num_selected_tokens,
    block_size=64,
    BLOCK_M=64,
    BLOCK_N=64,
) -> torch.Tensor
```

## Steps

1. **Block table construction** (already done in `block_table.py`).
   Convert selected semantic ranges to a list of physical page IDs.

2. **Triton kernel** — mask-less iteration:
   - `pid_m` indexes the query block.
   - Outer loop over physical page IDs from `block_table`.
   - Each page maps to `BLOCK_N` tokens — load K/V from those addresses.
   - Compute local QK^T, softmax (online safe softmax), PV accumulate.

3. **No causal mask needed** — blocks are loaded in logical token order
   so the natural token order of the pages preserves causality if desired.

## Current Status

The repo contains **no real GPU kernel code** — only a stub function
(`semantic_block_attention_triton`) that falls back to the CPU reference
when Triton is absent and raises `NotImplementedError` when Triton/CUDA
are present.  The actual Triton kernel body has not been written yet.
Every statement in this document describes a future design goal, not a
current capability.
