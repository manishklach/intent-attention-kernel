# GPU Kernel Plan

## Goal

Implement a Triton kernel that iterates only over physical KV pages
corresponding to selected semantic blocks, skipping unused pages
entirely.

## Kernel Design

The planned GPU kernel maps as follows:

- **One kernel launch** per (batch, head, query-block).
- **Block metadata** (or a flattened block table) is passed to the kernel
  as a device-side array of physical page IDs.
- The kernel iterates **only** over pages in the block table, skipping
  pages belonging to `SKIP` blocks.
- Each loaded page contributes `BLOCK_N` tokens to the local QK^T
  computation.  Online safe softmax accumulates across pages.
- No causal mask is needed because pages are loaded in logical token
  order (the block table preserves sequence order).

## Proposed Signature

```python
def semantic_attention_triton(
    q, k, v,
    block_table,          # physical page IDs for selected blocks
    num_selected_tokens,
    block_size=64,
    BLOCK_M=64,
    BLOCK_N=64,
) -> torch.Tensor
```

## Implementation Steps

1. **Block table construction** (already done in `block_table.py`).
   Convert selected semantic ranges to a list of physical page IDs.

2. **Triton kernel** — mask-less iteration:
   - `pid_m` indexes the query block.
   - Outer loop over physical page IDs from `block_table`.
   - Each page maps to `BLOCK_N` tokens — load K/V from those addresses.
   - Compute local QK^T, softmax (online safe softmax), PV accumulate.

3. **Support paged KV cache** — the block table maps logical page IDs
   to physical page IDs, enabling the kernel to work with a standard
   paged KV cache (e.g., vLLM-style page tables).

4. **Future fusion opportunities** — RoPE application and KV append
   could be fused with the attention kernel to reduce global memory
   round-trips.

## Validation Required

The kernel must be validated on **real NVIDIA hardware** (A100, H100,
or similar).  Performance claims can only be made after:

- Correctness checks against the CPU reference.
- Memory bandwidth profiling (achieved vs. peak HBM).
- End-to-end latency comparison with FlashAttention on equivalent
  workloads.

## Current Status

The repo contains **no real GPU kernel code** — only a stub function
(`semantic_block_attention_triton`) that falls back to the CPU reference
when Triton is absent and raises `NotImplementedError` when Triton/CUDA
are present.  The actual Triton kernel body has not been written yet.

**Every statement in this document describes a future design goal, not a
current capability.**
