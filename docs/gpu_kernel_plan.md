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

### Causal Masking

> The block table alone is not sufficient for causal masking.

For **single-token decode**, if the query position is strictly after all
selected KV tokens, causal masking may be unnecessary.  For **prefill**
or **multi-query blocks**, logical page order is not sufficient — the GPU
kernel must still support causal masking or query-position-aware masking.

The current CPU reference does **not** implement causal selected-block
attention because the compacted selected-KV tensor loses the original
token-position relationship.  A future GPU kernel will need explicit
query-position-aware masking (e.g., per-page token bounds or
query-position comparison) to support causal semantics on selected KV
pages.

### Partial Pages and Block Boundaries

- If a selected semantic block starts or ends in the middle of a KV page,
  the kernel needs per-page token bounds or per-element masks.
- Adjacent or overlapping physical pages should be de-duplicated while
  preserving logical token order.
- The current `BlockTable` is a CPU simulation — it returns page IDs
  without token-level masks.

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

The repo contains a Triton GPU kernel defined under the
`if _triton_available` guard in `triton_kernel.py`.  The kernel is
compiled and launched when Triton and CUDA are present; on CPU-only
machines it falls back to the PyTorch reference implementation.
Performance measurements on real NVIDIA hardware are still needed.

### Optional KV Dequant in the Attention Loop

If the KV cache is stored in INT8 format, the kernel must dequantize
each page before or during the attention computation:

- Dequant inside the attention loop adds ALU and register pressure.
- Dequant before the loop (materialize fp16 pages) adds memory pressure.
- The ideal trade-off depends on page reuse, dequant cost, and
  bandwidth savings from reading INT8 instead of fp16.

### Quantization Timing

Quantization (fp16 to INT8) should ideally happen:

- At page creation time (prefill) to avoid redundant conversion.
- At cold-page transition (when a page is first selected after a long
  period of disuse).
- **Not** repeatedly every step — re-quantizing a hot page every step
  would waste compute and may not save bandwidth.

### Page Deduplication

When multiple logical blocks map to the same physical page, the kernel
must deduplicate page IDs while preserving the logical token order.
Arbitrary sorting of page IDs would break causal masking and
position-dependent computations.

**Every statement in this document describes a design goal or
analysis direction, not a validated performance claim.**
