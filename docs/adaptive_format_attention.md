# Adaptive Format KV Attention Reference

## Motivation

In agentic long-context inference, different parts of the KV cache may benefit from different physical storage formats.
For example:
- System prompts and recent conversation might be best kept in dense FP16 for highest precision.
- Retrieved documents or tool outputs might be suitable for INT8 quantization to save memory.
- Very sparse regions (e.g., scratchpads with infrequent updates) might be stored sparsely to save both memory and compute.

The Adaptive Format KV Attention reference provides a CPU implementation that demonstrates how the attention kernel can handle multiple storage formats per KV page, enabling the runtime to make format-aware decisions.

## KV page formats

The reference supports three physical storage formats for each KV page:

1. **Dense FP16 (format tag 0)**
   - Stores the full page in 16-bit floating point without compression.
   - Highest precision, highest memory bandwidth.

2. **Dense INT8 + scale (format tag 1)**
   - Stores the page as 8-bit integers with a per-page scaling factor.
   - Memory bandwidth reduced by 2x (vs FP16) at the cost of quantization error.
   - Dequantization: `value = int8_value * scale`.

3. **Sparse top-k (format tag 2)**
   - Stores only the `k` largest-magnitude elements (by absolute value) in the page, along with their indices.
   - All other elements are treated as zero.
   - Memory savings depend on sparsity level `k` relative to page size.
   - Reconstruction: scatter the `k` values into a zero page at the specified indices.

Each page has an associated format tag (0, 1, or 2) that tells the attention reference how to interpret the stored data.

## Dense FP16 path

When a page is in format 0 (FP16 dense), the reference:
- Reads the page directly from the `kv_pages_dense_fp16` tensor.
- Treats the data as FP16 keys and values (no conversion needed).
- Computes attention scores as `query @ key^T / sqrt(D)` in FP32 for numerical stability.

## INT8 path

When a page is in format 1 (INT8 dense + scale), the reference:
- Reads the INT8 data from `kv_pages_dense_i8`.
- Reads the per-page scale from `kv_pages_dense_scales`.
- Dequantizes to FP32: `page_fp32 = page_i8.to(float32) * scale`.
- Uses the resulting FP32 tensor for keys and values in attention computation.

## Sparse page path

When a page is in format 2 (sparse top-k), the reference:
- Allocates a zero page of shape `[page_size, D]` in FP32.
- Reads the indices tensor (`kv_pages_sparse_indices[page_id]`) and values tensor (`kv_pages_sparse_values[page_id]`).
- Scatters each value into the zero page at the corresponding position (overwriting if duplicates exist; last write wins).
- Uses the resulting FP32 tensor for keys and values in attention computation.

## Runtime format metadata

The adaptive format attention reference expects the following inputs:
- `kv_pages_formats`: `[num_pages]` tensor of format tags (0, 1, or 2).
- Separate storage tensors for each format:
  - `kv_pages_dense_fp16`: `[num_pages, page_size, D]` (FP16)
  - `kv_pages_dense_i8`: `[num_pages, page_size, D]` (INT8)
  - `kv_pages_dense_scales`: `[num_pages]` (FP16, for INT8 dequant)
  - `kv_pages_sparse_indices`: `[num_pages, sparsity_k]` (INT64)
  - `kv_pages_sparse_values`: `[num_pages, sparsity_k]` (FP16)

The runtime must populate these tensors and the format tags according to its format assignment policy.

## Relationship to fused selected-quant decode

The adaptive format attention reference extends the idea of per-page precision (as in the IntentQuant path) to support multiple storage formats, not just precision levels.
While the IntentQuant path chooses between FP16 and INT8 (or other quantizations) per page, the adaptive format reference additionally supports a sparse format.
This allows the runtime to further reduce memory bandwidth for pages that are both low-precision and sparse.

In a future GPU kernel, the adaptive format logic could be fused with the selected-page gather and attention computation, similar to how the fused selected-quant decode kernel combines page selection, dequantization, and attention.

## Limitations

- This is a CPU reference implementation only. No GPU kernel is provided, and no GPU speedup is claimed.
- The sparse format reconstruction scatter may have race conditions in a parallel setting (last write wins for duplicate indices).
- The reference does not perform any format conversion or quantization; it assumes the input tensors are already in the correct format.
- No model quality or perplexity preservation is claimed; this is a format-agnostic attention mechanism.

## Future GPU kernel direction

A future Triton or CUDA kernel could:
1. Accept the same metadata (page tables, format tags, and format-specific storage tensors).
2. For each selected page, branch on the format tag to load and decompress the page data appropriately.
3. Compute the attention scores and values in a fused manner, avoiding intermediate storage of decompressed pages.
4. Use shared memory or registers to hold decompressed page tiles for efficient access.

Such a kernel would still be bounded by memory bandwidth, but could reduce traffic by:
- Using 2x less bandwidth for INT8 pages (vs FP16).
- Using even less bandwidth for sparse pages (proportional to `k / page_size`).

However, the actual speedup would depend on the GPU's memory subsystem, the ability to hide decompression latency, and the overhead of format branching.
