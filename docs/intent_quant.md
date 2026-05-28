# IntentQuant-KV: Intent-Aware Mixed-Precision KV Quantization

## Motivation

Long-context inference is often KV-bandwidth and KV-capacity constrained.
Standard approaches apply uniform quantization (e.g., INT8 for all KV pages)
or use heuristic grouping (e.g., KIVI's per-channel or per-token grouping).

But not every KV block deserves the same precision. Agentic context contains
semantically different regions:

- system prompts, always attended to
- recent conversation, attended to every step
- retrieved documents, attended to only when relevant
- tool outputs, scratchpads, and intermediate reasoning traces, often
  attended to once or never

## Why Uniform KV Quantization Is Suboptimal

- **Over-preserving low-value blocks**: low-score ATTEND blocks and cold
  KV pages are quantized to the same precision as critical blocks, wasting
  capacity.
- **Under-preserving critical blocks**: system prompts and recent context
  may suffer accuracy loss under aggressive uniform quantization.
- **No semantic awareness**: the runtime has access to block metadata
  (policy, score, recency) but does not use it for precision assignment.

## Semantic Precision Policies

IntentQuant-KV assigns precision per block using:

- **Block policy**: ALWAYS/GLOBAL blocks default to FP16; RECENT to FP8;
  SKIP blocks contribute zero bytes.
- **Block score**: high-scoring ATTEND blocks retain higher precision;
  low-scoring blocks are downgraded to INT4 or SKIP.
- **Memory pressure**: a knob in `[0, 1]` that downgrades non-critical
  blocks more aggressively as pressure increases.
- **Preserve flags**: `preserve_recent` and `preserve_global` keep recent
  and global blocks at higher precision even under moderate pressure.

### Precision Levels

| Precision | Bytes/value | Use case |
|---|---|---|
| FP16 | 2.0 | Critical blocks — ALWAYS, GLOBAL under low pressure |
| FP8 | 1.0 | RECENT, high-score ATTEND under moderate pressure |
| INT8 | 1.0 | High-score ATTEND, ALWAYS/GLOBAL under extreme pressure |
| INT4 | 0.5 | Low-score ATTEND, cold pages |
| INT4_RESIDUAL | 1.0 | Medium-score ATTEND — base INT4 + lightweight residual |
| SKIP | 0.0 | Skipped or ignored blocks |

## Relation to Existing Work

IntentQuant-KV is **not** a replacement for KIVI, KVQuant, or TurboQuant.
It is a **policy-abstraction prototype** that sits above the quantization
kernel. Key differences:

- **KIVI** (Liu et al.) introduces per-channel INT4 grouping with
  residual KV cache. IntentQuant-KV does not claim comparable accuracy.
- **KVQuant** (Hooper et al.) explores per-channel, per-token, and
  per-vector quantization with non-uniform grids. IntentQuant-KV
  uses simple symmetric absmax only.
- **TurboQuant** uses calibration to optimize scale factors.
  IntentQuant-KV does no calibration or training.

IntentQuant-KV focuses on the **control-plane question**: *given semantic
block metadata, what precision should each block get?* The actual
quantization kernel is a separate concern.

## Future Kernel Plan

A future GPU kernel would:

- Accept a per-page precision map (from `IntentQuantizer`) alongside the
  block table.
- Load KV pages at their assigned precision (fp16, int8, int4, etc.) and
  dequantize/decode inline.
- Fuse dequant with the attention loop to avoid materializing full-precision
  pages.
- Support INT4 residuals as a second buffer with smaller page size.

## Limitations

- **No real GPU kernel**: precision assignment is analytical only.
- **No accuracy validation**: perplexity, downstream accuracy, and model
  sensitivity to mixed-precision KV have not been evaluated.
- **No calibration**: scale factors use simple absmax without
  calibration or optimization.
- **Fake quantisation**: `fake_quantize_tensor` is a CPU simulation that
  materializes reconstructed tensors. It does not measure real dequant
  overhead or bandwidth.
- **No page-layout modelling**: real hardware benefit depends on whether
  mixed-precision pages can be stored contiguously or require extra
  indirection.
- **No production claim**: this is a research prototype. Real benefit
  depends on dequant overhead, memory bandwidth pressure, page reuse,
  attention fusion, and hardware-level support for mixed-precision
  layouts.
