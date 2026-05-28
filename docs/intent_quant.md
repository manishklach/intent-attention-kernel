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

> Quantization should not be only a global model setting; it can also be an
> execution policy driven by runtime intent.

## Why Uniform KV Quantization Is Not Enough

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

## Policy Inputs

The `IntentQuantizer` accepts:

| Parameter | Type | Default | Description |
|---|---|---|---|
| `memory_pressure` | float [0, 1] | 0.0 | Higher values force more aggressive downgrades |
| `preserve_recent` | bool | True | Keep recent blocks at FP8 or above |
| `preserve_global` | bool | True | Keep global blocks at FP16 or above |
| `high_score_threshold` | float | 0.75 | Attend score above this = high relevance |
| `medium_score_threshold` | float | 0.40 | Attend score between this and high = medium |

## Analytical Byte Model

`estimate_layout_bytes` computes:

- `dense_fp16_bytes`: bytes if all KV were stored in FP16
- `intent_quant_bytes`: bytes after per-block precision assignment
- `bytes_saved` / `bytes_saved_pct`: analytical savings
- `precision_distribution`: number of tokens per precision level
- `critical_full_precision_tokens`: tokens retained at FP16

Byte counting uses:

| Precision | Bytes per value |
|---|---|
| FP16 | 2.0 |
| FP8 | 1.0 |
| INT8 | 1.0 |
| INT4 | 0.5 |
| INT4_RESIDUAL | 1.0 |
| SKIP | 0.0 |

These are analytical estimates. Real GPU memory savings depend on page
layout, alignment, scale factor storage, and kernel fusion.

## Fake Quant/Dequant Simulation

`fake_quantize_tensor` and `compute_quant_error` provide a CPU-only
simulation of the quantization process:

1. Compute symmetric absmax scale over the last dimension.
2. Divide by scale, clamp to representable range, round.
3. Reconstruct: multiply by scale.
4. Compute error metrics: MSE, max absolute error, cosine similarity.

For `INT4_RESIDUAL`, the process runs twice: once for the base INT4
quantization and once for the residual (difference between original and
base reconstruction).

This simulation:

- Does **not** produce real low-bit tensors (fp8, int8, int4).
- Does **not** measure dequant overhead or bandwidth.
- Does **not** guarantee that real hardware quantization would produce
  similar error characteristics.

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

- **No real GPU kernel**: precision assignment is analytical only. No
  GPU kernel accepts per-block precision maps yet.
- **No accuracy validation**: perplexity, downstream accuracy, and model
  sensitivity to mixed-precision KV have not been evaluated.
- **No perplexity validation**: the effect of mixed-precision KV on
  generation quality has not been measured.
- **No calibration**: scale factors use simple absmax without
  calibration, optimization, or non-uniform grids.
- **Fake quantisation**: `fake_quantize_tensor` is a CPU simulation that
  materializes reconstructed tensors. It does not measure real dequant
  overhead or bandwidth.
- **Dequant overhead may dominate**: in many scenarios, the cost of
  dequantizing INT4 or INT8 pages during the attention loop may exceed
  the bandwidth savings from reading fewer bytes.
- **No page-layout modelling**: real hardware benefit depends on whether
  mixed-precision pages can be stored contiguously or require extra
  indirection. Page alignment and scale factor storage are not modelled.
- **No production claim**: this is a research prototype. Real benefit
  depends on dequant overhead, memory bandwidth pressure, page reuse,
  attention fusion, and hardware-level support for mixed-precision
  layouts.
