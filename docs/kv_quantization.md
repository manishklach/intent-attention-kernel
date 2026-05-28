# KV Quantization Modeling

## Purpose

Long-context inference is often KV-bandwidth and KV-capacity constrained.
Selected or cold KV pages may benefit from reduced-precision storage
(e.g., INT8) to lower memory pressure and bandwidth requirements.

## Current Approach

The prototype models per-channel absmax quantization:

- For each KV page, compute the absolute maximum per channel dimension.
- Scale values to INT8 range; store as `torch.int8`.
- Store per-channel scale factors as `torch.float16`.
- On dequant, multiply INT8 values by the corresponding scale.

The model currently counts bytes for fp16 vs int8+scale for a given
number of selected pages. It does **not** run dequant or attention in
the quantized space.

## Limitations

- **No accuracy validation**: perplexity, downstream task accuracy, or
  model sensitivity to INT8 KV quantization has not been evaluated.
- **No GPU throughput claim**: real benefit depends on dequant overhead,
  page reuse patterns, memory bandwidth pressure, and hardware-level
  support (e.g., INT8 MMA in Hopper).
- **No integration with attention**: the current reference attention
  always works in fp16. A future kernel would fuse dequant into the
  attention loop to avoid materializing dequantized pages.
- **Page-level, not token-level**: quantization boundaries are
  page-aligned (group size = page size).
- **Per-channel, not per-token or per-group**: finer quantization
  granularity may improve accuracy but adds overhead.

## When Quantization Could Help

- Cold KV pages that are rarely attended to.
- Selected pages where the runtime can tolerate lower precision.
- Prefill-phase KV pages that are stored once and read many times
  during decode.

## When Quantization May Not Help

- Hot pages that are attended to every step — dequant overhead may
  outweigh bandwidth savings.
- Small batch sizes where bandwidth is not the bottleneck.
- Hardware without fast INT8 dequant or INT8 MMA support.

## Usage

```bash
python benchmarks/bench_kv_quant.py
```

This prints analytical byte savings for fp16 dense vs int8+scale
selected pages at varying skip ratios and KV lengths.
