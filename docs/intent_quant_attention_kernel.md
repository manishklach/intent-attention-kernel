# IntentQuant Attention Kernel

**Per-Block Mixed-Precision Attention Reference — CPU Prototype**

Extends the intent-aware quantization idea from `IntentQuantizer` into the
attention path itself: selected KV blocks are quantized per-block, then
dequantized before concatenation and dense attention, all on CPU.

## Motivation

`semantic_block_attention` already avoids loading skipped KV blocks by
gathering only selected tokens. But even selected blocks may not all need
FP16 precision:

- **Always/Global blocks** — system prompt, global memory — attended to
  every step; likely benefit from FP16.
- **High-score Attend blocks** — important retrieved documents — could use
  FP8.
- **Low-score Attend blocks** — less relevant context — could use INT8 or
  even INT4.
- **Skipped blocks** — already excluded.

`intent_quant_attention_reference()` applies per-block fake quant/dequant
to the gathered K/V slices, then runs dense attention over the
reconstructed K/V tensor.

## API

```python
from intent_attention import (
    BlockLayout, BlockPolicy, SemanticBlock, IntentQuantizer,
    intent_quant_attention_reference, compare_intent_quant_to_fp16_selected,
)

q = torch.randn(1, 4, 8, 64)
k = torch.randn(1, 4, 256, 64)
v = torch.randn(1, 4, 256, 64)

layout = BlockLayout([
    SemanticBlock("always",  0,   64,  BlockPolicy.ALWAYS),
    SemanticBlock("attend",  64,  192, BlockPolicy.ATTEND, score=0.6),
    SemanticBlock("skip",    192, 256, BlockPolicy.SKIP),
])

quantizer = IntentQuantizer(memory_pressure=0.5)

out = intent_quant_attention_reference(q, k, v, layout, quantizer)
# torch.Size([1, 4, 8, 64])

out, debug = intent_quant_attention_reference(
    q, k, v, layout, quantizer, return_debug=True
)
# debug contains:
#   selected_block_names, selected_tokens,
#   precision_by_block, bytes_saved_pct,
#   reconstruction_mse_k, reconstruction_mse_v,
#   output_mse_vs_fp16_selected, output_cosine_vs_fp16_selected, ...

result = compare_intent_quant_to_fp16_selected(q, k, v, layout, quantizer)
# dict with output_mse, output_cosine_similarity, bytes_saved_pct, ...
```

## Limitations

- **CPU-only, research prototype.** No GPU speedup is claimed.
- **Causal not supported.** `causal=True` raises `NotImplementedError`
  because selected KV indices are in original context coordinates. A
  future version could accept `query_positions` for causal masking.
- **Per-block quant/dequant overhead.** Doing individual
  `fake_quantize_tensor` + `fake_dequantize_tensor` per block is expensive
  on CPU — the quantized path is slower than the FP16-selected path.
- **No model accuracy or perplexity validation.** Reconstruction error
  metrics (`mse`, `cosine_similarity`, `max_abs_error`) are analytical
  only.

## Benchmark

```bash
python benchmarks/bench_intent_quant_attention.py
```

Output includes a table comparing FP16-selected vs IntentQuant-selected
attention runtime, cosine similarity, memory savings, and memory pressure
settings.
