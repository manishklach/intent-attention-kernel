# Validation Plan

**Why analytical savings are not enough**

The repo currently demonstrates:

- Semantic block metadata and routing
- Analytical KV byte and FLOP savings estimates
- CPU reference attention paths with fake quant/dequant
- Optional Triton decode kernel prototype

None of these prove that the routing decisions or quantized precision levels
preserve model quality on real LLM outputs. Without quality validation, the
savings estimates are hypothetical.

---

## Quality validation ladder

### Level 1: Proxy reconstruction metrics (done)

`compute_quant_error()` reports MSE, max-abs-error, and cosine similarity
between original and fake-quantized tensors. These are necessary but not
sufficient — they measure numerical divergence per tensor, not whether the
model's output distribution shifts.

### Level 2: Small LLM perplexity validation (this experiment)

`experiments/llm_quality_validation.py` runs a small HuggingFace causal LM
(SmolLM2-135M, SmolLM2-360M, or TinyLlama-1.1B) through:

1. Baseline forward pass — measure perplexity on Wikitext-2
2. KV-cache quantization proxy — apply `fake_quantize_tensor` /
   `fake_dequantize_tensor` to `past_key_values` after each prefill step
3. Compare perplexity across policies: conservative, balanced, aggressive

**Limitations of this experiment:**

- The quantization is applied outside the native model forward pass.
  Real KV-cache quantization would require patching the attention modules
  or replacing the KV cache storage.
- Only small models are feasible on CPU.
- The results are a proxy, not a production guarantee.

### Level 3: Real KV-cache patching (future work)

The `hf_patch.py` module already supports patching attention modules for
selected-block execution. A future extension could replace the KV cache
tensors inside patched attention modules with quantized variants, running
the full model with real quantization rather than post-hoc manipulation.

This would provide more trustworthy perplexity and downstream task metrics.

---

## What would count as publishable evidence

1. **Perplexity within 1% of baseline at 30%+ estimated KV savings**
   on multiple small models (SmolLM2, TinyLlama, GPT-2) and multiple
   datasets (Wikitext-2, C4, PG-19).

2. **Repeatable across policies** — higher memory pressure should correlate
   with higher perplexity, and the relationship should be predictable.

3. **Ablation: precision matters** — demonstrating that FP16-selected blocks
   outperform INT8-selected blocks on the same model, confirming the
   precision assignment has measurable effect.

4. **Ablation: selection matters** — demonstrating that selected-block
   attention (skipping low-score blocks) outperforms random-block selection
   at the same token count.

---

## What is not claimed

- No GPU speedup is claimed from these CPU experiments.
- No production-quality quantization guarantee is made.
- The KV-cache proxy does not reproduce real hardware quantization effects
  (e.g., no INT4 packed storage, no GPU tensor-core dequant).
- Perplexity on Wikitext-2 is not a comprehensive quality evaluation.
  Downstream task evaluation (MMLU, GSM8K, etc.) would be needed for
  stronger claims.
- No comparison against KIVI, KVQuant, or other KV quantization methods
  has been performed.
