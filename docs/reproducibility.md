# Reproducibility Guide

## Prerequisites

- Python 3.10+
- PyTorch 2.x
- No CUDA or Triton required for CPU paths

---

## Install

```bash
git clone https://github.com/manishklach/intent-attention-kernel.git
cd intent-attention-kernel
pip install -e ".[dev]"
```

---

## CPU Validation (no GPU required)

```bash
# Compile-check all source files
python -m py_compile src/intent_attention/*.py

# Run all tests
pytest -q

# Run end-to-end router demo
python examples/end_to_end_router_demo.py

# Run KV Block Router benchmark (routing + cost model)
python benchmarks/bench_block_router.py

# Run IntentQuant mixed-precision policy benchmark
python benchmarks/bench_intent_quant.py

# Run per-block IntentQuant attention reference benchmark
python benchmarks/bench_intent_quant_attention.py
```

Expected: all benchmarks exit with `== 0`, no errors, no GPU required.

---

## Dry-Run Validation (no model downloads, no GPU)

These validate that the experiment harnesses import and parse correctly without
downloading models or requiring CUDA:

```bash
# LLM quality validation — dry run
python experiments/llm_quality_validation.py --dry-run

# GPU decode benchmark — dry run
python experiments/gpu_decode_benchmark.py --dry-run
```

Expected:
- `llm_quality_validation.py --dry-run` prints its config and exits.
- `gpu_decode_benchmark.py --dry-run` prints hardware detection and exits.

---

## Small LLM Quality Validation (requires transformers + datasets)

```bash
pip install transformers datasets

python experiments/llm_quality_validation.py \
    --model HuggingFaceTB/SmolLM2-135M \
    --max-samples 8 \
    --max-length 256 \
    --policies baseline conservative balanced aggressive
```

This runs:
1. Baseline forward pass — perplexity on Wikitext-2 (or synthetic fallback).
2. Quantized KV-cache proxy — applies fake quant/dequant to `past_key_values`
   after each prefill step for each policy.
3. Comparison table — perplexity and reconstruction metrics per policy.

**Caveat:** This is a proxy. The quantization is applied outside the native
model forward pass and does not represent production KV-cache quantization.
See `docs/validation_plan.md` for details.

---

## GPU Decode Benchmark (requires CUDA)

```bash
python experiments/gpu_decode_benchmark.py \
    --batch 1 \
    --heads 32 \
    --head-dim 64 \
    --kv-len 16384 \
    --selected-frac 0.25
```

Additional configurations:

```bash
# Longer context
python experiments/gpu_decode_benchmark.py \
    --batch 1 --heads 32 --head-dim 64 \
    --kv-len 65536 --selected-frac 0.125

# Higher batch (simulating multi-request decode)
python experiments/gpu_decode_benchmark.py \
    --batch 4 --heads 32 --head-dim 64 \
    --kv-len 16384 --selected-frac 0.25
```

---

## Important Notes

### T4 / Turing GPUs (CC < 8.0)

- **Do not compare FlashAttention-2 on T4 as if it is a supported FA2 baseline.**
  FA2 requires Ampere or later (CC >= 8.0). Running FA2 on T4 may fall back to
  an unoptimized path or raise an error.
- Use PyTorch SDPA, eager mode, or xFormers as baselines on T4.
- If available, `flash-attention-turing` (separate package) may provide
  optimized attention for Turing hardware.

### Recommended hardware for FA2 comparisons

- A100 (Ampere, CC 8.0)
- A10G / A30 (Ampere, CC 8.0)
- L4 / L40S (Ada, CC 8.9)
- RTX 3090 / 4090 (Ampere/Ada, CC 8.6/8.9)
- H100 (Hopper, CC 9.0)

### Citation

When reporting results, include:

> *Measured on [GPU] with PyTorch [version], CUDA [version].*
> *SelectedKV uses torch SDPA after gather. TritonIntentQuant is an*
> *untuned prototype. FA2 baselines are reported only on Ampere+ hardware.*
> *These are local measurements. Results may vary.*
