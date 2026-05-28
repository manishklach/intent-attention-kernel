# Results Template

Copy and fill these tables when running experiments.

---

## LLM Quality Validation

**Command:** `python experiments/llm_quality_validation.py --model <model> --policies baseline conservative balanced aggressive`

| Model | Policy | Memory Pressure | Est. KV Saved (%) | Perplexity | PPL Δ (%) | Notes |
|---|---|---|---|---|---|---|
| HuggingFaceTB/SmolLM2-135M | baseline | 0.0 | 0 | | — | |
| HuggingFaceTB/SmolLM2-135M | conservative | 0.25 | | | | |
| HuggingFaceTB/SmolLM2-135M | balanced | 0.50 | | | | |
| HuggingFaceTB/SmolLM2-135M | aggressive | 0.75 | | | | |
| TinyLlama/TinyLlama-1.1B | baseline | 0.0 | 0 | | — | |
| TinyLlama/TinyLlama-1.1B | conservative | 0.25 | | | | |
| TinyLlama/TinyLlama-1.1B | balanced | 0.50 | | | | |
| TinyLlama/TinyLlama-1.1B | aggressive | 0.75 | | | | |

**Additional metrics to record:**

- MSE (mean squared error) per policy
- Max-abs error per policy
- Cosine similarity per policy
- Selected token count per policy
- Skipped token count per policy

---

## GPU Decode Benchmark

**Command:** `python experiments/gpu_decode_benchmark.py --batch <B> --heads <H> --head-dim <D> --kv-len <L> --selected-frac <F>`

### System

| Field | Value |
|---|---|
| GPU | |
| Compute Capability | |
| PyTorch version | |
| CUDA version | |
| Triton installed | |
| flash-attn installed | |
| xformers installed | |

### Latency (ms)

| Backend | KV Len | Sel Frac | Avg (ms) | Ratio vs SDPA | Notes |
|---|---|---|---|---|---|
| SDPA | 16384 | 1.00 | | 1.00 (baseline) | |
| SDPA | 65536 | 1.00 | | 1.00 (baseline) | |
| SelectedKV | 16384 | 0.25 | | | |
| SelectedKV | 65536 | 0.125 | | | |
| TritonIntentQuant | 16384 | 0.25 | | | |
| TritonIntentQuant | 65536 | 0.125 | | | |
| FlashAttention | 16384 | 1.00 | | | CC >= 8.0 only |
| FlashAttention | 65536 | 1.00 | | | CC >= 8.0 only |
| xFormers | 16384 | 1.00 | | | |
| xFormers | 65536 | 1.00 | | | |

> Ratio vs baseline = SDPA_ms / backend_ms. > 1.0 means backend is faster.

### Vary selected-frac (KV=16384, same GPU)

| Backend | 0.0625 | 0.125 | 0.25 | 0.50 | 1.00 |
|---|---|---|---|---|---|
| SelectedKV | | | | | |
| TritonIntentQuant | | | | | |

### Vary KV length (sel=0.25, same GPU)

| Backend | 4096 | 16384 | 65536 | 131072 |
|---|---|---|---|---|
| SDPA | | | | |
| SelectedKV | | | | |
| TritonIntentQuant | | | | |

---

## Router Benchmark

**Command:** `python benchmarks/bench_block_router.py`

| Scenario | Total Tokens | Selected Tokens | Selected Pages | Prefetch Pages | Bytes Saved (%) | Precision Distribution |
|---|---|---|---|---|---|---|
| Simple (64-token blocks, 16 pages) | | | | | | |
| Default config (no pressure) | | | | | | |
| Balanced config (0.5 pressure) | | | | | | |
| Aggressive config (0.75 pressure) | | | | | | |
| High pressure (0.95) | | | | | | |

**Precision distribution format:** `{FP16: N, FP8: M, INT8: P, INT4: Q, SKIP: R}`

---

## Reproducibility Checklist

- [ ] CPU tests pass (`pytest -q` outputs `... passed`)
- [ ] End-to-end demo runs (`python examples/end_to_end_router_demo.py`)
- [ ] All benchmarks run without errors
- [ ] Dry-run validation passes (`--dry-run` prints config and exits)
- [ ] Experiment command recorded with all flags
- [ ] GPU benchmark: hardware info recorded from `--dry-run`
- [ ] GPU benchmark: FA2 skipped on T4 (CC < 8.0) — no misleading comparison
- [ ] LLM validation: record model, dataset, max-samples, max-length
- [ ] LLM validation: note proxy limitation in results
