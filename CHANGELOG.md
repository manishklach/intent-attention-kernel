# Changelog

## 0.1.0 — CPU-first prototype

*Initial release — semantic block attention simulator for agentic long-context inference.*

### Added

- Added adaptive-format KV attention reference implementation.
- Added tests for adaptive-format page dispatch and output behavior.
- Added benchmark script for adaptive-format attention reference.
- Semantic block metadata model:
  - `BlockPolicy` enum (`ALWAYS`, `ATTEND`, `SKIP`, `RECENT`, `GLOBAL`)
  - `SemanticBlock` dataclass (name, start, end, policy, score)
  - `BlockLayout` with validation, selected-block filtering, token indices
- PyTorch reference attention paths:
  - `dense_attention` — standard scaled dot-product attention
  - `semantic_block_attention` — gather selected K/V, then dense
- Analytical cost model:
  - `attention_flops`, `kv_read_bytes`, `semantic_attention_cost`, `savings_report`
- Synthetic agentic layout generators:
  - `generate_agentic_layout` — realistic agentic trace (deterministic with seed)
  - `random_layout`, `layout_from_policy_dict`
- `BlockTable` — paged KV page mapping simulation
- Triton/CUDA placeholder (`triton_kernel.py`) with CPU fallback
- CPU benchmark scripts:
  - `bench_cost_model.py` — analytical FLOP/KV-byte savings table
  - `bench_cpu_reference.py` — CPU timing comparison (with disclaimer)
- Test suite (50+ pytest tests):
  - Metadata validation (empty names, overlaps, unsorted, ATTEND scores)
  - Reference correctness (semantic equals gathered dense)
  - Cost model (savings %, edge cases)
  - Synthetic trace determinism
  - Block table page mapping
  - Triton fallback on CPU
- Documentation:
  - `docs/architecture.md`
  - `docs/attention_layout.md`
  - `docs/gpu_kernel_plan.md`
  - `docs/results_cpu.md`
  - `docs/repo_metadata.md`
- GitHub Actions CI (Ubuntu, Python 3.10 & 3.11)
- `CHANGELOG.md`

### Known Limitations

- CPU-only simulator — no GPU kernel implementation.
- Causal masking not implemented (raises `NotImplementedError`).
- Triton kernel is a stub — raises `NotImplementedError` when hardware is present.
- No integration with real inference engines (HuggingFace, vLLM).
