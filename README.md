[![CI](https://github.com/manishklach/intent-attention-kernel/actions/workflows/tests.yml/badge.svg)](https://github.com/manishklach/intent-attention-kernel/actions/workflows/tests.yml)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

# Intent Attention Kernel

**Semantic Block Attention for Agentic Long-Context Inference — CPU-First Research Prototype**

> Attention should not pretend context is flat.

---

## Motivation

Long-context agentic inference should not treat context as a flat token stream.
The runtime often knows that tokens belong to semantic regions:

- system prompts
- recent conversation
- retrieved documents
- tool outputs
- memory summaries
- scratchpad

Standard attention computes scores over all past KV tokens.  Many blocks can be
safely ignored — saving memory bandwidth and compute — if the runtime exposes
block structure to the attention mechanism.

## Core Idea

Represent semantic regions as **blocks** with **attention policies**, then
compute attention only over selected KV blocks.

> Do not compute and then mask; expose structure early enough to avoid the work.

### Dense vs Masked vs Selected-Block Attention

| Approach | KV tokens read | Metadata required | Current repo behavior |
|---|---:|---|---|
| Dense attention | All | None | PyTorch CPU baseline |
| Masked attention | Often all | Token/block mask | Not the main design target |
| Selected-block attention | Selected subset | Block policy + bounds | Simulated with selected K/V gather |

- The current CPU implementation gathers selected K/V then runs dense attention.
- A future GPU kernel would avoid loading skipped K/V pages from HBM.
- **No GPU speedup is claimed.**

## Architecture

```
Runtime Context
    |
    v
Semantic Block Metadata
  (system_prompt: ALWAYS, retrieved_doc_0: ATTEND score=0.92, ...)
    |
    v
Selected KV Blocks (skip SKIP blocks)
    |
    v
Attention over selected blocks
    |
    v
Output
```

## Quickstart

```bash
# Install from source (editable, with dev dependencies)
pip install -e ".[dev]"

# Compile-check all source files
python -m py_compile src/intent_attention/*.py

# Run tests
pytest -q

# Run analytical cost model
python benchmarks/bench_cost_model.py

# Run CPU timing benchmark (not GPU-representative)
python benchmarks/bench_cpu_reference.py
```

## Formatting

```bash
# Auto-format with black
python -m black src tests benchmarks

# Lint with ruff
python -m ruff check src tests benchmarks
```

## Example Usage

```python
import torch
from intent_attention import (
    BlockLayout,
    BlockPolicy,
    SemanticBlock,
    semantic_block_attention,
    savings_report,
)

q = torch.randn(1, 4, 16, 64)
k = torch.randn(1, 4, 1024, 64)
v = torch.randn(1, 4, 1024, 64)

layout = BlockLayout([
    SemanticBlock("system_prompt",     0,   128, BlockPolicy.ALWAYS),
    SemanticBlock("retrieved_doc_0",  128, 512, BlockPolicy.ATTEND, score=0.85),
    SemanticBlock("retrieved_doc_1",  512, 768, BlockPolicy.SKIP),
    SemanticBlock("recent_context",    768, 1024, BlockPolicy.RECENT),
])

out, debug = semantic_block_attention(q, k, v, layout, return_debug=True)

print(out.shape)          # torch.Size([1, 4, 16, 64])
print(debug)
# {
#   'selected_token_count': 640,
#   'selected_block_names': ['system_prompt', 'retrieved_doc_0', 'recent_context'],
#   'total_kv_tokens': 1024,
#   'selected_kv_tokens': 640
# }

report = savings_report(1, 4, 16, 1024, debug["selected_kv_tokens"], 64)
print(f"FLOPs saved: {report['flops_saved_pct']:.1f}%")
print(f"KV bytes saved: {report['kv_bytes_saved_pct']:.1f}%")
```

## Running Tests

```bash
pytest -q          # quiet mode
pytest -v          # verbose mode
pytest tests/      # run all tests in the tests directory
```

## Running Benchmarks

### Analytical cost model

```bash
python benchmarks/bench_cost_model.py
```

Sample output:

```
|   Total Tokens |   Selected |   Fraction | FLOPs Saved %   | KV Bytes Saved %   |
|----------------|------------|------------|-----------------|--------------------|
|           1024 |       1024 |     1.0000 | 0.00%           | 0.00%              |
|           4096 |       2816 |     0.6875 | 31.25%          | 31.25%             |
|          16384 |       5376 |     0.3281 | 67.19%          | 67.19%             |
|          65536 |       5376 |     0.0820 | 91.80%          | 91.80%             |
```

These are **analytical** numbers — they count FLOPs and KV bytes for dense vs
selected-block attention using the same formula, with zero tensor execution.
They are **not** measured GPU performance.

### CPU timing benchmark

```bash
python benchmarks/bench_cpu_reference.py
```

Sample output:

```
============================================================
WARNING: CPU timing is not representative of GPU kernel performance.
These numbers measure Python + PyTorch overhead on CPU only.
============================================================

  Tokens    Dense (s)   Semantic (s)   CPU Ratio
-----------------------------------------------
     512       0.0032         0.0030       1.08x
    1024       0.0054         0.0058       0.94x
    2048       0.0089         0.0100       0.89x
    4096       0.0163         0.0140       1.16x
```

### Interpreting CPU Ratio

The **CPU Ratio** is `dense_time / semantic_time` on PyTorch's CPU backend —
an unoptimised, non-fused path.  This ratio:

- Is **not** a GPU speedup prediction.
- Can be below 1.0 for small cases due to gather overhead, cache effects,
  PyTorch dispatch overhead, or small-tensor matmul behaviour.
- Shows trend direction: as KV length grows, selected-block attention tends
  to spend less time proportionally.

The important claim at this stage is the **analytical reduction in selected KV
work**, not measured GPU acceleration.

## What Is Implemented

- [x] `BlockPolicy` enum (`ALWAYS`, `ATTEND`, `SKIP`, `RECENT`, `GLOBAL`)
- [x] `SemanticBlock` / `BlockLayout` with full validation
- [x] Dense attention reference (PyTorch, CPU)
- [x] Selected-block attention — gather K/V, then dense (CPU only)
- [x] Causal masking in dense attention
- [x] Analytical cost model — FLOPs, KV bytes, savings %
- [x] Synthetic trace generators (deterministic with `seed`)
- [x] `BlockTable` — paged KV mapping helper (CPU simulation)
- [x] Dynamic block scoring (`BlockScorer` — cosine-similarity per ATTEND block)
- [x] HuggingFace Transformers integration (`patch_model`)
- [x] vLLM-style paged-attention bridge
- [x] INT8 KV cache quantisation (per-channel absmax)
- [x] Speculative KV block prefetching (`BlockPrefetcher`)
- [x] Triton GPU kernel with CPU fallback
- [x] Comprehensive test suite

## What Is Not Claimed

This repo does **not**:

- Claim GPU speedups.  All performance numbers are analytical or CPU-only.
- Measure GPU memory bandwidth or kernel launch overhead.
- Support production inference workloads.

It **is** a **research prototype** that proves:

- Metadata representation (blocks, policies, validation)
- Correctness semantics (selected-block attention equals gathered dense)
- Synthetic agentic layout generation
- Analytical cost-model savings (FLOPs, KV bytes)
- Extensibility (HF patch, vLLM bridge, KV quant, prefetch)

… before a production-grade GPU kernel is implemented.

## Repository Layout

```
intent-attention-kernel/
├── .github/workflows/tests.yml   CI
├── benchmarks/
│   ├── bench_cost_model.py       Analytical cost model
│   ├── bench_cpu_reference.py    CPU timing (for development only)
│   ├── bench_dynamic_scoring.py  Dynamic block scoring evaluation
│   ├── bench_kv_quant.py         KV cache quantisation memory analysis
│   └── bench_prefetch.py         Speculative prefetch decode simulation
├── docs/
│   ├── architecture.md           Module design
│   ├── attention_layout.md       Block policies
│   ├── gpu_kernel_plan.md        Future GPU mapping
│   ├── repo_metadata.md          Suggestions for GitHub settings
│   └── results_cpu.md            Detailed CPU results notes
├── src/intent_attention/
│   ├── __init__.py               Public API
│   ├── _enum.py                  StrEnum base
│   ├── block_metadata.py         BlockPolicy, SemanticBlock, BlockLayout
│   ├── block_scorer.py           Dynamic block scoring (cosine similarity)
│   ├── block_table.py            Paged KV mapping simulation
│   ├── cost_model.py             Analytical FLOP/KV-byte model
│   ├── hf_patch.py               HuggingFace Transformers integration
│   ├── kv_quant.py               INT8 KV cache quantisation
│   ├── prefetch.py               Speculative KV block prefetching
│   ├── reference.py              Dense + selected-block attention
│   ├── synthetic_traces.py       Layout generators
│   ├── triton_kernel.py          Triton GPU kernel with CPU fallback
│   ├── triton_kernel_quant.py    INT8 quantised Triton kernel
│   └── vllm_bridge.py            vLLM-style paged-attention bridge
├── tests/                        Test suite
├── CHANGELOG.md
├── README.md
└── pyproject.toml
```

## Roadmap (Future Work)

- [ ] **Triton kernel** — iterate only over physical pages from block table
- [ ] **CUDA kernel** — minimal paged-attention with semantic skipping
- [ ] **Variable block sizes** — support non-uniform page sizes
- [ ] **Integration with HuggingFace / vLLM** — plug into real inference engines

## Disclaimer

This is research prototype code.  Interfaces may change.  Not production-ready.
No GPU speedups are claimed or implied.  All GPU-related statements describe
future design goals, not current capabilities.

## License

MIT
