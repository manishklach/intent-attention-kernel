# Intent Attention Simulator

**Semantic Block Attention for Agentic Long-Context Inference — CPU Simulator / Research Prototype**

[![CI](https://github.com/manishklach/intent-attention-kernel/actions/workflows/tests.yml/badge.svg)](https://github.com/manishklach/intent-attention-kernel/actions/workflows/tests.yml)

---

## Motivation

Long-context agentic inference should not treat context as a flat token stream. The
runtime often knows that tokens belong to semantic regions:

- system prompts
- recent conversation
- retrieved documents
- tool outputs
- memory summaries
- scratchpad

Standard attention computes scores over all past KV tokens. However, many blocks can
be safely ignored to save memory bandwidth and compute.

## Core Idea

Represent semantic regions as **blocks** with **attention policies**, then compute
attention only over selected KV blocks.

### Dense vs Masked vs Selected-Block Attention

| Approach | KV tokens read | Metadata required | GPU-friendly |
|----------|---------------|-------------------|--------------|
| Dense attention | All | None | Yes |
| Sparse/masked attention | All (mask applied post-QK) | Per-token mask | No |
| **Selected-block attention (design)** | Subset | Per-block policy + bounds | Aspires to be (needs paged KV kernel) |

Selected-block attention aims to skip loading unused KV pages from HBM; this repo simulates the metadata and cost model on CPU only.

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
# Install from source
pip install -e ".[dev]"

# Run tests
pytest -q

# Run cost model benchmark
python benchmarks/bench_cost_model.py

# Run CPU timing benchmark
python benchmarks/bench_cpu_reference.py
```

## Example Usage

```python
import torch
from intent_attention import (
    BlockPolicy, SemanticBlock, BlockLayout,
    dense_attention, semantic_block_attention,
    savings_report,
)

layout = BlockLayout([
    SemanticBlock("system_prompt", 0, 512, BlockPolicy.ALWAYS),
    SemanticBlock("retrieved_doc", 512, 1536, BlockPolicy.ATTEND, score=0.85),
    SemanticBlock("ignored_chunk", 1536, 2048, BlockPolicy.SKIP),
    SemanticBlock("recent_context", 2048, 4096, BlockPolicy.RECENT),
])

q = torch.randn(1, 32, 128, 128)
k = torch.randn(1, 32, 4096, 128)
v = torch.randn(1, 32, 4096, 128)

out, debug = semantic_block_attention(q, k, v, layout, return_debug=True)
print(f"Selected {debug['selected_kv_tokens']} of {debug['total_kv_tokens']} KV tokens")

report = savings_report(1, 32, 128, 4096, debug["selected_kv_tokens"], 128)
print(f"FLOPs saved: {report['flops_saved_pct']:.1f}%")
print(f"KV bytes saved: {report['kv_bytes_saved_pct']:.1f}%")
```

## Benchmark Commands

```bash
# Analytical cost model
python benchmarks/bench_cost_model.py

# CPU timing (not representative of GPU)
python benchmarks/bench_cpu_reference.py
```

## What Is Not Claimed

This repo does **not**:

- Claim GPU speedups. All performance numbers are analytical or CPU-only.
- Implement a real GPU kernel. The `triton_kernel.py` is a stub that falls back to PyTorch on CPU.
- Measure GPU memory bandwidth or kernel launch overhead.
- Support production inference workloads.

It **is** a **simulator-first prototype** that proves the interface, correctness,
metadata model, selected-block attention semantics, and analytical cost savings
before a real Triton/CUDA kernel is implemented.

## What Is Implemented

- [x] `BlockPolicy` enum (ALWAYS, ATTEND, SKIP, RECENT, GLOBAL)
- [x] `SemanticBlock` / `BlockLayout` with validation
- [x] Dense attention reference (PyTorch, CPU)
- [x] Selected-block attention (gather K/V, then dense — CPU only)
- [x] Analytical cost model (FLOPs, KV bytes, savings %)
- [x] Synthetic trace generators (deterministic with seed)
- [x] `BlockTable` helper for paged KV mapping (CPU simulation)
- [x] Triton stub with CPU fallback
- [x] Comprehensive test suite

## Roadmap (Future Work)

- [ ] **Triton kernel** — iterate only over physical pages from block table
- [ ] **CUDA kernel** — minimal paged-attention with semantic skipping
- [ ] **Variable block sizes** — support non-uniform page sizes
- [ ] **Integration with HuggingFace / vLLM** — plug into real inference engines

## Disclaimer

This is research prototype code. Interfaces may change. Not production-ready.
No GPU speedups are claimed or implied. All GPU-related statements describe
future design goals, not current capabilities.

## License

MIT
