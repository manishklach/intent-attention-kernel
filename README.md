[![CI](https://github.com/manishklach/intent-attention-kernel/actions/workflows/tests.yml/badge.svg)](https://github.com/manishklach/intent-attention-kernel/actions/workflows/tests.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

# Intent Attention Kernel

**Intent-Aware KV Execution for Agentic Long-Context Inference**

> Attention should not pretend context is flat — and KV execution should not
> pretend every block is equally useful.

---

**This repo is a CPU-first research prototype for exposing semantic runtime
intent to the KV execution layer. It does not claim GPU speedups yet.**

---

## Thesis

Long-context agentic inference is not just an attention problem. It is a KV
execution problem.

Agentic context contains structurally different regions:

- system prompts
- recent conversation
- retrieved documents
- tool outputs
- memory summaries
- scratchpads
- intermediate reasoning traces

A generic dense attention path treats all of these as one flat KV stream.
This repo explores a different interface: expose semantic block metadata to
the execution layer so the runtime can select, score, quantize, prefetch,
and eventually schedule KV blocks more intelligently.

---

## Five Pillars

### 1. Semantic KV Block Selection

`BlockLayout` and `SemanticBlock` describe context regions. `BlockPolicy`
controls whether a block is `ALWAYS`, `ATTEND`, `SKIP`, `RECENT`, or
`GLOBAL`. The CPU reference gathers selected K/V tokens and computes
attention over them.

> Do not compute and then mask; expose structure early enough to avoid the
> work.

### 2. Dynamic Block Scoring

Some blocks may be ambiguous. A lightweight scoring path can rank candidate
blocks using query-to-block similarity. This is a heuristic prototype, not
a trained router. It is meant to model the control-plane surface that a
future runtime or kernel could consume.

### 3. KV Quantization Modeling

Long-context inference is often KV-bandwidth and KV-capacity constrained.
Cold or selected KV pages may benefit from INT8-style quantization. Current
quantization work is modeling and prototype-level only. No model accuracy,
perplexity, or GPU throughput claim is made. Real benefit depends on
dequant overhead, page reuse, bandwidth pressure, and hardware support.

### 4. IntentQuant-KV: Intent-Aware Mixed-Precision KV Quantization

Not every KV block deserves the same precision. `IntentQuantizer` assigns
per-block precision (FP16, FP8, INT8, INT4, INT4_RESIDUAL, or SKIP) based
on block policy, score, recency, and memory pressure. This is a policy
simulator only — no real GPU quantization kernel is provided.

### 6. IntentQuant Attention Kernel — Per-Block Quantized Attention

Extends IntentQuant-KV into the selected-block attention path itself. Each
selected block is individually quantized (via `fake_quantize_tensor`) and
immediately dequantized (via `fake_dequantize_tensor`) before being
concatenated and passed to dense attention. This is a CPU reference — the
quantized path is intentionally slower to isolate reconstruction error
mechanics without hardware fusion.

```python
from intent_attention.intent_quant_attention import (
    intent_quant_attention_reference,
    compare_intent_quant_to_fp16_selected,
)
```

### 5. Speculative KV Prefetch Simulation

Agentic decode often reuses similar KV regions over adjacent steps. A
prefetcher can predict likely next-step KV pages. The current benchmark
simulates hit rate and latency-hiding potential. Prefetch must never affect
correctness. No real latency speedup is claimed without hardware validation.

---

## IntentQuant-KV

IntentQuant-KV explores the idea that **not every KV block deserves the
same precision**.

In agentic long-context inference, KV blocks have different semantic roles:

- **Critical blocks** — system prompts, global memory, recent context —
  are attended to every step and may need higher precision.
- **Lower-score blocks** — old retrieved documents, tool outputs,
  scratchpad regions — are attended to less frequently and can use
  lower precision or residual quantization.
- **Skipped blocks** contribute zero KV bytes.

`IntentQuantizer` assigns a `KVPrecision` (FP16, FP8, INT8, INT4,
INT4_RESIDUAL, or SKIP) to each block using:

- **Block policy**: ALWAYS/GLOBAL blocks default to FP16; RECENT to FP8.
- **Block score**: high-scoring ATTEND blocks retain higher precision;
  low-scoring blocks are downgraded.
- **Memory pressure**: a knob in [0, 1] that downgrades non-critical
  blocks as pressure increases.
- **Preserve flags**: `preserve_recent` and `preserve_global` keep
  important blocks at higher precision even under moderate pressure.

**This is CPU-first, analytical, and prototype-level.**

- No GPU speedup is claimed.
- No model accuracy or perplexity preservation is claimed.
- Fake quantize/dequantize is only a CPU simulation using symmetric
  absmax scaling.
- The real benefit depends on dequant overhead, memory bandwidth,
  page layout, page reuse, and attention fusion.

```bash
# Run the IntentQuant-KV benchmark
python benchmarks/bench_intent_quant.py
```

---

## Architecture

```text
Agentic runtime
    |
    v
Semantic context blocks
    |
    +--> block policy selection
    |
    +--> dynamic block scoring
    |
    +--> paged KV block table
    |
    +--> optional KV quantization model
    |
    +--> optional next-step prefetch prediction
    |
    v
Selected-KV attention reference
    |
    v
Future Triton/CUDA kernel path
```

---

## Dense vs Masked vs Intent-Aware

| Approach | What it knows | Work avoided today | Future GPU goal |
|---|---|---|---|
| Dense attention | Flat token stream | None | Baseline |
| Masked attention | Token/block mask | Usually limited | May still process masked regions |
| Selected-block attention | Semantic block bounds + policy | CPU gather over selected K/V | Avoid loading skipped KV pages |
| Intent-aware KV execution | Policy + score + quant + prefetch hints | Analytical/simulated today | Fuse selection, dequant, and prefetch into kernel/runtime |

> Do not compute and then mask; expose structure early enough to avoid the
> work.

---

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

# Run CPU timing benchmark
python benchmarks/bench_cpu_reference.py

# Run KV quantization memory model
python benchmarks/bench_kv_quant.py

# Run speculative prefetch simulation
python benchmarks/bench_prefetch.py

# Run dynamic scoring benchmark
python benchmarks/bench_dynamic_scoring.py

# Run intent-aware mixed-precision KV quantization benchmark
python benchmarks/bench_intent_quant.py

# Run per-block IntentQuant attention reference benchmark
python benchmarks/bench_intent_quant_attention.py

# Run optional Triton IntentQuant decode attention benchmark (requires GPU + Triton)
python benchmarks/bench_triton_intent_quant_attention.py
```

---

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

---

## Tests

```bash
pytest -q          # quiet mode
pytest -v          # verbose mode
pytest tests/      # run all tests
```

---

## Benchmarks

All benchmarks run on CPU and are safe to run without CUDA or Triton.

### bench_cost_model.py

Analytical FLOP and KV-byte savings from selected-block attention. Uses
zero-tensor arithmetic to compare dense vs selected-attention cost.

### bench_cpu_reference.py

CPU timing sanity check for dense vs selected-block reference paths.
Measures PyTorch overhead at small token counts on CPU only.

### bench_kv_quant.py

KV byte savings model for selected INT8-style KV pages. Compares fp16
dense storage vs int8+scale for selected pages. Purely analytical.

### bench_prefetch.py

Simulated next-step KV page prediction and hit-rate behavior for
speculative prefetch during agentic decode.

### bench_dynamic_scoring.py

Synthetic query-to-block cosine-similarity scoring behavior across
varying block counts.

### bench_intent_quant.py

Intent-aware mixed-precision KV quantization policy simulator. Assigns
per-block precision (FP16/FP8/INT8/INT4/INT4_RESIDUAL/SKIP) based on
block policy, score, recency, and memory pressure. Includes a fake
quant/dequant reconstruction error test.

### bench_intent_quant_attention.py

CPU reference for per-block mixed-precision fake quant/dequant within the
selected-block attention path. Compares FP16-selected vs quantized-selected
attention outputs and reports reconstruction error metrics.

### bench_triton_intent_quant_attention.py

Optional Triton prototype for single-token decode attention over selected
KV pages with per-page precision (FP16 or INT8). Skips cleanly on systems
without Triton or CUDA. No GPU speedup is claimed — this is a first kernel
prototype for hardware experimentation.

> CPU Ratio is not a GPU speedup claim. CPU timing is affected by PyTorch
> dispatch overhead, gather overhead, cache behavior, tensor size, and
> small-batch effects.

---

## What Is Implemented

- [x] SemanticBlock / BlockLayout metadata
- [x] BlockPolicy enum (ALWAYS, ATTEND, SKIP, RECENT, GLOBAL)
- [x] BlockTable page mapping helper
- [x] PyTorch dense attention baseline
- [x] PyTorch selected-block attention reference
- [x] Dynamic block scoring prototype (BlockScorer)
- [x] Analytical FLOP/KV-byte cost model
- [x] Synthetic agentic trace generator
- [x] KV quantization benchmark/model
- [x] Speculative prefetch simulator (BlockPrefetcher)
- [x] Triton/CUDA placeholder paths with CPU-safe fallback
- [x] HuggingFace Transformers integration (patch_model)
- [x] vLLM-style paged-attention bridge
- [x] Intent-aware mixed-precision KV quantization policy simulator (IntentQuantizer)
- [x] Fake quant/dequant reconstruction metrics (FP16/FP8/INT8/INT4/INT4_RESIDUAL)
- [x] pytest coverage (94 tests)
- [x] CPU benchmark scripts (8 benchmarks)
- [x] IntentQuant Attention Kernel — per-block fake quant/dequant in selected-block attention path
- [x] Triton IntentQuant decode attention prototype (optional, GPU-only)

---

## What Is Not Claimed

- No GPU speedups are claimed.
- No production-ready Triton/CUDA kernel is claimed.
- No real NVIDIA hardware validation has been performed.
- Quantization has not been validated for model accuracy or perplexity.
- No superiority over KIVI, KVQuant, or TurboQuant is claimed.
- No production quantization kernel is provided.
- No model quality guarantee is made.
- Prefetch has not been validated for real latency improvement.
- Dynamic scoring is a heuristic, not a trained routing model.
- CPU Ratio is not a GPU speedup.
- Analytical KV/FLOP savings are not measured GPU performance.

---

## Repository Layout

```
intent-attention-kernel/
    .github/workflows/tests.yml   CI
    benchmarks/
        bench_cost_model.py       Analytical cost model
        bench_cpu_reference.py    CPU timing (for development only)
        bench_dynamic_scoring.py  Dynamic block scoring evaluation
        bench_intent_quant.py     Intent-aware mixed-precision KV quantization
        bench_intent_quant_attention.py  Per-block quantized attention reference
        bench_triton_intent_quant_attention.py  Optional Triton decode attention
        bench_kv_quant.py         KV cache quantisation memory analysis
        bench_prefetch.py         Speculative prefetch decode simulation
    docs/
        architecture.md           Module design
        attention_layout.md       Block policies
        dynamic_scoring.md        Dynamic scoring design
        gpu_kernel_plan.md        Future GPU mapping
        intent_quant.md           Intent-aware mixed-precision KV quantization
        kv_quantization.md        KV quantization modeling
        prefetch.md               Speculative prefetch simulation
        repo_metadata.md          Suggestions for GitHub settings
        results_cpu.md            Detailed CPU results notes
    src/intent_attention/
        __init__.py               Public API
        _enum.py                  StrEnum base
        block_metadata.py         BlockPolicy, SemanticBlock, BlockLayout
        block_scorer.py           Dynamic block scoring (cosine similarity)
        block_table.py            Paged KV mapping simulation
        cost_model.py             Analytical FLOP/KV-byte model
        hf_patch.py               HuggingFace Transformers integration
        intent_quant.py           Intent-aware mixed-precision KV quantization
        intent_quant_attention.py Per-block quantized attention reference
        kv_quant.py               INT8 KV cache quantisation
        triton_intent_quant_attention.py Optional Triton IntentQuant decode attention               INT8 KV cache quantisation
        prefetch.py               Speculative KV block prefetching
        reference.py              Dense + selected-block attention
        synthetic_traces.py       Layout generators
        triton_kernel.py          Triton GPU kernel with CPU fallback
        triton_kernel_quant.py    INT8 quantised Triton kernel
        vllm_bridge.py            vLLM-style paged-attention bridge
    tests/                        Test suite
    CHANGELOG.md
    README.md
    pyproject.toml
```

---

## Formatting

```bash
# Auto-format with black
python -m black src tests benchmarks

# Lint with ruff
python -m ruff check src tests benchmarks
```

---

## Roadmap (Future Work)

- [x] **Triton IntentQuant decode kernel** — selected-page decode with per-page precision (FP16/INT8)
- [ ] **Triton kernel** — iterate only over physical pages from block table (general)
- [ ] **CUDA kernel** — minimal paged-attention with semantic skipping
- [ ] **Variable block sizes** — support non-uniform page sizes
- [ ] **Integration with HuggingFace / vLLM** — plug into real inference
      engines
- [ ] **Trained routing** — replace heuristic scoring with learned block
      selection

---

## Disclaimer

This is research prototype code. Interfaces may change. Not
production-ready. No GPU speedups are claimed or implied. All GPU-related
statements describe future design goals, not current capabilities.

## License

MIT
