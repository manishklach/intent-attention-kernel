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

## Current Status

| Area | Status |
|---|---|
| CPU-first prototype | Complete |
| Router-to-kernel metadata | Implemented |
| IntentQuant policy simulation | Implemented |
| IntentQuant attention reference | Implemented |
| Triton decode prototype | Optional, exists |
| LLM validation harness | Exists (proxy only) |
| GPU benchmark harness | Exists (no measured speedups yet) |
| Measured GPU speedup | Not claimed |
| Measured model quality | Not claimed |

**Key docs:**
- [Research Summary](docs/research_summary.md) — thesis, problem, proposed interface, limitations
- [Reproducibility Guide](docs/reproducibility.md) — exact commands for CPU, dry-run, LLM, and GPU
- [Validation Plan](docs/validation_plan.md) — quality ladder, proxy metrics, publishable-evidence bar
- [GPU Benchmarking](docs/gpu_benchmarking.md) — fair baselines, hardware matrix, T4 caveat
- [Results Template](docs/results_template.md) — tables to fill when running experiments

---

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

## System Components

### 1. Semantic KV Block Selection

`BlockLayout` and `SemanticBlock` describe context regions. `BlockPolicy`
controls whether a block is `ALWAYS`, `ATTEND`, `SKIP`, `RECENT`, or
`GLOBAL`. The CPU reference gathers selected K/V tokens and computes
attention over them.

> Do not compute and then mask; expose structure early enough to avoid the
> work.

### 2. KV Block Router

The KV Block Router is the missing **runtime-to-kernel policy layer**.
It converts semantic context blocks into flat kernel-ready metadata:

- selected pages
- skipped pages
- precision by page
- prefetch hints
- routing reasons

```python
from intent_attention import BlockRouter, RouterConfig

router = BlockRouter(RouterConfig(memory_pressure=0.5))
routed = router.route_layout(layout, total_tokens=1440)
summary = router.routing_summary(routed)
meta = routing_to_kernel_metadata(routed, page_size=16)
```

**The router is the policy layer. The kernel is the execution layer.**

### 3. Dynamic Block Scoring

Some blocks may be ambiguous. A lightweight scoring path can rank candidate
blocks using query-to-block similarity. This is a heuristic prototype, not
a trained router. It is meant to model the control-plane surface that a
future runtime or kernel could consume.

### 4. IntentQuant-KV: Intent-Aware Mixed-Precision KV Quantization

Not every KV block deserves the same precision. `IntentQuantizer` assigns
per-block precision (FP16, FP8, INT8, INT4, INT4_RESIDUAL, or SKIP) based
on block policy, score, recency, and memory pressure. This is a policy
simulator only — no real GPU quantization kernel is provided.

### 5. IntentQuant Attention Reference — Per-Block Quantized Attention

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

### 6. Speculative KV Prefetch Simulation

Agentic decode often reuses similar KV regions over adjacent steps. A
prefetcher can predict likely next-step KV pages. The current benchmark
simulates hit rate and latency-hiding potential. Prefetch must never affect
correctness. No real latency speedup is claimed without hardware validation.

### 7. Optional Triton Decode-Attention Prototype

An optional GPU-only kernel (`triton_intent_quant_attention.py`) implements
single-token decode attention over selected KV pages with per-page precision
(FP16 or INT8). It skips cleanly on systems without Triton or CUDA and is
not required for any CPU test or benchmark. **No GPU speedup is claimed.**

```bash
python benchmarks/bench_triton_intent_quant_attention.py
```

### 8. Validation Harness

Two experiment scripts validate the prototype pipeline without making claims:

- `experiments/llm_quality_validation.py` — proxy perplexity validation on
  small HuggingFace models. Applies fake quant/dequant to `past_key_values`
  across multiple routing policies. Dry-run mode validates imports without
  downloading models.
- `experiments/gpu_decode_benchmark.py` — decode-step attention latency
  benchmark across PyTorch SDPA, SelectedKV, Triton IntentQuant, xFormers,
  and FlashAttention-2 (Ampere+ only). Dry-run mode detects hardware without
  launching kernels.

See `docs/validation_plan.md` and `docs/gpu_benchmarking.md` for details.

### 9. Fused Selected-Quant Decode Kernel

An experimental Triton kernel (`fused_selected_quant_decode.py`) that fuses
runtime semantic page selection, mixed-precision (FP16/INT8/SKIP) page loading,
and decode-step attention into a single GPU kernel. It consumes BlockRouter
metadata directly and is the execution-layer backend for intent-aware KV
execution.

```bash
# Dry-run (validate imports, detect hardware)
python benchmarks/bench_fused_selected_quant_decode.py --dry-run

# Full benchmark on GPU (requires Triton + CUDA)
python benchmarks/bench_fused_selected_quant_decode.py \
    --batch 1 --heads 8 --head-dim 64 \
    --num-pages 64 --selected-frac 0.25
```

**No GPU speedup is claimed.** This is a research prototype.

### 10. Adaptive Format KV Attention Reference

The Adaptive Format KV Attention reference models KV cache pages stored in different physical formats, such as dense FP16 pages, INT8 quantized pages, and sparse pages.

This extends the repo's intent-aware KV execution model beyond page selection and precision tags. The runtime can now reason about the actual representation of each KV page and dispatch the attention path accordingly.

This is a CPU/reference implementation only. It does not claim GPU speedup or production-ready format dispatch.

### 11. Triton Adaptive-Format Decode Attention Kernel

An optional Triton kernel (`triton_adaptive_format_attention.py`) extending per-page format dispatch to GPU decode attention. Each selected KV page is tagged with a storage format (FP16, INT8, SPARSE, or SKIP). The kernel loads pages according to their format tag, applying INT8 dequantization as needed, and accumulates attention with online softmax.

The SPARSE Triton path is interface-first (CPU fallback). The kernel is a research prototype — **no GPU speedup is claimed**.

```bash
# Dry-run (validate imports, no GPU required)
python benchmarks/bench_triton_adaptive_format_attention.py --dry-run

# Full benchmark on GPU (requires Triton + CUDA)
python benchmarks/bench_triton_adaptive_format_attention.py
```

Related: `docs/triton_adaptive_format_attention.md`

### 12. CPU Adaptive KV Runtime — KVMemoryManager

An orchestrator (`kv_memory_manager.py`) that unifies per-page storage format assignment, access tracking, cold-page demotion (FP16→INT8), hot-page promotion (INT8→FP16), page selection, prefetch prediction, and adaptive-format attention into a single runtime interface.

Demonstrates the "smart KV cache memory" concept on CPU: each page carries metadata (format, policy, access count, recency), and the runtime makes format-transition decisions based on access patterns. No GPU speedup is claimed.

```bash
python examples/cpu_adaptive_kv_runtime_demo.py
```

Related: `docs/kv_memory_manager.md`

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

## Runtime-to-Kernel Contract

This repo models a contract where the runtime produces policy metadata
and the kernel consumes it selectively. The kernel does not magically
discover which context blocks are useful.

```text
Agentic runtime
    |
    v
Semantic block layout
    |
    v
KV Block Router
    |
    +--> block selection (policy + score + recency)
    +--> precision assignment (IntentQuantizer)
    +--> prefetch candidates
    |
    v
Kernel metadata
    |
    +--> selected_page_ids
    +--> block_precision_by_page
    +--> prefetch_page_ids
    +--> routing reasons
    |
    v
Selected-block / IntentQuant attention path
```

| Layer | Responsibility |
|---|---|
| Semantic block layout | Describe context regions, policies, scores, and token bounds |
| KV Block Router | Decide which blocks to select, skip, quantize, or prefetch |
| IntentQuantizer | Assign per-block precision such as FP16, FP8, INT8, INT4, or SKIP |
| Kernel metadata | Flatten routing output into selected page IDs, precision tags, and prefetch hints |
| Attention reference | Run CPU dense or selected-block attention over the selected metadata |
| Future Triton/CUDA kernel | Consume the same metadata in a fused GPU execution path |

The router is the policy layer. The kernel is the execution layer.

---

## Architecture

```text
Agentic runtime
    |
    v
KV Block Router (policy layer)
    |
    +--> semantic policy (ALWAYS, ATTEND, SKIP, RECENT, GLOBAL)
    +--> dynamic block score
    +--> recency window
    +--> memory pressure
    +--> optional query-to-block similarity
    |
    v
Kernel metadata
    |
    +--> selected pages
    +--> precision by page
    +--> prefetch hints
    |
    v
IntentQuant / selected-block attention kernels
    |
    v
Future Triton/CUDA kernel path
```

---

## Dense vs Masked vs Intent-Aware

| Approach | What it knows | Work avoided today | Future GPU goal |
|---|---|---:|---|
| Dense attention | Flat token stream | None | Baseline |
| Masked attention | Token/block mask | Usually limited | May still process masked regions |
| Selected-block attention | Semantic block bounds and policy | CPU gather over selected K/V | Avoid loading skipped KV pages |
| Intent-aware KV execution | Policy, score, quantization, and prefetch hints | Analytical/simulated today | Fuse selection, dequant, and prefetch into kernel/runtime |

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

# Run adaptive format KV attention reference benchmark
python benchmarks/bench_adaptive_format_attention.py

# Run optional Triton adaptive format decode attention benchmark (requires GPU + Triton)
python benchmarks/bench_triton_adaptive_format_attention.py

# Run optional Triton IntentQuant decode attention benchmark (requires GPU + Triton)
python benchmarks/bench_triton_intent_quant_attention.py

# Run KV Block Router benchmark (CPU)
python benchmarks/bench_block_router.py

# Run end-to-end router demo
python examples/end_to_end_router_demo.py

# Run CPU Adaptive KV Runtime demo
python examples/cpu_adaptive_kv_runtime_demo.py

# Dry-run LLM quality validation (validates imports only, no model download)
python experiments/llm_quality_validation.py --dry-run

# Dry-run GPU decode benchmark (validates imports, no GPU required)
python experiments/gpu_decode_benchmark.py --dry-run
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

### bench_triton_adaptive_format_attention.py

Optional Triton prototype for adaptive-format decode attention over KV pages with
per-page storage format tags (FP16, INT8, SPARSE, SKIP). Supports dry-run mode
for CI validation without GPU hardware. No GPU speedup is claimed.

### bench_block_router.py

CPU routing and cost-model benchmark for the KV Block Router. Generates
synthetic agentic layouts at 8K, 32K, and 128K tokens and reports block
selection, precision distribution, page IDs, and estimated KV byte savings
for multiple router configurations.

> This is a routing and cost-model benchmark, not a GPU speedup claim.

> CPU Ratio is not a GPU speedup claim. CPU timing is affected by PyTorch
> dispatch overhead, gather overhead, cache behavior, tensor size, and
> small-batch effects.

### Experiments

#### LLM Quality Validation (`experiments/llm_quality_validation.py`)

Proxy perplexity validation on small HuggingFace models (SmolLM2, TinyLlama).
Runs baseline vs quantized-pass_key_values comparison across multiple routing
policies. **This is a proxy only** — the quantization is applied outside the
native model forward pass and does not represent production KV-cache
quantization.

```bash
# Dry-run (validate imports, no model download)
python experiments/llm_quality_validation.py --dry-run

# Run with SmolLM2-135M on Wikitext-2 (requires transformers + datasets)
python experiments/llm_quality_validation.py --model HuggingFaceTB/SmolLM2-135M
```

Results include: baseline perplexity, quantized perplexity per policy,
reconstruction error metrics (MSE, max-abs, cosine), and selected/skipped
block counts per routing config.

| Policy | KV tokens kept | Est. bytes saved |
|---|---|---|
| conservative | 100% (no skip) | 0% |
| balanced | ~50% | ~50% |
| aggressive | ~25% | ~75% |

#### GPU Decode Benchmark (`experiments/gpu_decode_benchmark.py`)

Measures decode-step attention latency on available GPU hardware across
multiple backends: PyTorch SDPA, selected-KV gather + SDPA, optional Triton
IntentQuant decode, optional xFormers, and optional FlashAttention.

```bash
# Dry-run (validate imports, detect hardware)
python experiments/gpu_decode_benchmark.py --dry-run

# Full benchmark on GPU
python experiments/gpu_decode_benchmark.py \
    --batch 1 --heads 32 --head-dim 64 \
    --kv-len 65536 --selected-frac 0.25 \
    --iters 100 --warmup 20
```

**T4 caveat:** FlashAttention-2 is skipped on Turing GPUs (CC < 8.0). Use
PyTorch SDPA or xFormers as baselines on T4.

See `docs/gpu_benchmarking.md` for hardware matrix and fair-baseline guide.

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
- [x] pytest coverage (160 tests)
- [x] CPU benchmark scripts (10 benchmarks)
- [x] IntentQuant Attention Kernel — per-block fake quant/dequant in selected-block attention path
- [x] Triton IntentQuant decode attention prototype (optional, GPU-only)
- [x] CPU-first KV Block Router — runtime-to-kernel policy layer
- [x] routing-to-kernel metadata conversion (selected pages, precision, prefetch)
- [x] per-block routing decisions and reasons
- [x] End-to-end demo script (examples/end_to_end_router_demo.py)
- [x] LLM quality validation experiment (experiments/llm_quality_validation.py)
- [x] GPU decode benchmark experiment (experiments/gpu_decode_benchmark.py)
- [x] Validation plan docs (docs/validation_plan.md)
- [x] GPU benchmarking guide (docs/gpu_benchmarking.md)
- [x] Adaptive Format KV Attention Reference — CPU reference for heterogeneous KV page formats (FP16, INT8, sparse)
- [x] Triton Adaptive-Format Decode Attention Kernel — optional GPU decode with per-page FP16/INT8/SPARSE/SKIP dispatch
- [x] CPU Adaptive KV Runtime (KVMemoryManager) — orchestrator for format assignment, access tracking, demotion/promotion, and decode

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
- **The KV Block Router is heuristic, not learned.**
- **Selected pages are not guaranteed optimal.**
- **No accuracy or perplexity validation has been performed on routing decisions.**
- **Partial-page bounds are not implemented.** The router selects full pages
  even if a block starts or ends mid-page. A future kernel would need
  per-page token offset masks for correctness.
- CPU Ratio is not a GPU speedup.
- Analytical KV/FLOP savings are not measured GPU performance.
- **Validation experiments use proxy KV-cache quantization** — post-hoc
  quantize/dequantize on past_key_values, not real in-place KV cache
  quantization. Results do not guarantee production quality preservation.
- **GPU benchmarks are local measurements only.** No GPU speedup claim
  is made from any single config, GPU, or software version. Results vary
  by hardware, driver, CUDA version, and system load.

---

## Repository Layout

```
intent-attention-kernel/
    .github/workflows/tests.yml   CI
    benchmarks/
        bench_block_router.py     KV Block Router routing & cost model
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
            block_router.md           KV Block Router design and contract
            dynamic_scoring.md        Dynamic scoring design
            gpu_benchmarking.md       GPU benchmarking guide & fair baselines
            gpu_kernel_plan.md        Future GPU mapping
            intent_quant.md           Intent-aware mixed-precision KV quantization
            kv_quantization.md        KV quantization modeling
            prefetch.md               Speculative prefetch simulation
            repo_metadata.md          Suggestions for GitHub settings
            results_cpu.md            Detailed CPU results notes
            validation_plan.md        LLM quality validation plan
        experiments/
            gpu_decode_benchmark.py   GPU decode attention benchmark
            llm_quality_validation.py Proxy perplexity validation
    src/intent_attention/
        __init__.py               Public API
        _enum.py                  StrEnum base
        block_metadata.py         BlockPolicy, SemanticBlock, BlockLayout
        block_router.py           KV Block Router (policy layer)
        block_scorer.py           Dynamic block scoring (cosine similarity)
        block_table.py            Paged KV mapping simulation
        cost_model.py             Analytical FLOP/KV-byte model
        hf_patch.py               HuggingFace Transformers integration
        intent_quant.py           Intent-aware mixed-precision KV quantization
        intent_quant_attention.py Per-block quantized attention reference
        kv_quant.py               INT8 KV cache quantisation
        triton_intent_quant_attention.py Optional Triton IntentQuant decode attention
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

- [x] **KV Block Router** — runtime-to-kernel policy layer (CPU)
- [x] **Triton IntentQuant decode kernel** — selected-page decode with per-page precision (FP16/INT8)
- [x] **Adaptive Format KV Attention Reference** — CPU reference for heterogeneous KV page formats
- [x] **Triton Adaptive-Format Decode Kernel** — GPU decode with per-page FP16/INT8/SPARSE/SKIP dispatch
- [x] **CPU Adaptive KV Runtime** — smart KV cache memory manager with format tracking and demotion/promotion
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
