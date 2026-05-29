# Intent Attention Kernel: Research Summary

## One-Sentence Thesis

Long-context agentic inference should expose semantic runtime intent to the KV
execution layer so kernels can select, score, quantize, prefetch, and schedule
KV blocks more intelligently.

---

## Problem

Current attention kernels operate on flat KV buffers. They receive tensors and
attention masks but not *semantic context* about what those tensors represent.
This limits their ability to make intelligent execution decisions:

- **Dense attention treats all context as equally important.** System prompts,
  retrieved documents, tool outputs, and recent conversation are loaded into the
  same KV cache and attended to identically. The kernel does not know which
  blocks are critical and which are speculative.

- **Agentic context is structurally heterogeneous.** A single agentic turn may
  include system instructions, retrieved web pages, code interpreter outputs,
  memory summaries, intermediate reasoning traces, and the current user message.
  These regions have very different reuse patterns and precision requirements.

- **KV cache bandwidth and capacity are bottlenecks.** Long-context inference
  (64K–1M+ tokens) is often memory-bound. Every KV page that can be skipped or
  quantized reduces memory traffic and capacity pressure — but only if the
  kernel knows *which* pages to skip or quantize.

- **Quantization and prefetch decisions are made without semantic input.**
  Existing KV quantization methods (KIVI, KVQuant, etc.) apply uniform or
  statistical precision without knowledge of a block's semantic role or
  downstream importance.

- **Kernels receive tensors, not intent.** The runtime knows which context
  regions are recent, which are retrieved, and which are low-scoring — but this
  information is discarded before the kernel is called.

---

## Proposed Interface

We propose a **runtime-to-kernel contract** where the runtime produces policy
metadata and the kernel consumes it selectively:

### Runtime emits

| Artifact | Description |
|---|---|
| Semantic block layout | Named regions with policies (ALWAYS, ATTEND, SKIP, RECENT, GLOBAL) |
| Block scores | Query-to-block cosine similarity heuristic |
| Routing decisions | Which blocks to select, skip, or quantize |
| Selected page IDs | Flat page-level indices for gather |
| Precision metadata | Per-page FP16 or INT8 (policy-simulated) |
| Prefetch hints | Candidate pages for next-step speculative loading |
| Routing reasons | Human-readable why each block was routed as it was |

### Kernel consumes

| Metadata | Kernel path |
|---|---|
| Selected page IDs | Gather KV pages; compute attention only over selected tokens |
| Per-page precision | FP16 path or INT8 dequant path |
| Prefetch hints | Speculative page load (future) |
| Query-position-aware masks | Partial-page token offsets (future) |

---

## Current Prototype

```
Agentic Runtime
     |
     v
Semantic Block Layout
     |
     v
KV Block Router
     |        \
     |         +--> Dynamic Scoring (query-to-block cosine)
     |         +--> IntentQuant Precision (per-block FP16/INT8/INT4/SKIP)
      |         +--> Adaptive Format Metadata (per-page FP16/INT8/SPARSE)
      |         +--> KVMemoryManager (format tracking, demotion/promotion, decode)
      |         +--> Prefetch Hints (next-step candidates)
     v
Kernel Metadata
     |
     v
Selected / IntentQuant / Adaptive Format Attention Path (CPU reference)
     |
     v
Future Triton/CUDA Kernel (optional prototype exists for adaptive-format)
```

### Adaptive Format KV Attention
- models KV pages with different physical representations
- supports dense FP16, INT8-style, and sparse representations
- provides CPU/reference validation only
- does not claim GPU speedup

### Components implemented

| Component | File | Status |
|---|---|---|
| `SemanticBlock` / `BlockLayout` | `src/.../block_metadata.py` | Stable |
| `BlockPolicy` (ALWAYS, ATTEND, SKIP, RECENT, GLOBAL) | `src/.../block_metadata.py` | Stable |
| `BlockRouter` (runtime-to-kernel policy layer) | `src/.../block_router.py` | Stable |
| `BlockScorer` (cosine-similarity scoring) | `src/.../block_scorer.py` | Prototype |
| `IntentQuantizer` (per-block precision assignment) | `src/.../intent_quant.py` | Stable |
| `intent_quant_attention_reference` (CPU) | `src/.../intent_quant_attention.py` | Stable |
| `intent_quant_decode_attention_triton` (GPU prototype) | `src/.../triton_intent_quant_attention.py` | Optional |
| `BlockPrefetcher` (speculative prefetch simulation) | `src/.../prefetch.py` | Prototype |
| `routing_to_kernel_metadata` | `src/.../block_router.py` | Stable |
| LLM perplexity validation harness | `experiments/llm_quality_validation.py` | Functional |
| GPU decode benchmark harness | `experiments/gpu_decode_benchmark.py` | Functional |
| End-to-end router demo | `examples/end_to_end_router_demo.py` | Stable |
| Adaptive Format KV Attention Reference (CPU) | `src/.../adaptive_format_attention.py` | Stable |
| Triton Adaptive-Format Decode Attention Kernel (GPU prototype) | `src/.../triton_adaptive_format_attention.py` | Optional |
| CPU Adaptive KV Runtime (KVMemoryManager) | `src/.../kv_memory_manager.py` | Stable |

### Tests

- 200+ unit and integration tests (pytest) (Triton tests skip without GPU)
- All CPU tests pass with no CUDA or Triton required

---

## Evidence So Far

We are honest about what has and has not been measured:

**Measured:**
- Analytical FLOP savings from selected-block attention (cost model)
- Analytical KV byte savings from skipped and quantized pages (cost model)
- CPU reference correctness: selected-block attention matches dense attention on
  selected tokens
- Fake quant/dequant reconstruction error (MSE, max-abs, cosine similarity)
- Router metadata generation (page IDs, precision tags, prefetch hints)
- CPU routing latency at 8K–128K tokens
- Dry-run validation: experiment harnesses import and parse correctly

**Not yet measured (see "Limitations" below):**
- GPU latency or throughput
- Real LLM perplexity preservation
- Real KV-cache quantization effects (current quantization is post-hoc proxy)
- Trained routing vs heuristic routing
- Downstream task accuracy (MMLU, GSM8K, etc.)

---

## Limitations

| Limitation | Impact |
|---|---|
| No production Triton/CUDA kernel | All attention is CPU reference; no GPU speedup claim possible |
| Causal selected-block attention raises `NotImplementedError` | GPT-style decode with KV-cache selection via query-position masking is not yet supported |
| No real model quality validation | Perplexity proxy uses post-hoc quant/dequant on past_key_values, not real KV-cache replacement |
| Heuristic router (not learned) | Block scoring and routing are hardcoded heuristics; no guarantee of optimality |
| Fake quantization (not real INT8/INT4 storage) | Reconstruction metrics are analytical; real hardware quantization effects are not captured |
| Partial-page masks not implemented | Blocks that start or end mid-page select the full page; future kernel would need per-page token-offset masks for correctness |
| No comparison against baselines | KIVI, KVQuant, TurboQuant, or other KV quantization methods have not been compared |
| No GPU speedup measurements | All benchmarks and timing are CPU-only |

---

## Next Milestones

1. **Run small LLM perplexity validation** — execute
   `experiments/llm_quality_validation.py` with SmolLM2-135M or TinyLlama-1.1B
   on Wikitext-2 to obtain initial proxy perplexity numbers.

2. **Run GPU benchmark on target hardware** — execute
   `experiments/gpu_decode_benchmark.py` on A100, L4, A10G, or RTX 30xx+ to
   obtain decode-step latency for SDPA, SelectedKV, and Triton paths.

3. **Implement query-position-aware causal selected-block attention** — extend
   the reference to support causal masking over partial KV, enabling GPT-style
   decode without the current `NotImplementedError`.

4. **Learned router** — replace the heuristic `BlockScorer` with a lightweight
   trained model that predicts block relevance from query embedding similarity
   or learned weights.

5. **Real paged-KV integration** — integrate with vLLM's PagedAttention or
   HuggingFace's KV cache to demonstrate routing on live inference.

6. **Comparison against baselines** — measure perplexity and GPU throughput
   against KIVI, KVQuant, and dense attention on equivalent hardware.
