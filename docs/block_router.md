# KV Block Router

**CPU-first policy layer that converts semantic context blocks into
kernel-ready metadata.**

The router is the policy layer. The kernel is the execution layer.

---

## Motivation

The `IntentQuantizer` and `BlockLayout` modules already provide per-block
precision assignment and block selection. But there is no explicit
*policy layer* that converts runtime context (block metadata, scores,
memory pressure, recency) into a flat set of kernel-ready instructions.

The KV Block Router fills this gap. It takes semantic block metadata as
input and produces:

- which pages to load (selected pages)
- which pages to skip (skipped pages)
- what precision each page should use (FP16, INT8, INT4, ...)
- which pages are candidates for speculative prefetch
- human-readable reasons for each routing decision

---

## Why the kernel should not decide relevance alone

A kernel that decides relevance internally forces the policy into opaque
kernel parameters or hard-coded heuristics. This makes it hard to:

- **Audit decisions** — what was skipped and why?
- **Tune policies** — change routing without recompiling kernels.
- **Combine signals** — fuse policy + score + recency + pressure in one
  place before the kernel sees data.
- **Swap strategies** — replace heuristic routing with a learned router
  without touching the kernel.

The router/policy layer stays outside the kernel. The kernel just executes.

---

## Runtime-to-kernel contract

```text
Agentic runtime
    |
    v
KV Block Router
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
    +--> selected_page_ids
    +--> skipped_page_ids
    +--> block_precision_by_page
    +--> prefetch_page_ids
    +--> routing reasons
    |
    v
IntentQuant / selected-block attention kernels
```

---

## Routing inputs

| Input | Source | Description |
|---|---|---|
| `BlockLayout` | Runtime/contex manager | Ordered semantic blocks with policies |
| `SemanticBlock.score` | Dynamic scorer or LLM signal | Relevance score (0..1) |
| `RouterConfig` | User or system config | Top-k, thresholds, pressure |
| `query_vector` (optional) | Pooled query tensor | Query-to-block similarity |
| `block_representations` (optional) | Pooled block tensors | Per-block representations |

---

## Routing decisions

Each block receives a `BlockDecision`:

| Decision | Meaning |
|---|---|
| `SELECT` | Block is selected for attention. |
| `SKIP` | Block is skipped. Zero KV bytes. |
| `PREFETCH` | Block is a prefetch candidate for future steps. |
| `QUANTIZE` | Block is selected but should use reduced precision. |
| `PIN_HIGH_PRECISION` | Block is critical and must stay FP16. |

### Default routing rules

- **ALWAYS / GLOBAL** → `PIN_HIGH_PRECISION`, always selected.
- **RECENT** → `SELECT`, always selected.
- **ATTEND** with score >= threshold → `SELECT` or `QUANTIZE`.
- **ATTEND** with score < threshold or outside top-k → `SKIP`.
- **SKIP** → `SKIP` unless score is unexpectedly high (>= threshold).
- Under high memory pressure, low-score ATTEND blocks are aggressively
  skipped and non-critical precision is downgraded.

---

## Precision assignment

Each selected or quantized block receives a `KVPrecision` from
`IntentQuantizer.assign_block_precision()`:

- FP16 — critical or high-score blocks
- FP8 — moderate-score blocks under pressure
- INT8 — medium-score blocks, or high-pressure scenarios
- INT4 / INT4_RESIDUAL — low-score blocks, experimental
- SKIP — effectively zero bytes

The router passes `memory_pressure` from its own `RouterConfig` into the
quantizer to keep precision and routing consistent.

---

## Prefetch hints

`prefetch_page_ids()` returns page IDs that may be useful in the next
decode step:

- High-score blocks that were **not selected** this step (missed
  opportunities).
- Pages adjacent to selected blocks (neighbour prefetch).
- Deduplicated, sorted by score, limited by `prefetch_top_k`.

Prefetch is a hint, not a correctness requirement. The kernel must still
produce correct output if prefetch pages are not available.

---

## Kernel metadata output

```python
from intent_attention.block_router import routing_to_kernel_metadata

meta = routing_to_kernel_metadata(routed_blocks, page_size=16)
# {
#     "selected_page_ids": [0, 1, 2, ...],
#     "prefetch_page_ids": [5, 8],
#     "block_precision_by_page": {"0": "fp16", "1": "fp16", "2": "int8", ...},
#     "selected_block_names": ["system_prompt", "doc_high", ...],
#     "skipped_block_names": ["doc_low", "unused"],
#     "reasons_by_block": {"system_prompt": "ALWAYS block, always selected", ...},
# }
```

This metadata is designed to be consumed by:

- `intent_quant_attention_reference` (CPU reference)
- `intent_quant_decode_attention_triton` (optional GPU prototype)
- Future selected-block attention kernels

---

## Usage

```python
from intent_attention import (
    BlockLayout, BlockPolicy, SemanticBlock,
    BlockRouter, RouterConfig, routing_to_kernel_metadata,
)

layout = BlockLayout([
    SemanticBlock("system",     0,   128,  BlockPolicy.ALWAYS),
    SemanticBlock("doc_high",   128, 640,  BlockPolicy.ATTEND, score=0.85),
    SemanticBlock("doc_low",    640, 1024, BlockPolicy.ATTEND, score=0.20),
    SemanticBlock("recent",     1024, 1280, BlockPolicy.RECENT),
    SemanticBlock("unused",     1280, 1440, BlockPolicy.SKIP),
])

config = RouterConfig(
    top_k_blocks=4,
    score_threshold=0.35,
    memory_pressure=0.5,
)

router = BlockRouter(config)
routed = router.route_layout(layout, total_tokens=1440)

print(router.routing_summary(routed))
# {
#     "total_blocks": 5,
#     "selected_blocks": 3,
#     "skipped_blocks": 2,
#     "selected_tokens": 1024,
#     "estimated_fp16_mb": ..., "estimated_quant_mb": ...,
#     "bytes_saved_pct": ...,
# }

meta = routing_to_kernel_metadata(routed, page_size=16)
# Ready for kernel consumption.
```

---

## Benchmark

```bash
python benchmarks/bench_block_router.py
```

Sample output (CPU, Intel i7-13700H, Python 3.11):

```
===========================================================================
  KV Block Router — CPU Routing & Cost Benchmark
===========================================================================

  This is a routing and cost-model benchmark, not a GPU speedup claim.

---------------------------------------------------------------------------

Layout: 8,192 tokens, 7 blocks

  default (top_k=8, threshold=0.35, mem_pressure=0.5)
    Total blocks:    7
    Selected blocks: 6
    Skipped blocks:  1
    Selected tokens: 7168
    Selected pages:  448 (page_size=16)
    FP16 KV (MB):    1.75
    Routed KV (MB):  0.922
    Bytes saved %:   47.32
    Precision dist:  {'FP16': 384, 'FP8': 1664, 'INT8': 2048, 'INT4_RESIDUAL': 3072}

  aggro (top_k=4, threshold=0.5, mem_pressure=0.8)
    Routed KV (MB):  0.703
    Bytes saved %:   59.82
    Precision dist:  {'FP8': 3968, 'FP16': 128, 'INT4': 3072}

  relaxed (top_k=16, threshold=0.2, mem_pressure=0.1)
    Bytes saved %:   47.32
    Precision dist:  {'FP16': 384, 'FP8': 1664, 'INT8': 2048, 'INT4_RESIDUAL': 3072}

---------------------------------------------------------------------------

Layout: 131,072 tokens, 7 blocks

  default
    Selected tokens: 114688 (7168 pages)
    FP16 KV: 28.0 MB → Routed KV: 14.75 MB → 47.32% saved

  aggro
    Routed KV: 11.25 MB → 59.82% saved
    Precision: FP8 63K, INT4 49K, FP16 2K
```

All figures are analytical cost estimates on CPU. No GPU speedup is claimed
or implied.

---

## Limitations

- **Heuristic scoring only.** The router uses `score_threshold` and
  `top_k_blocks` — no learned routing.
- **No trained router.** Replacing `BlockRouter` with a learned policy is
  future work.
- **No model-quality validation.** Routing decisions have not been
  validated for model accuracy or perplexity preservation.
- **No GPU speedup claim.** The router runs on CPU and produces metadata.
- **Partial-page masks** and causal query positions remain future work.
- **Prefetch is speculative.** Prefetch hints may be wrong; the kernel
  must tolerate missing pages.

---

## Future work

- Learned router (trainable block scoring)
- Causal query-position masking support
- Partial-page precision (split precision within a page)
- Integration with real LLM inference pipelines
- Multi-step routing with temporal dependencies
