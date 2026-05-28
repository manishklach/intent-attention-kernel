# Fused Selected-Quant Decode Kernel

**Runtime semantic page selection + mixed-precision dequant + attention in one fused GPU kernel.**

---

## One-Sentence Thesis

A fused Triton kernel that consumes runtime semantic metadata (page IDs, per-page precision tags) and loads only selected pages — with FP16, INT8, or SKIP per page — fused into a single decode-step attention pass, eliminating memory traffic for skipped pages and separate dequant passes.

---

## Why This Is Novel

| What existing systems do | What this kernel does |
|---|---|
| Load all KV pages uniformly (FlashAttention, vLLM) | Loads only pages selected by runtime policy |
| Apply uniform precision to all blocks (KIVI, KVQuant) | Per-page precision: FP16, INT8, or SKIP |
| Separate dequant before attention | Fused dequant + matmul per page |
| Fixed sparsity mask at compile time | Dynamic page selection from per-instance metadata |
| CPU-side selection + separate kernel launch | Single kernel consumes all metadata directly |

**No production system today fuses runtime semantic page selection, mixed-precision
page loading, and attention into a single kernel.** This is the missing
execution-layer backend for the KV Block Router.

---

## Pipeline

```
Agentic Runtime
    |
    v
KV Block Router (CPU)
    |
    +--> selected_page_ids  [B, H, max_selected]
    +--> page_precision     [num_pages] (FP16 / INT8 / SKIP)
    +--> page_counts        [B, H]
    |
    v
Fused Selected-Quant Decode Kernel (GPU)
    |
    +--> For each selected page p in range(page_count[b,h]):
    |        precision = page_precision[page_ids[b,h,p]]
    |        if precision == SKIP:     continue        # zero memory traffic
    |        if precision == FP16:     load k, v fp16
    |        if precision == INT8:     load k, v i8 + dequant with per-page scale
    |        accumulate block-level attention
    |
    v
Output: [B, H, 1, D] attention result
```

---

## Kernel Interface

```python
def fused_selected_quant_decode(
    query: torch.Tensor,              # [B, H, 1, D] FP16
    k_pages_fp16: torch.Tensor,       # [num_pages, page_size, D] FP16
    v_pages_fp16: torch.Tensor,       # [num_pages, page_size, D] FP16
    k_pages_int8: torch.Tensor,       # [num_pages, page_size, D] INT8
    v_pages_int8: torch.Tensor,       # [num_pages, page_size, D] INT8
    k_scales: torch.Tensor,           # [num_pages] FP16 (per-page scale for INT8 dequant)
    v_scales: torch.Tensor,           # [num_pages] FP16
    page_table: torch.Tensor,         # [B, H, max_selected_pages] int32
    page_precision: torch.Tensor,     # [num_pages] int32 (0=FP16, 1=INT8, 2=SKIP)
    page_counts: torch.Tensor,        # [B, H] int32
    config: FusedDecodeConfig,
) -> torch.Tensor:                    # [B, H, 1, D] FP16
```

### Precision tags

| Value | Meaning | Memory traffic per page |
|---|---|---|
| 0 (FP16) | Load K/V from FP16 storage | `2 * page_size * D * 2 bytes` |
| 1 (INT8) | Load K/V from INT8 storage + dequant | `2 * page_size * D * 1 byte` |
| 2 (SKIP) | Skip page entirely | 0 bytes |

---

## Kernel Architecture (Triton)

- **1 program per (batch, head)**: `pid = tl.program_id(0)`
- **Outer loop over selected pages**: each program iterates from `0` to `page_counts[pid]`
- **Inner page loop**: for each page, load precision → conditional load FP16 or INT8+dequant → accumulate block-level matmul
- **Online safe softmax**: `m_i`/`l_i`/`acc` accumulator pattern for numerical stability
- **No inter-warp communication**: each program is independent

### Pseudocode

```
pid = program_id(0)
b = pid // H
h = pid % H

n_pages = page_counts[pid]
m_i = -inf, l_i = 0, acc = 0

for p in range(n_pages):
    page_id = page_table[b, h, p]
    precision = page_precision[page_id]
    
    if precision == SKIP:
        continue
    
    if precision == FP16:
        k_block = load(k_fp16[page_id, :, :])  # [page_size, D]
        v_block = load(v_fp16[page_id, :, :])  # [page_size, D]
    elif precision == INT8:
        k_i8_block = load(k_int8[page_id, :, :])  # [page_size, D]
        v_i8_block = load(v_int8[page_id, :, :])
        k_scale = load(k_scales[page_id])
        v_scale = load(v_scales[page_id])
        k_block = k_i8_block * k_scale
        v_block = v_i8_block * v_scale
    
    # Block-level attention
    s = q @ k_block^T / sqrt(D)        # [1, page_size]
    m_new = max(m_i, max(s))
    p = exp(s - m_new)
    l_i = exp(m_i - m_new) * l_i + sum(p)
    acc = exp(m_i - m_new) * acc + p @ v_block
    m_i = m_new

output = acc / l_i
```

---

## Integration with KV Block Router

The kernel consumes `routing_to_kernel_metadata()` output directly:

```python
from intent_attention import BlockRouter, RouterConfig, routing_to_kernel_metadata
from intent_attention.fused_selected_quant_decode import fused_selected_quant_decode

router = BlockRouter(RouterConfig(memory_pressure=0.5))
routed = router.route_layout(layout, total_tokens=1440)
meta = routing_to_kernel_metadata(routed, page_size=16)

# Convert metadata to kernel-ready tensors
page_table = ...  # from meta["selected_page_ids"]
page_precision = ...  # from meta["block_precision_by_page"]
```

---

## Novelty Summary

1. **Semantic page selection fused into the kernel**: The kernel reads per-page metadata at load time and decides dynamically which pages to load. No fixed mask, no CPU pre-filter — the metadata is consumed directly.

2. **Mixed-precision fused dequant**: Each page is independently loaded at FP16 or INT8 precision, with dequant fused into the load. No separate dequant kernel or memory pass.

3. **Skip-page zero-traffic**: Pages marked SKIP contribute zero HBM traffic. The kernel never touches them.

4. **End-to-end semantic pipeline**: BlockRouter → Kernel Metadata → Fused Kernel. The entire pipeline from runtime intent to GPU execution is connected.

---

## Limitations (Current)

- Single-token decode only (no batched prefill)
- Non-causal (query-position masks not yet implemented)
- No packed INT4 storage (uses FP16 placeholder for now)
- No tensor-core INT8 dequant path
- Host-side metadata conversion not fused into kernel
- Warp divergence from variable page counts within a warp
