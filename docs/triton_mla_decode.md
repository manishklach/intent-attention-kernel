# Triton MLA Decode Kernel

## Motivation

Multi-Head Latent Attention (MLA), introduced in DeepSeek-V2, compresses the KV
cache from `n_heads × d_head` elements per token to a much smaller latent
dimension `d_c`. At DeepSeek scale (`d_c=512` vs `n_heads×d_head=4096`), this
yields roughly **8× KV cache compression**.

The existing `mla.py` reference implements MLA on CPU but cannot exploit this
compression benefit during GPU decode. The `triton_mla_decode.py` kernel fills
this gap: it operates directly on the compressed latent dimension, avoiding the
need to expand back to full head dimension before attention.

## MLA Compressed KV Representation

Standard attention stores separate K and V tensors of shape
`[total_tokens, n_heads, d_head]`. MLA instead stores a single compressed
latent vector per token (plus optional RoPE side-car):

```
Standard:  K[t, h] ∈ R^{d_head}, V[t, h] ∈ R^{d_head}  →  2 × n_heads × d_head
MLA:       c[t]   ∈ R^{d_c},     rope[t] ∈ R^{d_rope}  →  d_c + d_rope
```

At inference, the latent `c[t]` is projected back to K/V heads inside the
attention kernel via absorbed weight matrices — no explicit expansion needed.

## Absorbed QK and VO Weights

The MLA reference fuses projections to avoid redundant computation:

```
W_QK_fused = W_UQ @ W_UK.T     (d_model × d_c)
W_VO_fused = W_UV @ W_O        (d_c × d_model)
```

The query is projected into latent space once: `q_absorb = q @ W_QK_fused`.
The scores are computed directly in latent space: `scores = q_absorb @ C.T`.
The context vector is projected to output once: `out = context @ W_VO_fused`.

## Page-Table Selected Latent Decode

The kernel does not attend to all latent tokens. Instead, a page table
(`page_table: [n_selected] int32`) selects which latent pages to process,
mirroring the block selection from `BlockRouter`/`BlockLayout`. The page table
is iterated sequentially:

```python
for i in range(n_selected):
    page_id = page_table[i]
    c_page = C[page_id * page_size : (page_id + 1) * page_size]
    # online softmax update
```

Pages are processed in page-table order (which may differ from physical order).

## Online Softmax

The kernel accumulates attention over selected pages using the standard online
softmax algorithm (safe softmax), which avoids materialising the full attention
matrix:

```
m_i = max over all seen scores
l_i = sum(exp(scores - m_i)) over all seen scores
acc = sum(exp(scores - m_i) * c) over all seen scores

final context = acc / l_i
```

This allows iterating an arbitrary number of selected KV pages without
pre-allocating a large attention score matrix.

## Empty Page-Table NaN Guard

When no pages are selected (`page_table` is empty), the online softmax state
never updates: `m_i` stays at `-inf`, `l_i` stays at 0, and `acc` stays at 0.
Without a guard, `acc / l_i` produces `0.0 / 0.0 = NaN`.

The kernel guards against this:

```python
# Triton path
context = tl.where(m_i > -float("inf"), acc / l_i,
                   tl.zeros([BLOCK_DC], dtype=tl.float32))

# CPU fallback
context = acc / l_i if m_i > -float("inf") else torch.zeros(d_c, ...)
```

When no pages are selected, the output is all zeros (no NaN).

## CPU Fallback

When Triton or CUDA is unavailable — or the input tensor is not on CUDA — the
kernel transparently falls back to `_mla_decode_cpu()`, which implements the
identical online softmax algorithm in pure PyTorch:

```python
def mla_decode_triton(q_absorb, C, W_VO_fused, page_table, page_size=64):
    if _can_run_gpu_kernel():
        return _gpu_path(...)
    return _mla_decode_cpu(...)
```

This ensures all CPU-only tests, CI, and development work without a GPU.

## Limitations

| Limitation | Impact |
|---|---|
| No causal masking | GPU Triton path does not support position-aware causal masking yet. CPU fallback processes all selected pages without masking. |
| Uniform page sizes | All pages in `C` must have `page_size` tokens. Non-uniform page sizes require padding or multiple kernel invocations. |
| No cross-attention support | The kernel is designed for decoder self-attention only. Cross-attention with separate encoder/output would need a different interface. |
| No learned routing | Pages are selected by score threshold. The kernel does not perform its own selection. |
| No real GPU validation | The GPU kernel compiles with Triton but has not been benchmarked on production hardware (A100, H100, etc.). **No GPU speedup is claimed.** |
| Half-dim RoPE not fused | The MLA RoPE side-car (rotary applied to a subset of dims) is not handled by this kernel. The reference `mla.py` manages RoPE separately. |

## Future GPU Validation

When GPU hardware is available, the following should be validated:

1. **End-to-end correctness** — match `mla_sparse_decode_reference` output at
   `atol=1e-2` across random seeds, batch sizes, and page selections.
2. **No NaN on empty selection** — `page_table=[]` produces zeros.
3. **Fused W_VO projection throughput** — compare kernel-fused vs separate
   `context @ W_VO_fused` matmul latency.
4. **Online softmax FP accumulation** — compare online vs global softmax MSE
   across varying `d_c` (64–512).
5. **Memory bandwidth utilisation** — measure against roofline model on target
   hardware. Expect ~8× KV traffic reduction vs dense MLA at DeepSeek scale.
