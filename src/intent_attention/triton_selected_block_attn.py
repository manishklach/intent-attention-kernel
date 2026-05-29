"""Triton selected-block attention kernel."""
from __future__ import annotations

import math
from typing import Optional, Tuple, Union

import torch

_triton_available: bool = False
_cuda_available: bool = torch.cuda.is_available()

def _probe_triton() -> bool:
    try:
        import triton
        return True
    except ImportError:
        return False

_triton_available = _probe_triton()

def is_triton_available() -> bool:
    return _triton_available

def is_cuda_available() -> bool:
    return _cuda_available

def _can_run_gpu_kernel() -> bool:
    if not _triton_available or not _cuda_available:
        return False
    try:
        from triton.runtime.driver import active
        return active.get_current_target().backend == "cuda"
    except Exception:
        return False


if _triton_available:
    import triton
    import triton.language as tl

    @triton.jit
    def _fwd_kernel_selected_block(
        Q, K, V, O,
        block_starts, block_ends, semantic_ids,
        stride_qb, stride_qh, stride_qm, stride_qd,
        stride_kb, stride_kh, stride_kn, stride_kd,
        stride_vb, stride_vh, stride_vn, stride_vd,
        stride_ob, stride_oh, stride_om, stride_od,
        q_len: tl.int32, n_blocks: tl.int32,
        scale: tl.float32,
        BLOCK_M: tl.constexpr, BLOCK_D: tl.constexpr,
    ):
        pid_b = tl.program_id(0)
        pid_h = tl.program_id(1)
        pid_m = tl.program_id(2)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, BLOCK_D)

        q_ptrs = Q + pid_b * stride_qb + pid_h * stride_qh \
                 + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
        q_mask = offs_m[:, None] < q_len
        q = tl.load(q_ptrs, mask=q_mask, other=0.0).to(tl.float32)

        acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
        m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)

        for i in range(n_blocks):
            start = tl.load(block_starts + i)
            end = tl.load(block_ends + i)
            n_kv = end - start

            offs_n = start + tl.arange(0, BLOCK_M)
            k_ptrs = K + pid_b * stride_kb + pid_h * stride_kh \
                     + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd
            k_mask = offs_n[:, None] < end
            k = tl.load(k_ptrs, mask=k_mask, other=0.0).to(tl.float32)

            s = tl.dot(q, tl.trans(k)) * scale

            m_ij = tl.max(s, axis=1)
            p = tl.exp(s - m_ij[:, None])
            l_ij = tl.sum(p, axis=1)

            m_new = tl.maximum(m_i, m_ij)
            alpha = tl.exp(m_i - m_new)
            beta = tl.exp(m_ij - m_new)

            v_ptrs = V + pid_b * stride_vb + pid_h * stride_vh \
                     + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd
            v_mask = offs_n[:, None] < end
            v = tl.load(v_ptrs, mask=v_mask, other=0.0).to(tl.float32)

            p = p.to(v.dtype)
            acc = alpha[:, None] * acc + beta[:, None] * tl.dot(p, v)
            l_i = alpha * l_i + beta * l_ij
            m_i = m_new

        acc = acc / l_i[:, None]
        o_ptrs = O + pid_b * stride_ob + pid_h * stride_oh \
                 + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od
        tl.store(o_ptrs, acc.to(O.dtype.element_ty), mask=q_mask)


def triton_semantic_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_starts: torch.Tensor,
    block_ends: torch.Tensor,
    scale: Optional[float] = None,
) -> torch.Tensor:
    if not _can_run_gpu_kernel():
        return _cpu_fallback(q, k, v, block_starts, block_ends, scale)
    if scale is None:
        scale = 1.0 / math.sqrt(q.size(-1))
    batch, heads, q_len, d_head = q.shape
    O = torch.empty_like(q)
    n_blocks = block_starts.shape[0]
    grid = (batch, heads, triton.cdiv(q_len, 128))
    _fwd_kernel_selected_block[grid](
        q, k, v, O,
        block_starts, block_ends, None,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        q_len, n_blocks, scale,
        BLOCK_M=128, BLOCK_D=d_head,
    )
    return O


def _cpu_fallback(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    block_starts: torch.Tensor, block_ends: torch.Tensor,
    scale: Optional[float] = None,
) -> torch.Tensor:
    if scale is None:
        scale = 1.0 / math.sqrt(q.size(-1))
    parts = []
    for s, e in zip(block_starts.tolist(), block_ends.tolist()):
        k_b = k[..., s:e, :]
        v_b = v[..., s:e, :]
        scores = torch.matmul(q, k_b.transpose(-2, -1)) * scale
        attn = torch.softmax(scores, dim=-1)
        parts.append(torch.matmul(attn, v_b))
    return sum(parts)
