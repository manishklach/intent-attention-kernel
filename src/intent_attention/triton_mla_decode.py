"""Triton MLA decode kernel — compressed latent attention on GPU."""
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
    def _mla_decode_kernel(
        Q_Absorb, C, W_VO, O,
        page_table,
        stride_qb, stride_qq, stride_qd,
        stride_cn, stride_cd,
        stride_wr, stride_wc,
        stride_ob, stride_oq, stride_od,
        q_len: tl.int32,
        n_selected: tl.int32,
        scale: tl.float32,
        BLOCK_N: tl.constexpr,
        BLOCK_DC: tl.constexpr,
        BLOCK_DO: tl.constexpr,
    ):
        pid = tl.program_id(0)
        batch = pid // q_len
        q_pos = pid % q_len

        offs_dc = tl.arange(0, BLOCK_DC)
        q_ptrs = Q_Absorb + batch * stride_qb + q_pos * stride_qq + offs_dc
        q = tl.load(q_ptrs, mask=offs_dc < BLOCK_DC, other=0.0).to(tl.float32)

        acc = tl.zeros([BLOCK_DC], dtype=tl.float32)
        m_i = tl.full([1], -float("inf"), dtype=tl.float32)
        l_i = tl.zeros([1], dtype=tl.float32)

        for i in range(n_selected):
            page_id = tl.load(page_table + i)
            c_start = page_id * BLOCK_N
            offs_n = c_start + tl.arange(0, BLOCK_N)

            c_ptrs = C + offs_n[:, None] * stride_cn + offs_dc[None, :] * stride_cd
            c = tl.load(c_ptrs, mask=offs_n[:, None] < q_len, other=0.0).to(tl.float32)

            s = tl.sum(q[None, :] * c, axis=1) * scale
            m_ij = tl.max(s, axis=0)
            p = tl.exp(s - m_ij)
            l_ij = tl.sum(p, axis=0)

            m_new = tl.maximum(m_i, m_ij)
            alpha = tl.exp(m_i - m_new)
            beta = tl.exp(m_ij - m_new)

            acc = alpha * acc + beta * tl.sum(p[:, None] * c, axis=0)
            l_i = alpha * l_i + beta * l_ij
            m_i = m_new

        context = tl.where(m_i > -float("inf"), acc / l_i, tl.zeros([BLOCK_DC], dtype=tl.float32))

        offs_do = tl.arange(0, BLOCK_DO)
        out = tl.zeros([BLOCK_DO], dtype=tl.float32)
        for k in range(0, BLOCK_DC, 32):
            dc_off = k + tl.arange(0, 32)
            mask = dc_off < BLOCK_DC
            c_val = tl.load(context + dc_off, mask=mask, other=0.0)
            w_ptrs = W_VO + dc_off[:, None] * stride_wr + offs_do[None, :] * stride_wc
            w = tl.load(w_ptrs, mask=mask[:, None] & (offs_do[None, :] < BLOCK_DO), other=0.0)
            out += tl.sum(c_val[:, None] * w, axis=0)

        o_ptrs = O + batch * stride_ob + q_pos * stride_oq + offs_do
        tl.store(o_ptrs, out.to(O.dtype.element_ty), mask=offs_do < BLOCK_DO)


def mla_decode_triton(
    q_absorb: torch.Tensor,
    C: torch.Tensor,
    W_VO_fused: torch.Tensor,
    page_table: torch.Tensor,
    page_size: int = 64,
) -> torch.Tensor:
    batch, q_len, d_c = q_absorb.shape
    d_out = W_VO_fused.shape[1]
    n_selected = page_table.shape[0]

    if _can_run_gpu_kernel():
        O = torch.empty(batch, q_len, d_out, device=q_absorb.device, dtype=q_absorb.dtype)
        grid = (batch * q_len,)
        _mla_decode_kernel[grid](
            q_absorb, C, W_VO_fused, O, page_table,
            q_absorb.stride(0), q_absorb.stride(1), q_absorb.stride(2),
            C.stride(0), C.stride(1),
            W_VO_fused.stride(0), W_VO_fused.stride(1),
            O.stride(0), O.stride(1), O.stride(2),
            q_len, n_selected,
            1.0 / math.sqrt(d_c),
            BLOCK_N=page_size, BLOCK_DC=d_c, BLOCK_DO=d_out,
        )
        return O
    return _mla_decode_cpu(q_absorb, C, W_VO_fused, page_table, page_size)


def _mla_decode_cpu(
    q_absorb: torch.Tensor,
    C: torch.Tensor,
    W_VO_fused: torch.Tensor,
    page_table: torch.Tensor,
    page_size: int = 64,
) -> torch.Tensor:
    batch, q_len, d_c = q_absorb.shape
    d_out = W_VO_fused.shape[1]
    out = torch.zeros(batch, q_len, d_out, device=q_absorb.device, dtype=q_absorb.dtype)
    scale = 1.0 / math.sqrt(d_c)
    for b in range(batch):
        for qp in range(q_len):
            q = q_absorb[b, qp]
            acc = torch.zeros(d_c, device=q_absorb.device)
            m_i = -float("inf")
            l_i = 0.0
            for i in range(page_table.shape[0]):
                pid = page_table[i].item()
                c_page = C[pid * page_size: (pid + 1) * page_size]
                s = (q[None, :] * c_page).sum(dim=-1) * scale
                m_ij = s.max()
                p = torch.exp(s - m_ij)
                l_ij = p.sum()
                m_new = max(m_i, m_ij)
                alpha = math.exp(m_i - m_new)
                beta = math.exp(m_ij - m_new)
                acc = alpha * acc + beta * (p[:, None] * c_page).sum(dim=0)
                l_i = alpha * l_i + beta * l_ij
                m_i = m_new
            context = acc / l_i if m_i > -float("inf") else torch.zeros(d_c, device=q_absorb.device)
            out[b, qp] = context @ W_VO_fused
    return out
