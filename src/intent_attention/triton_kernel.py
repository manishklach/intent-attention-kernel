from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple, Union

import torch

from .block_metadata import BlockLayout, BlockPolicy
from .block_table import BlockTable
from .prefetch import (
    BlockPrefetcher,
    get_prefetcher,
    get_prefetch_stream,
    launch_prefetch_pages,
    reset_prefetcher,
)

_triton_available: bool = False
_cuda_available: bool = torch.cuda.is_available()


def _probe_triton() -> bool:
    try:
        import triton  # noqa: F401
        import triton.language as tl  # noqa: F401
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

    # ------------------------------------------------------------------ #
    #  Original fp16 forward kernel (unchanged)
    # ------------------------------------------------------------------ #
    @triton.jit
    def _fwd_kernel(
        Q, K, V, O,
        page_table,
        stride_qb, stride_qh, stride_qm, stride_qd,
        stride_kb, stride_kh, stride_kn, stride_kd,
        stride_vb, stride_vh, stride_vn, stride_vd,
        stride_ob, stride_oh, stride_om, stride_od,
        q_len: tl.int32,
        kv_len: tl.int32,
        n_selected: tl.int32,
        scale: tl.float32,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_D: tl.constexpr,
        causal: tl.constexpr,
    ):
        pid_b = tl.program_id(0)
        pid_h = tl.program_id(1)
        pid_m = tl.program_id(2)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, BLOCK_D)

        q_ptrs = (Q + pid_b * stride_qb + pid_h * stride_qh
                  + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd)
        q_mask = offs_m[:, None] < q_len
        q = tl.load(q_ptrs, mask=q_mask, other=0.0).to(tl.float32)

        acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
        m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)

        for i in range(n_selected):
            page_id = tl.load(page_table + i)
            kv_start = page_id * BLOCK_N
            offs_n = kv_start + tl.arange(0, BLOCK_N)

            k_ptrs = (K + pid_b * stride_kb + pid_h * stride_kh
                      + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd)
            k_mask = offs_n[:, None] < kv_len
            k = tl.load(k_ptrs, mask=k_mask, other=0.0).to(tl.float32)

            s = tl.dot(q, tl.trans(k)) * scale

            if causal:
                q_pos = offs_m[:, None]
                kv_pos = offs_n[None, :]
                s = tl.where(q_pos >= kv_pos, s, float("-inf"))

            m_ij = tl.max(s, axis=1)
            p = tl.exp(s - m_ij[:, None])
            l_ij = tl.sum(p, axis=1)

            m_new = tl.maximum(m_i, m_ij)
            alpha = tl.exp(m_i - m_new)
            beta = tl.exp(m_ij - m_new)

            v_ptrs = (V + pid_b * stride_vb + pid_h * stride_vh
                      + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd)
            v_mask = offs_n[:, None] < kv_len
            v = tl.load(v_ptrs, mask=v_mask, other=0.0).to(tl.float32)

            p = p.to(v.dtype)
            acc = alpha[:, None] * acc + beta[:, None] * tl.dot(p, v)
            l_i = alpha * l_i + beta * l_ij
            m_i = m_new

        acc = acc / l_i[:, None]

        o_ptrs = (O + pid_b * stride_ob + pid_h * stride_oh
                  + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od)
        tl.store(o_ptrs, acc.to(O.dtype.element_ty), mask=q_mask)

    # ------------------------------------------------------------------ #
    #  INT8 quantisation kernel  (grid = num_pages, num_heads)
    # ------------------------------------------------------------------ #
    @triton.jit
    def _quant_kernel(
        K_fp16, V_fp16,
        K_int8, V_int8,
        K_scale, V_scale,
        stride_kp_in, stride_kh_in, stride_ks_in, stride_kd_in,
        stride_vp_in, stride_vh_in, stride_vs_in, stride_vd_in,
        stride_kp_out, stride_kh_out, stride_ks_out, stride_kd_out,
        stride_vp_out, stride_vh_out, stride_vs_out, stride_vd_out,
        stride_kp_s, stride_kh_s,
        stride_vp_s, stride_vh_s,
        PAGE_SIZE: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        pid_p = tl.program_id(0)
        pid_h = tl.program_id(1)

        offs_s = tl.arange(0, PAGE_SIZE)
        offs_d = tl.arange(0, BLOCK_D)

        k_fp16_ptrs = (K_fp16 + pid_p * stride_kp_in + pid_h * stride_kh_in
                       + offs_s[:, None] * stride_ks_in + offs_d[None, :] * stride_kd_in)
        k = tl.load(k_fp16_ptrs, mask=offs_s[:, None] < PAGE_SIZE, other=0.0).to(tl.float32)

        absmax = tl.max(tl.abs(k), axis=0)
        s_k = absmax / 127.0
        s_k = tl.where(s_k == 0.0, 1.0, s_k)

        k_q = tl.clamp(tl.libdevice.rint(k / s_k[None, :]), -127, 127).to(tl.int8)

        k_int8_ptrs = (K_int8 + pid_p * stride_kp_out + pid_h * stride_kh_out
                       + offs_s[:, None] * stride_ks_out + offs_d[None, :] * stride_kd_out)
        tl.store(k_int8_ptrs, k_q, mask=offs_s[:, None] < PAGE_SIZE)

        k_scale_ptrs = (K_scale + pid_p * stride_kp_s + pid_h * stride_kh_s + offs_d)
        tl.store(k_scale_ptrs, s_k.to(tl.float16), mask=offs_d < BLOCK_D)

        v_fp16_ptrs = (V_fp16 + pid_p * stride_vp_in + pid_h * stride_vh_in
                       + offs_s[:, None] * stride_vs_in + offs_d[None, :] * stride_vd_in)
        v = tl.load(v_fp16_ptrs, mask=offs_s[:, None] < PAGE_SIZE, other=0.0).to(tl.float32)

        absmax_v = tl.max(tl.abs(v), axis=0)
        s_v = absmax_v / 127.0
        s_v = tl.where(s_v == 0.0, 1.0, s_v)

        v_q = tl.clamp(tl.libdevice.rint(v / s_v[None, :]), -127, 127).to(tl.int8)

        v_int8_ptrs = (V_int8 + pid_p * stride_vp_out + pid_h * stride_vh_out
                       + offs_s[:, None] * stride_vs_out + offs_d[None, :] * stride_vd_out)
        tl.store(v_int8_ptrs, v_q, mask=offs_s[:, None] < PAGE_SIZE)

        v_scale_ptrs = (V_scale + pid_p * stride_vp_s + pid_h * stride_vh_s + offs_d)
        tl.store(v_scale_ptrs, s_v.to(tl.float16), mask=offs_d < BLOCK_D)

    # ------------------------------------------------------------------ #
    #  Forward kernel with inline INT8 dequant  (grid = B, H, cdiv(q,B))
    # ------------------------------------------------------------------ #
    @triton.jit
    def _fwd_kernel_quant(
        Q,
        K_int8, V_int8,
        K_scale, V_scale,
        O,
        page_table,
        stride_qb, stride_qh, stride_qm, stride_qd,
        stride_kp, stride_kh, stride_ks, stride_kd,
        stride_vp, stride_vh, stride_vs, stride_vd,
        stride_kb_s, stride_kh_s, stride_kp_s,
        stride_vb_s, stride_vh_s, stride_vp_s_s,
        stride_ob, stride_oh, stride_om, stride_od,
        q_len: tl.int32,
        kv_len: tl.int32,
        n_selected: tl.int32,
        scale_factor: tl.float32,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_D: tl.constexpr,
        causal: tl.constexpr,
    ):
        pid_b = tl.program_id(0)
        pid_h = tl.program_id(1)
        pid_m = tl.program_id(2)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, BLOCK_D)

        q_ptrs = (Q + pid_b * stride_qb + pid_h * stride_qh
                  + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd)
        q_mask = offs_m[:, None] < q_len
        q = tl.load(q_ptrs, mask=q_mask, other=0.0).to(tl.float32)

        acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
        m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)

        for i in range(n_selected):
            page_id = tl.load(page_table + i)
            offs_n = tl.arange(0, BLOCK_N)

            k_int8_ptrs = (K_int8 + pid_b * stride_kp + pid_h * stride_kh
                           + page_id * stride_kp + offs_n[:, None] * stride_ks
                           + offs_d[None, :] * stride_kd)
            k_mask = offs_n[:, None] < kv_len
            k_i8 = tl.load(k_int8_ptrs, mask=k_mask, other=0).to(tl.float32)

            k_scale_ptrs = (K_scale + pid_b * stride_kb_s + pid_h * stride_kh_s
                            + page_id * stride_kp_s + offs_d)
            k_s = tl.load(k_scale_ptrs, mask=offs_d < BLOCK_D, other=1.0)
            k = k_i8 * k_s[None, :]

            s = tl.dot(q, tl.trans(k)) * scale_factor

            if causal:
                q_pos = offs_m[:, None]
                kv_pos = offs_n[None, :]
                s = tl.where(q_pos >= kv_pos, s, float("-inf"))

            m_ij = tl.max(s, axis=1)
            p = tl.exp(s - m_ij[:, None])
            l_ij = tl.sum(p, axis=1)

            m_new = tl.maximum(m_i, m_ij)
            alpha = tl.exp(m_i - m_new)
            beta = tl.exp(m_ij - m_new)

            v_int8_ptrs = (V_int8 + pid_b * stride_vp + pid_h * stride_vh
                           + page_id * stride_vp + offs_n[:, None] * stride_vs
                           + offs_d[None, :] * stride_vd)
            v_i8 = tl.load(v_int8_ptrs, mask=k_mask, other=0).to(tl.float32)

            v_scale_ptrs = (V_scale + pid_b * stride_vb_s + pid_h * stride_vh_s
                            + page_id * stride_vp_s_s + offs_d)
            v_s = tl.load(v_scale_ptrs, mask=offs_d < BLOCK_D, other=1.0)
            v = v_i8 * v_s[None, :]

            p = p.to(tl.float16)
            acc = alpha[:, None] * acc + beta[:, None] * tl.dot(p, v)
            l_i = alpha * l_i + beta * l_ij
            m_i = m_new

        acc = acc / l_i[:, None]

        o_ptrs = (O + pid_b * stride_ob + pid_h * stride_oh
                  + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od)
        tl.store(o_ptrs, acc.to(O.dtype.element_ty), mask=q_mask)


# ------------------------------------------------------------------ #
#  Host helpers
# ------------------------------------------------------------------ #

def _triton_impl(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    layout: BlockLayout,
    causal: bool = False,
    threshold: float = 0.5,
) -> torch.Tensor:
    BLOCK_M = 128
    BLOCK_N = 128

    q_len = q.size(-2)
    kv_len = k.size(-2)
    head_dim = q.size(-1)

    selected_blocks = []
    for block in layout.blocks:
        if block.policy == BlockPolicy.SKIP:
            continue
        if block.policy == BlockPolicy.ATTEND:
            if block.score is None or block.score < threshold:
                continue
        selected_blocks.append(block)

    if not selected_blocks:
        return torch.zeros_like(q)

    selected_layout = BlockLayout(selected_blocks)
    bt = BlockTable(block_size=BLOCK_N)
    pages, _ = bt.create_block_table(selected_layout, kv_len)
    pages = torch.unique(pages).to(device=q.device, dtype=torch.int32)
    n_selected = pages.shape[0]

    batch, heads = q.shape[0], q.shape[1]
    O = torch.empty_like(q)
    grid = (batch, heads, triton.cdiv(q_len, BLOCK_M))

    _fwd_kernel[grid](
        q, k, v, O,
        pages,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        q_len, kv_len, n_selected,
        1.0 / math.sqrt(head_dim),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_D=head_dim,
        causal=causal,
    )

    return O


def _run_quant_gpu(
    k: torch.Tensor,
    v: torch.Tensor,
    page_size: int = 128,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch, heads, total_tokens, d_head = k.shape
    num_pages = (total_tokens + page_size - 1) // page_size

    k_int8 = torch.empty(batch, heads, num_pages, page_size, d_head, dtype=torch.int8, device=k.device)
    v_int8 = torch.empty(batch, heads, num_pages, page_size, d_head, dtype=torch.int8, device=v.device)
    k_scale = torch.empty(batch, heads, num_pages, d_head, dtype=torch.float16, device=k.device)
    v_scale = torch.empty(batch, heads, num_pages, d_head, dtype=torch.float16, device=v.device)

    k_padded = torch.zeros(batch, heads, num_pages * page_size, d_head, dtype=k.dtype, device=k.device)
    v_padded = torch.zeros_like(k_padded)
    k_padded[..., :total_tokens, :] = k
    v_padded[..., :total_tokens, :] = v

    k_4d = k_padded.view(batch, heads, num_pages, page_size, d_head)
    v_4d = v_padded.view(batch, heads, num_pages, page_size, d_head)

    grid = (num_pages, heads)
    _quant_kernel[grid](
        k_4d, v_4d,
        k_int8, v_int8,
        k_scale, v_scale,
        k_4d.stride(0), k_4d.stride(1), k_4d.stride(2), k_4d.stride(3),
        v_4d.stride(0), v_4d.stride(1), v_4d.stride(2), v_4d.stride(3),
        k_int8.stride(0), k_int8.stride(1), k_int8.stride(2), k_int8.stride(3),
        v_int8.stride(0), v_int8.stride(1), v_int8.stride(2), v_int8.stride(3),
        k_scale.stride(0), k_scale.stride(1),
        v_scale.stride(0), v_scale.stride(1),
        PAGE_SIZE=page_size,
        BLOCK_D=d_head,
    )
    return k_int8, v_int8, k_scale, v_scale


def _triton_impl_quant(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    layout: BlockLayout,
    causal: bool = False,
    threshold: float = 0.5,
    page_size: int = 128,
) -> torch.Tensor:
    q_len = q.size(-2)
    kv_len = k.size(-2)
    head_dim = q.size(-1)

    selected_blocks = []
    for block in layout.blocks:
        if block.policy == BlockPolicy.SKIP:
            continue
        if block.policy == BlockPolicy.ATTEND:
            if block.score is None or block.score < threshold:
                continue
        selected_blocks.append(block)

    if not selected_blocks:
        return torch.zeros_like(q)

    k_int8, v_int8, k_scale, v_scale = _run_quant_gpu(k, v, page_size)

    selected_layout = BlockLayout(selected_blocks)
    bt = BlockTable(block_size=page_size)
    pages, _ = bt.create_block_table(selected_layout, kv_len)
    pages = torch.unique(pages).to(device=q.device, dtype=torch.int32)
    n_selected = pages.shape[0]

    batch, heads = q.shape[0], q.shape[1]
    O = torch.empty_like(q)
    grid = (batch, heads, triton.cdiv(q_len, 128))

    _fwd_kernel_quant[grid](
        q,
        k_int8, v_int8, k_scale, v_scale,
        O,
        pages,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k_int8.stride(0), k_int8.stride(1), k_int8.stride(2), k_int8.stride(3),
        v_int8.stride(0), v_int8.stride(1), v_int8.stride(2), v_int8.stride(3),
        k_scale.stride(0), k_scale.stride(1), k_scale.stride(2),
        v_scale.stride(0), v_scale.stride(1), v_scale.stride(2),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        q_len, kv_len, n_selected,
        1.0 / math.sqrt(head_dim),
        BLOCK_M=128,
        BLOCK_N=page_size,
        BLOCK_D=head_dim,
        causal=causal,
    )

    return O


def _compute_selected_page_ids(
    layout: BlockLayout,
    kv_len: int,
    threshold: float = 0.5,
    page_size: int = 128,
) -> torch.Tensor:
    """Return sorted unique page IDs selected by *layout* under *threshold*."""
    selected = [
        b for b in layout.blocks
        if b.policy != BlockPolicy.SKIP
        and (b.policy != BlockPolicy.ATTEND or (b.score is not None and b.score >= threshold))
    ]
    if not selected:
        return torch.empty(0, dtype=torch.int32)
    selected_layout = BlockLayout(selected)
    bt = BlockTable(block_size=page_size)
    pages, _ = bt.create_block_table(selected_layout, kv_len)
    return torch.unique(pages).to(dtype=torch.int32)


def semantic_block_attention_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    layout: BlockLayout,
    causal: bool = False,
    threshold: float = 0.5,
    return_debug: bool = False,
    use_quant: bool = False,
    prefetch: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, Any]]]:
    kv_tokens = k.size(-2)
    layout.validate(kv_tokens)

    if _can_run_gpu_kernel():
        if use_quant:
            out = _triton_impl_quant(q, k, v, layout, causal=causal, threshold=threshold)
        else:
            out = _triton_impl(q, k, v, layout, causal=causal, threshold=threshold)
    else:
        from .reference import semantic_block_attention as _fallback
        out = _fallback(q, k, v, layout, causal=causal, return_debug=return_debug)
        if return_debug:
            out, debug = out
            debug.setdefault("prefetched_page_ids", [])
            return out, debug
        return out

    # ---- prefetch: predict next-step pages and launch async load ----------
    prefetched_page_ids: List[int] = []
    if prefetch and _can_run_gpu_kernel():
        prefetch_stream = get_prefetch_stream()
        if prefetch_stream is not None:
            prefetch_stream.synchronize()

        current_pages = _compute_selected_page_ids(layout, kv_tokens, threshold=threshold)
        current_pages_list = current_pages.tolist()

        prefetcher = get_prefetcher()
        predicted = prefetcher.predict_next(current_pages_list)
        prefetcher.record(current_pages_list)
        prefetched_page_ids = predicted

        if predicted and prefetch_stream is not None:
            predicted_t = torch.tensor(predicted, dtype=torch.int32, device=q.device)
            launch_prefetch_pages(k, v, predicted_t, stream=prefetch_stream)

    # ---- debug return ----------------------------------------------------
    if return_debug:
        selected = [
            b for b in layout.blocks
            if b.policy != BlockPolicy.SKIP
            and (b.policy != BlockPolicy.ATTEND or (b.score is not None and b.score >= threshold))
        ]
        selected_count = sum(b.end - b.start for b in selected)
        debug: Dict[str, Any] = {
            "selected_token_count": selected_count,
            "selected_block_names": [b.name for b in selected],
            "total_kv_tokens": kv_tokens,
            "selected_kv_tokens": selected_count,
            "prefetched_page_ids": prefetched_page_ids,
        }
        return out, debug

    return out
