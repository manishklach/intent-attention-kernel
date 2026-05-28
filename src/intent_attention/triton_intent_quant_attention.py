"""
Optional Triton prototype for IntentQuant decode attention.

This file is intentionally hardware-guarded:
- It does not require Triton at import time.
- It does not run in CPU-only CI.
- It provides a narrow first GPU target: single-token decode attention over selected KV pages.
- It is a prototype, not a production kernel.

Kernel idea:
    For each (batch, head), read a list of selected KV pages.
    Each page has precision metadata.
    The kernel loads only selected pages, dequantizes according to page precision,
    computes attention for one query token, and writes the output.

Current prototype constraints:
    - q shape: [B, H, D]
    - output shape: [B, H, D]
    - page_ids shape: [B, H, MAX_SELECTED_PAGES]
    - page_count shape: [B, H]
    - K/V page tensors are separate by precision family for simplicity.
    - FP16 and INT8 paths are implemented.
    - INT4/INT4_RESIDUAL are reserved in metadata but fall back to INT8-style handling
      unless you wire packed int4 storage later.
    - Non-causal single-token decode only.
    - Query position masking is not implemented here yet.

No GPU speedup is claimed.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Optional, Tuple

import torch


class TritonKVPrecision(IntEnum):
    """Integer precision tags passed to the Triton kernel."""

    FP16 = 0
    FP8 = 1
    INT8 = 2
    INT4 = 3
    INT4_RESIDUAL = 4
    SKIP = 5


@dataclass(frozen=True)
class IntentQuantKernelConfig:
    """Configuration for the prototype Triton kernel."""

    page_size: int = 16
    head_dim: int = 64
    max_selected_pages: int = 64
    block_d: int = 64


def is_triton_available() -> bool:
    """Return True if Triton can be imported."""

    try:
        import triton  # noqa: F401
        import triton.language as tl  # noqa: F401

        return True
    except Exception:
        return False


def is_cuda_available() -> bool:
    """Return True if PyTorch sees a CUDA device."""

    return torch.cuda.is_available()


def _require_triton_cuda() -> None:
    if not is_triton_available():
        raise RuntimeError(
            "Triton is not installed. Install Triton and run on a CUDA-capable system "
            "to use the optional IntentQuant Triton kernel."
        )

    if not is_cuda_available():
        raise RuntimeError(
            "CUDA is not available. The optional IntentQuant Triton kernel requires "
            "an NVIDIA GPU. CPU reference paths remain available."
        )


def _validate_decode_inputs(
    q: torch.Tensor,
    page_ids: torch.Tensor,
    page_count: torch.Tensor,
    page_precision: torch.Tensor,
    out: Optional[torch.Tensor],
) -> None:
    if q.ndim != 3:
        raise ValueError(f"q must have shape [B, H, D], got {tuple(q.shape)}")

    if page_ids.ndim != 3:
        raise ValueError(
            f"page_ids must have shape [B, H, max_selected_pages], got {tuple(page_ids.shape)}"
        )

    if page_count.ndim != 2:
        raise ValueError(f"page_count must have shape [B, H], got {tuple(page_count.shape)}")

    if page_precision.ndim != 1:
        raise ValueError(
            f"page_precision must have shape [num_pages], got {tuple(page_precision.shape)}"
        )

    if page_ids.shape[0] != q.shape[0] or page_ids.shape[1] != q.shape[1]:
        raise ValueError("page_ids batch/head dimensions must match q")

    if page_count.shape[0] != q.shape[0] or page_count.shape[1] != q.shape[1]:
        raise ValueError("page_count batch/head dimensions must match q")

    if out is not None and out.shape != q.shape:
        raise ValueError(f"out must have shape {tuple(q.shape)}, got {tuple(out.shape)}")


def make_page_tables_from_selected_pages(
    selected_pages: torch.Tensor,
    batch: int,
    heads: int,
    max_selected_pages: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convenience helper for simple experiments.

    Args:
        selected_pages:
            Either [N] shared across all B/H or [B, H, N].
        batch:
            Batch size.
        heads:
            Number of heads.
        max_selected_pages:
            Optional output page-table width.

    Returns:
        page_ids:
            int32 tensor [B, H, max_selected_pages]
        page_count:
            int32 tensor [B, H]
    """

    if selected_pages.ndim == 1:
        n_pages = selected_pages.numel()
        width = max_selected_pages or n_pages

        if n_pages > width:
            raise ValueError("selected_pages length exceeds max_selected_pages")

        page_ids = torch.full(
            (batch, heads, width),
            fill_value=-1,
            dtype=torch.int32,
            device=selected_pages.device,
        )
        page_ids[:, :, :n_pages] = selected_pages.to(torch.int32).view(1, 1, n_pages)

        page_count = torch.full(
            (batch, heads),
            fill_value=n_pages,
            dtype=torch.int32,
            device=selected_pages.device,
        )

        return page_ids, page_count

    if selected_pages.ndim == 3:
        if selected_pages.shape[0] != batch or selected_pages.shape[1] != heads:
            raise ValueError("selected_pages [B, H, N] must match batch and heads")

        n_pages = selected_pages.shape[2]
        width = max_selected_pages or n_pages

        if n_pages > width:
            raise ValueError("selected_pages width exceeds max_selected_pages")

        page_ids = torch.full(
            (batch, heads, width),
            fill_value=-1,
            dtype=torch.int32,
            device=selected_pages.device,
        )
        page_ids[:, :, :n_pages] = selected_pages.to(torch.int32)

        page_count = torch.full(
            (batch, heads),
            fill_value=n_pages,
            dtype=torch.int32,
            device=selected_pages.device,
        )

        return page_ids, page_count

    raise ValueError("selected_pages must have shape [N] or [B, H, N]")


if is_triton_available():
    import triton
    import triton.language as tl

    @triton.jit
    def _intent_quant_decode_attention_kernel(
        q_ptr,
        k_fp16_ptr,
        v_fp16_ptr,
        k_i8_ptr,
        v_i8_ptr,
        k_i8_scale_ptr,
        v_i8_scale_ptr,
        page_ids_ptr,
        page_count_ptr,
        page_precision_ptr,
        out_ptr,
        B: tl.constexpr,
        H: tl.constexpr,
        D: tl.constexpr,
        PAGE_SIZE: tl.constexpr,
        MAX_SELECTED_PAGES: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """
        One Triton program handles one (batch, head) decode query.

        Storage assumptions:
            q_ptr:          [B, H, D] fp16/fp32
            k_fp16_ptr:     [num_pages, PAGE_SIZE, D] fp16
            v_fp16_ptr:     [num_pages, PAGE_SIZE, D] fp16
            k_i8_ptr:       [num_pages, PAGE_SIZE, D] int8
            v_i8_ptr:       [num_pages, PAGE_SIZE, D] int8
            k_i8_scale_ptr: [num_pages] fp32
            v_i8_scale_ptr: [num_pages] fp32
            page_ids_ptr:   [B, H, MAX_SELECTED_PAGES] int32
            page_count_ptr: [B, H] int32
            page_precision_ptr: [num_pages] int32
            out_ptr:        [B, H, D]

        Implemented precision paths:
            FP16: direct load from fp16 page tensors
            INT8: load int8 and multiply by per-page scale

        Reserved precision tags:
            FP8, INT4, INT4_RESIDUAL currently route through INT8 path
            if the int8 buffers are populated. This keeps the first kernel simple.
        """

        bh = tl.program_id(0)
        b = bh // H
        h = bh - b * H

        d_offsets = tl.arange(0, BLOCK_D)
        d_mask = d_offsets < D

        q_base = (b * H + h) * D
        q = tl.load(q_ptr + q_base + d_offsets, mask=d_mask, other=0.0).to(tl.float32)

        page_count = tl.load(page_count_ptr + b * H + h)
        scale = 1.0 / tl.sqrt(D + 0.0)

        m_i = -float("inf")
        l_i = 0.0
        acc = tl.zeros((BLOCK_D,), dtype=tl.float32)

        for pidx in range(0, MAX_SELECTED_PAGES):
            active_page = pidx < page_count
            page_id = tl.load(
                page_ids_ptr + (b * H + h) * MAX_SELECTED_PAGES + pidx,
                mask=active_page,
                other=-1,
            )

            valid_page = active_page & (page_id >= 0)
            precision = tl.load(page_precision_ptr + page_id, mask=valid_page, other=5)

            valid_page = valid_page & (precision != 5)

            for t in range(0, PAGE_SIZE):
                token_base = (page_id * PAGE_SIZE + t) * D

                is_fp16 = precision == 0

                k_fp16 = tl.load(
                    k_fp16_ptr + token_base + d_offsets,
                    mask=valid_page & is_fp16 & d_mask,
                    other=0.0,
                ).to(tl.float32)

                v_fp16 = tl.load(
                    v_fp16_ptr + token_base + d_offsets,
                    mask=valid_page & is_fp16 & d_mask,
                    other=0.0,
                ).to(tl.float32)

                k_i8 = tl.load(
                    k_i8_ptr + token_base + d_offsets,
                    mask=valid_page & (~is_fp16) & d_mask,
                    other=0,
                ).to(tl.float32)

                v_i8 = tl.load(
                    v_i8_ptr + token_base + d_offsets,
                    mask=valid_page & (~is_fp16) & d_mask,
                    other=0,
                ).to(tl.float32)

                k_scale = tl.load(k_i8_scale_ptr + page_id, mask=valid_page, other=1.0)
                v_scale = tl.load(v_i8_scale_ptr + page_id, mask=valid_page, other=1.0)

                k_val = tl.where(is_fp16, k_fp16, k_i8 * k_scale)
                v_val = tl.where(is_fp16, v_fp16, v_i8 * v_scale)

                score = tl.sum(q * k_val, axis=0) * scale
                score = tl.where(valid_page, score, -float("inf"))

                m_new = tl.maximum(m_i, score)
                alpha = tl.exp(m_i - m_new)
                beta = tl.exp(score - m_new)

                acc = acc * alpha + v_val * beta
                l_i = l_i * alpha + beta
                m_i = m_new

        result = acc / l_i
        result = tl.where(l_i > 0.0, result, 0.0)

        out_base = (b * H + h) * D
        tl.store(out_ptr + out_base + d_offsets, result, mask=d_mask)

else:
    triton = None
    tl = None


def intent_quant_decode_attention_triton(
    q: torch.Tensor,
    k_pages_fp16: torch.Tensor,
    v_pages_fp16: torch.Tensor,
    k_pages_i8: torch.Tensor,
    v_pages_i8: torch.Tensor,
    k_i8_scales: torch.Tensor,
    v_i8_scales: torch.Tensor,
    page_ids: torch.Tensor,
    page_count: torch.Tensor,
    page_precision: torch.Tensor,
    *,
    config: Optional[IntentQuantKernelConfig] = None,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Run the optional Triton prototype for single-token decode attention.

    Args:
        q:
            [B, H, D] query tensor.
        k_pages_fp16:
            [num_pages, page_size, D] fp16/bf16 tensor.
        v_pages_fp16:
            [num_pages, page_size, D] fp16/bf16 tensor.
        k_pages_i8:
            [num_pages, page_size, D] int8 tensor.
        v_pages_i8:
            [num_pages, page_size, D] int8 tensor.
        k_i8_scales:
            [num_pages] float scale for K int8 pages.
        v_i8_scales:
            [num_pages] float scale for V int8 pages.
        page_ids:
            [B, H, max_selected_pages] int32 selected page IDs.
        page_count:
            [B, H] int32 number of selected pages per batch/head.
        page_precision:
            [num_pages] int32 precision tag for each page.
        config:
            Optional kernel config.
        out:
            Optional output tensor [B, H, D].

    Returns:
        out:
            [B, H, D]

    Notes:
        This is a prototype kernel. It has not been performance tuned.
        It does not implement causal/query-position masking.
        It does not implement native packed INT4 yet.
    """

    _require_triton_cuda()

    cfg = config or IntentQuantKernelConfig()
    _validate_decode_inputs(q, page_ids, page_count, page_precision, out)

    if not q.is_cuda:
        raise ValueError("q must be a CUDA tensor")

    bsz, heads, dim = q.shape

    if dim != cfg.head_dim:
        raise ValueError(f"q head_dim={dim} does not match config.head_dim={cfg.head_dim}")

    if cfg.block_d < dim:
        raise ValueError(
            "This prototype expects block_d >= head_dim so one program handles the full head."
        )

    if k_pages_fp16.shape != v_pages_fp16.shape:
        raise ValueError("k_pages_fp16 and v_pages_fp16 must have the same shape")

    if k_pages_i8.shape != v_pages_i8.shape:
        raise ValueError("k_pages_i8 and v_pages_i8 must have the same shape")

    if k_pages_fp16.ndim != 3:
        raise ValueError("page tensors must have shape [num_pages, page_size, D]")

    if k_pages_fp16.shape[1] != cfg.page_size:
        raise ValueError("page_size mismatch")

    if k_pages_fp16.shape[2] != dim:
        raise ValueError("page head_dim mismatch")

    if page_ids.shape[2] != cfg.max_selected_pages:
        raise ValueError("page_ids width must match config.max_selected_pages")

    if out is None:
        out = torch.empty_like(q)

    grid = (bsz * heads,)

    _intent_quant_decode_attention_kernel[grid](
        q,
        k_pages_fp16,
        v_pages_fp16,
        k_pages_i8,
        v_pages_i8,
        k_i8_scales,
        v_i8_scales,
        page_ids,
        page_count,
        page_precision,
        out,
        B=bsz,
        H=heads,
        D=dim,
        PAGE_SIZE=cfg.page_size,
        MAX_SELECTED_PAGES=cfg.max_selected_pages,
        BLOCK_D=cfg.block_d,
    )

    return out


def fake_int8_pages_from_fp16(
    pages: torch.Tensor,
    *,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert page tensor to simple symmetric int8 pages for experimentation.

    Args:
        pages:
            Tensor [num_pages, page_size, D]
        eps:
            Numerical stability epsilon.

    Returns:
        q_pages:
            int8 tensor with same shape.
        scales:
            float32 tensor [num_pages], where reconstructed page = q_page * scale.

    This is intentionally simple. Real kernels would likely use per-channel,
    per-group, or asymmetric scales depending on K/V and hardware constraints.
    """

    if pages.ndim != 3:
        raise ValueError("pages must have shape [num_pages, page_size, D]")

    num_pages = pages.shape[0]
    flat = pages.float().reshape(num_pages, -1)

    max_abs = flat.abs().amax(dim=1).clamp_min(eps)
    scales = max_abs / 127.0

    q = torch.round(flat / scales[:, None]).clamp(-127, 127).to(torch.int8)
    q = q.reshape_as(pages)

    return q, scales.to(torch.float32)


def make_precision_tensor(
    num_pages: int,
    fp16_pages: Optional[torch.Tensor] = None,
    int8_pages: Optional[torch.Tensor] = None,
    *,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Build a page_precision tensor for experiments.

    Args:
        num_pages:
            Total number of pages.
        fp16_pages:
            Optional tensor/list of page IDs to mark as FP16.
        int8_pages:
            Optional tensor/list of page IDs to mark as INT8.
        device:
            Output device.

    Returns:
        int32 tensor [num_pages]
    """

    precision = torch.full(
        (num_pages,),
        fill_value=int(TritonKVPrecision.INT8),
        dtype=torch.int32,
        device=device,
    )

    if fp16_pages is not None:
        precision[fp16_pages.to(device=device, dtype=torch.long)] = int(TritonKVPrecision.FP16)

    if int8_pages is not None:
        precision[int8_pages.to(device=device, dtype=torch.long)] = int(TritonKVPrecision.INT8)

    return precision
