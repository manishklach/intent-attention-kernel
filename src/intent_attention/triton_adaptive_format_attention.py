"""
Optional Triton prototype for adaptive-format decode attention.

This module extends the per-page precision idea (FP16 / INT8 / SKIP) to
per-page storage format:

    FP16   — dense FP16 page (direct load)
    INT8   — dense INT8 page + per-page scale (dequantize on load)
    SPARSE — sparse tile defined by index/value pairs (interface-first)
    SKIP   — no load, no contribution

It is the GPU-side counterpart to
``adaptive_format_attention.py`` (CPU reference).

**No GPU speedup is claimed.**  This is an experimental research prototype.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Optional, Tuple

import torch
from .adaptive_format_attention import adaptive_format_attention_reference


# ────────────────────────────────────────────────────
#  Enum & config
# ────────────────────────────────────────────────────


class AdaptivePageFormat(IntEnum):
    """Storage format tag for one physical KV page."""

    FP16 = 0
    INT8 = 1
    SPARSE = 2
    SKIP = 3


@dataclass(frozen=True)
class AdaptiveFormatKernelConfig:
    """Configuration for the adaptive-format Triton decode kernel."""

    page_size: int = 16
    head_dim: int = 64
    max_selected_pages: int = 64
    block_d: int = 64
    sparse_max_nnz: int = 256


# ────────────────────────────────────────────────────
#  Hardware detection
# ────────────────────────────────────────────────────


def is_triton_available() -> bool:
    try:
        import triton  # noqa: F401
        import triton.language as tl  # noqa: F401
        return True
    except Exception:
        return False


def is_cuda_available() -> bool:
    return torch.cuda.is_available()


def _require_triton_cuda() -> None:
    if not is_triton_available():
        raise RuntimeError(
            "Triton is not installed.  Install Triton and run on a CUDA-capable system "
            "to use the optional adaptive-format Triton kernel."
        )
    if not is_cuda_available():
        raise RuntimeError(
            "CUDA is not available.  CPU reference paths remain available."
        )


# ────────────────────────────────────────────────────
#  Triton kernel (guarded import)
# ────────────────────────────────────────────────────

if is_triton_available():
    import triton
    import triton.language as tl

    @triton.jit
    def _adaptive_format_decode_kernel(
        q_ptr,
        page_ids_ptr,
        page_formats_ptr,
        fp16_k_ptr,
        fp16_v_ptr,
        int8_k_ptr,
        int8_v_ptr,
        int8_k_scale_ptr,
        int8_v_scale_ptr,
        sparse_k_idx_ptr,
        sparse_k_val_ptr,
        sparse_v_idx_ptr,
        sparse_v_val_ptr,
        sparse_nnz_ptr,
        out_ptr,
        B: tl.constexpr,
        H: tl.constexpr,
        D: tl.constexpr,
        PAGE_SIZE: tl.constexpr,
        MAX_SELECTED_PAGES: tl.constexpr,
        BLOCK_D: tl.constexpr,
        SPARSE_MAX_NNZ: tl.constexpr,
    ):
        """
        One program per (batch, head).

        Storage assumptions:
            q:              [B, H, D]                        fp16
            page_ids:       [B, H, MAX_SELECTED_PAGES]        int32
            page_formats:   [num_pages]                       int32
            fp16_k/v:       [num_pages, PAGE_SIZE, D]         fp16
            int8_k/v:       [num_pages, PAGE_SIZE, D]         int8
            int8_k/v_scale: [num_pages]                       fp32
            sparse_k/v_idx: [num_pages, H, SPARSE_MAX_NNZ, 2] int32
            sparse_k/v_val: [num_pages, H, SPARSE_MAX_NNZ]    fp16
            sparse_nnz:     [num_pages, H]                    int32
            out:            [B, H, D]                         fp16

        Implemented paths:
            FP16  – direct load from fp16 page tensors
            INT8  – load int8 tile, multiply by per-page scale
            SPARSE– not yet implemented in Triton (interface-first)
            SKIP  – no memory traffic, no contribution
        """
        bh = tl.program_id(0)
        b = bh // H
        h = bh - b * H

        d_offsets = tl.arange(0, BLOCK_D)
        d_mask = d_offsets < D

        q_base = (b * H + h) * D
        q = tl.load(q_ptr + q_base + d_offsets, mask=d_mask, other=0.0).to(tl.float32)

        scale = 1.0 / tl.sqrt(D + 0.0)

        m_i = -float("inf")
        l_i = 0.0
        acc = tl.zeros((BLOCK_D,), dtype=tl.float32)

        for pidx in range(0, MAX_SELECTED_PAGES):
            page_id = tl.load(
                page_ids_ptr + (b * H + h) * MAX_SELECTED_PAGES + pidx,
                mask=True,
                other=-1,
            )
            valid_page = page_id >= 0

            fmt = tl.load(page_formats_ptr + page_id, mask=valid_page, other=3)
            valid_page = valid_page & (fmt != 3)

            is_fp16 = fmt == 0
            is_int8 = fmt == 1
            is_sparse = fmt == 2

            for t in range(0, PAGE_SIZE):
                token_base = (page_id * PAGE_SIZE + t) * D

                # FP16 path
                k_fp16 = tl.load(
                    fp16_k_ptr + token_base + d_offsets,
                    mask=valid_page & is_fp16 & d_mask,
                    other=0.0,
                ).to(tl.float32)

                v_fp16 = tl.load(
                    fp16_v_ptr + token_base + d_offsets,
                    mask=valid_page & is_fp16 & d_mask,
                    other=0.0,
                ).to(tl.float32)

                # INT8 path
                k_i8 = tl.load(
                    int8_k_ptr + token_base + d_offsets,
                    mask=valid_page & is_int8 & d_mask,
                    other=0,
                ).to(tl.float32)

                v_i8 = tl.load(
                    int8_v_ptr + token_base + d_offsets,
                    mask=valid_page & is_int8 & d_mask,
                    other=0,
                ).to(tl.float32)

                k_scale = tl.load(
                    int8_k_scale_ptr + page_id, mask=valid_page & is_int8, other=1.0
                )
                v_scale = tl.load(
                    int8_v_scale_ptr + page_id, mask=valid_page & is_int8, other=1.0
                )

                # SPARSE path – not yet implemented in Triton
                # For sparse pages the tile remains zero (interface-first)

                k_val = tl.where(is_fp16, k_fp16, k_i8 * k_scale)
                v_val = tl.where(is_fp16, v_fp16, v_i8 * v_scale)

                score = tl.sum(q * k_val, axis=0) * scale
                score = tl.where(valid_page & ~is_sparse, score, -float("inf"))

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


# ────────────────────────────────────────────────────
#  Public wrapper (Triton path)
# ────────────────────────────────────────────────────


def adaptive_format_decode_attention_triton(
    q: torch.Tensor,
    page_ids: torch.Tensor,
    page_formats: torch.Tensor,
    fp16_k_pages: torch.Tensor,
    fp16_v_pages: torch.Tensor,
    int8_k_pages: Optional[torch.Tensor] = None,
    int8_v_pages: Optional[torch.Tensor] = None,
    int8_k_scales: Optional[torch.Tensor] = None,
    int8_v_scales: Optional[torch.Tensor] = None,
    sparse_k_indices: Optional[torch.Tensor] = None,
    sparse_k_values: Optional[torch.Tensor] = None,
    sparse_v_indices: Optional[torch.Tensor] = None,
    sparse_v_values: Optional[torch.Tensor] = None,
    sparse_nnz: Optional[torch.Tensor] = None,
    *,
    config: Optional[AdaptiveFormatKernelConfig] = None,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Run the optional Triton adaptive-format decode kernel.

    Args:
        q:           [B, H, D] query.
        page_ids:    [B, H, max_selected_pages] selected physical page IDs (-1 unused).
        page_formats:[num_pages] format tag per page (0=FP16,1=INT8,2=SPARSE,3=SKIP).
        fp16_k_pages: [num_pages, page_size, D] FP16 K pages.
        fp16_v_pages: [num_pages, page_size, D] FP16 V pages.
        int8_k_pages: [num_pages, page_size, D] int8 K pages.
        int8_v_pages: [num_pages, page_size, D] int8 V pages.
        int8_k_scales: [num_pages] scale for each int8 K page.
        int8_v_scales: [num_pages] scale for each int8 V page.
        sparse_k_indices: [num_pages, H, sp_max_nnz, 2] sparse K indices.
        sparse_k_values:  [num_pages, H, sp_max_nnz]     sparse K values.
        sparse_v_indices: [num_pages, H, sp_max_nnz, 2] sparse V indices.
        sparse_v_values:  [num_pages, H, sp_max_nnz]     sparse V values.
        sparse_nnz:       [num_pages, H] actual non-zero count per page/head.

    Returns:
        out: [B, H, D] attention output.

    **No GPU speedup is claimed.**
    """
    _require_triton_cuda()
    cfg = config or AdaptiveFormatKernelConfig()

    if not q.is_cuda:
        raise ValueError("q must be a CUDA tensor")

    B, H, D = q.shape
    if cfg.block_d < D:
        raise ValueError("block_d must be >= D")

    if out is None:
        out = torch.empty_like(q)

    grid = (B * H,)

    # Provide empty tensors for optional sparse buffers if None
    if sparse_k_indices is None:
        sparse_k_indices = torch.zeros(0, dtype=torch.int32, device=q.device)
    if sparse_v_indices is None:
        sparse_v_indices = torch.zeros(0, dtype=torch.int32, device=q.device)
    if sparse_k_values is None:
        sparse_k_values = torch.zeros(0, dtype=torch.float16, device=q.device)
    if sparse_v_values is None:
        sparse_v_values = torch.zeros(0, dtype=torch.float16, device=q.device)
    if sparse_nnz is None:
        sparse_nnz = torch.zeros(0, dtype=torch.int32, device=q.device)

    _adaptive_format_decode_kernel[grid](
        q,
        page_ids,
        page_formats,
        fp16_k_pages,
        fp16_v_pages,
        int8_k_pages if int8_k_pages is not None else torch.zeros_like(fp16_k_pages, dtype=torch.int8, device=q.device),
        int8_v_pages if int8_v_pages is not None else torch.zeros_like(fp16_v_pages, dtype=torch.int8, device=q.device),
        int8_k_scales if int8_k_scales is not None else torch.ones(1, dtype=torch.float32, device=q.device),
        int8_v_scales if int8_v_scales is not None else torch.ones(1, dtype=torch.float32, device=q.device),
        sparse_k_indices,
        sparse_k_values,
        sparse_v_indices,
        sparse_v_values,
        sparse_nnz,
        out,
        B=B, H=H, D=D,
        PAGE_SIZE=cfg.page_size,
        MAX_SELECTED_PAGES=cfg.max_selected_pages,
        BLOCK_D=cfg.block_d,
        SPARSE_MAX_NNZ=cfg.sparse_max_nnz,
    )
    return out


# ────────────────────────────────────────────────────
#  CPU reference dispatch
# ────────────────────────────────────────────────────


def adaptive_format_decode_attention_reference_dispatch(
    q: torch.Tensor,
    page_ids: torch.Tensor,
    page_formats: torch.Tensor,
    fp16_k_pages: torch.Tensor,
    fp16_v_pages: torch.Tensor,
    int8_k_pages: Optional[torch.Tensor] = None,
    int8_v_pages: Optional[torch.Tensor] = None,
    int8_k_scales: Optional[torch.Tensor] = None,
    int8_v_scales: Optional[torch.Tensor] = None,
    sparse_k_indices: Optional[torch.Tensor] = None,
    sparse_k_values: Optional[torch.Tensor] = None,
    sparse_v_indices: Optional[torch.Tensor] = None,
    sparse_v_values: Optional[torch.Tensor] = None,
    sparse_nnz: Optional[torch.Tensor] = None,
    *,
    config: Optional[AdaptiveFormatKernelConfig] = None,
) -> torch.Tensor:
    """
    CPU reference dispatch for adaptive-format decode attention.

    Converts the Triton-style input contract into the
    ``adaptive_format_attention_reference`` calling convention so that
    tests can validate behavior on CPU-only machines.

    Returns:
        out: [B, H, D].
    """
    cfg = config or AdaptiveFormatKernelConfig()
    B, H, D = q.shape
    num_pages = fp16_k_pages.shape[0]
    PS = cfg.page_size

    # Build a matching page_table and page_counts from page_ids
    max_sel = page_ids.shape[2]
    page_table = page_ids.clone().to(torch.int32)  # [B, H, max_sel]
    page_counts = (page_ids >= 0).sum(dim=2).to(torch.int32)  # [B, H]

    # Prepare format-specific storage tensors
    kv_pages_fp16 = fp16_k_pages  # reuse for both K/V in the reference (shared KV pages)
    kv_pages_i8 = torch.zeros(num_pages, PS, D, dtype=torch.int8)
    kv_pages_scales = torch.ones(num_pages, dtype=torch.float16)
    sp_indices = torch.zeros(num_pages, 4, dtype=torch.int64)
    sp_values = torch.zeros(num_pages, 4, dtype=torch.float16)

    # Populate int8 storage
    if int8_k_pages is not None and int8_k_scales is not None:
        kv_pages_i8 = int8_k_pages.to(torch.int8)
        kv_pages_scales = int8_k_scales.to(torch.float16)

    # Populate sparse storage
    if sparse_k_indices is not None and sparse_k_values is not None:
        sp_k = sparse_k_indices.shape[2]  # sparse_max_nnz
        sp_indices = sparse_k_indices[:, 0, :, 0].to(torch.int64)  # [num_pages, sp_max]  token-offset only
        sp_values = sparse_k_values[:, 0, :].to(torch.float16)

    class _RefConfig:
        page_size = PS
        head_dim = D

    # Reshape q to [B, H, 1, D] for the reference
    q_4d = q.unsqueeze(2)  # [B, H, 1, D]

    out_4d = adaptive_format_attention_reference(
        q_4d,
        kv_pages_fp16,
        kv_pages_i8,
        kv_pages_scales,
        sp_indices,
        sp_values,
        page_formats,
        page_table,
        page_counts,
        _RefConfig(),
    )
    return out_4d.squeeze(2)  # [B, H, D]


# ────────────────────────────────────────────────────
#  Helper utilities
# ────────────────────────────────────────────────────


def make_adaptive_page_tables(
    selected_pages: torch.Tensor,
    batch: int,
    heads: int,
    max_selected_pages: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build page_ids and page_counts tensors for experiments.

    Args:
        selected_pages: [N] shared across B/H or [B, H, N].
        batch: B.
        heads: H.
        max_selected_pages: width of output page_ids.

    Returns:
        page_ids:   int32 [B, H, max_selected_pages]
        page_counts: int32 [B, H]
    """
    if selected_pages.ndim == 1:
        n_pages = selected_pages.numel()
        width = max_selected_pages or n_pages
        if n_pages > width:
            raise ValueError("selected_pages exceeds max_selected_pages")
        page_ids = torch.full(
            (batch, heads, width), fill_value=-1, dtype=torch.int32, device=selected_pages.device,
        )
        page_ids[:, :, :n_pages] = selected_pages.to(torch.int32).view(1, 1, n_pages)
        page_counts = torch.full(
            (batch, heads), fill_value=n_pages, dtype=torch.int32, device=selected_pages.device,
        )
        return page_ids, page_counts

    if selected_pages.ndim == 3:
        if selected_pages.shape[0] != batch or selected_pages.shape[1] != heads:
            raise ValueError("selected_pages [B, H, N] must match batch and heads")
        n_pages = selected_pages.shape[2]
        width = max_selected_pages or n_pages
        if n_pages > width:
            raise ValueError("selected_pages width exceeds max_selected_pages")
        page_ids = torch.full(
            (batch, heads, width), fill_value=-1, dtype=torch.int32, device=selected_pages.device,
        )
        page_ids[:, :, :n_pages] = selected_pages.to(torch.int32)
        page_counts = torch.full(
            (batch, heads), fill_value=n_pages, dtype=torch.int32, device=selected_pages.device,
        )
        return page_ids, page_counts

    raise ValueError("selected_pages must have shape [N] or [B, H, N]")