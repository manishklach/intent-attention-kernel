"""
Fused Selected-Quant Decode Kernel.

A Triton GPU kernel that consumes runtime semantic metadata (page IDs, per-page
precision tags) and loads only selected pages — with FP16, INT8, or SKIP per
page — fused into a single decode-step attention pass.

This kernel is the execution-layer backend for the KV Block Router.
It is the missing piece between runtime intent and GPU execution.

No GPU speedup is claimed. This is a research prototype.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, List, Optional, Tuple

import torch


# ------------------------------------------------------------------ #
#  Precision enum
# ------------------------------------------------------------------ #


class FusedKVPrecision(IntEnum):
    """Precision tags for per-page load decisions inside the kernel."""

    FP16 = 0
    INT8 = 1
    SKIP = 2


# ------------------------------------------------------------------ #
#  Config
# ------------------------------------------------------------------ #


@dataclass(frozen=True)
class FusedDecodeConfig:
    """Configuration for the fused selected-quant decode kernel."""

    page_size: int = 16
    head_dim: int = 64
    max_selected_pages: int = 64
    block_d: int = 64


# ------------------------------------------------------------------ #
#  Triton availability guards
# ------------------------------------------------------------------ #


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
        raise RuntimeError("Triton is not available")
    if not is_cuda_available():
        raise RuntimeError("CUDA is not available")


# ------------------------------------------------------------------ #
#  Triton kernel (conditional)
# ------------------------------------------------------------------ #

TRITON_AVAILABLE = is_triton_available()

if TRITON_AVAILABLE:
    import triton  # noqa: F811
    import triton.language as tl  # noqa: F811

    @triton.jit
    def _fused_selected_quant_decode_kernel(
        # Query
        q_ptr,
        # KV pages (FP16)
        k_fp16_ptr,
        v_fp16_ptr,
        # KV pages (INT8)
        k_i8_ptr,
        v_i8_ptr,
        # Per-page INT8 scales
        k_scale_ptr,
        v_scale_ptr,
        # Page metadata
        page_table_ptr,
        page_precision_ptr,
        page_count_ptr,
        # Output
        out_ptr,
        # Shape constants
        B: tl.constexpr,
        H: tl.constexpr,
        D: tl.constexpr,
        PAGE_SIZE: tl.constexpr,
        MAX_SELECTED_PAGES: tl.constexpr,
        BLOCK_D: tl.constexpr,
        # Precision tag constants (must match FusedKVPrecision)
        PREC_FP16: tl.constexpr,
        PREC_INT8: tl.constexpr,
        PREC_SKIP: tl.constexpr,
    ):
        """One program per (batch, head)."""

        pid = tl.program_id(0)
        b = pid // H
        h = pid % H

        n_selected = tl.load(page_count_ptr + pid)
        # Clamp to max to avoid out-of-bounds
        n_selected = tl.minimum(n_selected, MAX_SELECTED_PAGES)

        # Load query for this head: [1, D]
        q_off = q_ptr + b * H * D + h * D + tl.arange(0, BLOCK_D)
        q_mask = tl.arange(0, BLOCK_D) < D
        q = tl.load(q_off, mask=q_mask, other=0.0)
        q = tl.where(tl.arange(0, BLOCK_D) < D, q, 0.0)

        # Online safe-softmax accumulators
        m_i = tl.full([1], -float("inf"), dtype=tl.float32)
        l_i = tl.zeros([1], dtype=tl.float32)
        acc = tl.zeros([BLOCK_D], dtype=tl.float32)

        # Pre-compute 1/sqrt(D)
        inv_sqrt_d = 1.0 / math.sqrt(D)

        # Iterate over selected pages
        for p in range(MAX_SELECTED_PAGES):
            is_valid = p < n_selected

            # Load page_id
            page_id_off = page_table_ptr + b * H * MAX_SELECTED_PAGES + h * MAX_SELECTED_PAGES + p
            page_id = tl.load(page_id_off, mask=is_valid, other=0)

            # Load precision tag for this page
            prec_off = page_precision_ptr + page_id
            prec = tl.load(prec_off, mask=is_valid, other=PREC_SKIP)

            # ---- FP16 path ----
            # Load K block: [PAGE_SIZE, BLOCK_D]
            k_block = tl.zeros([PAGE_SIZE, BLOCK_D], dtype=tl.float16)
            for i in range(PAGE_SIZE):
                row_off = k_fp16_ptr + page_id * PAGE_SIZE * D + i * D + tl.arange(0, BLOCK_D)
                row_mask = (is_valid & (prec == PREC_FP16)) & (tl.arange(0, BLOCK_D) < D)
                row = tl.load(row_off, mask=row_mask, other=0.0)
                k_block = tl.where(
                    (tl.arange(0, PAGE_SIZE) == i)[:, None] & (prec == PREC_FP16),
                    row[None, :],
                    k_block,
                )

            # Load V block: [PAGE_SIZE, BLOCK_D]
            v_block = tl.zeros([PAGE_SIZE, BLOCK_D], dtype=tl.float16)
            for i in range(PAGE_SIZE):
                row_off = v_fp16_ptr + page_id * PAGE_SIZE * D + i * D + tl.arange(0, BLOCK_D)
                row_mask = (is_valid & (prec == PREC_FP16)) & (tl.arange(0, BLOCK_D) < D)
                row = tl.load(row_off, mask=row_mask, other=0.0)
                v_block = tl.where(
                    (tl.arange(0, PAGE_SIZE) == i)[:, None] & (prec == PREC_FP16),
                    row[None, :],
                    v_block,
                )

            # ---- INT8 path ----
            for i in range(PAGE_SIZE):
                row_off = k_i8_ptr + page_id * PAGE_SIZE * D + i * D + tl.arange(0, BLOCK_D)
                row_mask = (is_valid & (prec == PREC_INT8)) & (tl.arange(0, BLOCK_D) < D)
                row_i8 = tl.load(row_off, mask=row_mask, other=0)
                row_fp16 = row_i8.to(tl.float16)
                # Load per-page scale
                scale_off = k_scale_ptr + page_id
                scale = tl.load(scale_off, mask=(is_valid & (prec == PREC_INT8)), other=1.0)
                row_deq = row_fp16 * scale
                k_block = tl.where(
                    (tl.arange(0, PAGE_SIZE) == i)[:, None] & (prec == PREC_INT8),
                    row_deq[None, :],
                    k_block,
                )

            for i in range(PAGE_SIZE):
                row_off = v_i8_ptr + page_id * PAGE_SIZE * D + i * D + tl.arange(0, BLOCK_D)
                row_mask = (is_valid & (prec == PREC_INT8)) & (tl.arange(0, BLOCK_D) < D)
                row_i8 = tl.load(row_off, mask=row_mask, other=0)
                row_fp16 = row_i8.to(tl.float16)
                scale_off = v_scale_ptr + page_id
                scale = tl.load(scale_off, mask=(is_valid & (prec == PREC_INT8)), other=1.0)
                row_deq = row_fp16 * scale
                v_block = tl.where(
                    (tl.arange(0, PAGE_SIZE) == i)[:, None] & (prec == PREC_INT8),
                    row_deq[None, :],
                    v_block,
                )

            # ---- SKIP path: nothing to load, k_block/v_block remain 0 ----

            # Block-level attention: s = q @ k^T * inv_sqrt_d
            # k_block: [PAGE_SIZE, D], q: [D] -> s: [PAGE_SIZE]
            s = tl.zeros([PAGE_SIZE], dtype=tl.float32)
            for d_off in range(0, D, BLOCK_D):
                d_idx = d_off + tl.arange(0, BLOCK_D)
                d_mask = d_idx < D
                q_chunk = tl.load(q_ptr + b * H * D + h * D + d_idx, mask=d_mask, other=0.0)
                k_chunk = k_block[:, d_idx]
                s += tl.sum(q_chunk[None, :] * k_chunk, axis=1)
            s = s * inv_sqrt_d

            # Mask out invalid or SKIP pages
            is_loaded = (prec == PREC_FP16) | (prec == PREC_INT8)
            s = tl.where(
                (is_loaded & is_valid),
                s,
                tl.full([PAGE_SIZE], -float("inf"), dtype=tl.float32),
            )

            # Online softmax update
            m_new = tl.maximum(m_i, tl.max(s, axis=0))
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(s - m_new)
            l_i = alpha * l_i + tl.sum(p, axis=0)
            acc = alpha * acc

            # Accumulate: p @ v_block
            # p: [PAGE_SIZE], v_block: [PAGE_SIZE, D] -> acc: [D]
            for d_off in range(0, D, BLOCK_D):
                d_idx = d_off + tl.arange(0, BLOCK_D)
                d_mask = d_idx < D
                v_chunk = v_block[:, d_idx]
                acc_chunk = tl.sum(p[:, None] * v_chunk, axis=0)
                cur = tl.load(acc + d_idx, mask=d_mask, other=0.0)
                tl.store(acc + d_idx, cur + acc_chunk, mask=d_mask)

            m_i = m_new

        # Write final output: acc / l_i
        out_off = out_ptr + b * H * D + h * D + tl.arange(0, BLOCK_D)
        out_mask = tl.arange(0, BLOCK_D) < D
        result = tl.load(acc + tl.arange(0, BLOCK_D), mask=out_mask, other=0.0)
        l_i_safe = tl.maximum(l_i, 1e-8)
        result = result / l_i_safe
        tl.store(out_off, result.to(tl.float16), mask=out_mask)

else:
    # Placeholder when Triton is not available
    pass


# ------------------------------------------------------------------ #
#  CPU reference for correctness validation
# ------------------------------------------------------------------ #


def fused_selected_quant_decode_reference(
    query: torch.Tensor,
    k_pages_fp16: torch.Tensor,
    v_pages_fp16: torch.Tensor,
    k_pages_int8: torch.Tensor,
    v_pages_int8: torch.Tensor,
    k_scales: torch.Tensor,
    v_scales: torch.Tensor,
    page_table: torch.Tensor,
    page_precision: torch.Tensor,
    page_counts: torch.Tensor,
    config: FusedDecodeConfig,
) -> torch.Tensor:
    """CPU reference for the fused selected-quant decode kernel.

    Validates correctness by running the same algorithm on CPU.
    """
    B, H, _ = page_table.shape[:3]
    D = config.head_dim
    page_size = config.page_size

    out = torch.zeros(B, H, 1, D, dtype=torch.float16, device=query.device)

    for b_idx in range(B):
        for h_idx in range(H):
            n_pages = int(page_counts[b_idx, h_idx].item())
            q = query[b_idx, h_idx, 0, :].float()

            m_i = -float("inf")
            l_i = 0.0
            acc = torch.zeros(D, dtype=torch.float32)

            for p in range(n_pages):
                page_id = int(page_table[b_idx, h_idx, p].item())
                prec = int(page_precision[page_id].item())

                if prec == FusedKVPrecision.SKIP:
                    continue

                if prec == FusedKVPrecision.FP16:
                    k_block = k_pages_fp16[page_id].float()
                    v_block = v_pages_fp16[page_id].float()
                elif prec == FusedKVPrecision.INT8:
                    k_i8 = k_pages_int8[page_id].float()
                    v_i8 = v_pages_int8[page_id].float()
                    k_s = k_scales[page_id].float()
                    v_s = v_scales[page_id].float()
                    k_block = k_i8 * k_s
                    v_block = v_i8 * v_s
                else:
                    continue

                k_block = k_block.view(-1, D)
                v_block = v_block.view(-1, D)

                # Block-level attention
                s = (q @ k_block.T) / math.sqrt(D)
                s = s.squeeze(0)

                m_new = max(m_i, s.max().item())
                alpha = math.exp(m_i - m_new)
                p = torch.exp(s - m_new)
                l_i = alpha * l_i + p.sum()
                acc = alpha * acc
                acc = acc + (p @ v_block)
                m_i = m_new

            out[b_idx, h_idx, 0, :] = (acc / max(l_i, 1e-8)).half()

    return out


# ------------------------------------------------------------------ #
#  Metadata conversion (BlockRouter integration)
# ------------------------------------------------------------------ #


def metadata_to_kernel_tensors(
    meta: Dict,
    num_pages: int,
    B: int = 1,
    H: int = 1,
    config: Optional[FusedDecodeConfig] = None,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert BlockRouter metadata to kernel-ready tensors.

    Args:
        meta: Output of ``routing_to_kernel_metadata()``.
        num_pages: Total number of KV pages.
        B: Batch dimension.
        H: Head dimension.
        config: Kernel config (used for max_selected_pages).
        device: Target device.

    Returns:
        (page_table, page_precision, page_counts) tensors.
    """
    if config is None:
        config = FusedDecodeConfig()
    if device is None:
        device = torch.device("cpu")

    max_selected = config.max_selected_pages

    page_table = torch.zeros(B, H, max_selected, dtype=torch.int32, device=device)
    page_counts = torch.zeros(B, H, dtype=torch.int32, device=device)
    page_precision = torch.full(
        (num_pages,), FusedKVPrecision.FP16, dtype=torch.int32, device=device
    )

    # Populate precision from metadata
    for page_id_str, prec_str in meta.get("block_precision_by_page", {}).items():
        try:
            page_id = int(page_id_str)
        except ValueError:
            continue
        if prec_str == "SKIP":
            page_precision[page_id] = FusedKVPrecision.SKIP
        elif prec_str == "INT8" or prec_str == "INT4" or prec_str == "INT4_RESIDUAL":
            page_precision[page_id] = FusedKVPrecision.INT8
        else:
            page_precision[page_id] = FusedKVPrecision.FP16

    # Populate page table from selected_page_ids
    sel_ids = meta.get("selected_page_ids", [])
    n_selected = min(len(sel_ids), max_selected)
    for b_idx in range(B):
        for h_idx in range(H):
            for p in range(n_selected):
                page_table[b_idx, h_idx, p] = sel_ids[p]
            page_counts[b_idx, h_idx] = n_selected

    return page_table, page_precision, page_counts


# ------------------------------------------------------------------ #
#  Public API
# ------------------------------------------------------------------ #


def fused_selected_quant_decode(
    query: torch.Tensor,
    k_pages_fp16: torch.Tensor,
    v_pages_fp16: torch.Tensor,
    k_pages_int8: torch.Tensor,
    v_pages_int8: torch.Tensor,
    k_scales: torch.Tensor,
    v_scales: torch.Tensor,
    page_table: torch.Tensor,
    page_precision: torch.Tensor,
    page_counts: torch.Tensor,
    config: FusedDecodeConfig = FusedDecodeConfig(),
) -> torch.Tensor:
    """Fused selected-quant decode attention.

    Launches the Triton kernel if available, otherwise falls back to CPU
    reference.

    Args:
        query: [B, H, 1, D] FP16 query tensor.
        k_pages_fp16: [num_pages, page_size, D] FP16 key pages.
        v_pages_fp16: [num_pages, page_size, D] FP16 value pages.
        k_pages_int8: [num_pages, page_size, D] INT8 key pages.
        v_pages_int8: [num_pages, page_size, D] INT8 value pages.
        k_scales: [num_pages] FP16 per-page key scales.
        v_scales: [num_pages] FP16 per-page value scales.
        page_table: [B, H, max_selected_pages] int32 page IDs.
        page_precision: [num_pages] int32 per-page precision tags.
        page_counts: [B, H] int32 per-(batch,head) page counts.
        config: Kernel configuration.

    Returns:
        [B, H, 1, D] FP16 attention output.
    """
    B, H, _, D = query.shape
    device = query.device

    # Validate shapes
    if D != config.head_dim:
        raise ValueError(f"head_dim {D} != config.head_dim {config.head_dim}")

    if is_triton_available() and is_cuda_available() and device.type == "cuda":
        return _launch_triton_kernel(
            query, k_pages_fp16, v_pages_fp16,
            k_pages_int8, v_pages_int8,
            k_scales, v_scales,
            page_table, page_precision, page_counts,
            config,
        )
    else:
        return fused_selected_quant_decode_reference(
            query, k_pages_fp16, v_pages_fp16,
            k_pages_int8, v_pages_int8,
            k_scales, v_scales,
            page_table, page_precision, page_counts,
            config,
        )


# ------------------------------------------------------------------ #
#  Triton kernel launch
# ------------------------------------------------------------------ #


def _launch_triton_kernel(
    query: torch.Tensor,
    k_pages_fp16: torch.Tensor,
    v_pages_fp16: torch.Tensor,
    k_pages_int8: torch.Tensor,
    v_pages_int8: torch.Tensor,
    k_scales: torch.Tensor,
    v_scales: torch.Tensor,
    page_table: torch.Tensor,
    page_precision: torch.Tensor,
    page_counts: torch.Tensor,
    config: FusedDecodeConfig,
) -> torch.Tensor:
    B, H, _, D = query.shape
    out = torch.zeros(B, H, 1, D, dtype=torch.float16, device=query.device)

    grid = (B * H,)

    _fused_selected_quant_decode_kernel[grid](
        query,
        k_pages_fp16,
        v_pages_fp16,
        k_pages_int8,
        v_pages_int8,
        k_scales,
        v_scales,
        page_table,
        page_precision,
        page_counts,
        out,
        B,
        H,
        D,
        config.page_size,
        config.max_selected_pages,
        config.block_d,
        FusedKVPrecision.FP16,
        FusedKVPrecision.INT8,
        FusedKVPrecision.SKIP,
    )

    return out


# ------------------------------------------------------------------ #
#  Utility: fake INT8 page conversion
# ------------------------------------------------------------------ #


def fake_int8_pages_from_fp16(
    pages: torch.Tensor,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert FP16 pages to fake INT8 + scales.

    Each page is independently scaled by its per-row absmax.

    Args:
        pages: [num_pages, page_size, D] FP16.
        eps: Small constant to avoid division by zero.

    Returns:
        (int8_pages, scales) where int8_pages is int8 and scales is FP16.
    """
    num_pages = pages.shape[0]
    pages_flat = pages.view(num_pages, -1)
    absmax = pages_flat.abs().max(dim=1, keepdim=True).values.clamp(min=eps)
    scales = absmax / 127.0  # symmetric int8 range
    int8_pages = (pages_flat / scales).round().clamp(-128, 127).to(torch.int8)
    int8_pages = int8_pages.view(pages.shape)
    return int8_pages, scales.squeeze(-1)
