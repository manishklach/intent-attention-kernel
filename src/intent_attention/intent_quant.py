from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import auto
from typing import Any, Dict, List, Optional, Tuple

import torch

from ._enum import StrEnum
from .block_metadata import BlockLayout, BlockPolicy, SemanticBlock


class KVPrecision(StrEnum):
    FP16 = auto()
    FP8 = auto()
    INT8 = auto()
    INT4 = auto()
    INT4_RESIDUAL = auto()
    SKIP = auto()


_BYTES_PER_VALUE: Dict[KVPrecision, float] = {
    KVPrecision.FP16: 2.0,
    KVPrecision.FP8: 1.0,
    KVPrecision.INT8: 1.0,
    KVPrecision.INT4: 0.5,
    KVPrecision.INT4_RESIDUAL: 1.0,
    KVPrecision.SKIP: 0.0,
}


@dataclass
class QuantPolicy:
    precision: KVPrecision
    reason: str = ""
    estimated_bytes_per_value: float = field(init=False)
    requires_scale: bool = field(init=False)
    uses_residual: bool = field(init=False)

    def __post_init__(self) -> None:
        self.estimated_bytes_per_value = _BYTES_PER_VALUE[self.precision]
        self.requires_scale = self.precision in (
            KVPrecision.INT8,
            KVPrecision.INT4,
            KVPrecision.INT4_RESIDUAL,
        )
        self.uses_residual = self.precision == KVPrecision.INT4_RESIDUAL


class IntentQuantizer:
    """Assigns mixed-precision KV quantization policies based on semantic
    block metadata, score, recency, and memory pressure.

    This is a CPU-first research prototype. No GPU speedups,
    model accuracy, or perplexity preservation is claimed.
    """

    def __init__(
        self,
        memory_pressure: float = 0.0,
        preserve_recent: bool = True,
        preserve_global: bool = True,
        high_score_threshold: float = 0.75,
        medium_score_threshold: float = 0.40,
    ) -> None:
        if not 0.0 <= memory_pressure <= 1.0:
            raise ValueError(f"memory_pressure must be in [0, 1], got {memory_pressure}")
        self.memory_pressure = memory_pressure
        self.preserve_recent = preserve_recent
        self.preserve_global = preserve_global
        self.high_score_threshold = high_score_threshold
        self.medium_score_threshold = medium_score_threshold

    def assign_block_precision(self, block: SemanticBlock) -> QuantPolicy:
        pol = block.policy
        score = block.score
        mp = self.memory_pressure

        # --- ALWAYS / GLOBAL -------------------------------------------------
        if pol in (BlockPolicy.ALWAYS, BlockPolicy.GLOBAL):
            if pol == BlockPolicy.GLOBAL and self.preserve_global:
                if mp > 0.8:
                    return QuantPolicy(KVPrecision.INT8, "global, extreme pressure")
                return QuantPolicy(KVPrecision.FP16, "global, preserved")
            if mp > 0.9:
                return QuantPolicy(KVPrecision.INT8, "always, extreme pressure")
            if mp > 0.7:
                return QuantPolicy(KVPrecision.FP8, "always, high pressure")
            return QuantPolicy(KVPrecision.FP16, "always, full precision")

        # --- RECENT ----------------------------------------------------------
        if pol == BlockPolicy.RECENT:
            if self.preserve_recent:
                if mp > 0.8:
                    return QuantPolicy(KVPrecision.INT8, "recent, extreme pressure")
                if mp > 0.5:
                    return QuantPolicy(KVPrecision.FP8, "recent, moderate pressure")
                return QuantPolicy(KVPrecision.FP8, "recent, preserved")
            if mp > 0.6:
                return QuantPolicy(KVPrecision.INT4_RESIDUAL, "recent, high pressure, not preserved")
            return QuantPolicy(KVPrecision.INT8, "recent, not preserved")

        # --- ATTEND ---------------------------------------------------------
        if pol == BlockPolicy.ATTEND:
            if score is None:
                baseline = KVPrecision.INT8 if mp > 0.5 else KVPrecision.FP16
                return QuantPolicy(baseline, "attend, no score")
            if score >= self.high_score_threshold:
                if mp > 0.8:
                    return QuantPolicy(KVPrecision.INT8, "high-score, extreme pressure")
                if mp > 0.5:
                    return QuantPolicy(KVPrecision.FP8, "high-score, moderate pressure")
                return QuantPolicy(KVPrecision.INT8, "high-score attend")
            if score >= self.medium_score_threshold:
                if mp > 0.6:
                    return QuantPolicy(KVPrecision.INT4, "medium-score, high pressure")
                return QuantPolicy(KVPrecision.INT4_RESIDUAL, "medium-score attend")
            if mp > 0.3:
                return QuantPolicy(KVPrecision.SKIP, "low-score, pressure")
            return QuantPolicy(KVPrecision.INT4, "low-score attend")

        # --- SKIP ------------------------------------------------------------
        if pol == BlockPolicy.SKIP:
            return QuantPolicy(KVPrecision.SKIP, "skipped block")

        return QuantPolicy(KVPrecision.FP16, "default")

    def assign_layout_precision(self, layout: BlockLayout) -> Dict[str, QuantPolicy]:
        policies: Dict[str, QuantPolicy] = {}
        for block in layout.blocks:
            policies[block.name] = self.assign_block_precision(block)
        return policies

    def estimate_layout_bytes(
        self,
        layout: BlockLayout,
        heads: int,
        head_dim: int,
        kv_tensors: int = 2,
    ) -> Dict[str, Any]:
        policies = self.assign_layout_precision(layout)
        dense_fp16: int = 0
        intent_bytes: float = 0.0
        precision_counts: Dict[str, int] = {}
        critical_fp16_tokens: int = 0

        for block in layout.blocks:
            policy = policies[block.name]
            n_tokens = block.end - block.start
            per_token = policy.estimated_bytes_per_value

            dense_body = kv_tensors * n_tokens * head_dim * 2  # fp16 baseline
            dense_fp16 += dense_body
            intent_body = kv_tensors * n_tokens * head_dim * per_token
            intent_bytes += intent_body

            precision_counts[policy.precision.value] = (
                precision_counts.get(policy.precision.value, 0) + n_tokens
            )

            if policy.precision == KVPrecision.FP16:
                critical_fp16_tokens += n_tokens

        bytes_saved = dense_fp16 - intent_bytes
        bytes_saved_pct = (
            (1.0 - intent_bytes / dense_fp16) * 100.0 if dense_fp16 > 0 else 0.0
        )

        total_tokens = layout.total_token_count()

        return {
            "total_tokens": total_tokens,
            "dense_fp16_bytes": dense_fp16,
            "intent_quant_bytes": int(round(intent_bytes)),
            "bytes_saved": int(round(bytes_saved)),
            "bytes_saved_pct": round(bytes_saved_pct, 2),
            "precision_distribution": precision_counts,
            "critical_full_precision_tokens": critical_fp16_tokens,
        }

    def summary_table(
        self,
        layout: BlockLayout,
        heads: int,
        head_dim: int,
    ) -> List[Dict[str, Any]]:
        policies = self.assign_layout_precision(layout)
        rows: List[Dict[str, Any]] = []
        for block in layout.blocks:
            policy = policies[block.name]
            n_tokens = block.end - block.start
            dense_bytes = 2 * n_tokens * head_dim * 2  # K+V fp16
            quant_bytes = int(round(2 * n_tokens * head_dim * policy.estimated_bytes_per_value))
            rows.append(
                {
                    "block": block.name,
                    "policy": block.policy.value,
                    "score": block.score,
                    "tokens": n_tokens,
                    "precision": policy.precision.value,
                    "reason": policy.reason,
                    "dense_fp16_bytes": dense_bytes,
                    "quant_bytes": quant_bytes,
                }
            )
        return rows


# --------------------------------------------------------------------------- #
#  Fake quantisation / dequantisation for CPU simulation only
# --------------------------------------------------------------------------- #


def _absmax_scale(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Symmetric per-channel (last-dim) absmax scale."""
    absmax = x.abs().amax(dim=tuple(range(x.ndim - 1)), keepdim=True)
    scale = absmax / 127.0
    scale = scale.clamp(min=1e-8)
    return scale, absmax


def fake_quantize_tensor(
    x: torch.Tensor,
    precision: KVPrecision,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Fake-quantize *x* to the requested precision using symmetric absmax scaling.

    Returns a reconstructed tensor (same shape, same dtype) and a metadata dict.
    This is a CPU simulation only — it does not produce real low-bit tensors.
    """
    if precision == KVPrecision.FP16:
        return x.half(), {"precision": "fp16", "scale": None}

    if precision == KVPrecision.FP8:
        scale, absmax = _absmax_scale(x)
        q = (x / scale).to(torch.float8_e4m3fn)
        reconstructed = q.float() * scale
        error = (x - reconstructed).float()
        return reconstructed, {
            "precision": "fp8",
            "scale": scale,
            "absmax": absmax,
            "mse": float(error.pow(2).mean()),
            "max_abs_error": float(error.abs().max()),
            "cosine_similarity": float(
                torch.nn.functional.cosine_similarity(
                    x.flatten().unsqueeze(0),
                    reconstructed.flatten().unsqueeze(0),
                ).item()
            ),
        }

    if precision in (KVPrecision.INT8, KVPrecision.INT4):
        n_bits = 8 if precision == KVPrecision.INT8 else 4
        scale, absmax = _absmax_scale(x)
        max_q = 2 ** (n_bits - 1) - 1
        q = (x / scale).clamp(-max_q, max_q).round().to(torch.int8)
        reconstructed = q.float() * scale
        error = (x - reconstructed).float()
        return reconstructed, {
            "precision": precision.value,
            "n_bits": n_bits,
            "scale": scale,
            "absmax": absmax,
            "mse": float(error.pow(2).mean()),
            "max_abs_error": float(error.abs().max()),
            "cosine_similarity": float(
                torch.nn.functional.cosine_similarity(
                    x.flatten().unsqueeze(0),
                    reconstructed.flatten().unsqueeze(0),
                ).item()
            ),
        }

    if precision == KVPrecision.INT4_RESIDUAL:
        scale, absmax = _absmax_scale(x)
        max_q = 7  # INT4 range
        q_base = (x / scale).clamp(-max_q, max_q).round().to(torch.int8)
        base_recon = q_base.float() * scale
        residual = x - base_recon
        res_scale, _ = _absmax_scale(residual)
        max_rq = 7
        rq = (residual / res_scale).clamp(-max_rq, max_rq).round().to(torch.int8)
        res_recon = rq.float() * res_scale
        reconstructed = base_recon + res_recon
        error = (x - reconstructed).float()
        return reconstructed, {
            "precision": "int4_residual",
            "scale": scale,
            "residual_scale": res_scale,
            "absmax": absmax,
            "mse": float(error.pow(2).mean()),
            "max_abs_error": float(error.abs().max()),
            "cosine_similarity": float(
                torch.nn.functional.cosine_similarity(
                    x.flatten().unsqueeze(0),
                    reconstructed.flatten().unsqueeze(0),
                ).item()
            ),
        }

    if precision == KVPrecision.SKIP:
        return torch.zeros_like(x), {"precision": "skip"}

    raise ValueError(f"Unknown precision: {precision}")


def fake_dequantize_tensor(
    qx: torch.Tensor,
    metadata: Dict[str, Any],
) -> torch.Tensor:
    """Reconstruct a fake-quantized tensor from its metadata (no-op for simulation)."""
    return qx


def compute_quant_error(
    original: torch.Tensor,
    reconstructed: torch.Tensor,
) -> Dict[str, float]:
    error = (original - reconstructed).float()
    mse = float(error.pow(2).mean())
    max_err = float(error.abs().max())
    cos_sim = float(
        torch.nn.functional.cosine_similarity(
            original.flatten().unsqueeze(0),
            reconstructed.flatten().unsqueeze(0),
        ).item()
    )
    return {
        "mse": mse,
        "max_abs_error": max_err,
        "cosine_similarity": cos_sim,
    }
