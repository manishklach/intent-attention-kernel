"""rope.py — Rotary Position Embedding utilities."""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch


def precompute_rope_freqs(
    seq_len: int, d_head: int, base: float = 10000.0,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert d_head % 2 == 0, "d_head must be even for RoPE"
    half = d_head // 2
    inv_freq = 1.0 / (base ** (torch.arange(0, half, dtype=torch.float32, device=device) / half))
    t = torch.arange(seq_len, dtype=torch.float32, device=device)
    freqs = torch.outer(t, inv_freq)
    return freqs.cos(), freqs.sin()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
    position_ids: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    half = cos.shape[-1]
    if cos.shape[-1] == x.shape[-1] // 2:
        cos = torch.cat([cos, cos], dim=-1)
        sin = torch.cat([sin, sin], dim=-1)
    if position_ids is None:
        position_ids = torch.arange(x.shape[-2], device=x.device)
    cos, sin = cos[position_ids], sin[position_ids]
    while cos.dim() < x.dim():
        cos, sin = cos.unsqueeze(0), sin.unsqueeze(0)
    return x * cos + rotate_half(x) * sin
