"""Tests for rope.py."""
from __future__ import annotations

import torch

from intent_attention.rope import precompute_rope_freqs, apply_rope, rotate_half


def test_precompute_rope_freqs_shape():
    cos, sin = precompute_rope_freqs(128, 64)
    assert cos.shape == (128, 32)
    assert sin.shape == (128, 32)


def test_precompute_rope_freqs_dtype():
    cos, sin = precompute_rope_freqs(32, 64)
    assert cos.dtype == torch.float32
    assert sin.dtype == torch.float32


def test_rotate_half():
    x = torch.randn(4, 8)
    rotated = rotate_half(x)
    assert rotated.shape == (4, 8)
    half = 4
    assert torch.allclose(rotated[..., :half], -x[..., half:])
    assert torch.allclose(rotated[..., half:], x[..., :half])


def test_apply_rope_shape():
    cos, sin = precompute_rope_freqs(64, 64)
    x = torch.randn(2, 4, 16, 64)
    out = apply_rope(x, cos, sin)
    assert out.shape == x.shape


def test_apply_rope_with_position_ids():
    cos, sin = precompute_rope_freqs(64, 64)
    x = torch.randn(2, 4, 16, 64)
    position_ids = torch.randint(0, 64, (16,))
    out = apply_rope(x, cos, sin, position_ids=position_ids)
    assert out.shape == x.shape


def test_apply_rope_norm_preserving():
    cos, sin = precompute_rope_freqs(128, 64)
    x = torch.randn(2, 4, 16, 64)
    out = apply_rope(x, cos[:16], sin[:16])
    in_norm = x.norm(dim=-1)
    out_norm = out.norm(dim=-1)
    assert torch.allclose(in_norm, out_norm, atol=1e-5)


def test_apply_rope_half_dim_handling():
    cos, sin = precompute_rope_freqs(64, 64)
    half = 32
    cos_half, sin_half = cos[..., :half], sin[..., :half]
    x = torch.randn(2, 4, 16, 64)
    out = apply_rope(x, cos_half, sin_half)
    assert out.shape == x.shape
