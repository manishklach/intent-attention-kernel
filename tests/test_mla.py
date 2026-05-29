"""Tests for mla.py."""
from __future__ import annotations

import torch

from intent_attention.mla import MLAConfig, MLABlockTable, mla_sparse_decode_reference


def test_mla_config_default():
    cfg = MLAConfig(d_model=4096, d_c=512, n_heads=32, d_head=128)
    assert cfg.d_model == 4096
    assert cfg.d_c == 512


def test_block_table_append_get():
    cfg = MLAConfig(d_model=4096, d_c=512, n_heads=32, d_head=128)
    table = MLABlockTable(cfg)
    table.append(0, torch.randn(64, 512))
    c = table.get_latent(0)
    assert c is not None
    assert c.shape == (64, 512)


def test_block_table_memory_bytes():
    cfg = MLAConfig(d_model=4096, d_c=512, n_heads=32, d_head=128)
    table = MLABlockTable(cfg)
    table.append(0, torch.randn(64, 512))
    assert table.memory_bytes() == 64 * 512 * 2


def test_block_table_missing_block():
    cfg = MLAConfig(d_model=4096, d_c=512, n_heads=32, d_head=128)
    table = MLABlockTable(cfg)
    assert table.get_latent(99) is None


def test_block_table_rope():
    cfg = MLAConfig(d_model=4096, d_c=512, n_heads=32, d_head=128, d_rope=64)
    table = MLABlockTable(cfg)
    table.append(0, torch.randn(64, 512), rope_page=torch.randn(64, 64))
    rope = table.get_rope(0)
    assert rope is not None
    assert rope.shape == (64, 64)


def test_absorb_weights():
    from intent_attention.mla import absorb_weights
    W_UQ = torch.randn(4096, 512)
    W_UK = torch.randn(4096, 512)
    W_UV = torch.randn(4096, 512)
    W_O = torch.randn(512, 4096)
    W_QK, W_VO = absorb_weights(W_UQ, W_UK, W_UV, W_O)
    assert W_QK.shape == (4096, 512)
    assert W_VO.shape == (4096, 4096)


def test_mla_sparse_decode_reference_returns_tuple():
    from intent_attention.block_metadata import BlockLayout, SemanticBlock, BlockPolicy
    cfg = MLAConfig(d_model=4096, d_c=512, n_heads=32, d_head=128)
    table = MLABlockTable(cfg)
    table.append(0, torch.randn(64, 512))
    W_QK = torch.randn(4096, 512)
    W_VO = torch.randn(512, 4096)
    q = torch.randn(2, 32, 8, 128)
    block = SemanticBlock("b0", 0, 64, BlockPolicy.ATTEND, score=0.9)
    layout = BlockLayout([block])
    out, debug = mla_sparse_decode_reference(q, table, W_QK, W_VO, layout, threshold=0.5, return_debug=True)
    assert out.shape == (2, 8, 4096)
    assert debug is not None
    assert "selected_latent_tokens" in debug
