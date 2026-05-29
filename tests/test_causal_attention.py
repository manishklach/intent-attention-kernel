"""Tests for causal selected-block attention."""
from __future__ import annotations

import torch

from intent_attention.reference import dense_attention, semantic_block_attention
from intent_attention.block_metadata import BlockLayout, SemanticBlock, BlockPolicy


def test_causal_dense_vs_selected_identical_layout():
    """When layout selects all tokens, causal selected-block matches dense causal."""
    q = torch.randn(2, 4, 8, 64)
    k = torch.randn(2, 4, 32, 64)
    v = torch.randn(2, 4, 32, 64)
    layout = BlockLayout([SemanticBlock("all", 0, 32, BlockPolicy.ALWAYS)])
    dense_out = dense_attention(q, k, v, causal=True)
    sel_out, _ = semantic_block_attention(q, k, v, layout, causal=True, return_debug=True)
    assert torch.allclose(dense_out, sel_out, atol=1e-5)


def test_causal_prevents_future_attention():
    """Query at position 5 can attend to KV at 0-5 but not 6+."""
    q = torch.randn(1, 1, 6, 8)
    k = torch.randn(1, 1, 12, 8)
    v = torch.randn(1, 1, 12, 8)
    layout = BlockLayout([SemanticBlock("b", 0, 10, BlockPolicy.ALWAYS)])
    out, _ = semantic_block_attention(q, k, v, layout, causal=True, return_debug=True)
    idx = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    k_sel = k.index_select(-2, idx)
    v_sel = v.index_select(-2, idx)
    out_dense = dense_attention(q, k_sel, v_sel, causal=True, original_kv_positions=idx)
    assert torch.allclose(out, out_dense, atol=1e-5)


def test_causal_all_queries_have_visible_kv():
    """Every query position has at least some visible KV (no degenerate NaN)."""
    q = torch.randn(1, 1, 4, 8)
    k = torch.randn(1, 1, 8, 8)
    v = torch.randn(1, 1, 8, 8)
    layout = BlockLayout([SemanticBlock("early", 0, 3, BlockPolicy.ALWAYS),
                          SemanticBlock("late", 3, 8, BlockPolicy.ALWAYS)])
    out, _ = semantic_block_attention(q, k, v, layout, causal=True, return_debug=True)
    assert not torch.isnan(out).any(), "Causal attention produced NaN"


def test_causal_with_skip_middle():
    """Causal selected-block with discontinuous selected positions."""
    q = torch.randn(1, 1, 6, 8)
    k = torch.randn(1, 1, 16, 8)
    v = torch.randn(1, 1, 16, 8)
    layout = BlockLayout([
        SemanticBlock("first", 0, 4, BlockPolicy.ALWAYS),
        SemanticBlock("skip", 4, 10, BlockPolicy.SKIP),
        SemanticBlock("last", 10, 16, BlockPolicy.ALWAYS),
    ])
    out, _ = semantic_block_attention(q, k, v, layout, causal=True, return_debug=True)
    idx = torch.tensor([0, 1, 2, 3, 10, 11, 12, 13, 14, 15])
    k_sel = k.index_select(-2, idx)
    v_sel = v.index_select(-2, idx)
    out_dense = dense_attention(q, k_sel, v_sel, causal=True, original_kv_positions=idx)
    assert torch.allclose(out, out_dense, atol=1e-5)


def test_causal_decode_single_token():
    """Decode step: single query at position 8, only KV positions <= 8 visible."""
    q = torch.randn(1, 1, 1, 8)
    k = torch.randn(1, 1, 16, 8)
    v = torch.randn(1, 1, 16, 8)
    block = SemanticBlock("b", 0, 16, BlockPolicy.ALWAYS)
    layout = BlockLayout([block])
    out, _ = semantic_block_attention(q, k, v, layout, causal=True, return_debug=True)
    out_dense = dense_attention(q, k, v, causal=True)
    assert torch.allclose(out, out_dense, atol=1e-5)


def test_causal_all_blocks_selected():
    """Multiple blocks selecting all KV tokens gives same result as dense causal."""
    q = torch.randn(2, 4, 16, 64)
    k = torch.randn(2, 4, 64, 64)
    v = torch.randn(2, 4, 64, 64)
    layout = BlockLayout([
        SemanticBlock("a", 0, 32, BlockPolicy.ALWAYS),
        SemanticBlock("b", 32, 64, BlockPolicy.ATTEND, score=0.9),
    ])
    out_dense = dense_attention(q, k, v, causal=True)
    out_sel, _ = semantic_block_attention(q, k, v, layout, causal=True, return_debug=True, threshold=0.5)
    assert torch.allclose(out_dense, out_sel, atol=1e-5)


def test_causal_non_contiguous_masking():
    """Query at position 5 masked from KV at position 8 when tokens 4-6 are skipped."""
    q = torch.randn(1, 1, 6, 8)
    k = torch.randn(1, 1, 12, 8)
    v = torch.randn(1, 1, 12, 8)
    layout = BlockLayout([
        SemanticBlock("early", 0, 3, BlockPolicy.ALWAYS),
        SemanticBlock("late", 8, 12, BlockPolicy.ALWAYS),
    ])
    out, _ = semantic_block_attention(q, k, v, layout, causal=True, return_debug=True)
    idx = torch.tensor([0, 1, 2, 8, 9, 10, 11])
    k_sel = k.index_select(-2, idx)
    v_sel = v.index_select(-2, idx)
    out_dense = dense_attention(q, k_sel, v_sel, causal=True, original_kv_positions=idx)
    assert torch.allclose(out, out_dense, atol=1e-5)


def test_causal_debug_returns_selected_block_names():
    """Causal mode still returns debug metadata."""
    q = torch.randn(1, 1, 4, 8)
    k = torch.randn(1, 1, 16, 8)
    v = torch.randn(1, 1, 16, 8)
    layout = BlockLayout([SemanticBlock("test", 0, 16, BlockPolicy.ALWAYS)])
    out, debug = semantic_block_attention(q, k, v, layout, causal=True, return_debug=True)
    assert "selected_block_names" in debug
    assert "total_kv_tokens" in debug
