import pytest
import torch
from intent_attention.block_metadata import SemanticBlock, BlockPolicy, BlockLayout
from intent_attention.reference import dense_attention, semantic_block_attention


def test_semantic_equals_dense_over_selected():
    q = torch.randn(2, 4, 16, 64)
    k = torch.randn(2, 4, 128, 64)
    v = torch.randn(2, 4, 128, 64)
    layout = BlockLayout(
        [
            SemanticBlock("a", 0, 64, BlockPolicy.ALWAYS),
            SemanticBlock("b", 64, 128, BlockPolicy.SKIP),
        ]
    )
    out, dbg = semantic_block_attention(q, k, v, layout, return_debug=True)

    selected_k = k[:, :, :64, :]
    selected_v = v[:, :, :64, :]
    expected = dense_attention(q, selected_k, selected_v)

    assert torch.allclose(out, expected)
    assert dbg["selected_token_count"] == 64
    assert out.shape == (2, 4, 16, 64)


def test_zero_selected_blocks_returns_zeros():
    q = torch.randn(1, 1, 8, 32)
    k = torch.randn(1, 1, 16, 32)
    v = torch.randn(1, 1, 16, 32)
    layout = BlockLayout(
        [
            SemanticBlock("a", 0, 16, BlockPolicy.SKIP),
        ]
    )
    out = semantic_block_attention(q, k, v, layout)
    assert torch.allclose(out, torch.zeros_like(q))


def test_zero_selected_blocks_returns_debug():
    q = torch.randn(1, 1, 8, 32)
    k = torch.randn(1, 1, 16, 32)
    v = torch.randn(1, 1, 16, 32)
    layout = BlockLayout(
        [
            SemanticBlock("a", 0, 16, BlockPolicy.SKIP),
        ]
    )
    out, dbg = semantic_block_attention(q, k, v, layout, return_debug=True)
    assert torch.allclose(out, torch.zeros_like(q))
    assert dbg["selected_token_count"] == 0
    assert dbg["selected_block_names"] == []


def test_all_policies_integration():
    q = torch.randn(1, 2, 8, 32)
    k = torch.randn(1, 2, 128, 32)
    v = torch.randn(1, 2, 128, 32)
    layout = BlockLayout(
        [
            SemanticBlock("always", 0, 32, BlockPolicy.ALWAYS),
            SemanticBlock("attend", 32, 64, BlockPolicy.ATTEND, score=0.9),
            SemanticBlock("skip", 64, 96, BlockPolicy.SKIP),
            SemanticBlock("recent", 96, 112, BlockPolicy.RECENT),
            SemanticBlock("global", 112, 128, BlockPolicy.GLOBAL),
        ]
    )
    out, dbg = semantic_block_attention(q, k, v, layout, return_debug=True)
    assert out.shape == (1, 2, 8, 32)
    assert dbg["selected_token_count"] == 96
    assert dbg["total_kv_tokens"] == 128


def test_causal_attention_shape():
    q = torch.randn(1, 2, 8, 32)
    k = torch.randn(1, 2, 16, 32)
    v = torch.randn(1, 2, 16, 32)
    out = dense_attention(q, k, v, causal=True)
    assert out.shape == (1, 2, 8, 32)
    assert torch.isfinite(out).all()
    assert not torch.isnan(out).any()


def test_causal_does_not_look_ahead():
    q = torch.randn(1, 1, 4, 8)
    k = torch.eye(4, 8).unsqueeze(0).unsqueeze(0)
    v = torch.ones(1, 1, 4, 8)
    out = dense_attention(q, k, v, causal=True)
    with torch.no_grad():
        manual = torch.matmul(
            torch.softmax(
                torch.matmul(q, k.transpose(-2, -1)) / (8**0.5)
                + torch.triu(torch.full((4, 4), float("-inf")), diagonal=1),
                dim=-1,
            ),
            v,
        )
    assert torch.allclose(out, manual, atol=1e-5)


def test_semantic_causal_matches_dense():
    q = torch.randn(1, 2, 8, 32)
    k = torch.randn(1, 2, 16, 32)
    v = torch.randn(1, 2, 16, 32)
    layout = BlockLayout([SemanticBlock("a", 0, 16, BlockPolicy.ALWAYS)])
    out_sel, _ = semantic_block_attention(q, k, v, layout, causal=True, return_debug=True)
    out_dense = dense_attention(q, k, v, causal=True)
    assert torch.allclose(out_sel, out_dense, atol=1e-5)
