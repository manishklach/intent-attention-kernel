import torch
from intent_attention.block_metadata import SemanticBlock, BlockPolicy, BlockLayout
from intent_attention.reference import dense_attention, semantic_block_attention

def test_semantic_equals_dense_selected():
    q = torch.randn(2, 4, 16, 64)
    k = torch.randn(2, 4, 128, 64)
    v = torch.randn(2, 4, 128, 64)
    layout = BlockLayout([
        SemanticBlock("a", 0, 64, BlockPolicy.ALWAYS),
        SemanticBlock("b", 64, 128, BlockPolicy.SKIP)
    ])
    out, dbg = semantic_block_attention(q, k, v, layout, return_debug=True)
    selected_k = k[:, :, :64, :]
    selected_v = v[:, :, :64, :]
    expected = dense_attention(q, selected_k, selected_v)
    assert torch.allclose(out, expected)
    assert dbg["selected_token_count"] == 64
    assert out.shape == (2, 4, 16, 64)
