import torch
from intent_attention.triton_kernel import (
    is_cuda_available,
    is_triton_available,
    semantic_block_attention_triton,
)
from intent_attention.block_metadata import SemanticBlock, BlockPolicy, BlockLayout


def test_triton_not_available_on_cpu():
    assert isinstance(is_triton_available(), bool)
    assert isinstance(is_cuda_available(), bool)


def test_semantic_block_attention_triton_fallback():
    """On CPU-only machines, fallback to reference should succeed."""
    q = torch.randn(1, 1, 8, 32)
    k = torch.randn(1, 1, 16, 32)
    v = torch.randn(1, 1, 16, 32)
    layout = BlockLayout([
        SemanticBlock("a", 0, 16, BlockPolicy.ALWAYS),
    ])
    out = semantic_block_attention_triton(q, k, v, layout)
    assert out.shape == (1, 1, 8, 32)
    assert not torch.isnan(out).any()


def test_triton_fallback_on_cpu():
    layout = BlockLayout([
        SemanticBlock("a", 0, 10, BlockPolicy.ALWAYS),
        SemanticBlock("b", 10, 16, BlockPolicy.SKIP),
    ])
    q = torch.randn(1, 1, 4, 32)
    k = torch.randn(1, 1, 16, 32)
    v = torch.randn(1, 1, 16, 32)
    out1 = semantic_block_attention_triton(q, k, v, layout)

    from intent_attention.reference import semantic_block_attention
    out2 = semantic_block_attention(q, k, v, layout)
    assert torch.allclose(out1, out2)
