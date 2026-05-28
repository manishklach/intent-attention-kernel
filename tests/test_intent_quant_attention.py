import pytest
import torch

from intent_attention.block_metadata import BlockLayout, BlockPolicy, SemanticBlock
from intent_attention.intent_quant import IntentQuantizer
from intent_attention.intent_quant_attention import (
    compare_intent_quant_to_fp16_selected,
    intent_quant_attention_reference,
)


def _basic_layout() -> BlockLayout:
    return BlockLayout([
        SemanticBlock("always", 0, 16, BlockPolicy.ALWAYS),
        SemanticBlock("attend_high", 16, 32, BlockPolicy.ATTEND, score=0.9),
        SemanticBlock("attend_low", 32, 48, BlockPolicy.ATTEND, score=0.3),
        SemanticBlock("skip", 48, 64, BlockPolicy.SKIP),
    ])


def _qkv(batch=1, heads=2, q_len=8, kv_len=64, d_head=32):
    return (
        torch.randn(batch, heads, q_len, d_head),
        torch.randn(batch, heads, kv_len, d_head),
        torch.randn(batch, heads, kv_len, d_head),
    )


def test_output_shape_matches_selected_block():
    q, k, v = _qkv()
    layout = _basic_layout()
    quantizer = IntentQuantizer(memory_pressure=0.3)
    out = intent_quant_attention_reference(q, k, v, layout, quantizer)
    assert out.shape == q.shape


def test_debug_contains_precision_and_bytes():
    q, k, v = _qkv()
    layout = _basic_layout()
    quantizer = IntentQuantizer(memory_pressure=0.3)
    out, debug = intent_quant_attention_reference(
        q, k, v, layout, quantizer, return_debug=True
    )
    assert "precision_by_block" in debug
    assert "bytes_saved_pct" in debug
    assert "reconstruction_mse_k" in debug
    assert "reconstruction_mse_v" in debug
    assert "output_mse_vs_fp16_selected" in debug
    assert len(debug["selected_block_names"]) == 3
    assert debug["selected_tokens"] == 48


def test_causal_raises_not_implemented():
    q, k, v = _qkv()
    layout = _basic_layout()
    quantizer = IntentQuantizer()
    with pytest.raises(NotImplementedError, match="query_positions"):
        intent_quant_attention_reference(q, k, v, layout, quantizer, causal=True)


def test_output_error_metrics_are_finite():
    q, k, v = _qkv()
    layout = _basic_layout()
    quantizer = IntentQuantizer(memory_pressure=0.5)
    out, debug = intent_quant_attention_reference(
        q, k, v, layout, quantizer, return_debug=True
    )
    assert torch.isfinite(out).all()
    assert debug["output_mse_vs_fp16_selected"] >= 0.0
    assert -1.0 <= debug["output_cosine_vs_fp16_selected"] <= 1.0


def test_bytes_saved_pct_non_negative():
    q, k, v = _qkv()
    layout = _basic_layout()
    quantizer = IntentQuantizer(memory_pressure=0.8)
    out, debug = intent_quant_attention_reference(
        q, k, v, layout, quantizer, return_debug=True
    )
    assert debug["bytes_saved_pct"] >= 0.0


def test_always_global_higher_precision_than_low_score():
    layout = BlockLayout([
        SemanticBlock("always", 0, 16, BlockPolicy.ALWAYS),
        SemanticBlock("low_score", 16, 32, BlockPolicy.ATTEND, score=0.1),
        SemanticBlock("skip_me", 32, 48, BlockPolicy.SKIP),
    ])
    q, k, v = _qkv(kv_len=48)
    quantizer = IntentQuantizer(memory_pressure=0.7)
    out, debug = intent_quant_attention_reference(
        q, k, v, layout, quantizer, return_debug=True
    )
    always_prec = debug["precision_by_block"]["always"]
    low_prec = debug["precision_by_block"]["low_score"]
    int_order = {"fp16": 0, "fp8": 1, "int8": 2, "int4_residual": 3, "int4": 4, "skip": 5}
    assert int_order.get(always_prec, 99) <= int_order.get(low_prec, 99)


def test_zero_selected_blocks_returns_zeros():
    q, k, v = _qkv(kv_len=16)
    layout = BlockLayout([SemanticBlock("skip", 0, 16, BlockPolicy.SKIP)])
    quantizer = IntentQuantizer()
    out = intent_quant_attention_reference(q, k, v, layout, quantizer)
    assert torch.allclose(out, torch.zeros_like(q))


def test_cpu_only_execution():
    q, k, v = _qkv()
    layout = _basic_layout()
    quantizer = IntentQuantizer()
    out = intent_quant_attention_reference(q, k, v, layout, quantizer)
    assert out.device.type == "cpu"


def test_compare_function_returns_expected_keys():
    q, k, v = _qkv()
    layout = _basic_layout()
    quantizer = IntentQuantizer(memory_pressure=0.4)
    result = compare_intent_quant_to_fp16_selected(q, k, v, layout, quantizer)
    assert "output_mse" in result
    assert "output_cosine_similarity" in result
    assert "bytes_saved_pct" in result
    assert "precision_by_block" in result
    assert result["bytes_saved_pct"] >= 0.0
    assert result["fp16_selected_tokens"] == result["quant_selected_tokens"]
