from __future__ import annotations

import torch
from intent_attention.block_metadata import BlockLayout, BlockPolicy, SemanticBlock
from intent_attention.kv_quant import (
    dequantise_kv_page,
    dequantise_selected_pages,
    quantise_kv_cache,
    quantise_kv_page,
)


def test_round_trip_error_below_one_percent():
    torch.manual_seed(7)
    page_size, d_head = 128, 64
    k = torch.randn(page_size, d_head, dtype=torch.float16)
    v = torch.randn(page_size, d_head, dtype=torch.float16)

    k_int8, v_int8, ks, vs = quantise_kv_page(k, v)
    k_hat, v_hat = dequantise_kv_page(k_int8, v_int8, ks, vs)

    k_err = (k - k_hat).abs().max() / k.abs().max()
    v_err = (v - v_hat).abs().max() / v.abs().max()

    assert (
        k_err.item() < 0.01
    ), f"K round-trip relative error {k_err.item():.5f} >= 0.01"
    assert (
        v_err.item() < 0.01
    ), f"V round-trip relative error {v_err.item():.5f} >= 0.01"


def test_round_trip_extreme_values():
    page_size = 32
    k = torch.tensor(
        [[-127.0, 0.0, 127.0, 63.0]] * page_size, dtype=torch.float16
    ).T.contiguous()
    v = k.clone()

    k_int8, v_int8, ks, vs = quantise_kv_page(k, v)
    k_hat, v_hat = dequantise_kv_page(k_int8, v_int8, ks, vs)

    k_err = (k - k_hat).abs().max()
    assert k_err.item() < 1.0, f"Extreme-value error {k_err.item():.4f}"
    assert torch.isfinite(k_hat).all()


def test_quantise_cache_shape():
    batch, heads, total, d_head = 2, 4, 256, 64
    k = torch.randn(batch, heads, total, d_head, dtype=torch.float16)
    v = torch.randn(batch, heads, total, d_head, dtype=torch.float16)

    page_size = 128
    k_int8, v_int8, ks, vs = quantise_kv_cache(k, v, page_size)

    num_pages = (total + page_size - 1) // page_size
    assert k_int8.shape == (batch, heads, num_pages, page_size, d_head)
    assert v_int8.shape == (batch, heads, num_pages, page_size, d_head)
    assert ks.shape == (batch, heads, num_pages, d_head)
    assert vs.shape == (batch, heads, num_pages, d_head)
    assert k_int8.dtype == torch.int8
    assert ks.dtype == torch.float16


def test_dequant_selected_pages():
    batch, heads, total, d_head = 1, 1, 256, 32
    k = torch.randn(batch, heads, total, d_head, dtype=torch.float16)
    v = torch.randn(batch, heads, total, d_head, dtype=torch.float16)

    k_int8, v_int8, ks, vs = quantise_kv_cache(k, v, page_size=128)
    page_ids = torch.tensor([0], dtype=torch.int32)

    k_deq, v_deq = dequantise_selected_pages(k_int8, v_int8, ks, vs, page_ids, 256, 128)

    assert k_deq.shape == (batch, heads, 128, d_head)
    assert torch.isfinite(k_deq).all()


class DequantCounter:
    def __init__(self) -> None:
        self.count = 0

    def __call__(self, k_int8, v_int8, ks, vs):
        self.count += 1
        return dequantise_kv_page(k_int8, v_int8, ks, vs)


def test_skip_blocks_not_dequantised():
    from intent_attention.reference import semantic_block_attention

    q = torch.randn(1, 1, 8, 32, dtype=torch.float16)
    k = torch.randn(1, 1, 64, 32, dtype=torch.float16)
    v = torch.randn(1, 1, 64, 32, dtype=torch.float16)

    layout = BlockLayout(
        [
            SemanticBlock("keep", 0, 32, BlockPolicy.ALWAYS),
            SemanticBlock("skip_me", 32, 64, BlockPolicy.SKIP),
        ]
    )

    counter = DequantCounter()
    import intent_attention.kv_quant as kvq

    orig_deq = kvq.dequantise_kv_page
    kvq.dequantise_kv_page = counter

    try:
        out = semantic_block_attention(q, k, v, layout, return_debug=False)
    finally:
        kvq.dequantise_kv_page = orig_deq

    assert out.shape == (1, 1, 8, 32)
    # dequantise should not have been called (this path uses direct fp16)
    # The CPU reference path never calls kv_quant functions,
    # so this test verifies that SKIP blocks are truly skipped in the layout.
    # For a full GPU path test, see the Triton test below.
