"""Integration tests for the full KV Block Router → kernel metadata → attention pipeline."""

import torch

from intent_attention import (
    BlockLayout,
    BlockPolicy,
    BlockRouter,
    RouterConfig,
    SemanticBlock,
    routing_to_kernel_metadata,
    semantic_block_attention,
)
from intent_attention.intent_quant import IntentQuantizer
from intent_attention.intent_quant_attention import (
    intent_quant_attention_reference,
)


def _layout() -> BlockLayout:
    return BlockLayout([
        SemanticBlock("system",     0,   64,   BlockPolicy.ALWAYS),
        SemanticBlock("doc_high",   64,  192,  BlockPolicy.ATTEND, score=0.85),
        SemanticBlock("doc_low",    192, 320,  BlockPolicy.ATTEND, score=0.15),
        SemanticBlock("recent",     320, 384,  BlockPolicy.RECENT),
        SemanticBlock("unused",     384, 448,  BlockPolicy.SKIP),
    ])


def test_router_to_kernel_metadata_chain():
    layout = _layout()
    total_tokens = 448
    router = BlockRouter(RouterConfig(score_threshold=0.3))
    routed = router.route_layout(layout, total_tokens)
    meta = routing_to_kernel_metadata(routed, page_size=16)
    assert len(meta["selected_page_ids"]) > 0
    assert isinstance(meta["selected_page_ids"], list)
    assert all(isinstance(pid, int) for pid in meta["selected_page_ids"])


def test_selected_page_ids_non_empty_for_always_recent():
    layout = _layout()
    router = BlockRouter(RouterConfig())
    routed = router.route_layout(layout, 448)
    pids = router.selected_page_ids(routed, page_size=16)
    assert len(pids) > 0


def test_precision_metadata_exists_for_selected_pages():
    layout = _layout()
    router = BlockRouter(RouterConfig())
    routed = router.route_layout(layout, 448)
    meta = routing_to_kernel_metadata(routed, page_size=16)
    assert len(meta["block_precision_by_page"]) > 0
    for k, v in meta["block_precision_by_page"].items():
        assert isinstance(k, str)
        assert isinstance(v, str)


def test_selected_block_attention_from_routed_metadata():
    B, H, Q, D = 1, 2, 4, 32
    total_tokens = 448
    q = torch.randn(B, H, Q, D)
    k = torch.randn(B, H, total_tokens, D)
    v = torch.randn(B, H, total_tokens, D)

    layout = _layout()
    router = BlockRouter(RouterConfig())
    routed = router.route_layout(layout, total_tokens)
    meta = routing_to_kernel_metadata(routed, page_size=16)

    # Use selected_block_names to verify semantic_block_attention output
    out, debug = semantic_block_attention(q, k, v, layout, return_debug=True)
    assert tuple(out.shape) == (B, H, Q, D)
    assert debug["selected_token_count"] > 0
    # Router-selected blocks should be a subset of layout-selected blocks
    layout_selected = {b.name for b in layout.selected_blocks()}
    router_selected = set(meta["selected_block_names"])
    assert router_selected.issubset(layout_selected)
    assert len(router_selected) > 0


def test_intent_quant_attention_from_routed_precision():
    B, H, Q, D = 1, 2, 4, 32
    total_tokens = 448
    q = torch.randn(B, H, Q, D)
    k = torch.randn(B, H, total_tokens, D)
    v = torch.randn(B, H, total_tokens, D)

    layout = _layout()
    config = RouterConfig(memory_pressure=0.5)
    router = BlockRouter(config)
    routed = router.route_layout(layout, total_tokens)

    quantizer = IntentQuantizer(memory_pressure=config.memory_pressure)
    out, debug = intent_quant_attention_reference(
        q, k, v, layout, quantizer, return_debug=True
    )
    assert tuple(out.shape) == (B, H, Q, D)
    assert "precision_by_block" in debug

    meta = routing_to_kernel_metadata(routed, page_size=16)
    for block_name, precision in debug["precision_by_block"].items():
        if block_name in meta["selected_block_names"]:
            assert precision.lower() in ("fp16", "fp8", "int8", "int4", "int4_residual")


def test_cpu_only_execution():
    layout = _layout()
    router = BlockRouter(RouterConfig())
    routed = router.route_layout(layout, 448)
    meta = routing_to_kernel_metadata(routed, page_size=16)
    assert len(meta["selected_page_ids"]) > 0
    assert len(meta["selected_block_names"]) >= 3
