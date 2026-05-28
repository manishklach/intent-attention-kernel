import pytest
import torch

from intent_attention.block_metadata import BlockLayout, BlockPolicy, SemanticBlock
from intent_attention.block_router import (
    BlockDecision,
    BlockRouter,
    KVPrecision,
    RouterConfig,
    RoutedBlock,
    compute_block_scores,
    routing_to_kernel_metadata,
)
from intent_attention.intent_quant import IntentQuantizer


def _basic_layout() -> BlockLayout:
    return BlockLayout([
        SemanticBlock("system", 0, 64, BlockPolicy.ALWAYS),
        SemanticBlock("doc_high", 64, 192, BlockPolicy.ATTEND, score=0.9),
        SemanticBlock("doc_low", 192, 320, BlockPolicy.ATTEND, score=0.1),
        SemanticBlock("recent", 320, 384, BlockPolicy.RECENT),
        SemanticBlock("unused", 384, 448, BlockPolicy.SKIP),
    ])


def _make_router(**kw) -> BlockRouter:
    return BlockRouter(RouterConfig(**kw))


def test_always_block_always_selected():
    router = _make_router()
    layout = BlockLayout([SemanticBlock("sys", 0, 64, BlockPolicy.ALWAYS)])
    routed = router.route_layout(layout, total_tokens=64)
    assert routed[0].decision == BlockDecision.PIN_HIGH_PRECISION


def test_global_block_always_selected():
    router = _make_router()
    layout = BlockLayout([SemanticBlock("global_mem", 0, 64, BlockPolicy.GLOBAL)])
    routed = router.route_layout(layout, total_tokens=64)
    assert routed[0].decision == BlockDecision.PIN_HIGH_PRECISION


def test_recent_block_always_selected():
    router = _make_router()
    layout = BlockLayout([SemanticBlock("recent", 0, 64, BlockPolicy.RECENT)])
    routed = router.route_layout(layout, total_tokens=64)
    assert routed[0].decision == BlockDecision.SELECT


def test_low_score_attend_may_be_skipped():
    router = _make_router(score_threshold=0.5, top_k_blocks=1)
    layout = BlockLayout([
        SemanticBlock("good", 0, 32, BlockPolicy.ATTEND, score=0.9),
        SemanticBlock("bad", 32, 64, BlockPolicy.ATTEND, score=0.1),
    ])
    routed = router.route_layout(layout, total_tokens=64)
    decisions = {b.name: b.decision for b in routed}
    assert decisions["good"] != BlockDecision.SKIP
    assert decisions["bad"] == BlockDecision.SKIP


def test_high_score_attend_selected():
    router = _make_router(score_threshold=0.5)
    layout = BlockLayout([SemanticBlock("important", 0, 64, BlockPolicy.ATTEND, score=0.9)])
    routed = router.route_layout(layout, total_tokens=64)
    assert routed[0].decision in (BlockDecision.SELECT, BlockDecision.QUANTIZE)


def test_top_k_limits_ordinary_attend():
    router = _make_router(top_k_blocks=2)
    layout = BlockLayout([
        SemanticBlock("a", 0, 16, BlockPolicy.ATTEND, score=0.9),
        SemanticBlock("b", 16, 32, BlockPolicy.ATTEND, score=0.8),
        SemanticBlock("c", 32, 48, BlockPolicy.ATTEND, score=0.7),
        SemanticBlock("d", 48, 64, BlockPolicy.ATTEND, score=0.6),
    ])
    routed = router.route_layout(layout, total_tokens=64)
    selected = router.selected_blocks(routed)
    assert len(selected) <= 2


def test_selected_page_ids_preserve_logical_order():
    router = _make_router()
    layout = BlockLayout([
        SemanticBlock("a", 0, 48, BlockPolicy.ALWAYS),
        SemanticBlock("b", 48, 96, BlockPolicy.ATTEND, score=0.9),
    ])
    routed = router.route_layout(layout, total_tokens=96)
    pids = router.selected_page_ids(routed, page_size=16)
    assert pids == sorted(pids)
    assert len(pids) == len(set(pids))


def test_duplicate_pages_removed_preserving_order():
    router = _make_router()
    layout = BlockLayout([
        SemanticBlock("a", 0, 32, BlockPolicy.ALWAYS),
        SemanticBlock("b", 32, 64, BlockPolicy.ATTEND, score=0.9),
    ])
    routed = router.route_layout(layout, total_tokens=64)
    pids = router.selected_page_ids(routed, page_size=16)
    assert len(pids) == len(set(pids))
    assert pids == sorted(pids)


def test_high_memory_pressure_lowers_non_critical_precision():
    router_low = _make_router(memory_pressure=0.1)
    router_high = _make_router(memory_pressure=0.9)
    layout = BlockLayout([
        SemanticBlock("always", 0, 32, BlockPolicy.ALWAYS),
        SemanticBlock("doc", 32, 64, BlockPolicy.ATTEND, score=0.5),
    ])
    routed_low = router_low.route_layout(layout, total_tokens=64)
    routed_high = router_high.route_layout(layout, total_tokens=64)

    def bpe_by_name(routed):
        from intent_attention.intent_quant import _BYTES_PER_VALUE
        return {b.name: _BYTES_PER_VALUE.get(b.precision, 2.0) for b in routed}

    bpe_low = bpe_by_name(routed_low)
    bpe_high = bpe_by_name(routed_high)
    # High memory pressure should not increase BPE for non-critical blocks
    assert bpe_high["doc"] <= bpe_low["doc"]


def test_routing_to_kernel_metadata():
    router = _make_router()
    layout = _basic_layout()
    routed = router.route_layout(layout, total_tokens=448)
    meta = routing_to_kernel_metadata(routed, page_size=16)
    assert "selected_page_ids" in meta
    assert "prefetch_page_ids" in meta
    assert "block_precision_by_page" in meta
    assert "selected_block_names" in meta
    assert "skipped_block_names" in meta
    assert "reasons_by_block" in meta
    assert len(meta["selected_page_ids"]) > 0
    assert len(meta["selected_block_names"]) >= 3


def test_cpu_only_execution():
    router = _make_router()
    layout = _basic_layout()
    routed = router.route_layout(layout, total_tokens=448)
    assert len(routed) == 5
    assert all(isinstance(b, RoutedBlock) for b in routed)


def test_routing_summary():
    router = _make_router()
    layout = _basic_layout()
    routed = router.route_layout(layout, total_tokens=448)
    summary = router.routing_summary(routed)
    assert summary["selected_tokens"] > 0
    assert summary["skipped_tokens"] >= 0
    assert summary["total_blocks"] == 5
    assert summary["precision_distribution"]
    assert summary["bytes_saved_pct"] >= 0.0


def test_skip_block_with_high_score_selected():
    router = _make_router(score_threshold=0.5)
    layout = BlockLayout([
        SemanticBlock("marked_skip", 0, 32, BlockPolicy.SKIP, score=0.8),
    ])
    routed = router.route_layout(layout, total_tokens=32)
    assert routed[0].decision == BlockDecision.SELECT


def test_prefetch_page_ids_returns_candidates():
    router = _make_router(prefetch_top_k=2)
    layout = BlockLayout([
        SemanticBlock("a", 0, 32, BlockPolicy.ALWAYS),
        SemanticBlock("b", 32, 64, BlockPolicy.ATTEND, score=0.2),
    ])
    routed = router.route_layout(layout, total_tokens=64)
    prefetch = router.prefetch_page_ids(routed, page_size=16)
    assert isinstance(prefetch, list)
    assert len(prefetch) <= 2


def test_compute_block_scores():
    q = torch.randn(64)
    reps = {"a": torch.randn(64), "b": torch.randn(64)}
    scores = compute_block_scores(q, reps)
    assert "a" in scores
    assert "b" in scores
    assert 0.0 <= scores["a"] <= 1.0
    assert 0.0 <= scores["b"] <= 1.0


def test_selected_token_indices_in_order():
    router = _make_router()
    layout = BlockLayout([
        SemanticBlock("a", 0, 16, BlockPolicy.ALWAYS),
        SemanticBlock("b", 32, 48, BlockPolicy.ATTEND, score=0.9),
    ])
    routed = router.route_layout(layout, total_tokens=48)
    indices = router.selected_token_indices(routed)
    assert indices == sorted(indices)
    # Expect no gap for the skipped middle region
    assert 16 not in indices


def test_skipped_blocks_filters_correctly():
    router = _make_router(score_threshold=0.5, top_k_blocks=1)
    layout = BlockLayout([
        SemanticBlock("a", 0, 16, BlockPolicy.ATTEND, score=0.9),
        SemanticBlock("b", 16, 32, BlockPolicy.ATTEND, score=0.1),
    ])
    routed = router.route_layout(layout, total_tokens=32)
    skipped = router.skipped_blocks(routed)
    assert len(skipped) == 1
    assert skipped[0].name == "b"
