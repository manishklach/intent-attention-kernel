from __future__ import annotations

import torch

from intent_attention.block_metadata import BlockLayout, BlockPolicy, SemanticBlock
from intent_attention.prefetch import BlockPrefetcher, reset_prefetcher
from intent_attention.triton_kernel import semantic_block_attention_triton

# ------------------------------------------------------------------ #
#  BlockPrefetcher unit tests
# ------------------------------------------------------------------ #


class TestBlockPrefetcher:

    def test_predict_returns_empty_with_no_history(self):
        p = BlockPrefetcher(history_size=4, min_frequency=3)
        assert p.predict_next([1, 2, 3]) == []

    def test_predict_returns_empty_with_fewer_entries_than_min_frequency(self):
        p = BlockPrefetcher(history_size=4, min_frequency=3)
        p.record([1])
        assert p.predict_next([1]) == []

    def test_frequency_prediction_returns_blocks_in_all_steps(self):
        p = BlockPrefetcher(history_size=4, min_frequency=3)
        p.record([10, 20])
        p.record([10, 20])
        p.record([10, 20])
        predicted = p.predict_next([10, 20])
        assert predicted == [10, 20]

    def test_blocks_in_less_than_min_frequency_are_excluded(self):
        p = BlockPrefetcher(history_size=4, min_frequency=3)
        p.record([1, 2])
        p.record([1, 3])
        p.record([1, 4])
        predicted = p.predict_next([1, 5])
        assert predicted == [1]

    def test_result_is_sorted(self):
        p = BlockPrefetcher(history_size=4, min_frequency=3)
        p.record([3, 1, 2])
        p.record([3, 1, 2])
        p.record([3, 1, 2])
        predicted = p.predict_next([3, 1, 2])
        assert predicted == [1, 2, 3]

    def test_single_block_always_selected(self):
        p = BlockPrefetcher(history_size=4, min_frequency=3)
        p.record([99])
        p.record([99])
        p.record([99])
        predicted = p.predict_next([99])
        assert predicted == [99]

    def test_empty_history_is_cleared_on_reset(self):
        p = BlockPrefetcher(history_size=4, min_frequency=3)
        p.record([1])
        p.record([1])
        p.record([1])
        p.reset()
        assert p.predict_next([1]) == []


# ------------------------------------------------------------------ #
#  Edge: all blocks skip  →  no pages selected  →  prefetch no-op
# ------------------------------------------------------------------ #


def test_all_skip_layout_prefetch_noop():
    layout = BlockLayout(
        [
            SemanticBlock("skip1", 0, 64, BlockPolicy.SKIP),
            SemanticBlock("skip2", 64, 128, BlockPolicy.SKIP),
        ]
    )
    kv = torch.randn(1, 1, 128, 32)
    q = torch.randn(1, 1, 1, 32)
    reset_prefetcher()
    out = semantic_block_attention_triton(q, kv, kv, layout, prefetch=True)
    assert out is not None
    assert out.shape == q.shape


# ------------------------------------------------------------------ #
#  Output invariance: enabling prefetch must not change attention
#  output (values are determined purely by selected KV, not by
#  what we prefetch for the next step).
# ------------------------------------------------------------------ #


def _fixed_layout() -> BlockLayout:
    return BlockLayout(
        [
            SemanticBlock("always", 0, 64, BlockPolicy.ALWAYS),
            SemanticBlock("attend", 64, 128, BlockPolicy.ATTEND, score=1.0),
        ]
    )


def test_prefetch_does_not_change_output():
    import pytest

    if not torch.cuda.is_available():
        pytest.skip("GPU required for full prefetch integration test")

    layout = _fixed_layout()
    q = torch.randn(1, 2, 8, 64, device="cuda")
    k = torch.randn(1, 2, 128, 64, device="cuda")
    v = torch.randn(1, 2, 128, 64, device="cuda")

    reset_prefetcher()
    out_no = semantic_block_attention_triton(q, k, v, layout, prefetch=False)

    reset_prefetcher()
    out_yes = semantic_block_attention_triton(q, k, v, layout, prefetch=True)

    assert torch.allclose(out_no, out_yes, atol=1e-5, rtol=1e-5)


def test_prefetch_does_not_change_output_cpu():
    layout = _fixed_layout()
    q = torch.randn(1, 2, 8, 64)
    k = torch.randn(1, 2, 128, 64)
    v = torch.randn(1, 2, 128, 64)

    reset_prefetcher()
    out_no = semantic_block_attention_triton(q, k, v, layout, prefetch=False)

    reset_prefetcher()
    out_yes = semantic_block_attention_triton(q, k, v, layout, prefetch=True)

    assert torch.allclose(out_no, out_yes, atol=1e-5, rtol=1e-5)


# ------------------------------------------------------------------ #
#  Prefetch returns debug info when return_debug=True
# ------------------------------------------------------------------ #


def test_prefetch_debug_contains_prefetched_page_ids():
    layout = BlockLayout(
        [
            SemanticBlock("a", 0, 128, BlockPolicy.ALWAYS),
        ]
    )
    q = torch.randn(1, 1, 4, 32)
    k = torch.randn(1, 1, 128, 32)
    v = torch.randn(1, 1, 128, 32)

    reset_prefetcher()
    _, debug = semantic_block_attention_triton(
        q,
        k,
        v,
        layout,
        return_debug=True,
        prefetch=True,
    )
    assert "prefetched_page_ids" in debug
