import pytest
from intent_attention.block_metadata import (
    BlockPolicy,
    BlockLayout,
    SemanticBlock,
    SELECTED_POLICIES,
)


def test_valid_layout():
    layout = BlockLayout([SemanticBlock("a", 0, 10, BlockPolicy.ALWAYS)])
    layout.validate(10)
    assert layout.selected_token_count() == 10


class TestValidation:
    def test_empty_name_fails(self):
        layout = BlockLayout([SemanticBlock("", 0, 10, BlockPolicy.ALWAYS)])
        with pytest.raises(ValueError, match="empty name"):
            layout.validate(10)

    def test_negative_start_fails(self):
        layout = BlockLayout([SemanticBlock("a", -1, 10, BlockPolicy.ALWAYS)])
        with pytest.raises(ValueError, match="start.*< 0"):
            layout.validate(10)

    def test_end_not_greater_than_start_fails(self):
        layout = BlockLayout([SemanticBlock("a", 5, 5, BlockPolicy.ALWAYS)])
        with pytest.raises(ValueError, match="end.*<= start"):
            layout.validate(10)

    def test_end_less_than_start_fails(self):
        layout = BlockLayout([SemanticBlock("a", 10, 5, BlockPolicy.ALWAYS)])
        with pytest.raises(ValueError, match="end.*<= start"):
            layout.validate(10)

    def test_end_exceeds_total_fails(self):
        layout = BlockLayout([SemanticBlock("a", 0, 11, BlockPolicy.ALWAYS)])
        with pytest.raises(ValueError, match="end.*> total_tokens"):
            layout.validate(10)

    def test_unsorted_blocks_fails(self):
        layout = BlockLayout(
            [
                SemanticBlock("b", 10, 20, BlockPolicy.SKIP),
                SemanticBlock("a", 0, 10, BlockPolicy.ALWAYS),
            ]
        )
        with pytest.raises(ValueError, match="not sorted"):
            layout.validate(20)

    def test_overlapping_blocks_fails(self):
        layout = BlockLayout(
            [
                SemanticBlock("a", 0, 10, BlockPolicy.ALWAYS),
                SemanticBlock("b", 5, 15, BlockPolicy.SKIP),
            ]
        )
        with pytest.raises(ValueError, match="overlap"):
            layout.validate(15)

    def test_attend_without_score_fails(self):
        layout = BlockLayout(
            [
                SemanticBlock("a", 0, 10, BlockPolicy.ATTEND, score=None),
            ]
        )
        with pytest.raises(ValueError, match="score is None"):
            layout.validate(10)

    def test_empty_blocks_validates_ok(self):
        layout = BlockLayout([])
        layout.validate(100)


class TestSelectedBlocks:
    def test_selected_blocks_filters_skip(self):
        layout = BlockLayout(
            [
                SemanticBlock("a", 0, 10, BlockPolicy.ALWAYS),
                SemanticBlock("b", 10, 20, BlockPolicy.SKIP),
                SemanticBlock("c", 20, 30, BlockPolicy.ATTEND, score=0.8),
            ]
        )
        names = [b.name for b in layout.selected_blocks()]
        assert names == ["a", "c"]

    def test_selected_token_indices(self):
        layout = BlockLayout(
            [
                SemanticBlock("a", 0, 3, BlockPolicy.ALWAYS),
                SemanticBlock("b", 3, 6, BlockPolicy.SKIP),
                SemanticBlock("c", 6, 9, BlockPolicy.ATTEND, score=0.5),
            ]
        )
        assert layout.selected_token_indices() == [0, 1, 2, 6, 7, 8]

    def test_selected_token_count(self):
        layout = BlockLayout(
            [
                SemanticBlock("a", 0, 100, BlockPolicy.ALWAYS),
                SemanticBlock("b", 100, 200, BlockPolicy.SKIP),
                SemanticBlock("c", 200, 300, BlockPolicy.ATTEND, score=0.5),
            ]
        )
        assert layout.selected_token_count() == 200

    def test_total_token_count(self):
        layout = BlockLayout(
            [
                SemanticBlock("a", 0, 100, BlockPolicy.ALWAYS),
                SemanticBlock("b", 100, 250, BlockPolicy.SKIP),
            ]
        )
        assert layout.total_token_count() == 250


class TestSummary:
    def test_summary_keys(self):
        layout = BlockLayout([SemanticBlock("a", 0, 10, BlockPolicy.ALWAYS)])
        s = layout.summary()
        assert s["total_blocks"] == 1
        assert s["selected_blocks"] == 1
        assert s["total_token_count"] == 10
        assert s["selected_token_count"] == 10


def test_all_policies_are_in_selected_or_skipped():
    for p in BlockPolicy:
        if p == BlockPolicy.SKIP:
            assert p not in SELECTED_POLICIES
        else:
            assert p in SELECTED_POLICIES
