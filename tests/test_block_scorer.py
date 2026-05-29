"""Tests for block_scorer.py."""
from __future__ import annotations

import torch

from intent_attention.block_scorer import BlockScorer, score_blocks, score_layout
from intent_attention.block_metadata import BlockLayout, SemanticBlock, BlockPolicy


def test_block_scorer_class():
    scorer = BlockScorer()
    q = torch.randn(2, 4, 8, 64)
    key_reps = [torch.randn(64) for _ in range(3)]
    scores = scorer.score_blocks(q, key_reps, threshold=0.3)
    assert len(scores) == 3


def test_block_scorer_scores_in_range():
    scorer = BlockScorer()
    q = torch.randn(1, 1, 4, 64)
    key_reps = [torch.randn(64) for _ in range(5)]
    scores = scorer.score_blocks(q, key_reps)
    for s in scores:
        assert 0.0 <= s <= 1.0


def test_block_scorer_empty():
    scorer = BlockScorer()
    q = torch.randn(1, 1, 4, 64)
    scores = scorer.score_blocks(q, [])
    assert scores == []


def test_score_blocks_function():
    q = torch.randn(2, 4, 8, 64)
    k = torch.randn(2, 4, 64, 64)
    block_starts = torch.tensor([0, 32])
    block_ends = torch.tensor([32, 64])
    scores = score_blocks(q, k, block_starts, block_ends)
    assert len(scores) == 2


def test_score_blocks_scores_in_range():
    q = torch.randn(1, 1, 4, 64)
    k = torch.randn(1, 1, 64, 64)
    block_starts = torch.tensor([0, 16, 32, 48])
    block_ends = torch.tensor([16, 32, 48, 64])
    scores = score_blocks(q, k, block_starts, block_ends)
    for s in scores:
        assert 0.0 <= s <= 1.0


def test_score_blocks_empty():
    q = torch.randn(1, 1, 4, 64)
    k = torch.randn(1, 1, 64, 64)
    scores = score_blocks(q, k, torch.zeros(0, dtype=torch.long), torch.zeros(0, dtype=torch.long))
    assert scores == []


def test_score_layout():
    q = torch.randn(1, 1, 4, 64)
    k = torch.randn(1, 1, 64, 64)
    blocks = [
        SemanticBlock("a", 0, 16, BlockPolicy.ATTEND),
        SemanticBlock("b", 16, 32, BlockPolicy.ATTEND),
        SemanticBlock("c", 32, 64, BlockPolicy.ALWAYS),
    ]
    layout = BlockLayout(blocks)
    scores = score_layout(q, k, layout)
    assert len(scores) == 3
    assert scores["c"] >= 0.0
