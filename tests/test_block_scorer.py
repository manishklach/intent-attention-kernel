import torch
from intent_attention.block_scorer import BlockScorer
from intent_attention.block_metadata import SemanticBlock, BlockPolicy, BlockLayout
from intent_attention.reference import semantic_block_attention


def test_orthogonal_block_scores_near_zero():
    """Query with zero key block => cosine similarity is 0."""
    head_dim = 64
    q = torch.randn(1, 1, 16, head_dim)
    rep = torch.zeros(head_dim)

    scorer = BlockScorer()
    scores = scorer.score_blocks(q, [rep])
    assert scores[0] == 0.0, f"Expected zero score, got {scores[0]:.4f}"


def test_aligned_block_scores_near_one():
    """Query aligned with block mean key => score near 1."""
    head_dim = 64
    q = torch.randn(1, 1, 16, head_dim)

    rep = q.mean(dim=-2).mean(dim=(0, 1))

    scorer = BlockScorer()
    scores = scorer.score_blocks(q, [rep])
    assert scores[0] > 0.85, f"Expected near-one score, got {scores[0]:.4f}"


def test_threshold_filtering_via_scorer():
    """Scores above threshold should be >= threshold; those below should be <."""
    head_dim = 64
    q = torch.randn(1, 2, 8, head_dim)

    rep_aligned = q.mean(dim=-2).mean(dim=(0, 1))
    rep_ortho = torch.zeros(head_dim)

    scorer = BlockScorer()
    scores = scorer.score_blocks(q, [rep_aligned, rep_ortho], threshold=0.5)

    assert scores[0] >= 0.5, f"Aligned block below threshold: {scores[0]:.4f}"
    assert scores[1] < 0.5, f"Orthogonal block above threshold: {scores[1]:.4f}"


def test_dynamic_scoring_populates_debug():
    """ATTEND blocks with score=None get populated and reported in debug."""
    q = torch.randn(1, 1, 8, 32)
    k = torch.randn(1, 1, 64, 32)
    v = torch.randn(1, 1, 64, 32)

    layout = BlockLayout(
        [
            SemanticBlock("always", 0, 32, BlockPolicy.ALWAYS),
            SemanticBlock("dynamic_attend", 32, 64, BlockPolicy.ATTEND, score=None),
        ]
    )

    out, dbg = semantic_block_attention(q, k, v, layout, return_debug=True)

    assert out.shape == (1, 1, 8, 32)
    assert "dynamic_scores" in dbg
    assert dbg["dynamic_scores"]["dynamic_attend"] >= 0.0
    assert dbg["selected_block_names"] == ["always", "dynamic_attend"]


def test_static_score_still_works():
    """Blocks with explicit static score are unchanged."""
    q = torch.randn(1, 1, 8, 32)
    k = torch.randn(1, 1, 64, 32)
    v = torch.randn(1, 1, 64, 32)

    layout = BlockLayout(
        [
            SemanticBlock("a", 0, 32, BlockPolicy.ALWAYS),
            SemanticBlock("b", 32, 64, BlockPolicy.ATTEND, score=0.75),
        ]
    )

    out, dbg = semantic_block_attention(q, k, v, layout, return_debug=True)
    assert "dynamic_scores" not in dbg
    assert dbg["selected_block_names"] == ["a", "b"]
    assert out.shape == (1, 1, 8, 32)
