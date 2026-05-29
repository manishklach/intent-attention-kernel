"""Tests for specattn.py."""
from __future__ import annotations

import torch

from intent_attention.specattn import SpecAttnController
from intent_attention.block_metadata import BlockLayout, BlockPolicy, SemanticBlock


def test_controller_init():
    ctrl = SpecAttnController(top_k_blocks=4, k_draft=2)
    assert ctrl.top_k_blocks == 4
    assert ctrl.k_draft == 2
    assert ctrl.mean_acceptance_rate() == 0.0


def test_init_layout_preserves_always():
    ctrl = SpecAttnController()
    blocks = [
        SemanticBlock("a", 0, 64, BlockPolicy.ALWAYS),
        SemanticBlock("b", 64, 128, BlockPolicy.SKIP),
        SemanticBlock("c", 128, 192, BlockPolicy.ATTEND),
        SemanticBlock("d", 192, 256, BlockPolicy.RECENT),
    ]
    layout = BlockLayout(blocks)
    new_layout = ctrl.init_layout(layout)
    assert len(new_layout.blocks) == len(blocks)
    assert new_layout.blocks[0].policy == BlockPolicy.ALWAYS
    assert new_layout.blocks[1].policy == BlockPolicy.SKIP
    assert new_layout.blocks[2].policy == BlockPolicy.ATTEND
    assert new_layout.blocks[3].policy == BlockPolicy.RECENT


def test_update_from_verification():
    ctrl = SpecAttnController(top_k_blocks=2, ema_alpha=1.0)
    blocks = [
        SemanticBlock("a", 0, 8, BlockPolicy.ATTEND),
        SemanticBlock("b", 8, 16, BlockPolicy.ATTEND),
        SemanticBlock("c", 16, 24, BlockPolicy.ATTEND),
    ]
    layout = BlockLayout(blocks)
    attn_weights = torch.randn(1, 1, 1, 24)
    attn_weights[..., 0:8] += 2.0
    new_layout = ctrl.update_from_verification(attn_weights, layout)
    attend_count = sum(1 for b in new_layout.blocks if b.policy == BlockPolicy.ATTEND)
    assert attend_count <= ctrl.top_k_blocks or attend_count == 0


def test_speculative_accept_all():
    ctrl = SpecAttnController()
    draft = [10, 20, 30]
    verify = torch.randn(len(draft), 100)
    verify[0, 10] = 100.0
    verify[1, 20] = 100.0
    verify[2, 30] = 100.0
    accepted = ctrl.speculative_accept(draft, verify)
    assert len(accepted) == len(draft)
    assert accepted == draft


def test_mean_acceptance_rate():
    ctrl = SpecAttnController()
    ctrl._acceptance_history.extend([0.5, 0.75, 1.0])
    assert abs(ctrl.mean_acceptance_rate() - 0.75) < 1e-6


def test_stats():
    ctrl = SpecAttnController(top_k_blocks=6, k_draft=3)
    stats = ctrl.stats()
    assert stats["top_k_blocks"] == 6
    assert stats["k_draft"] == 3
    assert "mean_acceptance_rate" in stats


def test_block_importance_scores_none():
    ctrl = SpecAttnController()
    assert ctrl.block_importance_scores() is None


def test_block_importance_scores_after_update():
    ctrl = SpecAttnController(top_k_blocks=2, ema_alpha=1.0)
    blocks = [SemanticBlock("a", 0, 8, BlockPolicy.ATTEND)]
    layout = BlockLayout(blocks)
    ctrl.update_from_verification(torch.randn(1, 1, 1, 8), layout)
    scores = ctrl.block_importance_scores()
    assert scores is not None
    assert scores.shape[0] == 1
