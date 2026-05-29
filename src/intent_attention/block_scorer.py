"""Block scoring — lightweight probe that scores ATTEND blocks."""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import torch

from .block_metadata import BlockLayout, BlockPolicy


class BlockScorer:
    """Lightweight probe that scores ATTEND blocks by cosine similarity.

    Expects pre-computed key block representations (mean-pooled per block).
    """

    def score_blocks(
        self,
        query: torch.Tensor,
        key_block_reps: Sequence[torch.Tensor],
        threshold: float = 0.5,
    ) -> List[float]:
        q_pooled = query.mean(dim=-2)
        q_global = q_pooled.mean(dim=(0, 1))
        q_norm = q_global / (q_global.norm(p=2) + 1e-10)

        scores: List[float] = []
        for rep in key_block_reps:
            r_norm = rep / (rep.norm(p=2) + 1e-10)
            sim = float(torch.dot(q_norm, r_norm).item())
            sim = max(0.0, sim)
            scores.append(sim)

        return scores


def score_blocks(
    q: torch.Tensor,
    k: torch.Tensor,
    block_starts: torch.Tensor,
    block_ends: torch.Tensor,
) -> List[float]:
    """Score each contiguous block via cosine similarity between
    mean-pooled query and mean-pooled block keys."""
    q_pooled = q.mean(dim=-2)
    q_global = q_pooled.mean(dim=(0, 1))
    q_norm = q_global / (q_global.norm(p=2) + 1e-10)

    scores: List[float] = []
    for s, e in zip(block_starts.tolist(), block_ends.tolist()):
        k_block = k[..., s:e, :].float().mean(dim=-2)
        k_block = k_block.mean(dim=(0, 1))
        k_norm = k_block / (k_block.norm(p=2) + 1e-10)
        sim = float(torch.dot(q_norm, k_norm).item())
        scores.append(max(0.0, sim))
    return scores


def score_layout(
    q: torch.Tensor,
    k: torch.Tensor,
    layout: BlockLayout,
) -> Dict[str, float]:
    """Score all ATTEND/ALWAYS blocks in a BlockLayout by name."""
    scores: Dict[str, float] = {}
    blocks_to_score = [b for b in layout.blocks if b.policy != BlockPolicy.SKIP]
    if not blocks_to_score:
        return scores

    bs = torch.tensor([b.start for b in blocks_to_score], dtype=torch.long)
    be = torch.tensor([b.end for b in blocks_to_score], dtype=torch.long)
    raw = score_blocks(q, k, bs, be)

    for block, s in zip(blocks_to_score, raw):
        scores[block.name] = s
    return scores
