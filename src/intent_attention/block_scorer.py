from __future__ import annotations

from typing import List

import torch


class BlockScorer:
    """Lightweight probe that scores ATTEND blocks by cosine similarity
    between a pooled query vector and each block's mean-pooled key."""

    def score_blocks(
        self,
        query: torch.Tensor,
        key_block_reps: List[torch.Tensor],
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
