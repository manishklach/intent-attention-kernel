"""specattn.py — verification-guided sparse KV selection for self-speculative decoding."""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from .block_metadata import BlockLayout, BlockPolicy, SemanticBlock
from .reference import dense_attention


class SpecAttnController:
    def __init__(self, top_k_blocks: int = 8, k_draft: int = 4,
                 history_len: int = 16, ema_alpha: float = 0.1):
        self.top_k_blocks = top_k_blocks
        self.k_draft = k_draft
        self.ema_alpha = ema_alpha
        self._acceptance_history: deque = deque(maxlen=history_len)
        self._block_ema: Optional[torch.Tensor] = None

    def init_layout(self, layout: BlockLayout) -> BlockLayout:
        new_blocks = []
        for b in layout.blocks:
            if b.policy == BlockPolicy.SKIP:
                new_blocks.append(b)
            elif b.policy in (BlockPolicy.ALWAYS, BlockPolicy.RECENT, BlockPolicy.GLOBAL):
                new_blocks.append(b)
            else:
                new_blocks.append(SemanticBlock(b.name, b.start, b.end, BlockPolicy.ATTEND))
        return BlockLayout(new_blocks)

    def update_from_verification(self, attn_weights: torch.Tensor, layout: BlockLayout) -> BlockLayout:
        attn_mean = attn_weights.float().mean(dim=(0, 1, 2))
        block_scores = torch.zeros(len(layout.blocks))
        for i, b in enumerate(layout.blocks):
            end = min(b.end, attn_mean.shape[0])
            if b.start < end:
                block_scores[i] = attn_mean[b.start:b.end].sum().item()
        if self._block_ema is None or self._block_ema.shape[0] != len(layout.blocks):
            self._block_ema = block_scores.clone()
        else:
            self._block_ema = self.ema_alpha * block_scores + (1 - self.ema_alpha) * self._block_ema
        attend_eligible = [i for i, b in enumerate(layout.blocks) if b.policy == BlockPolicy.ATTEND]
        top_global = set()
        if attend_eligible:
            eligible_scores = self._block_ema[attend_eligible]
            k = min(self.top_k_blocks, len(attend_eligible))
            _, top_local = eligible_scores.topk(k)
            top_global = {attend_eligible[j.item()] for j in top_local}
        new_blocks = []
        for i, b in enumerate(layout.blocks):
            if b.policy in (BlockPolicy.ALWAYS, BlockPolicy.RECENT, BlockPolicy.GLOBAL, BlockPolicy.SKIP):
                new_blocks.append(b)
            else:
                score = self._block_ema[i].item()
                if i in top_global:
                    new_blocks.append(SemanticBlock(b.name, b.start, b.end, BlockPolicy.ATTEND, score=score))
                else:
                    new_blocks.append(SemanticBlock(b.name, b.start, b.end, BlockPolicy.SKIP))
        return BlockLayout(new_blocks)

    def speculative_accept(self, draft_tokens: List[int], verify_logits: torch.Tensor,
                           draft_probs: Optional[torch.Tensor] = None) -> List[int]:
        accepted = []
        verify_probs = F.softmax(verify_logits.float(), dim=-1)
        for i, tok in enumerate(draft_tokens):
            v_prob = verify_probs[i, tok].item()
            if draft_probs is None:
                if verify_probs[i].argmax().item() == tok:
                    accepted.append(tok)
                else:
                    accepted.append(verify_probs[i].argmax().item())
                    break
            else:
                d_prob = draft_probs[i, tok].item()
                accept_prob = min(1.0, v_prob / max(d_prob, 1e-10))
                if torch.rand(1).item() < accept_prob:
                    accepted.append(tok)
                else:
                    corrected = (verify_probs[i] - draft_probs[i]).clamp(min=0)
                    corrected = corrected / corrected.sum().clamp(min=1e-10)
                    accepted.append(torch.multinomial(corrected, 1).item())
                    break
        rate = len(accepted) / max(len(draft_tokens), 1)
        self._acceptance_history.append(rate)
        return accepted

    def mean_acceptance_rate(self) -> float:
        return sum(self._acceptance_history) / max(len(self._acceptance_history), 1) if self._acceptance_history else 0.0

    def block_importance_scores(self) -> Optional[torch.Tensor]:
        return self._block_ema

    def stats(self) -> Dict:
        return {"mean_acceptance_rate": self.mean_acceptance_rate(),
                "top_k_blocks": self.top_k_blocks, "k_draft": self.k_draft,
                "n_acceptance_samples": len(self._acceptance_history)}
