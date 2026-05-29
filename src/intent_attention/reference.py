from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple, Union

import torch

from .block_metadata import BlockLayout, BlockPolicy
from .triton_selected_block_attn import (
    is_triton_available as _tsb_triton_avail,
    is_cuda_available as _tsb_cuda_avail,
    triton_semantic_attention as _triton_selected_block_attn,
)


def dense_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    scale = 1.0 / math.sqrt(q.size(-1))
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale

    if causal:
        q_len = q.size(-2)
        kv_len = k.size(-2)
        causal_mask = torch.triu(
            torch.full(
                (q_len, kv_len), float("-inf"), device=q.device, dtype=scores.dtype
            ),
            diagonal=1,
        )
        scores = scores + causal_mask

    if mask is not None:
        scores = scores + mask

    attn_weights = torch.softmax(scores, dim=-1)
    return torch.matmul(attn_weights, v)


def _resolve_dynamic_scores(
    q: torch.Tensor,
    k: torch.Tensor,
    layout: BlockLayout,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Populate ATTEND blocks whose score is None via BlockScorer.
    Returns a dict mapping block name -> computed score."""
    from .block_scorer import BlockScorer

    targets = [
        b for b in layout.blocks if b.policy == BlockPolicy.ATTEND and b.score is None
    ]
    if not targets:
        return {}

    key_reps: List[torch.Tensor] = []
    for block in targets:
        rep = k[..., block.start : block.end, :].mean(dim=-2)
        rep = rep.mean(dim=(0, 1))
        key_reps.append(rep)

    scorer = BlockScorer()
    scores = scorer.score_blocks(q, key_reps, threshold)

    dynamic: Dict[str, float] = {}
    for block, score in zip(targets, scores):
        block.score = score
        dynamic[block.name] = score

    return dynamic


def semantic_block_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    layout: BlockLayout,
    causal: bool = False,
    return_debug: bool = False,
    threshold: float = 0.5,
) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, Any]]]:
    kv_tokens = k.size(-2)

    dynamic_scores = _resolve_dynamic_scores(q, k, layout, threshold)

    layout.validate(kv_tokens)

    selected_indices = layout.selected_token_indices()

    if not selected_indices:
        out = torch.zeros_like(q)
        if return_debug:
            debug: Dict[str, Any] = {
                "selected_token_count": 0,
                "selected_block_names": [],
                "total_kv_tokens": kv_tokens,
                "selected_kv_tokens": 0,
            }
            if dynamic_scores:
                debug["dynamic_scores"] = dynamic_scores
            return out, debug
        return out

    idx = torch.tensor(selected_indices, dtype=torch.long, device=k.device)
    selected_k = k.index_select(-2, idx)
    selected_v = v.index_select(-2, idx)

    if causal:
        raise NotImplementedError(
            "Causal selected-block attention requires explicit query_positions "
            "because selected KV indices are in original context coordinates. "
            "The current CPU reference supports non-causal selected-block "
            "attention only."
        )

    output = dense_attention(q, selected_k, selected_v, causal=False)

    if return_debug:
        debug = {
            "selected_token_count": layout.selected_token_count(),
            "selected_block_names": [b.name for b in layout.selected_blocks()],
            "total_kv_tokens": kv_tokens,
            "selected_kv_tokens": len(selected_indices),
        }
        if dynamic_scores:
            debug["dynamic_scores"] = dynamic_scores
        return output, debug

    return output


def selected_block_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_starts: torch.Tensor,
    block_ends: torch.Tensor,
    semantic_ids: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
    use_gpu: bool = True,
) -> torch.Tensor:
    """Selected block attention via block boundaries.

    Tries the Triton kernel if GPU available, falls back to CPU loop.
    """
    if use_gpu and _tsb_triton_avail() and _tsb_cuda_avail() and q.is_cuda:
        try:
            return _triton_selected_block_attn(q, k, v, block_starts, block_ends, scale=scale)
        except Exception:
            pass
    return _selected_block_attention_cpu(q, k, v, block_starts, block_ends, scale=scale)


def _selected_block_attention_cpu(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_starts: torch.Tensor,
    block_ends: torch.Tensor,
    scale: Optional[float] = None,
) -> torch.Tensor:
    if scale is None:
        scale = 1.0 / math.sqrt(q.size(-1))
    out = torch.zeros_like(q)
    for s, e in zip(block_starts.tolist(), block_ends.tolist()):
        k_b = k[..., s:e, :]
        v_b = v[..., s:e, :]
        scores = torch.matmul(q, k_b.transpose(-2, -1)) * scale
        attn_w = torch.softmax(scores, dim=-1)
        out += torch.matmul(attn_w, v_b)
    return out
