from __future__ import annotations

from dataclasses import dataclass
from enum import auto
from typing import Dict, List, Optional, Tuple

import torch

from ._enum import StrEnum
from .block_metadata import BlockLayout, BlockPolicy, SemanticBlock
from .intent_quant import (
    KVPrecision,
    IntentQuantizer,
    _BYTES_PER_VALUE,
)


class BlockDecision(StrEnum):
    SELECT = auto()
    SKIP = auto()
    PREFETCH = auto()
    QUANTIZE = auto()
    PIN_HIGH_PRECISION = auto()


@dataclass
class RoutedBlock:
    name: str
    start: int
    end: int
    policy: BlockPolicy
    score: float | None
    decision: BlockDecision
    precision: KVPrecision
    reason: str


@dataclass
class RouterConfig:
    top_k_blocks: int = 8
    score_threshold: float = 0.35
    recent_window_tokens: int = 2048
    memory_pressure: float = 0.5
    prefetch_top_k: int = 4
    preserve_system_prompt: bool = True
    preserve_recent: bool = True


def compute_block_scores(
    query_vector: torch.Tensor,
    block_representations: Dict[str, torch.Tensor],
) -> Dict[str, float]:
    """
    Compute cosine similarity between a query vector and per-block
    representations.  Scores are normalised into [0, 1].

    Args:
        query_vector:
            1-D or 2-D tensor.
        block_representations:
            Mapping from block name to a 1-D representation tensor.

    Returns:
        Dict[str, float]:  block name mapped to a score in [0, 1].
    """
    q = query_vector.flatten().float()
    q_norm = q / (q.norm(p=2) + 1e-10)

    scores: Dict[str, float] = {}
    for name, rep in block_representations.items():
        r = rep.flatten().float()
        r_norm = r / (r.norm(p=2) + 1e-10)
        sim = float(torch.dot(q_norm, r_norm).item())
        sim = max(0.0, min(1.0, sim))
        scores[name] = sim

    return scores


class BlockRouter:
    """
    CPU-first policy layer that converts semantic context blocks into
    kernel-ready metadata: selected pages, skipped pages, precision choices,
    prefetch hints, and routing reasons.

    The router is the policy layer. The kernel is the execution layer.
    """

    def __init__(
        self,
        config: RouterConfig,
        quantizer: IntentQuantizer | None = None,
    ) -> None:
        self.config = config
        self.quantizer = quantizer or IntentQuantizer(
            memory_pressure=config.memory_pressure,
            preserve_recent=config.preserve_recent,
            preserve_global=config.preserve_system_prompt,
        )

    def route_layout(
        self,
        layout: BlockLayout,
        total_tokens: int,
        query_vector: Optional[torch.Tensor] = None,
        block_representations: Optional[Dict[str, torch.Tensor]] = None,
    ) -> List[RoutedBlock]:
        """
        Route every block in *layout* into one or more routing decisions.

        Returns a list of ``RoutedBlock`` records, one per block (in order).
        """
        layout.validate(total_tokens)
        cfg = self.config

        # ---- Optional query-to-block scoring -------------------------------
        query_scores: Dict[str, float] = {}
        if query_vector is not None and block_representations is not None:
            query_scores = compute_block_scores(query_vector, block_representations)

        # ---- Classify blocks -----------------------------------------------
        high_priority: List[SemanticBlock] = []
        attend_blocks: List[SemanticBlock] = []
        skip_blocks: List[SemanticBlock] = []

        for block in layout.blocks:
            if block.policy in (BlockPolicy.ALWAYS, BlockPolicy.GLOBAL, BlockPolicy.RECENT):
                high_priority.append(block)
            elif block.policy == BlockPolicy.ATTEND:
                attend_blocks.append(block)
            else:
                skip_blocks.append(block)

        # ---- Update missing ATTEND scores from query similarity ------------
        for i, block in enumerate(attend_blocks):
            if block.score is None and block.name in query_scores:
                attend_blocks[i] = SemanticBlock(
                    name=block.name,
                    start=block.start,
                    end=block.end,
                    policy=block.policy,
                    score=query_scores[block.name],
                )

        # ---- Score ranking for ATTEND blocks -------------------------------
        scored_attends = sorted(
            [b for b in attend_blocks if b.score is not None],
            key=lambda b: (b.score or 0.0),
            reverse=True,
        )
        top_k = scored_attends[: cfg.top_k_blocks]
        bottom = scored_attends[cfg.top_k_blocks:]

        # ---- Build result -------------------------------------------------
        routed: List[RoutedBlock] = []

        for block in high_priority:
            precision = self.quantizer.assign_block_precision(block).precision
            if block.policy in (BlockPolicy.ALWAYS, BlockPolicy.GLOBAL):
                routed.append(
                    RoutedBlock(
                        name=block.name,
                        start=block.start,
                        end=block.end,
                        policy=block.policy,
                        score=block.score,
                        decision=BlockDecision.PIN_HIGH_PRECISION,
                        precision=precision,
                        reason=f"{block.policy.value} block, always selected",
                    )
                )
            else:
                routed.append(
                    RoutedBlock(
                        name=block.name,
                        start=block.start,
                        end=block.end,
                        policy=block.policy,
                        score=block.score,
                        decision=BlockDecision.SELECT,
                        precision=precision,
                        reason="recent context, always selected",
                    )
                )

        for block in top_k:
            if block.score is None:
                continue
            precision = self.quantizer.assign_block_precision(block).precision
            if precision == KVPrecision.SKIP:
                routed.append(
                    RoutedBlock(
                        name=block.name,
                        start=block.start,
                        end=block.end,
                        policy=block.policy,
                        score=block.score,
                        decision=BlockDecision.SKIP,
                        precision=precision,
                        reason="low-score attend, skipped under memory pressure",
                    )
                )
                continue
            routed.append(
                RoutedBlock(
                    name=block.name,
                    start=block.start,
                    end=block.end,
                    policy=block.policy,
                    score=block.score,
                    decision=BlockDecision.QUANTIZE if precision != KVPrecision.FP16 else BlockDecision.SELECT,
                    precision=precision,
                    reason=f"attend score={block.score:.3f} >= threshold={cfg.score_threshold}",
                )
            )

        for block in bottom:
            precision = self.quantizer.assign_block_precision(block).precision
            skip_reason = "below top-k"
            if cfg.memory_pressure > 0.5 and block.score is not None and block.score < cfg.score_threshold:
                skip_reason += ", low score with pressure"
            routed.append(
                RoutedBlock(
                    name=block.name,
                    start=block.start,
                    end=block.end,
                    policy=block.policy,
                    score=block.score,
                    decision=BlockDecision.SKIP,
                    precision=precision,
                    reason=skip_reason,
                )
            )

        for block in skip_blocks:
            precision = self.quantizer.assign_block_precision(block).precision
            agent_skip = False
            if block.score is not None and block.score >= cfg.score_threshold:
                routed.append(
                    RoutedBlock(
                        name=block.name,
                        start=block.start,
                        end=block.end,
                        policy=block.policy,
                        score=block.score,
                        decision=BlockDecision.SELECT,
                        precision=precision,
                        reason="SKIP block with unexpectedly high score, selected",
                    )
                )
                agent_skip = True
            if not agent_skip:
                routed.append(
                    RoutedBlock(
                        name=block.name,
                        start=block.start,
                        end=block.end,
                        policy=block.policy,
                        score=block.score,
                        decision=BlockDecision.SKIP,
                        precision=KVPrecision.SKIP,
                        reason="SKIP block, not selected",
                    )
                )

        return routed

    def selected_blocks(self, routed_blocks: List[RoutedBlock]) -> List[RoutedBlock]:
        return [
            b for b in routed_blocks
            if b.decision in (BlockDecision.SELECT, BlockDecision.PIN_HIGH_PRECISION, BlockDecision.QUANTIZE)
        ]

    def skipped_blocks(self, routed_blocks: List[RoutedBlock]) -> List[RoutedBlock]:
        return [b for b in routed_blocks if b.decision == BlockDecision.SKIP]

    def selected_token_indices(self, routed_blocks: List[RoutedBlock]) -> List[int]:
        indices: List[int] = []
        for b in self.selected_blocks(routed_blocks):
            indices.extend(range(b.start, b.end))
        return indices

    def selected_page_ids(
        self,
        routed_blocks: List[RoutedBlock],
        page_size: int,
    ) -> List[int]:
        ids: List[int] = []
        seen: set = set()
        for b in self.selected_blocks(routed_blocks):
            for t in range(b.start, b.end):
                pid = t // page_size
                if pid not in seen:
                    seen.add(pid)
                    ids.append(pid)
        return ids

    def prefetch_page_ids(
        self,
        routed_blocks: List[RoutedBlock],
        page_size: int,
    ) -> List[int]:
        cfg = self.config
        # Gather high-score blocks that were NOT selected as prefetch candidates
        prefetch_candidates: List[Tuple[float, int]] = []

        for b in routed_blocks:
            if b.decision == BlockDecision.SKIP and b.score is not None and b.score >= cfg.score_threshold:
                for t in range(b.start, b.end):
                    pid = t // page_size
                    prefetch_candidates.append((b.score, pid))

        # Also include neighbour pages around selected blocks
        for b in self.selected_blocks(routed_blocks):
            left = max(0, b.start // page_size - 1)
            right = (b.end + page_size - 1) // page_size
            for pid in range(left, right + 1):
                prefetch_candidates.append((1.0, pid))

        if not prefetch_candidates:
            return []

        # Deduplicate keeping highest score
        seen: Dict[int, float] = {}
        for score, pid in prefetch_candidates:
            if pid not in seen or score > seen[pid]:
                seen[pid] = score

        sorted_pids = sorted(seen.items(), key=lambda x: (-x[1], x[0]))
        deduped = []
        seen_pids: set = set()
        for score, pid in sorted_pids:
            if pid not in seen_pids:
                seen_pids.add(pid)
                deduped.append(pid)

        return deduped[: cfg.prefetch_top_k]

    def routing_summary(self, routed_blocks: List[RoutedBlock]) -> Dict:
        """
        Return a summary dict of routing decisions.

        Byte estimates assume head_dim=64, K+V tensors, and fp16 baseline.
        These are analytical cost estimates, not measured performance.
        """
        selected = self.selected_blocks(routed_blocks)
        skipped = self.skipped_blocks(routed_blocks)
        total_tokens = sum(b.end - b.start for b in routed_blocks)
        selected_tokens = sum(b.end - b.start for b in selected)
        skipped_tokens = total_tokens - selected_tokens

        precision_dist: Dict[str, int] = {}
        total_fp16_bytes: int = 0
        total_quant_bytes: float = 0.0
        for b in selected:
            bpe = _BYTES_PER_VALUE.get(b.precision, 2.0)
            n = b.end - b.start
            total_fp16_bytes += 2 * n * 64 * 2  # kv=2, head_dim=64, fp16=2B/value
            total_quant_bytes += 2 * n * 64 * bpe  # kv=2, head_dim=64, bpe from policy
            k = b.precision.value
            precision_dist[k] = precision_dist.get(k, 0) + n

        bytes_saved_pct = (
            (1.0 - total_quant_bytes / max(total_fp16_bytes, 1)) * 100.0
            if total_fp16_bytes > 0
            else 0.0
        )

        return {
            "total_blocks": len(routed_blocks),
            "selected_blocks": len(selected),
            "skipped_blocks": len(skipped),
            "total_tokens": total_tokens,
            "selected_tokens": selected_tokens,
            "skipped_tokens": skipped_tokens,
            "estimated_fp16_mb": round(total_fp16_bytes / (1024 * 1024), 3),
            "estimated_quant_mb": round(total_quant_bytes / (1024 * 1024), 3),
            "bytes_saved_pct": round(bytes_saved_pct, 2),
            "precision_distribution": precision_dist,
        }


def routing_to_kernel_metadata(
    routed_blocks: List[RoutedBlock],
    page_size: int,
) -> Dict:
    """
    Convert routing decisions into flat metadata for the next execution layer.

    Returns a dict with:
        - selected_page_ids
        - prefetch_page_ids
        - block_precision_by_page
        - selected_block_names
        - skipped_block_names
        - reasons_by_block
    """
    router = BlockRouter(RouterConfig())
    selected = router.selected_blocks(routed_blocks)
    skipped = router.skipped_blocks(routed_blocks)

    selected_page_ids = router.selected_page_ids(routed_blocks, page_size)
    prefetch_page_ids = router.prefetch_page_ids(routed_blocks, page_size)

    block_precision_by_page: Dict[int, KVPrecision] = {}
    for b in selected:
        for t in range(b.start, b.end):
            pid = t // page_size
            if pid not in block_precision_by_page:
                block_precision_by_page[pid] = b.precision

    return {
        "selected_page_ids": selected_page_ids,
        "prefetch_page_ids": prefetch_page_ids,
        "block_precision_by_page": {
            str(k): v.value for k, v in sorted(block_precision_by_page.items())
        },
        "selected_block_names": [b.name for b in selected],
        "skipped_block_names": [b.name for b in skipped],
        "reasons_by_block": {b.name: b.reason for b in routed_blocks},
    }
