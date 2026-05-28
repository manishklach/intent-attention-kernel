from __future__ import annotations

import random
from typing import Dict, List, Optional

from .block_metadata import BlockPolicy, BlockLayout, SemanticBlock


def generate_agentic_layout(
    total_tokens: int,
    doc_blocks: int = 4,
    tool_blocks: int = 2,
    recent_tokens: int = 2048,
    score_threshold: float = 0.5,
    seed: Optional[int] = None,
) -> BlockLayout:
    rng = random.Random(seed)
    blocks: List[SemanticBlock] = []
    curr = 0

    if curr < total_tokens:
        sz = min(512, total_tokens - curr)
        blocks.append(
            SemanticBlock("system_prompt", curr, curr + sz, BlockPolicy.ALWAYS)
        )
        curr += sz

    if curr < total_tokens:
        sz = min(256, total_tokens - curr)
        blocks.append(
            SemanticBlock(
                "memory_summary", curr, curr + sz, BlockPolicy.ATTEND, score=1.0
            )
        )
        curr += sz

    for i in range(doc_blocks):
        if curr >= total_tokens:
            break
        sz = min(1024, total_tokens - curr)
        score = rng.random()
        policy = BlockPolicy.ATTEND if score >= score_threshold else BlockPolicy.SKIP
        blocks.append(
            SemanticBlock(f"retrieved_doc_{i}", curr, curr + sz, policy, score=score)
        )
        curr += sz

    for i in range(tool_blocks):
        if curr >= total_tokens:
            break
        sz = min(512, total_tokens - curr)
        score = rng.random()
        policy = BlockPolicy.ATTEND if score >= score_threshold else BlockPolicy.SKIP
        blocks.append(
            SemanticBlock(f"tool_output_{i}", curr, curr + sz, policy, score=score)
        )
        curr += sz

    if curr < total_tokens:
        recent_start = max(curr, total_tokens - recent_tokens)
        if recent_start > curr:
            blocks.append(
                SemanticBlock("ignored_context", curr, recent_start, BlockPolicy.SKIP)
            )
        blocks.append(
            SemanticBlock(
                "recent_context", recent_start, total_tokens, BlockPolicy.RECENT
            )
        )

    return BlockLayout(blocks)


def random_layout(
    total_tokens: int,
    num_blocks: int = 10,
    seed: Optional[int] = None,
) -> BlockLayout:
    rng = random.Random(seed)
    blocks: List[SemanticBlock] = []
    policies = list(BlockPolicy)
    per_block = total_tokens // num_blocks
    curr = 0

    for i in range(num_blocks):
        end = curr + per_block if i < num_blocks - 1 else total_tokens
        pol = rng.choice(policies)
        score = rng.random() if pol == BlockPolicy.ATTEND else None
        blocks.append(SemanticBlock(f"block_{i}", curr, end, pol, score=score))
        curr = end

    return BlockLayout(blocks)


def layout_from_policy_dict(
    policy_dict: Dict[str, BlockPolicy],
    block_sizes: Dict[str, int],
) -> BlockLayout:
    blocks: List[SemanticBlock] = []
    curr = 0

    for name, policy in policy_dict.items():
        sz = block_sizes.get(name, 0)
        score = 0.5 if policy == BlockPolicy.ATTEND else None
        blocks.append(SemanticBlock(name, curr, curr + sz, policy, score=score))
        curr += sz

    return BlockLayout(blocks)
