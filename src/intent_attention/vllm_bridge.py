from __future__ import annotations

from typing import List, Optional, Tuple

import torch

from .block_metadata import BlockLayout, BlockPolicy, SemanticBlock
from .block_table import BlockTable
from .reference import semantic_block_attention, dense_attention


def semantic_paged_attention(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_table: torch.Tensor,
    layout: BlockLayout,
    block_size: int = 128,
    causal: bool = True,
) -> torch.Tensor:
    """Semantic paged attention bridge for vLLM-style paged KV cache.

    Parameters
    ----------
    query        : (num_tokens, num_heads, head_size)
    key_cache    : (num_physical_blocks, num_heads, head_size, block_size)
    value_cache  : (num_physical_blocks, num_heads, head_size, block_size)
    block_table  : (num_logical_blocks,)  physical block index per logical block
    layout       : BlockLayout describing which logical KV ranges to select
    block_size   : tokens per physical block
    causal       : whether to apply causal masking

    Returns
    -------
    output       : (num_tokens, num_heads, head_size)
    """
    num_logical = block_table.shape[0]
    total_kv = num_logical * block_size

    physical_blocks = []
    for block in layout.blocks:
        if block.policy == BlockPolicy.SKIP:
            continue
        if block.policy == BlockPolicy.ATTEND and block.score is not None and block.score < 0.5:
            continue
        logical_start = block.start // block_size
        logical_end = (block.end + block_size - 1) // block_size
        for logical_idx in range(logical_start, min(logical_end, num_logical)):
            phys = int(block_table[logical_idx])
            physical_blocks.append(phys)

    if not physical_blocks:
        num_tokens = query.size(0)
        num_heads = query.size(1)
        head_size = query.size(2)
        return torch.zeros(num_tokens, num_heads, head_size, device=query.device, dtype=query.dtype)

    unique_blocks = list(dict.fromkeys(physical_blocks))

    selected_k = torch.cat(
        [key_cache[phys].permute(2, 0, 1) for phys in unique_blocks], dim=0
    ).unsqueeze(0)
    selected_v = torch.cat(
        [value_cache[phys].permute(2, 0, 1) for phys in unique_blocks], dim=0
    ).unsqueeze(0)

    q_4d = query.unsqueeze(0)

    output = dense_attention(q_4d, selected_k, selected_v, causal=causal)

    return output.squeeze(0)


def create_vllm_layout(
    num_logical_blocks: int,
    block_size: int,
    system_blocks: int = 1,
    recent_blocks: int = 4,
    attend_threshold: float = 0.5,
) -> BlockLayout:
    blocks: List[SemanticBlock] = []
    curr = 0

    if system_blocks > 0:
        sz = system_blocks * block_size
        blocks.append(SemanticBlock("system", curr, curr + sz, BlockPolicy.ALWAYS))
        curr += sz

    body_start = curr
    body_blocks = num_logical_blocks - system_blocks - recent_blocks
    if body_blocks > 0:
        sz = body_blocks * block_size
        blocks.append(
            SemanticBlock("retrieved_docs", curr, curr + sz, BlockPolicy.ATTEND, score=attend_threshold)
        )
        curr += sz

    if recent_blocks > 0:
        sz = recent_blocks * block_size
        blocks.append(
            SemanticBlock("recent_context", curr, curr + sz, BlockPolicy.RECENT)
        )
        curr += sz

    if curr < num_logical_blocks * block_size:
        blocks.append(
            SemanticBlock("padding", curr, num_logical_blocks * block_size, BlockPolicy.SKIP)
        )

    return BlockLayout(blocks)


def block_table_to_layout(
    block_table: torch.Tensor,
    num_system_prompt_blocks: int = 1,
    num_recent_blocks: int = 4,
) -> Tuple[BlockLayout, BlockTable]:
    return create_vllm_layout(
        num_logical_blocks=block_table.shape[0],
        block_size=64,
        system_blocks=num_system_prompt_blocks,
        recent_blocks=num_recent_blocks,
    ), BlockTable(block_size=64)
