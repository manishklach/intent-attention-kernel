from __future__ import annotations

from typing import List, Tuple

import torch

from .block_metadata import BlockLayout


class BlockTable:
    """Maps selected semantic KV ranges to physical page IDs for paged attention."""

    def __init__(self, block_size: int = 64):
        self.block_size = block_size

    def create_block_table(
        self,
        layout: BlockLayout,
        total_tokens: int,
    ) -> Tuple[torch.Tensor, int]:
        selected = layout.selected_blocks()
        if not selected:
            return torch.empty(0, dtype=torch.int32), 0

        pages: List[int] = []
        num_tokens = 0

        for block in selected:
            start_page = block.start // self.block_size
            end_page = (block.end + self.block_size - 1) // self.block_size
            pages.extend(range(start_page, end_page))
            num_tokens += block.end - block.start

        return torch.tensor(pages, dtype=torch.int32), num_tokens
