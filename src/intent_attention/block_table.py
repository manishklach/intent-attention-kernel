from __future__ import annotations

from typing import Dict, List, Tuple

import torch

from .block_metadata import BlockLayout


class BlockTable:
    """Maps selected semantic KV ranges to physical page IDs for paged attention.

    This is a CPU simulation of the mapping between semantic block ranges and
    fixed-size physical pages.  A future GPU kernel would consume the page IDs
    directly and handle partial-page masks inside the kernel.

    Notes
    -----
    - Pages are returned in sorted logical order.
    - Duplicate page IDs are removed while preserving first-occurrence order.
    - If a selected block starts or ends mid-page, the entire page is included.
      A real kernel would need per-page token offset bounds for correctness.
    """

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

        seen: Dict[int, None] = {}
        pages: List[int] = []
        num_tokens = 0

        for block in selected:
            start_page = block.start // self.block_size
            end_page = (block.end + self.block_size - 1) // self.block_size
            for pid in range(start_page, end_page):
                if pid not in seen:
                    seen[pid] = None
                    pages.append(pid)
            num_tokens += block.end - block.start

        return torch.tensor(pages, dtype=torch.int32), num_tokens
