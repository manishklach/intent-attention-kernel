import torch
from typing import Tuple
from .block_metadata import BlockLayout

class BlockTable:
    def __init__(self, block_size: int = 64):
        self.block_size = block_size
        
    def create_block_table(self, layout: BlockLayout, total_tokens: int) -> Tuple[torch.Tensor, int]:
        """
        Creates a physical block table index array mapping selected semantic
        blocks to underlying KV cache pages.
        """
        selected_blocks = layout.selected_blocks()
        if not selected_blocks:
            return torch.empty(0, dtype=torch.int32), 0
            
        valid_pages = []
        num_tokens = 0
        for block in selected_blocks:
            start_page = block.start // self.block_size
            end_page = (block.end + self.block_size - 1) // self.block_size
            
            for p in range(start_page, end_page):
                valid_pages.append(p)
            num_tokens += (block.end - block.start)
            
        return torch.tensor(valid_pages, dtype=torch.int32), num_tokens
