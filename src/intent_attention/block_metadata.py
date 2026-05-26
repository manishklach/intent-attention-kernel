from enum import Enum, auto
from dataclasses import dataclass
from typing import List, Optional, Dict

class BlockPolicy(Enum):
    ALWAYS = auto()
    ATTEND = auto()
    SKIP = auto()
    RECENT = auto()
    GLOBAL = auto()

@dataclass
class SemanticBlock:
    name: str
    start: int
    end: int
    policy: BlockPolicy
    score: Optional[float] = None

@dataclass
class BlockLayout:
    blocks: List[SemanticBlock]

    def validate(self, total_tokens: int) -> None:
        if not self.blocks:
            return
            
        for i, block in enumerate(self.blocks):
            if not block.name:
                raise ValueError("Block name cannot be empty")
            if block.start < 0:
                raise ValueError(f"Block '{block.name}' start < 0")
            if block.end <= block.start:
                raise ValueError(f"Block '{block.name}' end <= start")
            if block.end > total_tokens:
                raise ValueError(f"Block '{block.name}' end > total_tokens ({block.end} > {total_tokens})")
                
            if i > 0:
                prev_block = self.blocks[i - 1]
                if block.start < prev_block.start:
                    raise ValueError(f"Blocks are not sorted: {prev_block.name} vs {block.name}")
                if block.start < prev_block.end:
                    raise ValueError(f"Blocks are overlapping: {prev_block.name} and {block.name}")

    def selected_blocks(self) -> List[SemanticBlock]:
        return [b for b in self.blocks if b.policy in (
            BlockPolicy.ALWAYS, 
            BlockPolicy.ATTEND, 
            BlockPolicy.RECENT, 
            BlockPolicy.GLOBAL
        )]

    def selected_token_indices(self) -> List[int]:
        indices = []
        for block in self.selected_blocks():
            indices.extend(range(block.start, block.end))
        return indices

    def selected_token_count(self) -> int:
        return sum(block.end - block.start for block in self.selected_blocks())

    def total_token_count(self) -> int:
        return sum(block.end - block.start for block in self.blocks)
        
    def summary(self) -> Dict:
        return {
            "total_blocks": len(self.blocks),
            "selected_blocks": len(self.selected_blocks()),
            "total_token_count": self.total_token_count(),
            "selected_token_count": self.selected_token_count(),
        }
