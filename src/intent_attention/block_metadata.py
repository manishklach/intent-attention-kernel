from __future__ import annotations

from dataclasses import dataclass, field
from enum import auto
from typing import Dict, List, Optional

from ._enum import StrEnum


class BlockPolicy(StrEnum):
    ALWAYS = auto()
    ATTEND = auto()
    SKIP = auto()
    RECENT = auto()
    GLOBAL = auto()


SELECTED_POLICIES = frozenset(
    {
        BlockPolicy.ALWAYS,
        BlockPolicy.ATTEND,
        BlockPolicy.RECENT,
        BlockPolicy.GLOBAL,
    }
)


@dataclass
class SemanticBlock:
    name: str
    start: int
    end: int
    policy: BlockPolicy
    score: Optional[float] = None


@dataclass
class BlockLayout:
    blocks: List[SemanticBlock] = field(default_factory=list)

    def validate(self, total_tokens: int) -> None:
        for i, block in enumerate(self.blocks):
            if not block.name:
                raise ValueError(f"Block at index {i} has empty name")
            if block.start < 0:
                raise ValueError(f"Block '{block.name}' has start={block.start} < 0")
            if block.end <= block.start:
                raise ValueError(
                    f"Block '{block.name}' has end={block.end} <= start={block.start}"
                )
            if block.end > total_tokens:
                raise ValueError(
                    f"Block '{block.name}' has end={block.end} > total_tokens={total_tokens}"
                )
            if block.policy == BlockPolicy.ATTEND and block.score is None:
                raise ValueError(
                    f"Block '{block.name}' has policy ATTEND but score is None"
                )
            if i > 0:
                prev = self.blocks[i - 1]
                if block.start < prev.start:
                    raise ValueError(
                        f"Blocks not sorted: '{block.name}' start={block.start} "
                        f"< '{prev.name}' start={prev.start}"
                    )
                if block.start < prev.end:
                    raise ValueError(
                        f"Blocks overlap: '{prev.name}' end={prev.end} "
                        f"> '{block.name}' start={block.start}"
                    )

    def selected_blocks(self) -> List[SemanticBlock]:
        return [b for b in self.blocks if b.policy in SELECTED_POLICIES]

    def selected_token_indices(self) -> List[int]:
        indices: List[int] = []
        for block in self.selected_blocks():
            indices.extend(range(block.start, block.end))
        return indices

    def selected_token_count(self) -> int:
        return sum(b.end - b.start for b in self.selected_blocks())

    def total_token_count(self) -> int:
        return sum(b.end - b.start for b in self.blocks)

    def summary(self) -> Dict[str, int]:
        return {
            "total_blocks": len(self.blocks),
            "selected_blocks": len(self.selected_blocks()),
            "total_token_count": self.total_token_count(),
            "selected_token_count": self.selected_token_count(),
        }
