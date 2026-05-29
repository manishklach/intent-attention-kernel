"""mla.py — Multi-Head Latent Attention (DeepSeek-V2/V3 style)."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from .block_metadata import BlockLayout, BlockPolicy


@dataclass
class MLAConfig:
    d_model: int
    d_c: int
    n_heads: int
    d_head: int
    d_rope: int = 64


def absorb_weights(W_UQ: torch.Tensor, W_UK: torch.Tensor, W_UV: torch.Tensor,
                   W_O: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    W_QK_fused = W_UQ.float().half()
    W_VO_fused = torch.mm(W_UV.float(), W_O.float()).half()
    return W_QK_fused, W_VO_fused


class MLABlockTable:
    def __init__(self, config: MLAConfig, page_size: int = 64):
        self.config = config
        self.page_size = page_size
        self._latent_pages: Dict[int, List[torch.Tensor]] = {}
        self._rope_pages: Dict[int, List[torch.Tensor]] = {}

    def append(self, block_id: int, c_page: torch.Tensor, rope_page: Optional[torch.Tensor] = None) -> None:
        self._latent_pages.setdefault(block_id, []).append(c_page)
        if rope_page is not None:
            self._rope_pages.setdefault(block_id, []).append(rope_page)

    def get_latent(self, block_id: int) -> Optional[torch.Tensor]:
        pages = self._latent_pages.get(block_id)
        return torch.cat(pages, dim=0) if pages else None

    def get_rope(self, block_id: int) -> Optional[torch.Tensor]:
        pages = self._rope_pages.get(block_id)
        return torch.cat(pages, dim=0) if pages else None

    def memory_bytes(self) -> int:
        total = 0
        for pages in self._latent_pages.values():
            for p in pages:
                total += p.numel() * 2
        for pages in self._rope_pages.values():
            for p in pages:
                total += p.numel() * 2
        return total


def mla_sparse_decode_reference(q: torch.Tensor, block_table: MLABlockTable,
                                 W_QK_fused: torch.Tensor, W_VO_fused: torch.Tensor,
                                 layout: BlockLayout, threshold: float = 0.5,
                                 return_debug: bool = False):
    batch, n_heads, q_len, d_head = q.shape
    config = block_table.config
    q_flat = q.permute(0, 2, 1, 3).reshape(batch, q_len, n_heads * d_head).float()
    q_absorb = torch.matmul(q_flat, W_QK_fused.float())
    selected = [b for b in layout.selected_blocks()
                if b.score is None or b.score >= threshold]
    latent_parts = []
    for b in selected:
        c = block_table.get_latent(layout.blocks.index(b) if isinstance(b.name, str) else b.name)
        if c is not None:
            latent_parts.append(c)
    if not latent_parts:
        out = torch.zeros(batch, q_len, config.d_model)
        return (out, {"selected_latent_tokens": 0}) if return_debug else (out, None)
    C = torch.cat(latent_parts, dim=0).float()
    scale = 1.0 / math.sqrt(config.d_c)
    scores = torch.matmul(q_absorb, C.T) * scale
    attn_w = F.softmax(scores, dim=-1)
    context = torch.matmul(attn_w, C)
    out = torch.matmul(context, W_VO_fused.float()).half()
    if not return_debug:
        return out, None
    total_tokens = sum(b.end - b.start for b in layout.blocks)
    debug = {"selected_block_names": [b.name for b in selected],
             "selected_latent_tokens": C.shape[0],
             "total_kv_tokens": total_tokens,
             "mla_memory_bytes": block_table.memory_bytes()}
    return out, debug


def mla_triton_decode(q: torch.Tensor, block_table: MLABlockTable,
                      W_QK_fused: torch.Tensor, W_VO_fused: torch.Tensor,
                      layout: BlockLayout, threshold: float = 0.5,
                      return_debug: bool = False):
    if not q.is_cuda:
        return mla_sparse_decode_reference(q, block_table, W_QK_fused, W_VO_fused,
                                           layout, threshold, return_debug)
    try:
        import triton
    except ImportError:
        return mla_sparse_decode_reference(q, block_table, W_QK_fused, W_VO_fused,
                                           layout, threshold, return_debug)
    return mla_sparse_decode_reference(q, block_table, W_QK_fused, W_VO_fused,
                                       layout, threshold, return_debug)
