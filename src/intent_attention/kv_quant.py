"""kv_quant.py — KIVI-style asymmetric INT8 KV quantisation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch

INT8_MAX = 127
INT8_MIN = -128
RESIDUAL_R = 128


def quantise_k_perchannel(k_page: torch.Tensor, group_size: int = 128) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    k = k_page.float()
    page_len, d_head = k.shape
    n_groups = max(1, (page_len + group_size - 1) // group_size)
    pad = n_groups * group_size - page_len
    if pad > 0:
        k = torch.cat([k, torch.zeros(pad, d_head)], dim=0)
    k_grouped = k.reshape(n_groups, group_size, d_head)
    k_scale = k_grouped.abs().max(dim=1).values / INT8_MAX
    k_scale = k_scale.clamp(min=1e-8)
    k_quant = (k_grouped / k_scale[:, None, :]).clamp(INT8_MIN, INT8_MAX).round()
    k_int8 = k_quant.reshape(-1, d_head)[:page_len].to(torch.int8)
    return k_int8, k_scale, torch.zeros_like(k_scale)


def dequantise_k(k_int8: torch.Tensor, k_scale: torch.Tensor, k_zero: torch.Tensor,
                 group_size: int = 128) -> torch.Tensor:
    page_len, d_head = k_int8.shape
    n_groups = k_scale.shape[0]
    k = k_int8.float()
    pad = n_groups * group_size - page_len
    if pad > 0:
        k = torch.cat([k, torch.zeros(pad, d_head)], dim=0)
    k_grouped = k.reshape(n_groups, group_size, d_head)
    return (k_grouped * k_scale[:, None, :] + k_zero[:, None, :]).reshape(-1, d_head)[:page_len]


def quantise_v_pertoken(v_page: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    v = v_page.float()
    v_scale = v.abs().max(dim=-1).values / INT8_MAX
    v_scale = v_scale.clamp(min=1e-8)
    v_quant = (v / v_scale[:, None]).clamp(INT8_MIN, INT8_MAX).round().to(torch.int8)
    return v_quant, v_scale, torch.zeros_like(v_scale)


def dequantise_v(v_int8: torch.Tensor, v_scale: torch.Tensor, v_zero: torch.Tensor) -> torch.Tensor:
    return v_int8.float() * v_scale[:, None] + v_zero[:, None]


@dataclass
class QuantisedPage:
    k_int8: torch.Tensor
    v_int8: torch.Tensor
    k_scale: torch.Tensor
    k_zero: torch.Tensor
    v_scale: torch.Tensor
    v_zero: torch.Tensor
    group_size: int = 128

    def dequantise_k(self) -> torch.Tensor:
        return dequantise_k(self.k_int8, self.k_scale, self.k_zero, self.group_size)

    def dequantise_v(self) -> torch.Tensor:
        return dequantise_v(self.v_int8, self.v_scale, self.v_zero)

    @property
    def page_len(self) -> int:
        return self.k_int8.shape[0]

    @property
    def bytes_used(self) -> int:
        return (self.k_int8.numel() + self.v_int8.numel()) * 1 + \
               (self.k_scale.numel() + self.k_zero.numel() + self.v_scale.numel() + self.v_zero.numel()) * 4


class KVQuantStore:
    def __init__(self, page_size: int = 64, residual_r: int = RESIDUAL_R, group_size: int = 128):
        self.page_size = page_size
        self.residual_r = residual_r
        self.group_size = group_size
        self._pages: Dict[int, List[QuantisedPage]] = {}
        self._residual_k: Optional[torch.Tensor] = None
        self._residual_v: Optional[torch.Tensor] = None

    def append_page(self, block_id: int, k_fp16: torch.Tensor, v_fp16: torch.Tensor) -> None:
        k_int8, k_scale, k_zero = quantise_k_perchannel(k_fp16, self.group_size)
        v_int8, v_scale, v_zero = quantise_v_pertoken(v_fp16)
        self._pages.setdefault(block_id, []).append(
            QuantisedPage(k_int8, v_int8, k_scale, k_zero, v_scale, v_zero, self.group_size))

    def update_residual(self, k_fp16: torch.Tensor, v_fp16: torch.Tensor) -> None:
        def _cat_tail(existing, new, r):
            combined = torch.cat([existing, new], dim=0) if existing is not None else new
            return combined[-r:] if combined.shape[0] > r else combined
        self._residual_k = _cat_tail(self._residual_k, k_fp16, self.residual_r)
        self._residual_v = _cat_tail(self._residual_v, v_fp16, self.residual_r)

    def get_block_kv(self, block_id: int) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        pages = self._pages.get(block_id, [])
        if not pages:
            return None, None
        return torch.cat([p.dequantise_k() for p in pages], dim=0), \
               torch.cat([p.dequantise_v() for p in pages], dim=0)

    def get_residual(self) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        return self._residual_k, self._residual_v

    def memory_bytes(self) -> Dict[str, int]:
        quant = sum(p.bytes_used for pages in self._pages.values() for p in pages)
        res = 0
        if self._residual_k is not None:
            res = (self._residual_k.numel() + self._residual_v.numel()) * 2
        return {"quantised_bytes": quant, "residual_fp16_bytes": res, "total_bytes": quant + res}

    def snr_db(self, block_id: int, k_ref: torch.Tensor, v_ref: torch.Tensor) -> Dict:
        k_deq, v_deq = self.get_block_kv(block_id)
        if k_deq is None:
            return {}
        def _snr(orig, recon):
            s = orig.float().pow(2).mean()
            n = (orig.float() - recon.float()).pow(2).mean()
            return 10 * torch.log10(s / n.clamp(min=1e-12)).item()
        return {"k_snr_db": _snr(k_ref, k_deq), "v_snr_db": _snr(v_ref, v_deq)}
