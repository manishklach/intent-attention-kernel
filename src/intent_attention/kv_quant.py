"""
On-the-fly KV cache quantisation: INT8 per-channel (per-head-dim) absmax.

    quantise_kv_page  : (page_size, d_head) fp16 -> int8 + fp16 scales
    dequantise_kv_page: (page_size, d_head) int8 + fp16 scales -> fp16

TODO(fp8): add FP8 quantise/dequantise path once hardware support matures.
"""

from __future__ import annotations

from typing import Tuple

import torch


def quantise_kv_page(
    k: torch.Tensor,
    v: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    absmax_k = k.abs().amax(dim=0)
    absmax_v = v.abs().amax(dim=0)
    scale_k = (absmax_k / 127.0).clamp(min=1e-10).to(torch.float16)
    scale_v = (absmax_v / 127.0).clamp(min=1e-10).to(torch.float16)
    k_int8 = torch.clamp(
        torch.round(k.float() / scale_k.float().unsqueeze(0)), -127, 127
    ).to(torch.int8)
    v_int8 = torch.clamp(
        torch.round(v.float() / scale_v.float().unsqueeze(0)), -127, 127
    ).to(torch.int8)
    return k_int8, v_int8, scale_k, scale_v


def dequantise_kv_page(
    k_int8: torch.Tensor,
    v_int8: torch.Tensor,
    scale_k: torch.Tensor,
    scale_v: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    k_fp16 = k_int8.to(torch.float16) * scale_k.unsqueeze(0)
    v_fp16 = v_int8.to(torch.float16) * scale_v.unsqueeze(0)
    return k_fp16, v_fp16


def quantise_kv_cache(
    k: torch.Tensor,
    v: torch.Tensor,
    page_size: int = 128,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch, heads, total_tokens, d_head = k.shape
    num_pages = (total_tokens + page_size - 1) // page_size

    k_int8 = torch.empty(
        batch, heads, num_pages, page_size, d_head, dtype=torch.int8, device=k.device
    )
    v_int8 = torch.empty(
        batch, heads, num_pages, page_size, d_head, dtype=torch.int8, device=v.device
    )
    k_scale = torch.empty(
        batch, heads, num_pages, d_head, dtype=torch.float16, device=k.device
    )
    v_scale = torch.empty(
        batch, heads, num_pages, d_head, dtype=torch.float16, device=v.device
    )

    for p in range(num_pages):
        start = p * page_size
        end = min(start + page_size, total_tokens)
        k_page = k[..., start:end, :].contiguous()
        v_page = v[..., start:end, :].contiguous()
        k_pad = torch.zeros(
            batch, heads, page_size, d_head, dtype=k.dtype, device=k.device
        )
        v_pad = torch.zeros(
            batch, heads, page_size, d_head, dtype=v.dtype, device=v.device
        )
        k_pad[..., : k_page.size(-2), :] = k_page
        v_pad[..., : v_page.size(-2), :] = v_page

        for b in range(batch):
            for h in range(heads):
                kq, vq, ks, vs = quantise_kv_page(k_pad[b, h], v_pad[b, h])
                k_int8[b, h, p] = kq
                v_int8[b, h, p] = vq
                k_scale[b, h, p] = ks
                v_scale[b, h, p] = vs

    return k_int8, v_int8, k_scale, v_scale


def dequantise_selected_pages(
    k_int8: torch.Tensor,
    v_int8: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    page_ids: torch.Tensor,
    total_tokens: int,
    page_size: int = 128,
) -> Tuple[torch.Tensor, torch.Tensor]:
    batch, heads, num_pages, _, d_head = k_int8.shape
    selected_pages_list: list[torch.Tensor] = []
    selected_pages_v: list[torch.Tensor] = []

    for p_i in page_ids:
        p = int(p_i)
        kp = k_int8[..., p, :, :].to(torch.float16)
        vp = v_int8[..., p, :, :].to(torch.float16)
        ks = k_scale[..., p, :]
        vs = v_scale[..., p, :]
        kp_deq = kp * ks.unsqueeze(-2)
        vp_deq = vp * vs.unsqueeze(-2)
        selected_pages_list.append(kp_deq)
        selected_pages_v.append(vp_deq)

    k_deq = torch.cat(selected_pages_list, dim=-2)
    v_deq = torch.cat(selected_pages_v, dim=-2)
    k_deq = k_deq[..., : min(total_tokens, num_pages * page_size), :]
    v_deq = v_deq[..., : min(total_tokens, num_pages * page_size), :]
    return k_deq, v_deq
