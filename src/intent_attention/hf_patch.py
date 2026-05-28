from __future__ import annotations

from typing import Callable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .block_metadata import BlockLayout
from .cost_model import savings_report
from .reference import semantic_block_attention

_LLAMA_PROJ = frozenset({"q_proj", "k_proj", "v_proj", "o_proj"})
_GPT2_PROJ = frozenset({"c_attn", "c_proj"})


def _extract_layer_idx(name: str) -> int:
    for part in name.split("."):
        if part.isdigit():
            return int(part)
    return 0


def _is_attention_module(module: nn.Module) -> bool:
    sub = {n for n, _ in module.named_children()}
    if _LLAMA_PROJ.issubset(sub) or _GPT2_PROJ.issubset(sub):
        return True
    cls_name = type(module).__name__
    if "attention" in cls_name.lower() or "attn" in cls_name.lower():
        return sub != set()
    return False


def _log_savings(module_name: str, kv_tokens: int, layout: BlockLayout) -> None:
    selected = layout.selected_token_count()
    report = savings_report(
        batch=1,
        heads=1,
        query_tokens=1,
        total_kv_tokens=kv_tokens,
        selected_kv_tokens=selected,
        head_dim=1,
    )
    pct = report["flops_saved_pct"]
    print(
        f"  [{module_name}] KV: {kv_tokens} -> {selected} tokens"
        f" ({pct:.1f}% FLOPs saved)"
    )


def _patch_single(
    module: nn.Module,
    module_name: str,
    layer_idx: int,
    layout_fn: Callable[[int], Optional[BlockLayout]],
) -> None:
    _patch_sdpa_forward(module, module_name, layer_idx, layout_fn)
    _patch_attn_method(module, module_name, layer_idx, layout_fn)


def _patch_sdpa_forward(
    module: nn.Module,
    module_name: str,
    layer_idx: int,
    layout_fn: Callable[[int], Optional[BlockLayout]],
) -> None:
    orig_forward = module.forward

    def patched_forward(*args: object, **kwargs: object) -> object:
        layout = layout_fn(layer_idx)
        if layout is None:
            return orig_forward(*args, **kwargs)

        kv_tokens: Optional[int] = None
        orig_sdpa = F.scaled_dot_product_attention

        def sdpa_wrapper(
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            attn_mask: object = None,
            dropout_p: float = 0.0,
            is_causal: bool = False,
            **kw: object,
        ) -> torch.Tensor:
            nonlocal kv_tokens
            kv_tokens = key.size(-2)
            return semantic_block_attention(query, key, value, layout, causal=is_causal)

        F.scaled_dot_product_attention = sdpa_wrapper
        try:
            result = orig_forward(*args, **kwargs)
            if kv_tokens is not None:
                _log_savings(module_name, kv_tokens, layout)
            return result
        finally:
            F.scaled_dot_product_attention = orig_sdpa

    module.forward = patched_forward


def _patch_attn_method(
    module: nn.Module,
    module_name: str,
    layer_idx: int,
    layout_fn: Callable[[int], Optional[BlockLayout]],
) -> None:
    if not hasattr(module, "_attn"):
        return
    orig_attn = module._attn

    def patched_attn(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        layout = layout_fn(layer_idx)
        if layout is None:
            return orig_attn(query, key, value, attention_mask, head_mask)

        out = semantic_block_attention(query, key, value, layout, causal=True)
        _log_savings(module_name, key.size(-2), layout)
        return out, None

    module._attn = patched_attn


def patch_model(
    model: nn.Module,
    layout_fn: Callable[[int], Optional[BlockLayout]],
    verbose: bool = True,
) -> nn.Module:
    patched_count = 0
    for name, module in model.named_modules():
        if _is_attention_module(module):
            layer_idx = _extract_layer_idx(name)
            _patch_single(module, name, layer_idx, layout_fn)
            patched_count += 1
            if verbose:
                print(f"  Patched {name} (layer {layer_idx})")
    if verbose:
        print(f"Patched {patched_count} attention modules.")
    return model
