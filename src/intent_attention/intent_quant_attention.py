from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import torch

from .block_metadata import BlockLayout
from .intent_quant import (
    IntentQuantizer,
    KVPrecision,
    QuantPolicy,
    compute_quant_error,
    fake_dequantize_tensor,
    fake_quantize_tensor,
)
from .reference import dense_attention, semantic_block_attention


def intent_quant_attention_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    layout: BlockLayout,
    quantizer: IntentQuantizer,
    causal: bool = False,
    return_debug: bool = False,
) -> Any:
    if causal:
        raise NotImplementedError(
            "Causal IntentQuant attention requires explicit query_positions "
            "because selected KV indices are in original context coordinates. "
            "The current CPU reference supports non-causal selected-block "
            "attention only."
        )

    kv_tokens = k.size(-2)
    layout.validate(kv_tokens)

    selected_blocks = layout.selected_blocks()
    if not selected_blocks:
        out = torch.zeros_like(q)
        if return_debug:
            return out, _empty_debug(layout, kv_tokens)
        return out

    precision_by_block: Dict[str, str] = {}
    k_parts: List[torch.Tensor] = []
    v_parts: List[torch.Tensor] = []
    k_recon_err: Dict[str, float] = {}
    v_recon_err: Dict[str, float] = {}
    estimated_fp16_bytes = 0
    estimated_quant_bytes = 0
    head_dim = q.size(-1)

    for block in selected_blocks:
        block_k = k[..., block.start : block.end, :].contiguous()
        block_v = v[..., block.start : block.end, :].contiguous()
        n_tokens = block.end - block.start

        policy: QuantPolicy = quantizer.assign_block_precision(block)
        precision_by_block[block.name] = policy.precision.value

        fp16_bytes = 2 * n_tokens * head_dim * 2
        quant_bytes = int(round(2 * n_tokens * head_dim * policy.estimated_bytes_per_value))
        estimated_fp16_bytes += fp16_bytes
        estimated_quant_bytes += quant_bytes

        k_q, k_meta = fake_quantize_tensor(block_k, policy.precision)
        v_q, v_meta = fake_quantize_tensor(block_v, policy.precision)
        k_deq = fake_dequantize_tensor(k_q, k_meta)
        v_deq = fake_dequantize_tensor(v_q, v_meta)

        k_parts.append(k_deq)
        v_parts.append(v_deq)

        if policy.precision != KVPrecision.SKIP:
            k_err = compute_quant_error(block_k, k_deq)
            v_err = compute_quant_error(block_v, v_deq)
            k_recon_err[block.name] = k_err["mse"]
            v_recon_err[block.name] = v_err["mse"]

    reconstructed_k = torch.cat(k_parts, dim=-2)
    reconstructed_v = torch.cat(v_parts, dim=-2)

    output = dense_attention(q, reconstructed_k, reconstructed_v, causal=False)

    if not return_debug:
        return output

    fp16_selected_k = torch.cat(
        [k[..., b.start : b.end, :].contiguous() for b in selected_blocks], dim=-2
    )
    fp16_selected_v = torch.cat(
        [v[..., b.start : b.end, :].contiguous() for b in selected_blocks], dim=-2
    )
    fp16_output = dense_attention(q, fp16_selected_k, fp16_selected_v, causal=False)

    output_err = compute_quant_error(fp16_output, output)
    bytes_saved_pct = (
        (1.0 - estimated_quant_bytes / max(estimated_fp16_bytes, 1)) * 100.0
        if estimated_fp16_bytes > 0
        else 0.0
    )

    debug: Dict[str, Any] = {
        "selected_block_names": [b.name for b in selected_blocks],
        "selected_tokens": layout.selected_token_count(),
        "precision_by_block": precision_by_block,
        "estimated_fp16_bytes": estimated_fp16_bytes,
        "estimated_quant_bytes": estimated_quant_bytes,
        "bytes_saved_pct": round(bytes_saved_pct, 2),
        "reconstruction_mse_k": k_recon_err,
        "reconstruction_mse_v": v_recon_err,
        "output_mse_vs_fp16_selected": output_err["mse"],
        "output_cosine_vs_fp16_selected": output_err["cosine_similarity"],
        "output_max_abs_error_vs_fp16_selected": output_err["max_abs_error"],
    }
    return output, debug


def _empty_debug(layout: BlockLayout, kv_tokens: int) -> Dict[str, Any]:
    return {
        "selected_block_names": [],
        "selected_tokens": 0,
        "precision_by_block": {},
        "estimated_fp16_bytes": 0,
        "estimated_quant_bytes": 0,
        "bytes_saved_pct": 0.0,
        "reconstruction_mse_k": {},
        "reconstruction_mse_v": {},
        "output_mse_vs_fp16_selected": 0.0,
        "output_cosine_vs_fp16_selected": 1.0,
        "output_max_abs_error_vs_fp16_selected": 0.0,
    }


def compare_intent_quant_to_fp16_selected(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    layout: BlockLayout,
    quantizer: IntentQuantizer,
) -> Dict[str, Any]:
    fp16_out, fp16_debug = semantic_block_attention(
        q, k, v, layout, causal=False, return_debug=True
    )
    quant_out, quant_debug = intent_quant_attention_reference(
        q, k, v, layout, quantizer, causal=False, return_debug=True
    )
    err = compute_quant_error(fp16_out, quant_out)
    result: Dict[str, Any] = {
        "output_mse": err["mse"],
        "output_cosine_similarity": err["cosine_similarity"],
        "output_max_abs_error": err["max_abs_error"],
        "fp16_selected_tokens": fp16_debug["selected_token_count"],
        "quant_selected_tokens": quant_debug["selected_tokens"],
        "estimated_fp16_bytes": quant_debug["estimated_fp16_bytes"],
        "estimated_quant_bytes": quant_debug["estimated_quant_bytes"],
        "bytes_saved_pct": quant_debug["bytes_saved_pct"],
        "precision_by_block": quant_debug["precision_by_block"],
        "reconstruction_mse_k": quant_debug["reconstruction_mse_k"],
        "reconstruction_mse_v": quant_debug["reconstruction_mse_v"],
    }
    return result
