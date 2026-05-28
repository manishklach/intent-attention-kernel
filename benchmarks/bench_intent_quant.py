from __future__ import annotations

import torch

from intent_attention.block_metadata import BlockLayout, BlockPolicy, SemanticBlock
from intent_attention.intent_quant import (
    KVPrecision,
    IntentQuantizer,
    fake_quantize_tensor,
    compute_quant_error,
)


def _make_layout(total_tokens: int) -> BlockLayout:
    """Create a plausible agentic layout for *total_tokens*."""
    third = total_tokens // 3
    return BlockLayout(
        [
            SemanticBlock("system_prompt", 0, third // 2, BlockPolicy.ALWAYS),
            SemanticBlock(
                "retrieved_doc_0",
                third // 2,
                third,
                BlockPolicy.ATTEND,
                score=0.85,
            ),
            SemanticBlock(
                "retrieved_doc_1",
                third,
                2 * third,
                BlockPolicy.ATTEND,
                score=0.40,
            ),
            SemanticBlock(
                "retrieved_doc_2",
                2 * third,
                2 * third + third // 2,
                BlockPolicy.ATTEND,
                score=0.10,
            ),
            SemanticBlock("recent", 2 * third + third // 2, total_tokens, BlockPolicy.RECENT),
        ]
    )


def main() -> None:
    torch.set_num_threads(1)

    sizes = [8192, 32768, 131072]
    heads = 32
    head_dim = 128

    print("=" * 90)
    print("IntentQuant-KV: Intent-Aware Mixed-Precision KV Quantization")
    print("=" * 90)
    print()
    print(
        f"{'Total Tokens':>13} {'fp16 (MB)':>12} {'Quant (MB)':>12}"
        f" {'Saved %':>9} {'Critical FP16':>13}  Precision Distribution"
    )
    print("-" * 90)

    for size in sizes:
        quant = IntentQuantizer(memory_pressure=0.5)
        layout = _make_layout(size)
        est = quant.estimate_layout_bytes(layout, heads=heads, head_dim=head_dim)

        fp16_mb = est["dense_fp16_bytes"] / 1e6
        quant_mb = est["intent_quant_bytes"] / 1e6
        dist_str = ", ".join(
            f"{k}: {v}" for k, v in est["precision_distribution"].items()
        )
        print(
            f"{est['total_tokens']:>13} {fp16_mb:>11.1f} {quant_mb:>11.1f}"
            f" {est['bytes_saved_pct']:>8.1f}% {est['critical_full_precision_tokens']:>13}"
            f"  {dist_str}"
        )

    # ---- fake quant / dequant reconstruction test --------------------------
    print()
    print("Fake quantisation reconstruction test (random tensor)")
    print("-" * 50)
    x = torch.randn(1, 4, 128, 64)
    results = []
    for prec in (
        KVPrecision.FP16,
        KVPrecision.FP8,
        KVPrecision.INT8,
        KVPrecision.INT4,
        KVPrecision.INT4_RESIDUAL,
    ):
        recon, meta = fake_quantize_tensor(x, prec)
        err = compute_quant_error(x, recon)
        results.append((prec.value, err["mse"], err["max_abs_error"], err["cosine_similarity"]))

    print(f"{'Precision':>16} {'MSE':>14} {'Max Abs Err':>14} {'Cos Sim':>10}")
    print("-" * 56)
    for name, mse, max_abs, cs in results:
        print(f"{name:>16} {mse:>13.6e} {max_abs:>13.6e} {cs:>9.6f}")

    print()
    print("IntentQuant-KV is a CPU simulation. No GPU accuracy or throughput claim is made.")


if __name__ == "__main__":
    main()
