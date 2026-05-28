import time

import torch

from intent_attention.block_metadata import BlockLayout, BlockPolicy, SemanticBlock
from intent_attention.intent_quant import IntentQuantizer
from intent_attention.intent_quant_attention import (
    compare_intent_quant_to_fp16_selected,
)
from intent_attention.reference import semantic_block_attention


def _bench_layout(kv_tokens: int) -> BlockLayout:
    n = kv_tokens // 4
    return BlockLayout([
        SemanticBlock("always", 0, n, BlockPolicy.ALWAYS),
        SemanticBlock("attend_high", n, 2 * n, BlockPolicy.ATTEND, score=0.9),
        SemanticBlock("attend_mid", 2 * n, 3 * n, BlockPolicy.ATTEND, score=0.5),
        SemanticBlock("skip", 3 * n, 4 * n, BlockPolicy.SKIP),
    ])


def benchmark_intent_quant_attention():
    print("=" * 80)
    print("  IntentQuant Attention Kernel — CPU Benchmarks")
    print("=" * 80)
    print()

    configs = [
        ("tiny", 2, 4, 8, 32, 8),
        ("small", 1, 6, 16, 128, 16),
        ("medium", 1, 8, 32, 256, 32),
    ]

    header = (
        f"{'Config':<10} {'Batch':>6} {'Heads':>6} {'Q_Len':>6} {'KV_Len':>6} {'D_Head':>6} "
        f"{'FP16Sel(ms)':>12} {'QuantSel(ms)':>12} {'Speedup':>8} "
        f"{'CosSim':>8} {'Save%':>7} {'MemPres':>8}"
    )
    print(header)
    print("-" * len(header))

    for mem_pressure in [0.3, 0.5, 0.8]:
        for label, B, H, Q, KV, D in configs:
            q = torch.randn(B, H, Q, D)
            k = torch.randn(B, H, KV, D)
            v = torch.randn(B, H, KV, D)
            layout = _bench_layout(KV)
            quantizer = IntentQuantizer(memory_pressure=mem_pressure)

            torch_fp16 = 0.0
            torch_quant = 0.0
            trials = 3

            for _ in range(trials):
                t0 = time.perf_counter()
                semantic_block_attention(q, k, v, layout, causal=False)
                torch_fp16 += time.perf_counter() - t0

                t0 = time.perf_counter()
                compare_intent_quant_to_fp16_selected(q, k, v, layout, quantizer)
                torch_quant += time.perf_counter() - t0

            fp16_ms = 1000.0 * torch_fp16 / trials
            quant_ms = 1000.0 * torch_quant / trials
            speedup = fp16_ms / max(quant_ms, 1e-9)

            result = compare_intent_quant_to_fp16_selected(q, k, v, layout, quantizer)
            cos_sim = result["output_cosine_similarity"]
            save_pct = result["bytes_saved_pct"]

            print(
                f"{label+'_mp'+str(mem_pressure).replace('.',''):<10} "
                f"{B:>6} {H:>6} {Q:>6} {KV:>6} {D:>6} "
                f"{fp16_ms:>10.3f}  {quant_ms:>10.3f}  {speedup:>6.2f}x "
                f"{cos_sim:>7.5f}  {save_pct:>5.1f}%  {mem_pressure:>7.1f}"
            )

    print()
    print("Note: CPU-only prototype — speedup < 1 means quantized path is slower due to per-block quant/dequant overhead.")
    print()


if __name__ == "__main__":
    benchmark_intent_quant_attention()
