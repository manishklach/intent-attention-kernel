"""
Benchmark for the optional Triton IntentQuant decode attention kernel.

This benchmark runs only on CUDA systems with Triton installed.
On CPU-only systems it prints a message and exits cleanly.
"""

import sys
import time

import torch

from intent_attention.triton_intent_quant_attention import (
    IntentQuantKernelConfig,
    fake_int8_pages_from_fp16,
    intent_quant_decode_attention_triton,
    is_cuda_available,
    is_triton_available,
    make_page_tables_from_selected_pages,
    make_precision_tensor,
)


def benchmark():
    has_triton = is_triton_available()
    has_cuda = is_cuda_available()

    print("=" * 80)
    print("  IntentQuant Triton Decode Attention — GPU Benchmark (Optional)")
    print("=" * 80)
    print()

    if not has_triton:
        print("Triton not installed. Skipping GPU benchmark.")
        print("Install Triton and run on a CUDA-capable system.")
        print()
        return

    if not has_cuda:
        print("CUDA not available. Skipping GPU benchmark.")
        print()
        return

    if not torch.cuda.is_available():
        print("torch.cuda.is_available() returned False. Skipping.")
        print()
        return

    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(0)
    print(f"Device: {gpu_name}")
    print()

    configs = [
        ("tiny",   1,  4, 64,  16,  16,  16),
        ("small",  1,  8, 64,  16,  32,  32),
        ("medium", 1, 16, 64,  16,  64,  64),
    ]

    header = (
        f"{'Config':<10} {'B':>3} {'H':>4} {'D':>5} {'Page':>6} {'Pages':>6} "
        f"{'FP16(ms)':>10} {'Quant(ms)':>10} {'Speedup':>8} {'CosSim':>8}"
    )
    print(header)
    print("-" * len(header))

    for label, B, H, D, page_size, num_pages, max_pages in configs:
        q = torch.randn(B, H, D, device=device, dtype=torch.float16)
        k_fp16 = torch.randn(num_pages, page_size, D, device=device, dtype=torch.float16)
        v_fp16 = torch.randn(num_pages, page_size, D, device=device, dtype=torch.float16)

        k_i8, k_scales = fake_int8_pages_from_fp16(k_fp16.cpu())
        v_i8, v_scales = fake_int8_pages_from_fp16(v_fp16.cpu())
        k_i8 = k_i8.to(device=device)
        v_i8 = v_i8.to(device=device)
        k_scales = k_scales.to(device=device)
        v_scales = v_scales.to(device=device)

        selected_pages = torch.arange(num_pages, device=device)
        page_ids, page_count = make_page_tables_from_selected_pages(
            selected_pages, B, H, max_selected_pages=max_pages
        )

        fp16_pages = selected_pages[: num_pages // 2]
        page_precision = make_precision_tensor(
            num_pages, fp16_pages=fp16_pages, device=device
        )

        config = IntentQuantKernelConfig(
            page_size=page_size, head_dim=D,
            max_selected_pages=max_pages, block_d=64,
        )

        trials = 3
        fp16_ms = 0.0
        quant_ms = 0.0

        for _ in range(trials):
            t0 = time.perf_counter()
            _ = intent_quant_decode_attention_triton(
                q, k_fp16, v_fp16, k_i8, v_i8, k_scales, v_scales,
                page_ids, page_count, page_precision,
                config=config,
            )
            torch.cuda.synchronize()
            quant_ms += 1000.0 * (time.perf_counter() - t0)

        for _ in range(trials):
            fp16_precision = make_precision_tensor(
                num_pages, fp16_pages=selected_pages, device=device
            )
            t0 = time.perf_counter()
            _ = intent_quant_decode_attention_triton(
                q, k_fp16, v_fp16, k_i8, v_i8, k_scales, v_scales,
                page_ids, page_count, fp16_precision,
                config=config,
            )
            torch.cuda.synchronize()
            fp16_ms += 1000.0 * (time.perf_counter() - t0)

        quant_ms /= trials
        fp16_ms /= trials
        speedup = fp16_ms / max(quant_ms, 1e-9)

        out_quant = intent_quant_decode_attention_triton(
            q, k_fp16, v_fp16, k_i8, v_i8, k_scales, v_scales,
            page_ids, page_count, page_precision,
            config=config,
        )
        out_fp16 = intent_quant_decode_attention_triton(
            q, k_fp16, v_fp16, k_i8, v_i8, k_scales, v_scales,
            page_ids, page_count,
            make_precision_tensor(num_pages, fp16_pages=selected_pages, device=device),
            config=config,
        )

        cos_sim = torch.nn.functional.cosine_similarity(
            out_quant.float().flatten().unsqueeze(0),
            out_fp16.float().flatten().unsqueeze(0),
        ).item()

        print(
            f"{label:<10} {B:>3} {H:>4} {D:>5} {page_size:>6} {num_pages:>6} "
            f"{fp16_ms:>10.3f} {quant_ms:>10.3f} {speedup:>6.2f}x {cos_sim:>7.5f}"
        )

    print()
    print("Note: This is a prototype kernel, not performance tuned.")
    print("Speedup depends on page count, quant vs fp16 ratio, and memory bandwidth.")
    print("CPU Ratio is not a GPU speedup claim.")
    print()


if __name__ == "__main__":
    benchmark()
