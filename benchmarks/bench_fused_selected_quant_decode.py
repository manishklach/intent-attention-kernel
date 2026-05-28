"""Benchmark: fused selected-quant decode attention.

Measures decode-step latency for the fused selected-quant kernel against:
- PyTorch SDPA full attention baseline
- CPU reference path

Skips cleanly on systems without Triton or CUDA.
No GPU speedup is claimed.
"""

from __future__ import annotations

import argparse
import sys
import time

import torch

from intent_attention.fused_selected_quant_decode import (
    FusedDecodeConfig,
    FusedKVPrecision,
    fake_int8_pages_from_fp16,
    fused_selected_quant_decode,
    is_cuda_available,
    is_triton_available,
)


def _bench_sdpa(q, k, v, iters, warmup, device):
    """PyTorch SDPA full-attention baseline."""
    k_full = k.reshape(1, 1, -1, q.size(-1)).expand(q.size(0), q.size(1), -1, -1)
    v_full = v.reshape(1, 1, -1, q.size(-1)).expand(q.size(0), q.size(1), -1, -1)
    for _ in range(warmup):
        torch.nn.functional.scaled_dot_product_attention(q, k_full, v_full)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        torch.nn.functional.scaled_dot_product_attention(q, k_full, v_full)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return 1000.0 * (time.perf_counter() - t0) / iters


def _bench_fused(q, k_fp16, v_fp16, k_i8, v_i8, k_sc, v_sc,
                 pt, prec, pc, config, iters, warmup, device):
    for _ in range(warmup):
        fused_selected_quant_decode(
            q, k_fp16, v_fp16, k_i8, v_i8, k_sc, v_sc, pt, prec, pc, config,
        )
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fused_selected_quant_decode(
            q, k_fp16, v_fp16, k_i8, v_i8, k_sc, v_sc, pt, prec, pc, config,
        )
    if device.type == "cuda":
        torch.cuda.synchronize()
    return 1000.0 * (time.perf_counter() - t0) / iters


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark: fused selected-quant decode attention"
    )
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--num-pages", type=int, default=64)
    parser.add_argument("--page-size", type=int, default=16)
    parser.add_argument("--selected-frac", type=float, default=0.25)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    print("=" * 72)
    print("  Fused Selected-Quant Decode — Benchmark")
    print("=" * 72)
    print()
    print(f"  Triton:      {'yes' if is_triton_available() else 'no'}")
    print(f"  CUDA:        {'yes' if is_cuda_available() else 'no'}")
    print()

    if args.dry_run:
        print("DRY RUN — no kernels launched")
        print()
        print(f"  Config: B={args.batch} H={args.heads} D={args.head_dim} "
              f"pages={args.num_pages} page_size={args.page_size} "
              f"sel={args.selected_frac}")
        print()
        print("  Would benchmark:")
        print("    1. PyTorch SDPA (full KV)")
        print(f"    2. Fused selected-quant decode (sel_frac={args.selected_frac})")
        print()
        sys.exit(0)

    device = torch.device(args.device if args.device == "cuda" and is_cuda_available() else "cpu")

    B = args.batch
    H = args.heads
    D = args.head_dim
    PS = args.page_size
    NP = args.num_pages
    n_selected = max(1, int(NP * args.selected_frac))
    max_selected = n_selected

    config = FusedDecodeConfig(
        page_size=PS, head_dim=D, max_selected_pages=max_selected, block_d=64,
    )

    # Tensors
    q = torch.randn(B, H, 1, D, dtype=torch.float16, device=device)
    k_fp16 = torch.randn(NP, PS, D, dtype=torch.float16, device=device)
    v_fp16 = torch.randn(NP, PS, D, dtype=torch.float16, device=device)
    k_i8, k_sc = fake_int8_pages_from_fp16(k_fp16.cpu())
    v_i8, v_sc = fake_int8_pages_from_fp16(v_fp16.cpu())
    k_i8, k_sc = k_i8.to(device), k_sc.to(device)
    v_i8, v_sc = v_i8.to(device), v_sc.to(device)

    # Page table: select first n_selected pages
    pt = torch.zeros(B, H, max_selected, dtype=torch.int32, device=device)
    pc = torch.zeros(B, H, dtype=torch.int32, device=device)
    for b_idx in range(B):
        for h_idx in range(H):
            for p in range(n_selected):
                pt[b_idx, h_idx, p] = p
            pc[b_idx, h_idx] = n_selected

    prec = torch.full((NP,), FusedKVPrecision.FP16, dtype=torch.int32, device=device)
    prec[0:n_selected] = FusedKVPrecision.FP16

    print(f"  Benchmark: B={B} H={H} D={D} NP={NP} PS={PS} "
          f"selected={n_selected}/{NP} ({args.selected_frac:.0%})")
    print(f"  Device: {device}")
    print(f"  Iterations: {args.iters}  Warmup: {args.warmup}")
    print()

    # 1. SDPA baseline
    sdpa_ms = _bench_sdpa(q, k_fp16, v_fp16, args.iters, args.warmup, device)
    print(f"  SDPA (full KV):     {sdpa_ms:8.3f} ms")

    # 2. Fused selected-quant
    fused_ms = _bench_fused(
        q, k_fp16, v_fp16, k_i8, v_i8, k_sc, v_sc,
        pt, prec, pc, config, args.iters, args.warmup, device,
    )
    print(f"  Fused selected-quant: {fused_ms:8.3f} ms")

    # 3. Ratio
    ratio = sdpa_ms / fused_ms if fused_ms > 0 else float("inf")
    print(f"  Ratio (SDPA / Fused): {ratio:.2f}x")
    print()
    print("  Note: ratio > 1 means fused kernel is faster than full SDPA.")
    print("  This is an untuned prototype. No GPU speedup is claimed.")
    print("  Results depend on hardware, precision distribution,")
    print("  and selected-frac.")
    print()


if __name__ == "__main__":
    main()
