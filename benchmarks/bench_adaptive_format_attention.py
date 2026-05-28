"""Benchmark: adaptive format KV attention.

Measures performance of adaptive format KV attention reference against:
- Standard dense attention reference
- Standard selected-block attention reference

Skips cleanly on systems without required dependencies.
No GPU speedup is claimed.
"""

from __future__ import arguments

import argparse
import sys
import time

import torch

from intent_attention.adaptive_format_attention import (
    adaptive_format_attention_reference,
    adaptive_format_attention_reference_simple,
)
from intent_attention.reference import dense_attention, semantic_block_attention
from intent_attention.block_metadata import BlockLayout, BlockPolicy, SemanticBlock


def _bench_dense_attention(q, k, v, iters, warmup):
    """Dense attention baseline."""
    for _ in range(warmup):
        dense_attention(q, k, v)
    torch.cuda.synchronize() if q.is_cuda else None
    t0 = time.perf_counter()
    for _ in range(iters):
        dense_attention(q, k, v)
    torch.cuda.synchronize() if q.is_cuda else None
    return 1000.0 * (time.perf_counter() - t0) / iters


def _bench_semantic_block_attention(q, k, v, layout, iters, warmup):
    """Standard selected-block attention reference."""
    for _ in range(warmup):
        semantic_block_attention(q, k, v, layout)
    torch.cuda.synchronize() if q.is_cuda else None
    t0 = time.perf_counter()
    for _ in range(iters):
        semantic_block_attention(q, k, v, layout)
    torch.cuda.synchronize() if q.is_cuda else None
    return 1000.0 * (time.perf_counter() - t0) / iters


def _bench_adaptive_format(q, kv_pages_fp16, kv_pages_i8, kv_pages_scales,
                          kv_pages_indices, kv_pages_values, kv_pages_formats,
                          page_table, page_counts, config, iters, warmup):
    """Adaptive format attention reference."""
    for _ in range(warmup):
        adaptive_format_attention_reference(
            q, kv_pages_fp16, kv_pages_i8, kv_pages_scales,
            kv_pages_indices, kv_pages_values, kv_pages_formats,
            page_table, page_counts, config,
        )
    torch.cuda.synchronize() if q.is_cuda else None
    t0 = time.perf_counter()
    for _ in range(iters):
        adaptive_format_attention_reference(
            q, kv_pages_fp16, kv_pages_i8, kv_pages_scales,
            kv_pages_indices, kv_pages_values, kv_pages_formats,
            page_table, page_counts, config,
        )
    torch.cuda.synchronize() if q.is_cuda else None
    return 1000.0 * (time.perf_counter() - t0) / iters


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark: adaptive format KV attention reference"
    )
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--kv-pages", type=int, default=64)
    parser.add_argument("--page-size", type=int, default=16)
    parser.add_argument("--selected-frac", type=float, default=0.25)
    parser.add_argument("--sparsity-k", type=int, default=4)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    print("=" * 72)
    print("  Adaptive Format KV Attention — Benchmark")
    print("=" * 72)
    print()

    print(f"  Device: {args.device}")
    print()

    if args.dry_run:
        print("DRY RUN — no kernels launched")
        print()
        print(f"  Config: B={args.batch} H={args.heads} D={args.head_dim} "
              f"KV_pages={args.kv_pages} PS={args.page_size} "
              f"selected={args.selected_frac:.0%} sparsity_k={args.sparsity_k}")
        print()
        print("  Would benchmark:")
        print("    1. Dense attention reference (full KV)")
        print("    2. Semantic block attention reference (selected KV)")
        print("    3. Adaptive format attention reference (multi-format KV)")
        print()
        sys.exit(0)

    device = torch.device(args.device)
    B = args.batch
    H = args.heads
    D = args.head_dim
    PS = args.page_size
    NP = args.kv_pages
    n_selected = max(1, int(NP * args.selected_frac))
    K = args.sparsity_k

    # Tensors
    q = torch.randn(B, H, 1, D, dtype=torch.float16, device=device)
    kv_pages_fp16 = torch.randn(NP, PS, D, dtype=torch.float16, device=device)
    kv_pages_i8 = torch.randint(-128, 127, (NP, PS, D), dtype=torch.int8, device=device)
    kv_pages_scales = torch.rand(NP, dtype=torch.float16, device=device) * 0.01  # Small scales
    kv_pages_indices = torch.randint(0, PS, (NP, K), dtype=torch.int64, device=device)
    kv_pages_values = torch.randn(NP, K, dtype=torch.float16, device=device)
    kv_pages_formats = torch.zeros(NP, dtype=torch.int8, device=device)  # Start with all FP16
    
    # Set some pages to different formats for variety
    if NP >= 3:
        kv_pages_formats[::max(1, NP//3)] = 1  # Every third page: INT8
        kv_pages_formats[::max(1, NP//2)] = 2  # Every second page: SPARSE

    # Page table: select first n_selected pages
    page_table = torch.zeros(B, H, n_selected, dtype=torch.int32, device=device)
    page_counts = torch.full((B, H), n_selected, dtype=torch.int32, device=device)
    for b_idx in range(B):
        for h_idx in range(H):
            for p in range(n_selected):
                page_table[b_idx, h_idx, p] = p

    print(f"  Benchmark: B={B} H={H} D={D} KV_pages={NP} PS={PS}")
    print(f"  Selected: {n_selected}/{NP} ({args.selected_frac:.0%})")
    print(f"  Sparsity K: {K}")
    print(f"  Format distribution: FP16={(kv_pages_formats==0).sum().item()}, "
          f"INT8={(kv_pages_formats==1).sum().item()}, "
          f"SPARSE={(kv_pages_formats==2).sum().item()}")
    print(f"  Device: {device}")
    print(f"  Iterations: {args.iters}  Warmup: {args.warmup}")
    print()

    # Create a dummy layout for semantic block attention comparison
    # Just use the first selected block as a simple layout
    layout = BlockLayout([
        SemanticBlock("selected_region", 0, n_selected * PS, BlockPolicy.ATTEND, score=0.9)
    ])

    # 1. Dense attention baseline
    dense_ms = _bench_dense_attention(q, kv_pages_fp16, kv_pages_fp16, args.iters, args.warmup)
    print(f"  Dense attention:      {dense_ms:8.3f} ms")

    # 2. Standard selected-block attention
    semantic_ms = _bench_semantic_block_attention(q, kv_pages_fp16, kv_pages_fp16, layout, args.iters, args.warmup)
    print(f"  Semantic block attn:  {semantic_ms:8.3f} ms")

    # 3. Adaptive format attention
    adaptive_ms = _bench_adaptive_format(
        q, kv_pages_fp16, kv_pages_i8, kv_pages_scales,
        kv_pages_indices, kv_pages_values, kv_pages_formats,
        page_table, page_counts, None, args.iters, args.warmup,
    )
    print(f"  Adaptive format attn: {adaptive_ms:8.3f} ms")

    # 4. Ratios
    if dense_ms > 0:
        ratio_dense = dense_ms / adaptive_ms
        print(f"  Ratio (Dense/Adaptive): {ratio_dense:.2f}x")
    if semantic_ms > 0:
        ratio_semantic = semantic_ms / adaptive_ms
        print(f"  Ratio (Semantic/Adaptive): {ratio_semantic:.2f}x")
    print()
    print("  Note: ratio > 1 means adaptive format is faster than baseline.")
    print("  This is a CPU reference. No GPU speedup is claimed.")
    print("  Results depend on format distribution, sparsity, and hardware.")
    print()


if __name__ == "__main__":
    main()
