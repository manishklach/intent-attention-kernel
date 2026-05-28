"""
GPU decode benchmark harness for IntentQuant attention prototypes.

Measures decode-step attention latency on available GPU hardware across
multiple backends: PyTorch SDPA, selected-KV gather, optional Triton
IntentQuant decode, optional xFormers, and optional FlashAttention.

This script skips unavailable backends gracefully.
No GPU speedup claim is made from these measurements.

CLI::

    python experiments/gpu_decode_benchmark.py --dry-run

    python experiments/gpu_decode_benchmark.py \\
        --batch 1 --heads 32 --head-dim 64 \\
        --kv-len 65536 --selected-frac 0.25 \\
        --iters 100 --warmup 20
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch


# ------------------------------------------------------------------ #
#  Detection helpers
# ------------------------------------------------------------------ #


def _detect_hardware() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "gpu_name": None,
        "compute_capability": None,
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "torch_version": torch.__version__,
    }

    if torch.cuda.is_available():
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["compute_capability"] = f"{torch.cuda.get_device_capability(0)[0]}.{torch.cuda.get_device_capability(0)[1]}"

    return info


def _triton_available() -> bool:
    try:
        import triton  # noqa: F401
        import triton.language as tl  # noqa: F401

        return True
    except Exception:
        return False


def _flash_attn_available() -> bool:
    try:
        import flash_attn  # noqa: F401

        return True
    except Exception:
        return False


def _xformers_available() -> bool:
    try:
        import xformers  # noqa: F401
        import xformers.ops  # noqa: F401

        return True
    except Exception:
        return False


# ------------------------------------------------------------------ #
#  Benchmark helpers
# ------------------------------------------------------------------ #


@dataclass
class BenchResult:
    backend: str
    available: bool
    avg_ms: float = 0.0
    selected_frac: float = 0.0
    kv_tokens_read: int = 0
    notes: str = ""


def _bench_sdpa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    iters: int,
    warmup: int,
) -> BenchResult:
    """PyTorch SDPA full-attention baseline."""
    try:
        for _ in range(warmup):
            torch.nn.functional.scaled_dot_product_attention(q, k, v)
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(iters):
            torch.nn.functional.scaled_dot_product_attention(q, k, v)
        torch.cuda.synchronize()
        avg_ms = 1000.0 * (time.perf_counter() - t0) / iters
    except Exception as e:
        return BenchResult(
            backend="SDPA", available=False, notes=f"error: {e}"
        )

    return BenchResult(
        backend="SDPA",
        available=True,
        avg_ms=round(avg_ms, 3),
        selected_frac=1.0,
        kv_tokens_read=k.size(-2),
    )


def _bench_selected_kv(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    selected_frac: float,
    iters: int,
    warmup: int,
) -> BenchResult:
    """Selected-KV gather + SDPA baseline."""
    kv_len = k.size(-2)
    n_selected = max(1, int(kv_len * selected_frac))
    indices = torch.randperm(kv_len, device=k.device)[:n_selected].sort().values

    try:
        for _ in range(warmup):
            k_sel = k.index_select(-2, indices)
            v_sel = v.index_select(-2, indices)
            torch.nn.functional.scaled_dot_product_attention(q, k_sel, v_sel)
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(iters):
            k_sel = k.index_select(-2, indices)
            v_sel = v.index_select(-2, indices)
            torch.nn.functional.scaled_dot_product_attention(q, k_sel, v_sel)
        torch.cuda.synchronize()
        avg_ms = 1000.0 * (time.perf_counter() - t0) / iters
    except Exception as e:
        return BenchResult(
            backend="SelectedKV", available=False, notes=f"error: {e}"
        )

    return BenchResult(
        backend="SelectedKV",
        available=True,
        avg_ms=round(avg_ms, 3),
        selected_frac=selected_frac,
        kv_tokens_read=n_selected,
    )


def _bench_triton_intent_quant(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    selected_frac: float,
    page_size: int,
    iters: int,
    warmup: int,
) -> BenchResult:
    """Optional Triton IntentQuant decode attention."""
    from intent_attention.triton_intent_quant_attention import (
        IntentQuantKernelConfig,
        fake_int8_pages_from_fp16,
        intent_quant_decode_attention_triton,
        make_page_tables_from_selected_pages,
        make_precision_tensor,
    )

    kv_len = k.size(-2)
    n_pages = (kv_len + page_size - 1) // page_size
    n_selected_pages = max(1, int(n_pages * selected_frac))
    selected_page_ids = torch.randperm(n_pages, device=q.device)[:n_selected_pages].sort().values

    page_ids, page_count = make_page_tables_from_selected_pages(
        selected_page_ids, batch=1, heads=1, max_selected_pages=n_selected_pages
    )
    page_ids = page_ids.expand(q.size(0), q.size(1), -1).contiguous()
    page_count = page_count.expand(q.size(0), q.size(1)).contiguous()

    # Build page tensors
    num_pages = n_pages
    k_fp16 = k[:, :, : num_pages * page_size, :].contiguous()
    v_fp16 = v[:, :, : num_pages * page_size, :].contiguous()
    k_i8, k_scales = fake_int8_pages_from_fp16(k_fp16[0, 0].contiguous())
    v_i8, v_scales = fake_int8_pages_from_fp16(v_fp16[0, 0].contiguous())
    k_i8 = k_i8.unsqueeze(0).unsqueeze(0).expand(q.size(0), q.size(1), -1, -1).contiguous()
    v_i8 = v_i8.unsqueeze(0).unsqueeze(0).expand(q.size(0), q.size(1), -1, -1).contiguous()
    k_scales = k_scales.unsqueeze(0).unsqueeze(0).expand(q.size(0), q.size(1), -1).contiguous()
    v_scales = v_scales.unsqueeze(0).unsqueeze(0).expand(q.size(0), q.size(1), -1).contiguous()

    fp16_pages = selected_page_ids[: n_selected_pages // 2]
    page_precision = make_precision_tensor(
        num_pages, fp16_pages=fp16_pages, device=q.device
    )

    config = IntentQuantKernelConfig(
        page_size=page_size,
        head_dim=q.size(-1),
        max_selected_pages=n_selected_pages,
        block_d=max(64, q.size(-1)),
    )

    try:
        for _ in range(warmup):
            intent_quant_decode_attention_triton(
                q, k_fp16, v_fp16, k_i8, v_i8, k_scales, v_scales,
                page_ids, page_count, page_precision,
                config=config,
            )
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(iters):
            intent_quant_decode_attention_triton(
                q, k_fp16, v_fp16, k_i8, v_i8, k_scales, v_scales,
                page_ids, page_count, page_precision,
                config=config,
            )
        torch.cuda.synchronize()
        avg_ms = 1000.0 * (time.perf_counter() - t0) / iters
    except Exception as e:
        return BenchResult(
            backend="TritonIntentQuant",
            available=False,
            notes=f"error: {e}",
        )

    return BenchResult(
        backend="TritonIntentQuant",
        available=True,
        avg_ms=round(avg_ms, 3),
        selected_frac=selected_frac,
        kv_tokens_read=n_selected_pages * page_size,
    )


def _bench_flash_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    iters: int,
    warmup: int,
    hw_info: Dict[str, Any],
) -> BenchResult:
    """Optional FlashAttention baseline — skip on Turing (CC < 8.0)."""
    cc = hw_info.get("compute_capability")
    if cc is not None:
        major = int(cc.split(".")[0])
        if major < 8:
            return BenchResult(
                backend="FlashAttention",
                available=False,
                notes=f"CC {cc} < 8.0; FA2 typically unsupported on Turing. Use PyTorch SDPA.",
            )

    if not _flash_attn_available():
        return BenchResult(
            backend="FlashAttention",
            available=False,
            notes="flash-attn not installed",
        )

    try:
        from flash_attn import flash_attn_func

        q_contig = q.contiguous().to(torch.float16)
        k_contig = k.contiguous().to(torch.float16)
        v_contig = v.contiguous().to(torch.float16)

        for _ in range(warmup):
            flash_attn_func(q_contig, k_contig, v_contig, causal=False)
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(iters):
            flash_attn_func(q_contig, k_contig, v_contig, causal=False)
        torch.cuda.synchronize()
        avg_ms = 1000.0 * (time.perf_counter() - t0) / iters
    except Exception as e:
        return BenchResult(
            backend="FlashAttention",
            available=False,
            notes=f"error: {e}",
        )

    return BenchResult(
        backend="FlashAttention",
        available=True,
        avg_ms=round(avg_ms, 3),
        selected_frac=1.0,
        kv_tokens_read=k.size(-2),
    )


def _bench_xformers(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    iters: int,
    warmup: int,
) -> BenchResult:
    """Optional xFormers memory-efficient attention."""
    if not _xformers_available():
        return BenchResult(
            backend="xFormers", available=False, notes="xformers not installed"
        )

    try:
        import xformers.ops as xops

        for _ in range(warmup):
            xops.memory_efficient_attention(q, k, v)
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(iters):
            xops.memory_efficient_attention(q, k, v)
        torch.cuda.synchronize()
        avg_ms = 1000.0 * (time.perf_counter() - t0) / iters
    except Exception as e:
        return BenchResult(
            backend="xFormers", available=False, notes=f"error: {e}"
        )

    return BenchResult(
        backend="xFormers",
        available=True,
        avg_ms=round(avg_ms, 3),
        selected_frac=1.0,
        kv_tokens_read=k.size(-2),
    )


# ------------------------------------------------------------------ #
#  Main
# ------------------------------------------------------------------ #


def main():
    parser = argparse.ArgumentParser(
        description="GPU decode benchmark for IntentQuant attention prototypes"
    )
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=32)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--kv-len", type=int, default=65536)
    parser.add_argument("--selected-frac", type=float, default=0.25)
    parser.add_argument("--page-size", type=int, default=16)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    print("=" * 72)
    print("  IntentQuant — GPU Decode Benchmark")
    print("=" * 72)
    print()

    hw_info = _detect_hardware()

    print(f"  PyTorch:       {hw_info['torch_version']}")
    print(f"  CUDA:          {hw_info['cuda_version'] or 'N/A'}")
    print(f"  GPU:           {hw_info['gpu_name'] or 'N/A'}")
    print(f"  CC:            {hw_info['compute_capability'] or 'N/A'}")
    print(f"  Triton:        {'yes' if _triton_available() else 'no'}")
    print(f"  flash-attn:    {'yes' if _flash_attn_available() else 'no'}")
    print(f"  xformers:      {'yes' if _xformers_available() else 'no'}")
    print()

    if args.dry_run:
        print("DRY RUN — no GPU kernels will be launched")
        print()
        print(f"  Config: B={args.batch} H={args.heads} D={args.head_dim} "
              f"KV={args.kv_len} sel={args.selected_frac}")
        print(f"  Available backends would be benchmarked:")
        print(f"    - PyTorch SDPA (always)")
        print(f"    - SelectedKV   (always)")
        print(f"    - TritonIntentQuant ({'yes' if _triton_available() else 'no; skip'})")
        cc = hw_info.get("compute_capability")
        if cc:
            major = int(cc.split(".")[0])
            has_fa2 = major >= 8
            print(f"    - FlashAttention ({'yes (CC>=8.0)' if has_fa2 else 'no; CC<8.0; T4 Turing; skip FA2'})")
        print(f"    - xFormers      ({'yes' if _xformers_available() else 'no; skip'})")
        print()
        sys.exit(0)

    if not torch.cuda.is_available():
        print("CUDA not available. Cannot run GPU benchmark.")
        print("Run with --dry-run for configuration info, or use a CUDA-capable system.")
        sys.exit(1)

    device = torch.device(args.device)

    q = torch.randn(args.batch, args.heads, 1, args.head_dim, device=device, dtype=torch.float16)
    k = torch.randn(args.batch, args.heads, args.kv_len, args.head_dim, device=device, dtype=torch.float16)
    v = torch.randn(args.batch, args.heads, args.kv_len, args.head_dim, device=device, dtype=torch.float16)

    print(f"  Benchmark config: B={args.batch} H={args.heads} D={args.head_dim} "
          f"KV={args.kv_len} sel={args.selected_frac}")
    print(f"  Iterations: {args.iters}  Warmup: {args.warmup}")
    print()

    results: List[BenchResult] = []

    # A. PyTorch SDPA
    print("  [1/5] Running PyTorch SDPA ...")
    results.append(_bench_sdpa(q, k, v, args.iters, args.warmup))

    # B. Selected-KV
    print(f"  [2/5] Running SelectedKV (frac={args.selected_frac}) ...")
    results.append(
        _bench_selected_kv(q, k, v, args.selected_frac, args.iters, args.warmup)
    )

    # C. Triton IntentQuant
    if _triton_available():
        print("  [3/5] Running TritonIntentQuant ...")
        results.append(
            _bench_triton_intent_quant(
                q, k, v, args.selected_frac, args.page_size, args.iters, args.warmup
            )
        )
    else:
        results.append(BenchResult(
            backend="TritonIntentQuant", available=False, notes="Triton not installed"
        ))

    # D. FlashAttention (skip on Turing <= T4)
    cc = hw_info.get("compute_capability")
    if cc and int(cc.split(".")[0]) >= 8:
        print("  [4/5] Running FlashAttention ...")
        results.append(_bench_flash_attn(q, k, v, args.iters, args.warmup, hw_info))
    else:
        results.append(BenchResult(
            backend="FlashAttention",
            available=False,
            notes=f"CC {cc} < 8.0; T4 Turing; skip FA2",
        ))

    # E. xFormers
    print("  [5/5] Running xFormers ...")
    results.append(_bench_xformers(q, k, v, args.iters, args.warmup))

    # Print table
    print()
    print("-" * 90)
    print(f"  GPU: {hw_info['gpu_name']}")
    print("-" * 90)
    header = (
        f"{'Backend':<20} {'Avail':>6} {'Avg(ms)':>10} "
        f"{'SelFrac':>8} {'KV_read':>10} {'Notes':>30}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        ms_str = f"{r.avg_ms:.4f}" if r.available else "N/A"
        print(
            f"{r.backend:<20} "
            f"{'yes' if r.available else 'no':>6} "
            f"{ms_str:>10} "
            f"{r.selected_frac:>7.2f} "
            f"{r.kv_tokens_read:>10} "
            f"{r.notes[:30]:>30}"
        )
    print("-" * 90)
    print()

    print("  These are local measurements on this GPU/configuration only.")
    print("  No GPU speedup claim is made.  Results depend on hardware,")
    print("  driver version, CUDA version, PyTorch version, and system")
    print("  load.  For FA2 comparisons, use Ampere/Ada/Hopper hardware.")
    print()


if __name__ == "__main__":
    main()
