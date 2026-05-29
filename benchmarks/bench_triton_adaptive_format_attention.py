"""Benchmark: Triton adaptive-format decode attention kernel.

Usage:
    pytest benchmarks/bench_triton_adaptive_format_attention.py -v
    python benchmarks/bench_triton_adaptive_format_attention.py --dry-run
"""

from __future__ import annotations

import argparse
import sys
import time

import torch

from intent_attention.triton_adaptive_format_attention import (
    AdaptiveFormatKernelConfig,
    is_cuda_available,
    is_triton_available,
    adaptive_format_decode_attention_triton,
    adaptive_format_decode_attention_reference_dispatch,
)


# ────────────────────────────────────────────────────
#  Config
# ────────────────────────────────────────────────────

_BENCH_CONFIGS = {
    "micro": AdaptiveFormatKernelConfig(page_size=16, head_dim=32, max_selected_pages=8, block_d=32),
    "small": AdaptiveFormatKernelConfig(page_size=16, head_dim=64, max_selected_pages=16, block_d=64),
    "base": AdaptiveFormatKernelConfig(page_size=16, head_dim=128, max_selected_pages=32, block_d=128),
}

_B = 2
_H = 4
_NP = 64  # physical pages


def _make_scenario(
    cfg: AdaptiveFormatKernelConfig,
    fp16_frac: float,
    int8_frac: float,
    skip_frac: float,
    device: torch.device,
):
    """Create tensors for a given format mix."""
    D = cfg.head_dim
    PS = cfg.page_size
    MAX = cfg.max_selected_pages
    assert abs(fp16_frac + int8_frac + skip_frac - 1.0) < 1e-6

    num_fp16 = int(_NP * fp16_frac)
    num_int8 = int(_NP * int8_frac)

    q = torch.randn(_B, _H, D, dtype=torch.float16, device=device)
    fp16_kp = torch.randn(_NP, PS, D, dtype=torch.float16, device=device)
    fp16_vp = fp16_kp  # shared for benchmark
    int8_kp = torch.zeros(_NP, PS, D, dtype=torch.int8, device=device)
    int8_vp = torch.zeros(_NP, PS, D, dtype=torch.int8, device=device)
    int8_scales = torch.ones(_NP, dtype=torch.float32, device=device)

    if num_int8 > 0:
        int8_kp[num_fp16:num_fp16 + num_int8] = (fp16_kp[num_fp16:num_fp16 + num_int8] * 10).to(torch.int8)
        int8_vp[num_fp16:num_fp16 + num_int8] = (fp16_vp[num_fp16:num_fp16 + num_int8] * 10).to(torch.int8)
        int8_scales[num_fp16:num_fp16 + num_int8] = 0.1

    page_formats = torch.full((_NP,), 3, dtype=torch.int32, device=device)  # default SKIP
    if num_fp16 > 0:
        page_formats[:num_fp16] = 0
    if num_int8 > 0:
        page_formats[num_fp16:num_fp16 + num_int8] = 1

    # Select MAX pages (some will be SKIP)
    page_ids = torch.zeros(_B, _H, MAX, dtype=torch.int32, device=device)
    for b in range(_B):
        for h in range(_H):
            num_sel = min(MAX, num_fp16 + num_int8)
            for i in range(num_sel):
                page_ids[b, h, i] = i % _NP
    # Introduce some SKIP by selecting pages beyond the fp16+int8 range
    for b in range(_B):
        for h in range(_H):
            for i in range(num_fp16 + num_int8, MAX):
                page_ids[b, h, i] = _NP - 1  # a SKIP page

    return q, page_ids, page_formats, fp16_kp, fp16_vp, int8_kp, int8_vp, int8_scales, int8_scales


def _time_call(fn, *, warmup=3, repeat=20, **kwargs) -> float:
    """Time a callable (ms per call)."""
    for _ in range(warmup):
        fn(**kwargs)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start = time.perf_counter()
    for _ in range(repeat):
        fn(**kwargs)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    elapsed_ms = (time.perf_counter() - start) / repeat * 1000
    return elapsed_ms


# ────────────────────────────────────────────────────
#  Scenarios (pytest visible)
# ────────────────────────────────────────────────────


def _run_scenario(name: str, fp16_frac: float, int8_frac: float, skip_frac: float, *,
                  dry_run: bool = False):
    cfg = _BENCH_CONFIGS["base"]
    D = cfg.head_dim
    PS = cfg.page_size
    MAX = cfg.max_selected_pages

    if dry_run:
        print(f"[dry-run] Scenario '{name}' — format mix: "
              f"FP16={fp16_frac:.0%}, INT8={int8_frac:.0%}, SKIP={skip_frac:.0%}")
        print(f"[dry-run]   cfg: D={D}, PS={PS}, MAX_SEL={MAX}, NP={_NP}, B={_B}, H={_H}")
        print(f"[dry-run]   Module imports OK, layout validated.")
        return

    device = torch.device("cuda:0")
    tensors = _make_scenario(cfg, fp16_frac, int8_frac, skip_frac, device)
    q, page_ids, page_formats, fp16_kp, fp16_vp, int8_kp, int8_vp, kscales, vscales = tensors

    kwargs = dict(
        q=q, page_ids=page_ids, page_formats=page_formats,
        fp16_k_pages=fp16_kp, fp16_v_pages=fp16_vp,
        int8_k_pages=int8_kp, int8_v_pages=int8_vp,
        int8_k_scales=kscales, int8_v_scales=vscales,
        config=cfg,
    )

    # GPU time
    t_gpu = _time_call(adaptive_format_decode_attention_triton, **kwargs)

    # CPU reference time
    cpu_kwargs = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in kwargs.items()}
    t_cpu = _time_call(adaptive_format_decode_attention_reference_dispatch, **cpu_kwargs)

    print(f"  GPU:  {t_gpu:8.2f} ms")
    print(f"  CPU:  {t_cpu:8.2f} ms")
    print(f"  Ratio: {t_cpu/t_gpu:.2f}x")


# ────────────────────────────────────────────────────
#  Pytest entry points
# ────────────────────────────────────────────────────


def test_fp16_only():
    _run_scenario("fp16_only", 1.0, 0.0, 0.0)


def test_int8_only():
    _run_scenario("int8_only", 0.0, 1.0, 0.0)


def test_mixed():
    _run_scenario("mixed", 0.5, 0.3, 0.2)


def test_mixed_with_skip():
    _run_scenario("mixed_with_skip", 0.3, 0.3, 0.4)


# ────────────────────────────────────────────────────
#  CLI
# ────────────────────────────────────────────────────


def main():
    p = argparse.ArgumentParser(description="Benchmark Triton adaptive-format decode attention")
    p.add_argument("--dry-run", action="store_true", help="Validate imports and layout only")
    args = p.parse_args()

    if args.dry_run:
        print("Adaptive-Format Triton Benchmark — dry run")
        print(f"  Triton available:   {is_triton_available()}")
        print(f"  CUDA available:     {is_cuda_available()}")
    elif not is_cuda_available() or not is_triton_available():
        print("Benchmark requires CUDA + Triton. Use --dry-run for validation.")
        sys.exit(1)

    scenarios = [
        ("fp16_only", 1.0, 0.0, 0.0),
        ("int8_only", 0.0, 1.0, 0.0),
        ("mixed", 0.5, 0.3, 0.2),
        ("mixed_with_skip", 0.3, 0.3, 0.4),
    ]
    for name, fp16_frac, int8_frac, skip_frac in scenarios:
        print(f"\n=== {name} ===")
        _run_scenario(name, fp16_frac, int8_frac, skip_frac, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
