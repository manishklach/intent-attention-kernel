"""Benchmark: CPU Adaptive KV Runtime (KVMemoryManager).

Measures decode-step latency under different format mixes, demotion
rates, and access patterns.

Usage:
    pytest benchmarks/bench_kv_memory_manager.py -v
    python benchmarks/bench_kv_memory_manager.py
"""

from __future__ import annotations

import argparse
import time

import torch

from intent_attention.kv_memory_manager import (
    KVMemoryManager,
    PageFormatPolicy,
    PageStorageFormat,
)
from intent_attention.block_metadata import BlockLayout, SemanticBlock, BlockPolicy


def _make_dense_layout(n_tokens: int, page_size: int, blocks: int) -> BlockLayout:
    tokens_per_block = n_tokens // blocks
    layout_blocks = []
    for i in range(blocks):
        start = i * tokens_per_block
        end = (i + 1) * tokens_per_block if i < blocks - 1 else n_tokens
        policy = [BlockPolicy.ALWAYS, BlockPolicy.RECENT, BlockPolicy.ATTEND, BlockPolicy.GLOBAL][i % 4]
        score = 0.9 if policy == BlockPolicy.ATTEND else None
        layout_blocks.append(SemanticBlock(f"block_{i}", start, end, policy, score=score))
    return BlockLayout(layout_blocks)


def _time_steps(mgr: KVMemoryManager, n_steps: int, warmup: int = 3) -> float:
    q = torch.randn(1, 1, mgr.head_dim, dtype=torch.float16)
    for _ in range(warmup):
        mgr.step(q, demote=True, promote=True, warmup=True, adapt=True, mask=True)
    mgr2 = KVMemoryManager(
        num_pages=mgr.num_pages, page_size=mgr.page_size,
        head_dim=mgr.head_dim, policy=mgr.policy,
    )
    layout = _make_dense_layout(
        mgr.num_pages * mgr.page_size, mgr.page_size,
        min(mgr.num_pages // 2, 8),
    )
    mgr2.register_layout(layout)
    for pid in mgr2.pages:
        mgr2.write_page(pid, torch.randn(mgr.page_size, mgr.head_dim, dtype=torch.float16))
    q = torch.randn(1, 1, mgr.head_dim, dtype=torch.float16)
    start = time.perf_counter()
    for _ in range(n_steps):
        mgr2.step(q, demote=True, promote=True, warmup=True, adapt=True, mask=True)
    elapsed = (time.perf_counter() - start) / n_steps * 1000
    return elapsed


def _run_scenario(name: str, n_pages: int, page_size: int, head_dim: int,
                  n_steps: int = 10, dry_run: bool = False) -> float:
    if dry_run:
        print(f"  [dry-run] {name}: pages={n_pages}, ps={page_size}, D={head_dim}")
        return 0.0

    policy = PageFormatPolicy(
        always_format=PageStorageFormat.FP16,
        attend_high_format=PageStorageFormat.FP16,
        attend_low_format=PageStorageFormat.INT8,
    )

    mgr = KVMemoryManager(
        num_pages=n_pages, page_size=page_size, head_dim=head_dim,
        policy=policy, adapt_policy_every=5,
    )

    layout = _make_dense_layout(
        n_pages * page_size, page_size, min(n_pages // 2, 8),
    )
    mgr.register_layout(layout)
    for pid in mgr.pages:
        mgr.write_page(pid, torch.randn(page_size, head_dim, dtype=torch.float16))

    t = _time_steps(mgr, n_steps)
    summary = mgr.page_summary()
    fmt = summary["format_distribution"]
    print(f"  {name:25s}  {t:8.2f} ms/step  "
          f"FP16={fmt['FP16']:3d}  INT8={fmt['INT8']:3d}  "
          f"cold={summary['cold_pages']:3d}  "
          f"demote={summary['demote_after_steps']}")
    return t


def test_small_config():
    _run_scenario("small (16 pages, D=32)", 16, 8, 32)


def test_medium_config():
    _run_scenario("medium (64 pages, D=64)", 64, 16, 64)


def test_large_config():
    _run_scenario("large (128 pages, D=128)", 128, 16, 128)


def main():
    p = argparse.ArgumentParser(description="Benchmark CPU Adaptive KV Runtime")
    p.add_argument("--dry-run", action="store_true", help="Validate imports only")
    args = p.parse_args()

    print("CPU Adaptive KV Runtime Benchmark")
    print(f"{'='*60}")

    scenarios = [
        ("tiny (8 pages, D=16)", 8, 8, 16),
        ("small (16 pages, D=32)", 16, 8, 32),
        ("medium (64 pages, D=64)", 64, 16, 64),
        ("large (128 pages, D=128)", 128, 16, 128),
    ]

    for name, n_pages, ps, D in scenarios:
        _run_scenario(name, n_pages, ps, D, dry_run=args.dry_run)

    print(f"\n{'='*60}")
    print("  Note: CPU timing includes demotion, promotion, warmup,")
    print("  self-tuning adaptation, and partial-page masking overhead.")
    print("  No GPU speedup is claimed.")


if __name__ == "__main__":
    main()
