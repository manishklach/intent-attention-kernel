"""Benchmark: estimated savings from block sparsity + quant."""
from __future__ import annotations

import math
import sys

from intent_attention.cost_model import savings_report, semantic_attention_cost
from intent_attention.block_metadata import BlockLayout, SemanticBlock, BlockPolicy


def bench_savings(dry_run=False):
    print("=== Savings Benchmark ===")
    configs = [
        {"name": "dense (baseline)", "n_kv": 8192, "n_sel": 8192, "quant_pct": 0},
        {"name": "sparse 25%", "n_kv": 8192, "n_sel": 2048, "quant_pct": 0},
        {"name": "sparse 12.5%", "n_kv": 8192, "n_sel": 1024, "quant_pct": 0},
        {"name": "sparse 6.25%", "n_kv": 8192, "n_sel": 512, "quant_pct": 0},
        {"name": "quant 100%", "n_kv": 8192, "n_sel": 8192, "quant_pct": 100},
        {"name": "quant 100% + sparse 12.5%", "n_kv": 8192, "n_sel": 1024, "quant_pct": 100},
    ]
    for cfg in configs:
        report = savings_report(
            n_kv=cfg["n_kv"],
            n_selected=cfg["n_sel"],
            d_head=128,
            n_heads=32,
            batch=1,
            quant_pct=cfg["quant_pct"],
        )
        total = report.get("attn_flops_gflops", 0) + report.get("kv_read_gbytes", 0) * 1e9 / 1e9
        print(f'  {cfg["name"]}: {report["attn_flops_gflops"]:.3f} GFLOPs, '
              f'{report["kv_read_gbytes"]:.3f} GB read, '
              f'estimated speedup vs dense: {report.get("speedup_vs_dense", "N/A")}')


def bench_savings_main(dry_run=False):
    if dry_run:
        print("[dry-run] savings benchmark — skipping compute")
        return
    bench_savings()


if __name__ == "__main__":
    dry_run = "--dry-run" in sys.argv
    bench_savings_main(dry_run=dry_run)
