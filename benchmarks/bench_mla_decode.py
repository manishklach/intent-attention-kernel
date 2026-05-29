"""Benchmark: MLA Triton decode kernel throughput."""
from __future__ import annotations

import math
import time
import sys

import torch

from intent_attention.mla import MLAConfig, MLABlockTable, mla_triton_decode
from intent_attention.block_metadata import BlockLayout, SemanticBlock, BlockPolicy


def bench_mla_decode(batch=2, n_heads=8, d_head=64, d_c=256, d_model=512,
                     n_blocks=16, page_size=64, n_steps=10, dry_run=False):
    cfg = MLAConfig(d_model=d_model, d_c=d_c, n_heads=n_heads, d_head=d_head)
    table = MLABlockTable(cfg, page_size=page_size)
    for bid in range(n_blocks):
        table.append(bid, torch.randn(page_size, d_c))

    W_QK = torch.randn(n_heads * d_head, d_c)
    W_VO = torch.randn(d_c, d_model)

    q = torch.randn(batch, n_heads, 1, d_head)
    blocks = [SemanticBlock(f"b{i}", i * page_size, (i + 1) * page_size,
                            BlockPolicy.ATTEND, score=0.9) for i in range(n_blocks)]
    layout = BlockLayout(blocks)

    if dry_run:
        print(f"[dry-run] MLA decode: {batch=}, {n_heads=}, {d_c=}, "
              f"{n_blocks=} blocks, {page_size=}")
        return

    t0 = time.perf_counter()
    for _ in range(n_steps):
        mla_triton_decode(q, table, W_QK, W_VO, layout, threshold=0.5)
    elapsed = (time.perf_counter() - t0) / n_steps * 1000
    print(f"  MLA decode: {elapsed:.2f} ms/step ({n_blocks} latent blocks)")


def bench_mla_main(dry_run=False):
    print("=== MLA Triton Decode Benchmark ===")
    configs = [
        {"n_blocks": 8, "d_c": 128, "page_size": 32},
        {"n_blocks": 16, "d_c": 256, "page_size": 64},
        {"n_blocks": 32, "d_c": 512, "page_size": 64},
    ]
    for cfg in configs:
        bench_mla_decode(n_blocks=cfg["n_blocks"], d_c=cfg["d_c"],
                         page_size=cfg["page_size"], dry_run=dry_run)


if __name__ == "__main__":
    dry_run = "--dry-run" in sys.argv
    bench_mla_main(dry_run=dry_run)
