"""Benchmark: SpecAttn controller end-to-end throughput."""
from __future__ import annotations

import math
import time
import sys

import torch

from intent_attention.specattn import SpecAttnController
from intent_attention.block_metadata import BlockLayout, SemanticBlock, BlockPolicy


def bench_specattn_controller(n_blocks=64, seq_len=4096, n_steps=20, dry_run=False):
    print(f"=== SpecAttn Controller ({n_blocks} blocks, {n_steps} steps) ===")
    ctrl = SpecAttnController(top_k_blocks=max(4, n_blocks // 4), k_draft=4)
    blocks = [SemanticBlock(f"b{i}", i * (seq_len // n_blocks),
                            (i + 1) * (seq_len // n_blocks), BlockPolicy.ATTEND)
              for i in range(n_blocks)]
    layout = BlockLayout(blocks)
    layout = ctrl.init_layout(layout)
    if dry_run:
        print(f"  [dry-run] {n_steps} steps, {n_blocks} blocks — skipping simulation")
        return
    update_times = []
    accept_times = []
    for step in range(n_steps):
        t0 = time.perf_counter()
        attn_w = torch.randn(1, 1, 1, seq_len)
        layout = ctrl.update_from_verification(attn_w, layout)
        update_times.append(time.perf_counter() - t0)
        t1 = time.perf_counter()
        draft = list(range(1, 5))
        verify = torch.randn(4, 32000)
        ctrl.speculative_accept(draft, verify)
        accept_times.append(time.perf_counter() - t1)
    avg_update = sum(update_times) / len(update_times) * 1000
    avg_accept = sum(accept_times) / len(accept_times) * 1000
    print(f"  update: {avg_update:.3f} ms/step | accept: {avg_accept:.3f} ms/step "
          f"| accept rate: {ctrl.mean_acceptance_rate():.2f}")


def bench_specattn_main(dry_run=False):
    for blocks in [32, 64]:
        bench_specattn_controller(n_blocks=blocks, dry_run=dry_run)
        print()


if __name__ == "__main__":
    dry_run = "--dry-run" in sys.argv
    bench_specattn_main(dry_run=dry_run)
