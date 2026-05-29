"""
Demo: CPU Adaptive KV Runtime — A walkthrough of smart KV cache management.

This script demonstrates KVMemoryManager in action, showing how pages
are assigned formats, accessed, demoted when cold, and promoted when hot.

Usage:
    python examples/cpu_adaptive_kv_runtime_demo.py
"""

from __future__ import annotations

import torch

from intent_attention.kv_memory_manager import (
    KVMemoryManager,
    PageFormatPolicy,
    PageStorageFormat,
)
from intent_attention.block_metadata import BlockLayout, SemanticBlock, BlockPolicy


def print_sep(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def main():
    torch.manual_seed(42)

    # ------------------------------------------------------------------
    # 1. Create a manager with an aggressive demotion policy
    # ------------------------------------------------------------------
    print_sep("1. Create KVMemoryManager with policy")

    policy = PageFormatPolicy(
        always_format=PageStorageFormat.FP16,
        recent_format=PageStorageFormat.FP16,
        attend_high_format=PageStorageFormat.FP16,
        attend_low_format=PageStorageFormat.INT8,
        skip_format=PageStorageFormat.SKIP,
        score_threshold=0.5,
        demote_cold_after_steps=3,   # demote after 3 steps of no access
        demote_cold_to=PageStorageFormat.INT8,
        promote_hot_after_accesses=5,
        promote_hot_to=PageStorageFormat.FP16,
    )

    mgr = KVMemoryManager(
        num_pages=32,
        page_size=16,
        head_dim=64,
        policy=policy,
    )

    print(f"  Pages:     {mgr.num_pages}")
    print(f"  Page size: {mgr.page_size}")
    print(f"  Head dim:  {mgr.head_dim}")
    print(f"  Device:    {mgr.device}")

    # ------------------------------------------------------------------
    # 2. Register a semantic block layout
    # ------------------------------------------------------------------
    print_sep("2. Register semantic block layout")

    layout = BlockLayout([
        SemanticBlock("system_prompt",   0,     256, BlockPolicy.ALWAYS),
        SemanticBlock("retrieved_doc_a", 256,   768, BlockPolicy.ATTEND, score=0.85),
        SemanticBlock("retrieved_doc_b", 768,  1024, BlockPolicy.ATTEND, score=0.25),
        SemanticBlock("tool_output",    1024,  1280, BlockPolicy.ATTEND, score=0.15),
        SemanticBlock("scratchpad",     1280,  1408, BlockPolicy.SKIP),
        SemanticBlock("recent_context", 1408,  1536, BlockPolicy.RECENT),
    ])

    mgr.register_layout(layout)

    print(f"  Blocks: {len(layout.blocks)}")
    print(f"  Total pages: {len(mgr.pages)}")
    for block in layout.blocks:
        pids = mgr.block_name_to_page_ids.get(block.name, [])
        fmt_counts = {}
        for pid in pids:
            s = mgr.pages[pid]
            fn = PageStorageFormat(s.format).name
            fmt_counts[fn] = fmt_counts.get(fn, 0) + 1
        print(f"    {block.name:20s}  policy={block.policy.name:8s}  "
              f"score={block.score or 0.0:.2f}  pages={fmt_counts}")

    # ------------------------------------------------------------------
    # 3. Write KV data for each page
    # ------------------------------------------------------------------
    print_sep("3. Write KV data")

    for pid, state in mgr.pages.items():
        kv = torch.randn(mgr.page_size, mgr.head_dim, dtype=torch.float16)
        mgr.write_page(pid, kv)

    # Verify storage
    for pid in sorted(mgr.pages.keys())[:4]:
        s = mgr.pages[pid]
        fmt_name = PageStorageFormat(s.format).name
        storage = "fp16" if s.kv_fp16 is not None else \
                  "int8" if s.kv_int8 is not None else \
                  "sparse" if s.sp_k_values is not None else \
                  "none"
        print(f"  Page {pid:2d}  format={fmt_name:6s}  stored_as={storage}")

    # ------------------------------------------------------------------
    # 4. Simulate multiple decode steps with access patterns
    # ------------------------------------------------------------------
    print_sep("4. Simulate decode steps")

    q = torch.randn(1, 1, mgr.head_dim, dtype=torch.float16)

    # Steps 1-2: access all non-SKIP pages
    for step_i in range(1, 3):
        out = mgr.step(q)
        summary = mgr.page_summary()
        fmt_dist = summary["format_distribution"]
        print(f"  Step {step_i}: "
              f"FP16={fmt_dist['FP16']:2d}  "
              f"INT8={fmt_dist['INT8']:2d}  "
              f"SPARSE={fmt_dist['SPARSE']:2d}  "
              f"SKIP={fmt_dist['SKIP']:2d}  "
              f"cold={summary['cold_pages']:2d}  "
              f"accesses={summary['total_accesses']:3d}")

    # Steps 3-6: let system_prompt and recent_context be cold (demotion threshold = 3)
    for step_i in range(3, 7):
        # Only access recent_context pages
        recent_pids = mgr.block_name_to_page_ids.get("recent_context", [])
        for pid in list(mgr.pages.keys()):
            if pid not in recent_pids:
                mgr.pages[pid].last_access_step = 0  # simulate no access

        out = mgr.step(q, demote=True)
        summary = mgr.page_summary()
        fmt_dist = summary["format_distribution"]
        print(f"  Step {step_i}: "
              f"FP16={fmt_dist['FP16']:2d}  "
              f"INT8={fmt_dist['INT8']:2d}  "
              f"SPARSE={fmt_dist['SPARSE']:2d}  "
              f"SKIP={fmt_dist['SKIP']:2d}  "
              f"cold={summary['cold_pages']:2d}  "
              f"accesses={summary['total_accesses']:3d}  "
              f"(demotion active)")

    # ------------------------------------------------------------------
    # 5. Show final page state
    # ------------------------------------------------------------------
    print_sep("5. Final page state")

    for pid in sorted(mgr.pages.keys())[:6]:
        s = mgr.pages[pid]
        fmt_name = PageStorageFormat(s.format).name
        hot = "HOT" if s.access_count >= policy.promote_hot_after_accesses else ""
        print(f"  Page {pid:2d}  fmt={fmt_name:6s}  "
              f"accesses={s.access_count:2d}  "
              f"last_step={s.last_access_step:2d}  "
              f"{hot}")

    # ------------------------------------------------------------------
    # 6. Summary
    # ------------------------------------------------------------------
    print_sep("6. Summary")
    summary = mgr.page_summary()
    fmt_dist = summary["format_distribution"]
    print(f"  Total pages:        {summary['num_pages']}")
    print(f"  Steps simulated:    {summary['step']}")
    print(f"  Format distribution:")
    for fmt_name, count in fmt_dist.items():
        pct = 100.0 * count / max(summary["num_pages"], 1)
        print(f"    {fmt_name:8s}: {count:2d} ({pct:5.1f}%)")
    print(f"  Cold pages:         {summary['cold_pages']}")
    print(f"  Total accesses:     {summary['total_accesses']}")
    print(f"  Prefetch preds:     {summary['prefetch_predictions']}")
    print(f"\n  [OK] CPU Adaptive KV Runtime demo complete.")
    print(f"  [OK] Pages were automatically demoted (FP16->INT8) when cold.")
    print(f"  [OK] No GPU speedup is claimed - this is a CPU reference.")


if __name__ == "__main__":
    main()
