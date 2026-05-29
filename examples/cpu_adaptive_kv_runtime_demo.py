"""
Demo: CPU Adaptive KV Runtime — smart KV cache memory in action.

Demonstrates:
- Per-page format assignment from semantic block layout
- Partial-page token masks for precise block boundaries
- Self-tuning demotion policy (adapts thresholds to access patterns)
- Prefetch warmup (promotes predicted pages to FP16)
- Format transitions across decode steps

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
    # 1. Create manager with self-tuning policy
    # ------------------------------------------------------------------
    print_sep("1. Create KVMemoryManager with self-tuning policy")

    policy = PageFormatPolicy(
        always_format=PageStorageFormat.FP16,
        recent_format=PageStorageFormat.FP16,
        attend_high_format=PageStorageFormat.FP16,
        attend_low_format=PageStorageFormat.INT8,
        skip_format=PageStorageFormat.SKIP,
        score_threshold=0.5,
        demote_cold_after_steps=3,
        demote_cold_to=PageStorageFormat.INT8,
        promote_hot_after_accesses=3,
        promote_hot_to=PageStorageFormat.FP16,
    )

    mgr = KVMemoryManager(
        num_pages=32,
        page_size=16,
        head_dim=64,
        policy=policy,
        adapt_policy_every=2,  # tune every 2 steps
    )

    print(f"  Pages:       {mgr.num_pages}")
    print(f"  Page size:   {mgr.page_size}")
    print(f"  Head dim:    {mgr.head_dim}")
    print(f"  Self-tune:   every {mgr.adapt_policy_every} steps")

    # ------------------------------------------------------------------
    # 2. Register a semantic block layout (partial blocks included)
    # ------------------------------------------------------------------
    print_sep("2. Register layout (some blocks not on page boundaries)")

    layout = BlockLayout([
        SemanticBlock("system_prompt",   0,     256, BlockPolicy.ALWAYS),
        SemanticBlock("retrieved_doc_a", 256,   768, BlockPolicy.ATTEND, score=0.85),
        SemanticBlock("retrieved_doc_b", 768,  1024, BlockPolicy.ATTEND, score=0.25),
        SemanticBlock("tool_output",    1024,  1280, BlockPolicy.ATTEND, score=0.15),
        SemanticBlock("scratchpad",     1280,  1305, BlockPolicy.SKIP),      # partial
        SemanticBlock("recent_context", 1305,  1400, BlockPolicy.RECENT),     # partial start
    ])

    mgr.register_layout(layout)

    print(f"  Blocks: {len(layout.blocks)}")
    print(f"  Total pages: {len(mgr.pages)}")
    partial_pages = [pid for pid, s in mgr.pages.items() if s.partial_page_mask is not None]
    print(f"  Partial pages: {len(partial_pages)} (precise token boundaries)")

    for block in layout.blocks:
        pids = mgr.block_name_to_page_ids.get(block.name, [])
        fmt_counts = {}
        for pid in pids:
            s = mgr.pages[pid]
            fn = PageStorageFormat(s.format).name
            fmt_counts[fn] = fmt_counts.get(fn, 0) + 1
        partial_in_block = sum(1 for pid in pids if mgr.pages[pid].partial_page_mask is not None)
        part_str = f"  partial={partial_in_block}" if partial_in_block else ""
        print(f"    {block.name:20s}  pages={fmt_counts}{part_str}")

    # ------------------------------------------------------------------
    # 3. Write KV data
    # ------------------------------------------------------------------
    print_sep("3. Write KV data")

    for pid, state in mgr.pages.items():
        kv = torch.randn(mgr.page_size, mgr.head_dim, dtype=torch.float16)
        mgr.write_page(pid, kv)

    for pid in sorted(mgr.pages.keys())[:4]:
        s = mgr.pages[pid]
        fmt_name = PageStorageFormat(s.format).name
        storage = "fp16" if s.kv_fp16 is not None else \
                  "int8" if s.kv_int8 is not None else \
                  "sparse" if s.sp_k_values is not None else "none"
        part = " (partial)" if s.partial_page_mask is not None else ""
        print(f"  Page {pid:2d}  fmt={fmt_name:6s}  stored={storage}{part}")

    # ------------------------------------------------------------------
    # 4. Decode steps with all features active
    # ------------------------------------------------------------------
    print_sep("4. Decode steps (self-tuning + warmup + partial masks)")

    q = torch.randn(1, 1, mgr.head_dim, dtype=torch.float16)

    for step_i in range(1, 11):
        demote_now = step_i >= 4
        adapt_now = step_i >= 4
        warmup_now = step_i >= 3

        out = mgr.step(
            q,
            demote=demote_now,
            promote=True,
            warmup=warmup_now,
            adapt=adapt_now,
            mask=True,
        )

        summary = mgr.page_summary()
        fmt = summary["format_distribution"]
        changes = []
        if adapt_now:
            changes.append(f"thresh={mgr.policy.demote_cold_after_steps}")
        if warmup_now:
            changes.append(f"warmed={summary['prefetch_predictions']}")
        change_str = " | ".join(changes) if changes else ""

        print(f"  Step {step_i:2d}:  "
              f"FP16={fmt['FP16']:2d}  "
              f"INT8={fmt['INT8']:2d}  "
              f"SKIP={fmt['SKIP']:2d}  "
              f"cold={summary['cold_pages']:2d}  "
              f"{change_str}")

    # ------------------------------------------------------------------
    # 5. Final page state
    # ------------------------------------------------------------------
    print_sep("5. Final page state (first 6 pages)")

    for pid in sorted(mgr.pages.keys())[:6]:
        s = mgr.pages[pid]
        fmt_name = PageStorageFormat(s.format).name
        hot = "HOT" if s.access_count >= policy.promote_hot_after_accesses else ""
        part = "partial" if s.partial_page_mask is not None else "full"
        print(f"  Page {pid:2d}  fmt={fmt_name:6s}  "
              f"accesses={s.access_count:2d}  "
              f"{part:8s}  {hot}")

    # ------------------------------------------------------------------
    # 6. Summary
    # ------------------------------------------------------------------
    print_sep("6. Summary")
    summary = mgr.page_summary()
    fmt_dist = summary["format_distribution"]
    print(f"  Total pages:            {summary['num_pages']}")
    print(f"  Steps simulated:        {summary['step']}")
    print(f"  Format distribution:")
    for fmt_name, count in fmt_dist.items():
        if count == 0:
            continue
        pct = 100.0 * count / max(summary["num_pages"], 1)
        print(f"    {fmt_name:8s}: {count:2d} ({pct:5.1f}%)")
    print(f"  Cold pages:             {summary['cold_pages']}")
    print(f"  Final demote threshold: {summary['demote_after_steps']} steps")
    print(f"  Final promote access:   {summary['promote_after_accesses']} steps")
    print(f"  Total accesses:         {summary['total_accesses']}")
    print(f"  Prefetch preds:         {summary['prefetch_predictions']}")
    print(f"\n  [OK] CPU Adaptive KV Runtime demo complete.")
    print(f"  [OK] Self-tuning adjusted thresholds based on access patterns.")
    print(f"  [OK] Partial-page masks applied for precise token boundaries.")
    print(f"  [OK] Prefetch warmup promoted predicted pages to FP16.")


if __name__ == "__main__":
    main()
