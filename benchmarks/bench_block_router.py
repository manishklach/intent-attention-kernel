"""
CPU benchmark for the KV Block Router.

Generates synthetic agentic layouts and runs routing decisions, printing
token/page-level cost estimates.  No GPU speedup is claimed.
"""

import torch

from intent_attention.block_metadata import BlockLayout, BlockPolicy, SemanticBlock
from intent_attention.block_router import BlockRouter, RouterConfig


def _agentic_layout(total_tokens: int) -> BlockLayout:
    sizes = {
        "system_prompt": max(128, total_tokens // 32),
        "global_memory": max(64, total_tokens // 64),
        "retrieved_1": max(256, total_tokens // 4),
        "retrieved_2": max(256, total_tokens // 4),
        "tool_output": max(128, total_tokens // 8),
        "scratchpad": max(128, total_tokens // 8),
        "recent_context": max(256, total_tokens // 4),
        "leftover_1": max(64, total_tokens // 32),
        "leftover_2": max(64, total_tokens // 32),
    }

    start = 0
    blocks = []
    for i, (name, size) in enumerate(sizes.items()):
        if start + size > total_tokens:
            size = total_tokens - start
        if size <= 0:
            break

        if name == "system_prompt":
            blocks.append(SemanticBlock(name, start, start + size, BlockPolicy.ALWAYS))
        elif name == "global_memory":
            blocks.append(SemanticBlock(name, start, start + size, BlockPolicy.GLOBAL))
        elif name == "recent_context":
            blocks.append(SemanticBlock(name, start, start + size, BlockPolicy.RECENT))
        elif name in ("retrieved_1", "retrieved_2"):
            score = 0.85 if name == "retrieved_1" else 0.40
            blocks.append(SemanticBlock(name, start, start + size, BlockPolicy.ATTEND, score=score))
        elif name == "tool_output":
            blocks.append(SemanticBlock(name, start, start + size, BlockPolicy.ATTEND, score=0.60))
        elif name in ("leftover_1", "leftover_2"):
            blocks.append(
                SemanticBlock(name, start, start + size, BlockPolicy.ATTEND, score=0.15)
            )
        else:
            blocks.append(SemanticBlock(name, start, start + size, BlockPolicy.SKIP))
        start += size

    if start < total_tokens:
        blocks.append(SemanticBlock("filler", start, total_tokens, BlockPolicy.SKIP))

    return BlockLayout(blocks)


def _print_summary(label: str, config: RouterConfig, layout: BlockLayout):
    total_tokens = layout.total_token_count()
    router = BlockRouter(config)
    routed = router.route_layout(layout, total_tokens)
    summary = router.routing_summary(routed)

    print(f"\n  {label}")
    print(f"    Config:          top_k={config.top_k_blocks}, "
          f"threshold={config.score_threshold}, "
          f"mem_pressure={config.memory_pressure}")
    print(f"    Total blocks:    {summary['total_blocks']}")
    print(f"    Selected blocks: {summary['selected_blocks']}")
    print(f"    Skipped blocks:  {summary['skipped_blocks']}")
    print(f"    Total tokens:    {summary['total_tokens']}")
    print(f"    Selected tokens: {summary['selected_tokens']}")
    print(f"    Skipped tokens:  {summary['skipped_tokens']}")
    print(f"    Selected pages   (page_size=16): {len(router.selected_page_ids(routed, page_size=16))}")
    print(f"    FP16 KV (MB):    {summary['estimated_fp16_mb']}")
    print(f"    Routed KV (MB):  {summary['estimated_quant_mb']}")
    print(f"    Bytes saved %:   {summary['bytes_saved_pct']}")
    print(f"    Precision dist:  {summary['precision_distribution']}")
    print(f"    Prefetch pages:  {router.prefetch_page_ids(routed, page_size=16)}")


def benchmark():
    print("=" * 75)
    print("  KV Block Router — CPU Routing & Cost Benchmark")
    print("=" * 75)
    print()
    print("  This is a routing and cost-model benchmark, not a GPU speedup claim.")
    print()

    configs = [
        ("default", RouterConfig()),
        ("aggro", RouterConfig(top_k_blocks=4, score_threshold=0.5, memory_pressure=0.8)),
        ("relaxed", RouterConfig(top_k_blocks=16, score_threshold=0.2, memory_pressure=0.1)),
    ]

    sizes = [8192, 32768, 131072]

    for size in sizes:
        layout = _agentic_layout(size)
        print("-" * 75)
        print(f"\nLayout: {size:,} tokens, {len(layout.blocks)} blocks")
        for label, cfg in configs:
            _print_summary(label, cfg, layout)

    print()
    print("  Note: All figures are analytical cost estimates on CPU.")
    print("  No GPU speedup is claimed or implied.")
    print()


if __name__ == "__main__":
    benchmark()
