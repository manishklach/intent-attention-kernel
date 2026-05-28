"""
End-to-end demo: KV Block Router → kernel metadata → attention execution.

CPU-only. No GPU required. No speedup claimed.
"""

import torch

from intent_attention import (
    BlockDecision,
    BlockLayout,
    BlockPolicy,
    BlockRouter,
    KVPrecision,
    RouterConfig,
    SemanticBlock,
    compute_block_scores,
    dense_attention,
    routing_to_kernel_metadata,
    semantic_block_attention,
)
from intent_attention.intent_quant import IntentQuantizer
from intent_attention.intent_quant_attention import (
    intent_quant_attention_reference,
)


def _make_agentic_layout(total_tokens: int) -> BlockLayout:
    return BlockLayout([
        SemanticBlock("system_prompt",     0,      128,      BlockPolicy.ALWAYS),
        SemanticBlock("global_memory",     128,    256,      BlockPolicy.GLOBAL),
        SemanticBlock("retrieved_doc_a",   256,    768,      BlockPolicy.ATTEND, score=0.85),
        SemanticBlock("retrieved_doc_b",   768,    1280,     BlockPolicy.ATTEND, score=0.30),
        SemanticBlock("tool_output",       1280,   1536,     BlockPolicy.ATTEND, score=0.60),
        SemanticBlock("scratchpad",        1536,   1792,     BlockPolicy.SKIP),
        SemanticBlock("recent_context",    1792,   2048,     BlockPolicy.RECENT),
    ])


def _make_block_representations(layout, d=64):
    """Synthetic random block representations for query-to-block scoring."""
    reps = {}
    for block in layout.blocks:
        reps[block.name] = torch.randn(d)
    return reps


def demo():
    print("=" * 75)
    print("  KV Block Router — End-to-End Demo (CPU Only)")
    print("=" * 75)
    print()

    total_tokens = 2048
    B, H, Q_LEN, D = 1, 4, 8, 64
    PAGE_SIZE = 16

    # 1. Generate synthetic agentic layout
    layout = _make_agentic_layout(total_tokens)
    print(f"1. Layout: {len(layout.blocks)} blocks, {total_tokens} tokens")

    # 2. Create query vector and block representations
    query_vector = torch.randn(D)
    block_reps = _make_block_representations(layout, D)
    scores = compute_block_scores(query_vector, block_reps)
    print(f"2. Query-to-block scores: {scores}")

    # 3. Route blocks using BlockRouter
    config = RouterConfig(
        top_k_blocks=4,
        score_threshold=0.35,
        memory_pressure=0.5,
        prefetch_top_k=4,
    )
    router = BlockRouter(config)
    routed = router.route_layout(layout, total_tokens, query_vector, block_reps)
    print(f"3. Routed {len(routed)} blocks")

    # 4. Convert routed blocks to kernel metadata
    meta = routing_to_kernel_metadata(routed, PAGE_SIZE)
    print(f"4. Kernel metadata:")
    print(f"   selected_page_ids: {meta['selected_page_ids'][:10]}{'...' if len(meta['selected_page_ids']) > 10 else ''}")
    print(f"   selected_block_names: {meta['selected_block_names']}")
    print(f"   skipped_block_names: {meta['skipped_block_names']}")
    print(f"   prefetch_page_ids: {meta['prefetch_page_ids']}")

    # 5. Routing summary
    summary = router.routing_summary(routed)
    print(f"5. Routing summary:")
    print(f"   selected: {summary['selected_blocks']} blocks, {summary['selected_tokens']} tokens")
    print(f"   skipped:  {summary['skipped_blocks']} blocks, {summary['skipped_tokens']} tokens")
    print(f"   FP16 KV:  {summary['estimated_fp16_mb']} MB")
    print(f"   Routed:   {summary['estimated_quant_mb']} MB")
    print(f"   Saved:    {summary['bytes_saved_pct']}%")
    print(f"   Precision distribution: {summary['precision_distribution']}")

    # 6. Run selected-block attention reference
    q = torch.randn(B, H, Q_LEN, D)
    k = torch.randn(B, H, total_tokens, D)
    v = torch.randn(B, H, total_tokens, D)

    out_sa, debug_sa = semantic_block_attention(q, k, v, layout, return_debug=True)
    print(f"6. Selected-block attention: output shape {tuple(out_sa.shape)}, selected {debug_sa['selected_token_count']} tokens")

    # 7. Run IntentQuant attention reference
    quantizer = IntentQuantizer(memory_pressure=config.memory_pressure)
    out_iq, debug_iq = intent_quant_attention_reference(
        q, k, v, layout, quantizer, return_debug=True
    )
    print(f"7. IntentQuant attention:    output shape {tuple(out_iq.shape)}")
    print(f"   Cos sim vs FP16-selected: {debug_iq['output_cosine_vs_fp16_selected']:.5f}")
    print(f"   MSE vs FP16-selected:     {debug_iq['output_mse_vs_fp16_selected']:.6e}")

    # 8. Reasons for each block
    print(f"8. Routing reasons:")
    for name, reason in meta["reasons_by_block"].items():
        routed_block = [b for b in routed if b.name == name][0]
        decision = routed_block.decision.value
        print(f"   {name:25s} -> {decision:20s} ({reason})")

    print()
    print("  Warning:")
    print("  This is a CPU-only research prototype. No GPU speedup is claimed.")
    print("  No model quality or perplexity preservation is claimed.")
    print("  All routing decisions are heuristic, not learned.")
    print()


if __name__ == "__main__":
    demo()
