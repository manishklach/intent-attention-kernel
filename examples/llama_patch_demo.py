from __future__ import annotations

from typing import Optional

import torch

from intent_attention.block_metadata import BlockLayout, BlockPolicy, SemanticBlock
from intent_attention.cost_model import savings_report
from intent_attention.hf_patch import patch_model


def build_layout(seq_len: int) -> BlockLayout:
    """Mark first 20% as ALWAYS, next 30% as ATTEND (high score), rest SKIP."""
    a_end = max(1, seq_len * 20 // 100)
    b_end = max(a_end + 1, seq_len * 50 // 100)
    return BlockLayout([
        SemanticBlock("system", 0, a_end, BlockPolicy.ALWAYS),
        SemanticBlock("retrieved", a_end, b_end, BlockPolicy.ATTEND, score=0.9),
        SemanticBlock("filler", b_end, seq_len, BlockPolicy.SKIP),
    ])


def main() -> None:
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        print("transformers not installed — skipping demo.")
        return

    model_name = "hf-internal-testing/tiny-random-GPT2"
    print(f"Loading {model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()

    dummy_input = tokenizer("Hello, world! This is a long context for testing.", return_tensors="pt")
    seq_len = dummy_input.input_ids.size(1)
    print(f"  Input sequence length: {seq_len}")

    with torch.no_grad():
        orig_output = model(**dummy_input)
        orig_logits = orig_output.logits
        print(f"Original logits — shape: {orig_logits.shape},"
              f"  NaNs: {torch.isnan(orig_logits).any().item()},"
              f"  Infs: {torch.isinf(orig_logits).any().item()}")

    layout = build_layout(seq_len)
    print(f"  Layout: {layout.summary()}")

    def layout_fn(layer_idx: int) -> Optional[BlockLayout]:
        return layout

    print("\nPatching model with semantic block attention...")
    patch_model(model, layout_fn, verbose=True)

    with torch.no_grad():
        patched_output = model(**dummy_input)
        patched_logits = patched_output.logits
        print(f"\nPatched logits  — shape: {patched_logits.shape},"
              f"  NaNs: {torch.isnan(patched_logits).any().item()},"
              f"  Infs: {torch.isinf(patched_logits).any().item()}")

    total = seq_len
    selected = layout.selected_token_count()
    report = savings_report(
        batch=1, heads=1, query_tokens=1,
        total_kv_tokens=total,
        selected_kv_tokens=selected,
        head_dim=1,
    )
    print(f"\nAnalytical savings per layer:")
    print(f"  KV tokens: {total} -> {selected}")
    print(f"  FLOPs saved: {report['flops_saved_pct']:.1f}%")
    print(f"  KV bytes saved: {report['kv_bytes_saved_pct']:.1f}%")

    assert torch.isfinite(patched_logits).all(), "Patched output contains NaN/Inf!"
    print("\nSUCCESS: patched forward pass completed without NaN/Inf.")


if __name__ == "__main__":
    main()
