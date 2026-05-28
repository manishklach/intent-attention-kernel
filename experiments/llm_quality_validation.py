"""
LLM quality validation harness for IntentQuant-KV policies.

Purpose:
    Run a small HuggingFace causal LM on a text dataset, compute baseline
    perplexity, and compare against simulated IntentQuant-KV KV-cache
    quantization using ``fake_quantize_tensor`` / ``fake_dequantize_tensor``
    on the ``past_key_values`` after prefill.

This is a CPU/GPU research experiment.  No GPU speedup claim is made.
No production-quality accuracy guarantee is made.

CLI usage::

    python experiments/llm_quality_validation.py --dry-run
    python experiments/llm_quality_validation.py \\
        --model HuggingFaceTB/SmolLM2-135M \\
        --max-samples 32 --max-length 512 \\
        --policies baseline conservative balanced aggressive
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import torch


# ------------------------------------------------------------------ #
#  Dry-run text corpus (no ``datasets`` dependency)
# ------------------------------------------------------------------ #

_FALLBACK_TEXT = (
    "The quick brown fox jumps over the lazy dog. "
    "A journey of a thousand miles begins with a single step. "
    "To be or not to be, that is the question. "
    "All that glitters is not gold. "
    "The only thing we have to fear is fear itself. "
    "Ask not what your country can do for you. "
    "I have a dream that one day this nation will rise up. "
    "In the beginning God created the heavens and the earth. "
    "It was the best of times, it was the worst of times. "
    "Call me Ishmael. "
    "It is a truth universally acknowledged. "
    "Happy families are all alike; every unhappy family is unhappy in its own way. "
    "The universe is under no obligation to make sense to you. "
    "Science is a way of thinking much more than it is a body of knowledge. "
    "The important thing is not to stop questioning. "
    "Equations are more important to me, because politics is for the present, "
    "but an equation is something for eternity. "
) * 50  # ~7K tokens


def _load_text(max_samples: int, max_length: int) -> List[str]:
    """Load text samples — from ``datasets`` if available, else fallback."""

    try:
        from datasets import load_dataset

        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation", trust_remote_code=True)
        texts = [row["text"] for row in ds if row["text"].strip()]
        if not texts:
            raise ValueError("wikitext dataset returned empty samples")
        print(f"  [data] Loaded {len(texts)} samples from wikitext-2-raw-v1")
    except Exception as e:
        print(f"  [data] datasets unavailable or failed ({e}); using fallback text")
        texts = _FALLBACK_TEXT.split(". ")

    texts = texts[:max_samples]
    # Truncate each sample to approximate max_length tokens
    texts = [t[: max_length * 4] for t in texts]
    return texts


# ------------------------------------------------------------------ #
#  Model helpers
# ------------------------------------------------------------------ #

_SUPPORTED_MODELS = [
    "HuggingFaceTB/SmolLM2-135M",
    "HuggingFaceTB/SmolLM2-360M",
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
]


def _load_model_and_tokenizer(
    model_name: str, device: str
) -> Tuple[Any, Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"  [model] Loading {model_name} on {device} ...")
    t0 = time.time()

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32 if device == "cpu" else torch.float16,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()

    dt = time.time() - t0
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  [model] Loaded {n_params/1e6:.1f}M params in {dt:.1f}s")
    return model, tokenizer


# ------------------------------------------------------------------ #
#  KV-cache quantization proxy
# ------------------------------------------------------------------ #


def _quantize_past_key_values(
    past_key_values: Tuple[Tuple[torch.Tensor, torch.Tensor], ...],
    qkv: Any,
) -> Tuple[Tuple[torch.Tensor, torch.Tensor], ...]:
    """Apply ``fake_quantize_tensor`` to every K,V tensor in
    ``past_key_values`` using a per-layer precision from *qkv*.

    Returns a new ``past_key_values`` tuple with the same structure.
    """
    from intent_attention.intent_quant import fake_dequantize_tensor, fake_quantize_tensor

    new_pkv: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for i, (k, v) in enumerate(past_key_values):
        pol = qkv["layer_policies"][i]
        k_q, k_meta = fake_quantize_tensor(k, pol)
        v_q, v_meta = fake_quantize_tensor(v, pol)
        k_dq = fake_dequantize_tensor(k_q, k_meta)
        v_dq = fake_dequantize_tensor(v_q, v_meta)
        new_pkv.append((k_dq, v_dq))
    return tuple(new_pkv)


# ------------------------------------------------------------------ #
#  Policy definitions
# ------------------------------------------------------------------ #


def _policy_configs() -> Dict[str, Dict[str, Any]]:
    return {
        "baseline": {
            "memory_pressure": 0.0,
            "high_score_threshold": 0.9,
            "medium_score_threshold": 0.6,
            "preserve_recent": True,
            "preserve_global": True,
            "label": "no quantization, 0% bytes saved",
        },
        "conservative": {
            "memory_pressure": 0.2,
            "high_score_threshold": 0.8,
            "medium_score_threshold": 0.5,
            "preserve_recent": True,
            "preserve_global": True,
            "label": "preserve global/recent, FP16/FP8/INT8",
        },
        "balanced": {
            "memory_pressure": 0.5,
            "high_score_threshold": 0.75,
            "medium_score_threshold": 0.4,
            "preserve_recent": True,
            "preserve_global": True,
            "label": "INT8/INT4_RESIDUAL for lower-score blocks",
        },
        "aggressive": {
            "memory_pressure": 0.8,
            "high_score_threshold": 0.85,
            "medium_score_threshold": 0.5,
            "preserve_recent": True,
            "preserve_global": True,
            "label": "more INT4/SKIP proxy",
        },
    }


def _apply_policy_to_all_layers(model: Any, policy_kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Create a per-layer ``KVPrecision`` policy for every transformer
    layer using the same IntentQuantizer settings."""
    from intent_attention.block_metadata import BlockLayout, BlockPolicy, SemanticBlock
    from intent_attention.intent_quant import KVPrecision, IntentQuantizer

    quantizer = IntentQuantizer(**policy_kwargs)

    n_layers = model.config.num_hidden_layers
    layout = BlockLayout([
        SemanticBlock("k_proj", 0, 1, BlockPolicy.ATTEND, score=0.5),
    ])

    layer_policies: List[KVPrecision] = []

    for _ in range(n_layers):
        pol = quantizer.assign_block_precision(layout.blocks[0]).precision
        layer_policies.append(pol)

    return {
        "layer_policies": layer_policies,
        "quantizer": quantizer,
    }


# ------------------------------------------------------------------ #
#  Perplexity evaluation
# ------------------------------------------------------------------ #


@torch.no_grad()
def evaluate_perplexity(
    model: Any,
    tokenizer: Any,
    texts: List[str],
    max_length: int,
    policy: Optional[Dict[str, Any]] = None,
    device: str = "cpu",
) -> Dict[str, float]:
    """Compute perplexity over *texts*.

    If *policy* is provided, quantize ``past_key_values`` after each
    prefill step using the per-layer policies.
    """
    total_loss = 0.0
    total_tokens = 0
    n_batches = 0

    for text in texts:
        enc = tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(device)
        if input_ids.size(1) < 2:
            continue

        # Forward pass (no KV cache modification for baseline)
        if policy is None:
            outputs = model(input_ids, labels=input_ids)
            loss = outputs.loss
            n_tokens = (input_ids != tokenizer.pad_token_id).sum().item()
            total_loss += loss.item() * n_tokens
            total_tokens += n_tokens
            n_batches += 1
            continue

        # --- Quantised path ------------------------------------------
        # We prefill the first token, then for each subsequent token we
        # quantize the growing past_key_values before the forward pass.
        seq_len = input_ids.size(1)
        chunk_loss = 0.0
        chunk_tokens = 0

        # First token
        past_key_values = None
        for pos in range(seq_len - 1):
            inp = input_ids[:, pos: pos + 1]
            if past_key_values is not None:
                past_key_values = _quantize_past_key_values(past_key_values, policy)

            out = model(inp, past_key_values=past_key_values, use_cache=True, labels=inp)
            loss = out.loss
            past_key_values = out.past_key_values

            n_tok = (inp != tokenizer.pad_token_id).sum().item()
            chunk_loss += loss.item() * n_tok
            chunk_tokens += n_tok

        if chunk_tokens > 0:
            total_loss += chunk_loss
            total_tokens += chunk_tokens
            n_batches += 1

    if total_tokens == 0:
        return {"perplexity": float("inf"), "avg_loss": 0.0, "tokens": 0}

    avg_loss = total_loss / total_tokens
    ppl = math.exp(avg_loss) if avg_loss < 100 else float("inf")
    return {"perplexity": ppl, "avg_loss": avg_loss, "tokens": total_tokens, "batches": n_batches}


# ------------------------------------------------------------------ #
#  Bytes-saved estimation
# ------------------------------------------------------------------ #


def _estimate_bytes_saved(policy: Optional[Dict[str, Any]], n_layers: int, d_head: int = 64) -> float:
    """Return rough % bytes saved vs fp16 baseline for one K,V pair per layer."""
    from intent_attention.intent_quant import _BYTES_PER_VALUE

    if policy is None:
        return 0.0

    fp16_total = 0
    quant_total = 0.0
    for pol in policy["layer_policies"]:
        bpe = _BYTES_PER_VALUE.get(pol, 2.0)
        fp16_total += 2 * d_head * 2  # K+V per-layer
        quant_total += 2 * d_head * bpe

    if fp16_total == 0:
        return 0.0
    return (1.0 - quant_total / fp16_total) * 100.0


# ------------------------------------------------------------------ #
#  Main
# ------------------------------------------------------------------ #


def main():
    parser = argparse.ArgumentParser(
        description="LLM quality validation for IntentQuant-KV policies"
    )
    parser.add_argument(
        "--model",
        default="HuggingFaceTB/SmolLM2-135M",
        choices=_SUPPORTED_MODELS,
        help="HuggingFace causal LM",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=8,
        help="Number of text samples to evaluate (default: 8)",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=256,
        help="Max token length per sample (default: 256)",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to run on (default: cpu)",
    )
    parser.add_argument(
        "--policies",
        nargs="+",
        default=["baseline"],
        choices=["baseline", "conservative", "balanced", "aggressive"],
        help="Policies to evaluate",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be done without loading models or data",
    )
    args = parser.parse_args()

    print("=" * 72)
    print("  IntentQuant-KV — LLM Quality Validation")
    print("=" * 72)
    print()

    if args.dry_run:
        print("DRY RUN — no models or data will be loaded")
        print()
        print(f"  Model:         {args.model}")
        print(f"  Device:        {args.device}")
        print(f"  Max samples:   {args.max_samples}")
        print(f"  Max length:    {args.max_length}")
        print(f"  Policies:      {args.policies}")
        print()
        print("  This would:")
        print(f"    1. Load {args.model}")
        print(f"    2. Load wikitext-2-raw-v1 or fallback text ({args.max_samples} samples)")
        print(f"    3. Run baseline perplexity")
        for p in args.policies:
            if p == "baseline":
                continue
            print(f"    4. Run {p} policy (simulated KV-cache quantization)")
        print("    5. Print comparison table")
        print()
        sys.exit(0)

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available. Falling back to CPU.")
        args.device = "cpu"

    # Load model
    try:
        model, tokenizer = _load_model_and_tokenizer(args.model, args.device)
    except Exception as e:
        print(f"  [error] Failed to load model: {e}")
        print("  Try running with --dry-run to verify configuration.")
        sys.exit(1)

    n_layers = model.config.num_hidden_layers
    d_head = getattr(model.config, "hidden_size", 512) // getattr(model.config, "num_attention_heads", 8)

    # Load text
    texts = _load_text(args.max_samples, args.max_length)
    print(f"  [data] {len(texts)} samples loaded")

    # Evaluate policies
    all_policies = _policy_configs()
    results: List[Dict[str, Any]] = []

    for pname in args.policies:
        if pname == "baseline":
            print(f"\n  Evaluating baseline (no quantization) ...")
            t0 = time.time()
            base = evaluate_perplexity(
                model, tokenizer, texts, args.max_length, policy=None, device=args.device
            )
            dt = time.time() - t0
            results.append(
                {
                    "policy": "baseline",
                    "memory_pressure": 0.0,
                    "bytes_saved_pct": 0.0,
                    "perplexity": base["perplexity"],
                    "ppl_delta_pct": 0.0,
                    "avg_loss": base["avg_loss"],
                    "tokens": base["tokens"],
                    "time_s": round(dt, 1),
                    "notes": "no quantization",
                }
            )
            print(f"    Perplexity: {base['perplexity']:.4f}  "
                  f"(loss={base['avg_loss']:.4f}, {base['tokens']} tokens, {dt:.1f}s)")
            continue

        cfg = all_policies[pname]
        policy = _apply_policy_to_all_layers(model, cfg)
        bytes_saved = _estimate_bytes_saved(policy, n_layers, d_head)
        print(f"\n  Evaluating {pname} (mp={cfg['memory_pressure']}, ~{bytes_saved:.0f}% bytes saved) ...")

        t0 = time.time()
        result = evaluate_perplexity(
            model, tokenizer, texts, args.max_length, policy=policy, device=args.device
        )
        dt = time.time() - t0

        base_ppl = results[0]["perplexity"]
        ppl_delta = ((result["perplexity"] - base_ppl) / max(base_ppl, 1e-8)) * 100.0
        results.append(
            {
                "policy": pname,
                "memory_pressure": cfg["memory_pressure"],
                "bytes_saved_pct": round(bytes_saved, 1),
                "perplexity": result["perplexity"],
                "ppl_delta_pct": round(ppl_delta, 2),
                "avg_loss": result["avg_loss"],
                "tokens": result["tokens"],
                "time_s": round(dt, 1),
                "notes": cfg["label"],
            }
        )
        print(f"    Perplexity: {result['perplexity']:.4f}  "
              f"(Δ={ppl_delta:+.2f}%, loss={result['avg_loss']:.4f}, "
              f"{result['tokens']} tokens, {dt:.1f}s)")

    # Print table
    print()
    print("-" * 72)
    print(f"  Results — {args.model}")
    print("-" * 72)
    header = f"{'Policy':<16} {'MP':>5} {'Save%':>7} {'PPL':>9} {'ΔPPL%':>8} {'Loss':>8} {'Tokens':>8} {'Time(s)':>8}"
    print(header)
    print("-" * len(header))
    for r in results:
        ppl_str = f"{r['perplexity']:.2f}" if r['perplexity'] != float('inf') else "inf"
        print(
            f"{r['policy']:<16} "
            f"{r['memory_pressure']:>5.1f} "
            f"{r['bytes_saved_pct']:>6.1f}% "
            f"{ppl_str:>9} "
            f"{r['ppl_delta_pct']:>+7.2f}% "
            f"{r['avg_loss']:>8.4f} "
            f"{r['tokens']:>8} "
            f"{r['time_s']:>7.1f}s"
        )
    print("-" * 72)
    print()

    print("  This is a proxy experiment using fake quantization on")
    print("  past_key_values. It does not prove production KV")
    print("  quantization quality. No GPU speedup is claimed.")
    print()


if __name__ == "__main__":
    main()
