from __future__ import annotations

import random

from intent_attention.block_metadata import BlockLayout, BlockPolicy
from intent_attention.prefetch import BlockPrefetcher
from intent_attention.synthetic_traces import generate_agentic_layout


def simulate_decode(
    layout: BlockLayout,
    n_steps: int = 100,
    volatility: float = 0.0,
    seed: int = 0,
) -> float:
    """Run an autoregressive decode simulation and return the average prefetch hit rate.

    *volatility* controls how many ATTEND-eligible blocks change status per
    step (0 = stable, 0.5 = 50 % of eligible blocks flip each step).
    """
    rng = random.Random(seed)

    # Collect the IDs of blocks that *could* be selected (non-SKIP).
    eligible_ids = [
        i for i, b in enumerate(layout.blocks) if b.policy != BlockPolicy.SKIP
    ]
    if not eligible_ids:
        return float("nan")

    # Start with all eligible blocks active.
    active: set[int] = set(eligible_ids)

    # Generate the selection for each step.
    actuals: list[list[int]] = []
    for step in range(n_steps):
        if step > 0 and volatility > 0:
            for bid in eligible_ids:
                if rng.random() < volatility:
                    if bid in active:
                        active.remove(bid)
                    else:
                        active.add(bid)
        actuals.append(sorted(active))

    # Run prefetcher over the trace.
    prefetcher = BlockPrefetcher(history_size=4, min_frequency=3)
    hit_rates: list[float] = []

    for step in range(n_steps - 1):
        predicted = prefetcher.predict_next(actuals[step])
        prefetcher.record(actuals[step])
        actual_next = actuals[step + 1]

        if len(actual_next) == 0:
            hit = 1.0 if len(predicted) == 0 else 0.0
        else:
            hits = len(set(predicted) & set(actual_next))
            hit = hits / len(actual_next)
        hit_rates.append(hit)

    return sum(hit_rates) / len(hit_rates) if hit_rates else 0.0


def main() -> None:
    total_tokens = 1024
    n_steps = 100
    volatilities = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

    layout = generate_agentic_layout(total_tokens, seed=42)

    print("Speculative KV Block Prefetch — Decode Simulation")
    print(
        f"Layout: {total_tokens} tokens, {len(layout.blocks)} blocks, {n_steps} steps"
    )
    print()
    header = f"{'Volatility':>12}  {'Hit Rate':>10}  {'Latency Hidden %':>18}"
    print(header)
    print("-" * len(header))

    for vol in volatilities:
        hit_rate = simulate_decode(layout, n_steps=n_steps, volatility=vol, seed=0)
        hidden_pct = hit_rate * 100.0
        print(f"{vol:>12.1f}  {hit_rate:>10.3f}  {hidden_pct:>17.1f}%")


if __name__ == "__main__":
    main()
