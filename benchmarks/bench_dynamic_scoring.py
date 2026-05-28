from __future__ import annotations

import torch

from intent_attention.block_metadata import BlockPolicy
from intent_attention.block_scorer import BlockScorer
from intent_attention.synthetic_traces import generate_agentic_layout


def main() -> None:
    torch.set_num_threads(1)

    n_traces = 20
    threshold = 0.5
    head_dim = 64

    total_attend = 0
    total_above = 0
    total_below = 0

    print(f"Evaluating {n_traces} random agentic traces (threshold={threshold})...")
    print()

    for seed in range(n_traces):
        layout = generate_agentic_layout(1024, seed=seed)
        attend_blocks = [b for b in layout.blocks if b.policy == BlockPolicy.ATTEND]
        if not attend_blocks:
            continue

        q = torch.randn(1, 4, 32, head_dim)
        k = torch.randn(1, 4, 1024, head_dim)

        key_reps = []
        for block in attend_blocks:
            rep = k[..., block.start : block.end, :].mean(dim=-2).mean(dim=(0, 1))
            key_reps.append(rep)

        scorer = BlockScorer()
        scores = scorer.score_blocks(q, key_reps, threshold)

        n_above = sum(1 for s in scores if s >= threshold)
        n_below = len(scores) - n_above

        total_attend += len(scores)
        total_above += n_above
        total_below += n_below

    pct_above = 100.0 * total_above / total_attend if total_attend else 0.0
    pct_below = 100.0 * total_below / total_attend if total_attend else 0.0

    print(f"Dynamic scoring summary ({n_traces} synthetic traces)")
    print("=" * 50)
    print(f"  Total ATTEND blocks encountered:  {total_attend}")
    print(f"  Attended   (score >= {threshold}):  {total_above:>6}  ({pct_above:.1f}%)")
    print(f"  Skipped    (score <  {threshold}):  {total_below:>6}  ({pct_below:.1f}%)")


if __name__ == "__main__":
    main()
