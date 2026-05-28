# Dynamic Block Scoring

## Purpose

Not all ATTEND blocks come with reliable scores from the runtime. Dynamic
scoring provides a lightweight heuristic to rank candidate blocks at
decode time using query-to-block cosine similarity.

## Design

`BlockScorer` computes the cosine similarity between a query
representation and each candidate block's key representation:

```
score = cosine_similarity(query_rep, block_key_rep)
```

Blocks with scores above `threshold` are selected; those below are
treated as SKIP.

## Limitations

- This is a **heuristic**, not a trained routing model.
- The query representation is currently a mean-pool of the query tensor.
  This loses position-specific and head-specific information.
- Block key representations can be: mean-pool of the block's keys, a
  learned embedding, or a zero vector for orthogonality testing.
- Cosine similarity is a simple proxy that may not reflect true
  attention importance.
- No calibration, training, or accuracy evaluation has been performed.

## Usage

```python
from intent_attention import BlockScorer

scorer = BlockScorer()
scores = scorer.score_blocks(
    query,            # (1, n_heads, q_len, d_head)
    block_reps,       # list of (d_head,) tensors, one per ATTEND block
    threshold=0.5,
)
```

## Relation to the Full System

Dynamic scoring is an optional refinement layer between block policy
selection and the attention reference. It models a control-plane surface
that a future GPU runtime or kernel could consume to make per-step
selection decisions without host-device round trips.
