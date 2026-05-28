# Attention Layout

A layout is a list of `SemanticBlock` objects.  Each block has:

| Field   | Type            | Description                        |
|---------|-----------------|------------------------------------|
| `name`  | `str`           | Human-readable label               |
| `start` | `int`           | First token index (inclusive)      |
| `end`   | `int`           | Last token index (exclusive)       |
| `policy`| `BlockPolicy`   | Attention policy for this block    |
| `score` | `float \| None` | Relevance score (required for ATTEND) |

## Policies

| Policy    | Selected | Description                                     |
|-----------|----------|-------------------------------------------------|
| `ALWAYS`  | Yes      | Always attend to this block.                    |
| `ATTEND`  | Yes      | Attend based on a relevance `score`.            |
| `RECENT`  | Yes      | Attend — always selected (like a sliding window).|
| `GLOBAL`  | Yes      | Attend — always selected (like global tokens).  |
| `SKIP`    | No       | Ignore this block entirely.                     |

## Validation Rules

`BlockLayout.validate(total_tokens)` checks:

- Block names are non-empty.
- `start >= 0` and `end > start`.
- `end <= total_tokens`.
- Blocks are sorted by `start` (ascending).
- Blocks do not overlap.
- `ATTEND` blocks have a non-`None` `score`.
