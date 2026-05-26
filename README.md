# Intent Attention Kernel

**Semantic Block Attention for Agentic Long-Context Inference**

## Motivation
Long-context agentic inference should not treat context as a flat token stream. The runtime often knows that tokens belong to semantic regions such as system prompts, recent conversation, retrieved documents, tool outputs, memory summaries, and scratchpad.

## Why long-context is not flat
Standard attention computes scores over all past tokens. However, many blocks can be safely ignored to save memory bandwidth and compute.

## Core idea
This repo represents semantic regions as blocks with attention policies, then computes attention only over selected KV blocks.

## Architecture
```text
Runtime context
  -> semantic block metadata
  -> selected KV blocks
  -> attention over selected blocks
  -> output
```

## Quickstart
```bash
pip install -e .
pytest
python benchmarks/bench_cpu_reference.py
```

## Disclaimer
This repo does not claim GPU speedups yet. It is a simulator-first prototype that proves the interface, correctness, and cost model before implementing an optimized Triton/CUDA kernel.
