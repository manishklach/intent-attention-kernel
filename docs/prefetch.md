# Speculative KV Prefetch Simulation

## Purpose

During agentic decode, consecutive steps often attend to overlapping
sets of KV pages (e.g., repeating system prompt lookups, recent context
reads, or retrieved document re-reads). A prefetcher can predict the
next step's likely KV pages and launch asynchronous loads to hide HBM
latency.

## Current Approach

`BlockPrefetcher` maintains a history of page access patterns and
predicts the next step's pages:

1. Track which physical KV pages were accessed at each decode step.
2. Use a simple predictor (e.g., last-access, frequency-based, or
   Markov-chain) to rank likely next-step pages.
3. Compare predicted pages against actual pages accessed at the next
   step to compute hit rate.
4. The prefetch simulation measures **hit rate** and **fraction of
   latency theoretically hidden**, not real wall-clock speedup.

## Key Constraints

- **Prefetch must never affect correctness**: prefetched pages may be
  stale or incorrect. The kernel should always wait for the authoritative
  page list before proceeding.
- **Prefetch is a latency-hiding hint**: it does not change which pages
  are attended to. It only attempts to start page loads earlier.
- **Over-prefetch wastes bandwidth**: predicting too many pages can
  consume memory bandwidth that would otherwise be used for useful loads.

## Limitations

- Hit rate is simulated on synthetic decode traces, not real model
  inference.
- No actual GPU page-load latency measurement.
- No asynchronous load (e.g., CUDA memcpy async or CDP) is implemented.
- The predictor is a simple heuristic — no learned or ML-based predictor
  is used.
- Prefetch benefit is highly workload-dependent: workloads with random
  page access patterns will see near-zero hit rates.

## Usage

```bash
python benchmarks/bench_prefetch.py
```

This simulates decode-step page prediction across varying workload
volatility levels and reports hit rates and latency-hiding percentages.

## Relation to the Full System

Prefetch is an optional optimization layer that sits below block
selection and above the GPU kernel. It models a future path where the
runtime communicates predicted page IDs to a GPU kernel's launch
parameters, enabling the kernel to begin loading pages before the
authoritative block list is fully resolved.
