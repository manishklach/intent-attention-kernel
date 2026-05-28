Release v0.3.0: Fused Selected-Quant Decode Kernel and Research Artifact Polish

## New Features

### Fused Selected-Quant Decode Kernel (Core Innovation)
- src/intent_attention/fused_selected_quant_decode.py: Triton kernel that fuses:
  - Runtime semantic page selection from BlockRouter metadata
  - Mixed-precision (FP16/INT8/SKIP) page loading with per-page dequant
  - Decode-step attention computation in a single GPU pass
- tests/test_fused_selected_quant_decode.py: 15 tests covering correctness, precision paths, edge cases
- benchmarks/bench_fused_selected_quant_decode.py: Benchmark vs SDPA baseline with dry-run support
- docs/fused_selected_quant_kernel.md: Design document explaining novelty and integration
- Updated __init__.py to export new public API

### Documentation & Reproducibility
- docs/research_summary.md: Thesis, problem, proposed interface, current prototype, evidence, limitations, next milestones
- docs/reproducibility.md: Exact commands for CPU validation, dry-run, LLM quality, and GPU benchmark
- docs/results_template.md: Markdown tables for LLM quality, GPU latency, and router benchmark results
- Updated docs/results_cpu.md: Added sample output for fused selected-quant benchmark
- Updated README.md:
  - Added "Current Status" section near top
  - Renamed "Five Pillars" → "System Components" with clean 1-9 numbering
  - Fixed tables to proper Markdown format
  - Added links to all key documentation files
  - Added new system component for the fused kernel

### Validation & Benchmarking (Previous Work)
- experiments/llm_quality_validation.py: Proxy perplexity validation on small HF models
- experiments/gpu_decode_benchmark.py: GPU decode-step attention benchmark across backends
- docs/validation_plan.md: Quality validation ladder and publishable-evidence bar
- docs/gpu_benchmarking.md: Fair baselines, hardware matrix, T4 caveat
- tests/test_validation_imports.py: 6 import/smoke tests for experiments

## Key Improvements

### System Components (Now 9 total):
1. Semantic KV Block Selection
2. KV Block Router (runtime-to-kernel policy layer)  
3. Dynamic Block Scoring
4. IntentQuant-KV (mixed-precision quantization)
5. IntentQuant Attention Reference (per-block quantized attention)
6. Speculative KV Prefetch Simulation
7. Optional Triton Decode-Attention Prototype
8. Validation Harness (LLM quality + GPU benchmark)
9. Fused Selected-Quant Decode Kernel (NEW)

### Validation Progress
- 156 unit and integration tests passing
- CPU-first design: all tests/run without CUDA or Triton
- Dry-run modes for experiments allow CI validation without downloads/GPU
- No GPU speedup or model quality claims made anywhere
- All caution language preserved

## Technical Highlights

The fused selected-quant decode kernel is the missing execution-layer backend that connects:
Agentic Runtime → KV Block Router → [Fused Kernel] → Attention Output

It fuses three innovations:
1. Runtime semantic page selection: Kernel reads per-page metadata at load time
2. Mixed-precision fused dequant: Different pages loaded at different precisions in one kernel
3. Skip-page support: Pages marked SKIP generate zero memory traffic

This is the first implementation connecting semantic runtime intent directly to GPU execution in this repository.

## Validation
- py_compile: All source files pass
- pytest: 156 passed, 5 skipped (Triton/CUDA without GPU)
- All benchmarks run successfully
- Dry-run validation for all experiments functional

This release establishes the kernel as a research prototype for hardware experimentation. No GPU speedups or model quality claims are made.