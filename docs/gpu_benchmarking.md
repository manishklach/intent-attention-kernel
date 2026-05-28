# GPU Benchmarking Guide

**Fair baselines for decode-step attention on NVIDIA GPUs.**

---

## Why GPU benchmarking matters

The repo's analytical cost model and CPU benchmarks cannot predict GPU
performance. GPU attention is dominated by memory bandwidth (HBM), kernel
launch overhead, and tensor-core utilization — none of which are captured
by CPU timing. Real GPU measurements on target hardware are the only
reliable signal.

---

## Fair baselines

| Backend | When to use | Notes |
|---|---|---|
| PyTorch SDPA | Always available | Eager or flash via `torch.backends.cuda.sdp_kernel()`. The default safe baseline. |
| SelectedKV gather + SDPA | Always available | Measures gather overhead + reduced-math attention. Not a fused kernel. |
| Triton IntentQuant | Only with Triton | Prototype, not tuned. Expect higher latency than SDPA for most configs. |
| FlashAttention-2 | CC >= 8.0 only | Requires Ampere/Ada/Hopper. **Do not use on T4 (Turing).** |
| xFormers | Only if installed | Memory-efficient attention. Known to work well on Turing/Ampere. |

---

## Hardware matrix

| GPU | Arch | CC | FA2 support | Recommended baseline |
|---|---|---|---|---|
| T4 | Turing | 7.5 | No | PyTorch SDPA, xFormers, flash-attention-turing |
| V100 | Volta | 7.0 | No | PyTorch SDPA |
| A10G / A100 | Ampere | 8.0 | Yes | FA2, SDPA, xFormers |
| RTX 3090 / 4090 | Ampere/Ada | 8.6/8.9 | Yes | FA2, SDPA, xFormers |
| L4 / L40S | Ada | 8.9 | Yes | FA2, SDPA, xFormers |
| H100 | Hopper | 9.0 | Yes | FA2, SDPA |

---

## T4 caveat

**Do not compare against FlashAttention-2 on T4 as if it were a supported
FA2 baseline.** T4 is Turing (CC 7.5). FlashAttention-2 requires Ampere or
later (CC >= 8.0) for its fast kernel paths. Running FA2 on T4 may fall
back to an unoptimized path or raise an error.

If you must benchmark on T4:

- Use `torch.nn.functional.scaled_dot_product_attention` with
  `enable_flash=False` to force the eager or memory-efficient path.
- Use `xformers.ops.memory_efficient_attention` if installed.
- Optionally try `flash-attention-turing` if available (separate package).

Clearly label any T4 results: *"Measured on T4 (Turing). FlashAttention-2
baseline not available. SDPA eager/MEM_EFF used instead."*

---

## How to run the benchmark

```bash
# See available options
python experiments/gpu_decode_benchmark.py --dry-run

# Run on a single GPU with standard decode config
python experiments/gpu_decode_benchmark.py \
    --batch 1 \
    --heads 32 \
    --head-dim 64 \
    --kv-len 65536 \
    --selected-frac 0.25 \
    --iters 100 \
    --warmup 20

# Run with longer context
python experiments/gpu_decode_benchmark.py \
    --batch 1 \
    --heads 32 \
    --head-dim 128 \
    --kv-len 131072 \
    --selected-frac 0.125

# Run with higher batch (simulating multiple decode requests)
python experiments/gpu_decode_benchmark.py \
    --batch 4 \
    --heads 32 \
    --head-dim 64 \
    --kv-len 65536 \
    --selected-frac 0.25
```

---

## How to interpret results

- **SDPA (full KV)** is the baseline. Everything else should be compared
  against this, not against each other.
- **SelectedKV (gather + SDPA)** adds gather overhead. If gather dominates,
  consider page-aligned selection.
- **TritonIntentQuant** is a prototype. Expect high latency at small
  selected-frac values due to per-page loop overhead.
- **ratio = SDPA_ms / backend_ms** > 1.0 means the backend is faster than
  full SDPA. Ratio < 1.0 means it is slower.
- **Selected-frac matters.** A backend that is slower at 100% selection
  may be faster at 10% selection (fewer KV pages processed).

---

## What not to claim

- Do not claim "2× faster than FlashAttention" unless you ran FA2 on
  Ampere/Ada/Hopper hardware and the comparison is fair.
- Do not claim "GPU speedup" from a single config or a single GPU.
- Do not claim "production-ready" from an untuned prototype kernel.
- Do not claim "representative of all workloads" from decode-only
  benchmarking (prefill has different characteristics).
- Do not claim "optimal" unless you swept page size, block size, grid
  dimensions, and memory pressure.

---

## Recommended citation for results

> *Measured on [GPU name] with PyTorch [version], CUDA [version].*
> *SelectedKV uses torch SDPA after gather. TritonIntentQuant is an
> untuned prototype. FlashAttention-2 baselines are only reported on
> Ampere/Ada/Hopper hardware.*
> *These are local measurements. Results may vary.*
