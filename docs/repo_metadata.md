# Repo Metadata Suggestions

## GitHub Description

```
Semantic block attention prototype for agentic long-context inference:
runtime-aware KV block selection, CPU reference implementation, analytical
cost model, and future Triton/CUDA kernel path.
```

## GitHub Topics

```
attention
long-context
agentic-ai
kv-cache
block-attention
sparse-attention
pytorch
triton
gpu-kernels
llm-inference
ai-infrastructure
systems
cost-model
research
```

## Suggested X / Twitter Post

> I started a small research repo: Intent Attention Kernel.
>
> The idea: long-context agentic inference should not treat context as flat.
>
> System prompts, retrieved docs, tool outputs, memory summaries,
> scratchpads, and recent turns are semantically different regions.
>
> So the runtime should pass block metadata to attention.
>
> CPU-first prototype:
> * PyTorch reference implementation
> * semantic KV block selection
> * synthetic agentic traces
> * analytical FLOP/KV traffic cost model
> * optional Triton/CUDA roadmap
>
> No GPU speedup claims yet. First step is proving the interface and correctness.
>
> Attention should not pretend context is flat.
