import time
import torch
from intent_attention.synthetic_traces import generate_agentic_layout
from intent_attention.reference import dense_attention, semantic_block_attention


def main():
    print("=" * 60)
    print("WARNING: CPU timing is not representative of GPU kernel performance.")
    print("These numbers measure Python + PyTorch overhead on CPU only.")
    print("=" * 60)
    print()

    sizes = [512, 1024, 2048, 4096]
    header = f"{'Tokens':>8} {'Dense (s)':>12} {'Semantic (s)':>14} {'CPU Ratio':>10}"
    print(header)
    print("-" * len(header))

    for size in sizes:
        q = torch.randn(1, 8, 128, 64)
        k = torch.randn(1, 8, size, 64)
        v = torch.randn(1, 8, size, 64)
        layout = generate_agentic_layout(size, seed=0)

        t0 = time.perf_counter()
        dense_attention(q, k, v)
        dense_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        semantic_block_attention(q, k, v, layout)
        sem_time = time.perf_counter() - t0

        ratio = dense_time / sem_time if sem_time > 0 else float("inf")
        print(f"{size:>8} {dense_time:>12.4f} {sem_time:>14.4f} {ratio:>10.2f}x")

    print()
    print("Note: Actual GPU speedups depend on memory bandwidth, kernel fusion,")
    print("and the fraction of KV tokens skipped. The cost model in")
    print("bench_cost_model.py provides analytical estimates.")


if __name__ == "__main__":
    main()
