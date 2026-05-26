import time
import torch
from intent_attention.synthetic_traces import generate_agentic_layout
from intent_attention.reference import dense_attention, semantic_block_attention

def main():
    print("WARNING: CPU timing is not representative of GPU kernel performance.")
    sizes = [512, 1024, 2048, 4096]
    for size in sizes:
        q = torch.randn(1, 8, 128, 64)
        k = torch.randn(1, 8, size, 64)
        v = torch.randn(1, 8, size, 64)
        layout = generate_agentic_layout(size)
        
        t0 = time.time()
        dense_attention(q, k, v)
        dense_time = time.time() - t0
        
        t0 = time.time()
        semantic_block_attention(q, k, v, layout)
        sem_time = time.time() - t0
        print(f"Size: {size}, Dense: {dense_time:.4f}s, Semantic: {sem_time:.4f}s")

if __name__ == "__main__":
    main()
