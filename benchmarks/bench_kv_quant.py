"""Benchmark: KV quant roundtrip speed."""
from __future__ import annotations

import math
import time

import torch

from intent_attention.kv_quant import quantise_k_perchannel, dequantise_k, quantise_v_pertoken, dequantise_v


def bench_kv_quant(page_size=64, d_head=128, n_pages=256, dry_run=False):
    k = torch.randn(n_pages * page_size, d_head)
    v = torch.randn(n_pages * page_size, d_head)
    if dry_run:
        print(f"[dry-run] kv_quant: {n_pages} pages, {page_size=}, {d_head=}")
        return
    k_split = torch.split(k, page_size)
    v_split = torch.split(v, page_size)
    k_int8_list = []
    v_int8_list = []
    ks_list = []
    vs_list = []
    t0 = time.perf_counter()
    for kp, vp in zip(k_split, v_split):
        k_int8, ks, _ = quantise_k_perchannel(kp)
        v_int8, vs, _ = quantise_v_pertoken(vp)
        k_int8_list.append(k_int8)
        v_int8_list.append(v_int8)
        ks_list.append(ks)
        vs_list.append(vs)
    quant_t = time.perf_counter() - t0
    us_per_page = quant_t / n_pages * 1e6
    print(f"  quant: {quant_t*1000:.1f} ms ({us_per_page:.1f} us/page)")
    t1 = time.perf_counter()
    for k_int8, ks in zip(k_int8_list, ks_list):
        dequantise_k(k_int8, ks, ks)
    for v_int8, vs in zip(v_int8_list, vs_list):
        dequantise_v(v_int8, vs)
    deq_t = time.perf_counter() - t1
    print(f"  dequant: {deq_t*1000:.1f} ms")
    total_bytes = 0
    for k_int8, v_int8 in zip(k_int8_list, v_int8_list):
        total_bytes += (k_int8.numel() + v_int8.numel()) * 1
    mem_orig = n_pages * page_size * d_head * 2 * 2
    print(f"  memory: {total_bytes / 1e6:.1f} MB (vs {mem_orig / 1e6:.1f} MB fp16)")


def bench_kv_quant_main(dry_run=False):
    print("=== KV Quant Benchmark ===")
    for pages in [64, 256]:
        for ps in [64, 128]:
            print(f"\n  [{pages} pages x {ps} tokens, d=128]:")
            bench_kv_quant(page_size=ps, n_pages=pages, dry_run=dry_run)


if __name__ == "__main__":
    import sys
    dry_run = "--dry-run" in sys.argv
    bench_kv_quant_main(dry_run=dry_run)
