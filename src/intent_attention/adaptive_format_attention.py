"""
Adaptive Format KV Attention Reference.

A CPU reference for KV attention that supports multiple storage formats per page:
- Format 0: Dense FP16 (no compression)
- Format 1: Dense INT8 + per-page scale (linear quantization)
- Format 2: Sparse top-k (store k largest-magnitude elements, rest zero)

This demonstrates heterogeneous KV cache storage formats with format-aware
attention computation. The kernel detects each page's format and applies
the appropriate decompression strategy on-the-fly.

No GPU speedup or model quality preservation is claimed. This is a CPU
reference for algorithmic validation only.
"""

from __future__ import annotations

import math
from typing import Tuple

import torch


def adaptive_format_attention_reference(
    query: torch.Tensor,
    kv_pages_dense_fp16: torch.Tensor,
    kv_pages_dense_i8: torch.Tensor,
    kv_pages_dense_scales: torch.Tensor,
    kv_pages_sparse_indices: torch.Tensor,
    kv_pages_sparse_values: torch.Tensor,
    kv_pages_formats: torch.Tensor,
    page_table: torch.Tensor,
    page_counts: torch.Tensor,
    config,
) -> torch.Tensor:
    """
    Adaptive format KV attention reference implementation.

    Args:
        query: [B, H, 1, D] FP16 query tensor
        kv_pages_dense_fp16: [num_pages, page_size, D] FP16 dense pages
        kv_pages_dense_i8: [num_pages, page_size, D] INT8 dense pages
        kv_pages_dense_scales: [num_pages] FP16 per-page scales for INT8 dequant
        kv_pages_sparse_indices: [num_pages, sparsity_k] INT64 indices for sparse format
        kv_pages_sparse_values: [num_pages, sparsity_k] FP16 values for sparse format
        kv_pages_formats: [num_pages] INT8 format tags (0=FP16, 1=INT8, 2=SPARSE)
        page_table: [B, H, max_selected_pages] INT32 page IDs to attend to
        page_counts: [B, H] INT32 number of selected pages per batch/head
        config: Configuration object with page_size, head_dim, etc.

    Returns:
        [B, H, 1, D] FP16 attention output
    """
    B, H, _, D = query.shape
    page_size = config.page_size
    sparsity_k = kv_pages_sparse_indices.shape[-1]  # Assuming last dim is k

    out = torch.zeros(B, H, 1, D, dtype=torch.float16, device=query.device)

    for b_idx in range(B):
        for h_idx in range(H):
            n_pages = int(page_counts[b_idx, h_idx].item())
            query_vec = query[b_idx, h_idx, 0, :].float()  # [D]

            # Online softmax accumulators
            m_i = torch.tensor(-float("inf"), device=query_vec.device)
            l_i = torch.tensor(0.0, device=query_vec.device)
            acc = torch.zeros(D, dtype=torch.float32, device=query_vec.device)

            for p_idx in range(n_pages):
                page_id = int(page_table[b_idx, h_idx, p_idx].item())
                format_tag = int(kv_pages_formats[page_id].item())

                # Reconstruct page based on format
                if format_tag == 0:  # Dense FP16
                    page_k = kv_pages_dense_fp16[page_id].float()  # [page_size, D]
                    page_v = kv_pages_dense_fp16[page_id].float()  # [page_size, D]
                elif format_tag == 1:  # Dense INT8 + scale
                    page_i8 = kv_pages_dense_i8[page_id].float()
                    scale = kv_pages_dense_scales[page_id].float()
                    page_k = (page_i8 * scale).view(page_size, D)
                    page_v = (page_i8 * scale).view(page_size, D)
                elif format_tag == 2:  # Sparse top-k
                    # Reconstruct sparse page: scatter values into zero page
                    page_k = torch.zeros(page_size, D, dtype=torch.float32)
                    page_v = torch.zeros(page_size, D, dtype=torch.float32)
                    
                    indices = kv_pages_sparse_indices[page_id]  # [sparsity_k]
                    values = kv_pages_sparse_values[page_id]    # [sparsity_k]
                    
                    for k_idx in range(sparsity_k):
                        pos = int(indices[k_idx].item())
                        val = float(values[k_idx].item())
                        if 0 <= pos < page_size:
                            page_k[pos, :] = val
                            page_v[pos, :] = val
                else:
                    # Unknown format - treat as zero page
                    page_k = torch.zeros(page_size, D, dtype=torch.float32)
                    page_v = torch.zeros(page_size, D, dtype=torch.float32)

                    # Compute block-level attention: score = query @ page_key^T / sqrt(D)
                    # query_vec: [D], page_k: [page_size, D] -> scores: [page_size]
                    scores = torch.mm(query_vec.unsqueeze(0), page_k.transpose(0, 1)).squeeze(0) / math.sqrt(D)
                    
                    # Online softmax update
                    m_new = torch.maximum(m_i, torch.max(scores))
                    alpha = torch.exp(m_i - m_new)
                    p = torch.exp(scores - m_new)
                    l_i = alpha * l_i + torch.sum(p)
                    acc = alpha * acc + torch.mm(p.unsqueeze(0), page_v).squeeze(0)
                    m_i = m_new

            # Normalize and store output
            out[b_idx, h_idx, 0, :] = (acc / torch.max(l_i, torch.tensor(1e-8, device=l_i.device))).half()

    return out


def adaptive_format_attention_reference_simple(
    query: torch.Tensor,
    kv_pages: torch.Tensor,
    kv_pages_formats: torch.Tensor,
    page_table: torch.Tensor,
    page_counts: torch.Tensor,
    config,
) -> torch.Tensor:
    """
    Simplified adaptive format attention for testing.
    
    Args:
        query: [B, H, 1, D] FP16
        kv_pages: [num_pages, page_size, D] FP16 (dense storage for all formats in this simple version)
        kv_pages_formats: [num_pages] INT8 format tags
        page_table: [B, H, max_selected_pages] INT32
        page_counts: [B, H] INT32
        config: Configuration
    
    Returns:
        [B, H, 1, D] FP16
    """
    # For simple test, ignore format differences and just use dense storage
    # In real implementation, format would determine how to interpret kv_pages
    B, H, _, D = query.shape
    out = torch.zeros(B, H, 1, D, dtype=torch.float16, device=query.device)

    for b_idx in range(B):
        for h_idx in range(H):
            n_pages = int(page_counts[b_idx, h_idx].item())
            query_vec = query[b_idx, h_idx, 0, :].float()
            
            m_i = torch.tensor(-float("inf"), device=query_vec.device)
            l_i = torch.tensor(0.0, device=query_vec.device)
            acc = torch.zeros(D, dtype=torch.float32, device=query_vec.device)
            
            for p_idx in range(n_pages):
                page_id = int(page_table[b_idx, h_idx, p_idx].item())
                # In simple version, ignore format and just use dense storage
                page_k = kv_pages[page_id].float()
                page_v = kv_pages[page_id].float()
                
                scores = torch.mm(query_vec.unsqueeze(0), page_k.transpose(0, 1)).squeeze(0) / math.sqrt(D)
                m_new = torch.maximum(m_i, torch.max(scores))
                alpha = torch.exp(m_i - m_new)
                p = torch.exp(scores - m_new)
                l_i = alpha * l_i + torch.sum(p)
                acc = alpha * acc + torch.mm(p.unsqueeze(0), page_v).squeeze(0)
                m_i = m_new
            
            out[b_idx, h_idx, 0, :] = (acc / torch.max(l_i, torch.tensor(1e-8, device=l_i.device))).half()

    return out