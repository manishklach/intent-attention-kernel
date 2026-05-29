"""Tests for triton_mla_decode.py CPU fallback."""
from __future__ import annotations

import math

import pytest
import torch

from intent_attention.triton_mla_decode import (
    is_cuda_available,
    is_triton_available,
    mla_decode_triton,
)


class TestCPUFallback:
    """CPU fallback tests (no GPU required)."""

    def test_single_token(self):
        batch, q_len, d_c, d_out = 1, 1, 64, 128
        page_size = 16
        q_absorb = torch.randn(batch, q_len, d_c)
        C = torch.randn(64, d_c)
        W_VO = torch.randn(d_c, d_out)
        page_table = torch.tensor([0, 1, 2, 3], dtype=torch.int32)
        out = mla_decode_triton(q_absorb, C, W_VO, page_table, page_size=page_size)
        assert out.shape == (batch, q_len, d_out)
        assert not torch.isnan(out).any()

    def test_batch(self):
        batch, q_len, d_c, d_out = 2, 4, 64, 128
        page_size = 16
        q_absorb = torch.randn(batch, q_len, d_c)
        C = torch.randn(64, d_c)
        W_VO = torch.randn(d_c, d_out)
        page_table = torch.tensor([0, 1, 2, 3], dtype=torch.int32)
        out = mla_decode_triton(q_absorb, C, W_VO, page_table, page_size=page_size)
        assert out.shape == (batch, q_len, d_out)

    def test_multiple_page_sizes(self):
        for page_size in [8, 16, 32, 64]:
            batch, q_len, d_c, d_out = 1, 2, 64, 128
            n_pages = 4
            q_absorb = torch.randn(batch, q_len, d_c)
            C = torch.randn(n_pages * page_size, d_c)
            W_VO = torch.randn(d_c, d_out)
            page_table = torch.tensor(list(range(n_pages)), dtype=torch.int32)
            out = mla_decode_triton(q_absorb, C, W_VO, page_table, page_size=page_size)
            assert out.shape == (batch, q_len, d_out)
            assert not torch.isnan(out).any()

    def test_no_selected_pages(self):
        batch, q_len, d_c, d_out = 1, 1, 64, 128
        q_absorb = torch.randn(batch, q_len, d_c)
        C = torch.randn(64, d_c)
        W_VO = torch.randn(d_c, d_out)
        page_table = torch.tensor([], dtype=torch.int32)
        out = mla_decode_triton(q_absorb, C, W_VO, page_table)
        assert out.shape == (batch, q_len, d_out)
        assert torch.allclose(out, torch.zeros_like(out))

    def test_single_page(self):
        batch, q_len, d_c, d_out = 1, 2, 64, 128
        page_size = 32
        q_absorb = torch.randn(batch, q_len, d_c)
        C = torch.randn(32, d_c)
        W_VO = torch.randn(d_c, d_out)
        page_table = torch.tensor([0], dtype=torch.int32)
        out = mla_decode_triton(q_absorb, C, W_VO, page_table, page_size=page_size)
        assert out.shape == (batch, q_len, d_out)

    def test_dtype_device_consistency(self):
        batch, q_len, d_c, d_out = 1, 2, 64, 128
        page_size = 16
        q_absorb = torch.randn(batch, q_len, d_c)
        C = torch.randn(64, d_c)
        W_VO = torch.randn(d_c, d_out, dtype=torch.float32)
        page_table = torch.tensor([0, 1, 2, 3], dtype=torch.int32)
        out = mla_decode_triton(q_absorb, C, W_VO, page_table, page_size=page_size)
        assert out.dtype == q_absorb.dtype
        assert out.device == q_absorb.device

    def test_vs_reference(self):
        from intent_attention.mla import MLAConfig, MLABlockTable, mla_sparse_decode_reference
        from intent_attention.block_metadata import BlockLayout, SemanticBlock, BlockPolicy

        d_model, d_c, n_heads, d_head = 256, 64, 4, 64
        cfg = MLAConfig(d_model, d_c, n_heads, d_head)
        page_size = 16
        table = MLABlockTable(cfg, page_size=page_size)
        for bid in range(4):
            table.append(bid, torch.randn(page_size, d_c))

        W_QK = torch.randn(n_heads * d_head, d_c)
        W_VO = torch.randn(d_c, d_model)
        q = torch.randn(2, n_heads, 4, d_head)
        q_flat = q.permute(0, 2, 1, 3).reshape(2, 4, n_heads * d_head)
        q_absorb = q_flat @ W_QK
        blocks = [SemanticBlock(f"b{i}", i * page_size, (i + 1) * page_size,
                                BlockPolicy.ATTEND, score=0.9) for i in range(4)]
        layout = BlockLayout(blocks)

        ref_out, _ = mla_sparse_decode_reference(
            q, table, W_QK, W_VO, layout, threshold=0.5, return_debug=True)

        C = torch.cat([table.get_latent(i) for i in range(4)], dim=0)
        page_table = torch.tensor([0, 1, 2, 3], dtype=torch.int32)
        triton_out = mla_decode_triton(
            q_absorb.float(), C.float(), W_VO.float(), page_table, page_size=page_size)

        assert torch.allclose(ref_out.float(), triton_out.float(), atol=1e-2)


class TestGPUSkip:
    """GPU path detection tests (safe on CPU-only systems)."""

    def test_cpu_fallback_used_when_no_cuda(self):
        batch, q_len, d_c, d_out = 1, 1, 64, 128
        q_absorb = torch.randn(batch, q_len, d_c)
        C = torch.randn(64, d_c)
        W_VO = torch.randn(d_c, d_out)
        page_table = torch.tensor([0], dtype=torch.int32)
        out = mla_decode_triton(q_absorb, C, W_VO, page_table)
        assert out.shape == (batch, q_len, d_out)
        assert not torch.isnan(out).any()

    @pytest.mark.skipif(not is_cuda_available(), reason="CUDA not available")
    @pytest.mark.skipif(not is_triton_available(), reason="Triton not available")
    def test_gpu_kernel_runs_on_cuda(self):
        batch, q_len, d_c, d_out = 1, 1, 64, 128
        page_size = 16
        q_absorb = torch.randn(batch, q_len, d_c, device="cuda")
        C = torch.randn(64, d_c, device="cuda")
        W_VO = torch.randn(d_c, d_out, device="cuda")
        page_table = torch.tensor([0, 1, 2, 3], dtype=torch.int32, device="cuda")
        out = mla_decode_triton(q_absorb, C, W_VO, page_table, page_size=page_size)
        assert out.shape == (batch, q_len, d_out)
        assert not torch.isnan(out).any()
        assert out.device.type == "cuda"
