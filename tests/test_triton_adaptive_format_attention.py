"""Tests for Triton adaptive-format decode attention kernel."""

from __future__ import annotations

import math
import pytest
import torch

from intent_attention.triton_adaptive_format_attention import (
    AdaptivePageFormat,
    AdaptiveFormatKernelConfig,
    is_triton_available,
    is_cuda_available,
    make_adaptive_page_tables,
    adaptive_format_decode_attention_reference_dispatch,
    adaptive_format_decode_attention_triton,
)


class TestModuleImports:
    """Tests that the module imports safely on CPU-only machines."""

    def test_imports_without_triton(self):
        from intent_attention import triton_adaptive_format_attention as mod
        assert mod is not None
        assert hasattr(mod, "AdaptivePageFormat")
        assert hasattr(mod, "adaptive_format_decode_attention_reference_dispatch")

    def test_is_triton_available_returns_bool(self):
        val = is_triton_available()
        assert isinstance(val, bool)

    def test_is_cuda_available_returns_bool(self):
        val = is_cuda_available()
        assert isinstance(val, bool)


class TestAdaptivePageFormat:
    def test_enum_values_are_stable(self):
        assert AdaptivePageFormat.FP16.value == 0
        assert AdaptivePageFormat.INT8.value == 1
        assert AdaptivePageFormat.SPARSE.value == 2
        assert AdaptivePageFormat.SKIP.value == 3

    def test_enum_members(self):
        assert len(AdaptivePageFormat) == 4


class TestMakeAdaptivePageTables:
    def test_1d_preserves_order(self):
        pages = torch.tensor([3, 1, 2], dtype=torch.int32)
        page_ids, page_counts = make_adaptive_page_tables(pages, batch=1, heads=1)
        assert page_ids.shape == (1, 1, 3)
        assert page_ids[0, 0, 0].item() == 3
        assert page_ids[0, 0, 1].item() == 1
        assert page_ids[0, 0, 2].item() == 2
        assert page_counts[0, 0].item() == 3

    def test_minus_one_unused(self):
        pages = torch.tensor([0], dtype=torch.int32)
        page_ids, page_counts = make_adaptive_page_tables(pages, batch=2, heads=2, max_selected_pages=4)
        assert page_ids.shape == (2, 2, 4)
        assert page_ids[0, 0, 0].item() == 0
        for i in range(1, 4):
            assert page_ids[0, 0, i].item() == -1
        assert page_counts[0, 0].item() == 1

    def test_skip_pages_excluded_from_count(self):
        pages = torch.tensor([0, 2], dtype=torch.int32)
        page_ids, page_counts = make_adaptive_page_tables(pages, batch=1, heads=1)
        assert page_counts[0, 0].item() == 2

    def test_invalid_ndim_raises(self):
        with pytest.raises(ValueError):
            make_adaptive_page_tables(torch.zeros(2, 3, 4, 5), batch=2, heads=3)


class TestReferenceDispatch:
    def test_cpu_only_execution(self):
        assert not is_cuda_available(), "This test requires CPU-only execution"

    @pytest.fixture
    def small_fixture(self):
        B, H, D = 1, 2, 32
        PS = 8
        NP = 4
        MAX_SEL = 2
        q = torch.randn(B, H, D, dtype=torch.float16)
        fp16_kv = torch.randn(NP, PS, D, dtype=torch.float16)
        page_ids = torch.zeros(B, H, MAX_SEL, dtype=torch.int32)
        page_ids[0, 0, 0] = 0
        page_ids[0, 0, 1] = 1
        page_formats = torch.zeros(NP, dtype=torch.int32)
        return B, H, D, PS, NP, MAX_SEL, q, fp16_kv, page_ids, page_formats

    def test_fp16_only(self, small_fixture):
        B, H, D, PS, NP, MAX_SEL, q, fp16_kv, page_ids, page_formats = small_fixture
        page_formats[:] = int(AdaptivePageFormat.FP16)
        config = AdaptiveFormatKernelConfig(page_size=PS, head_dim=D, max_selected_pages=MAX_SEL)
        out = adaptive_format_decode_attention_reference_dispatch(
            q, page_ids, page_formats, fp16_kv, fp16_kv, config=config,
        )
        assert out.shape == (B, H, D)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    def test_int8_only(self, small_fixture):
        B, H, D, PS, NP, MAX_SEL, q, fp16_kv, page_ids, page_formats = small_fixture
        page_formats[:] = int(AdaptivePageFormat.INT8)
        int8_kv = (fp16_kv * 10).to(torch.int8)
        int8_scales = torch.full((NP,), 0.1, dtype=torch.float32)
        config = AdaptiveFormatKernelConfig(page_size=PS, head_dim=D, max_selected_pages=MAX_SEL)
        out = adaptive_format_decode_attention_reference_dispatch(
            q, page_ids, page_formats, fp16_kv, fp16_kv,
            int8_k_pages=int8_kv, int8_v_pages=int8_kv,
            int8_k_scales=int8_scales, int8_v_scales=int8_scales,
            config=config,
        )
        assert out.shape == (B, H, D)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    def test_mixed_fp16_int8_skip(self, small_fixture):
        B, H, D, PS, NP, MAX_SEL, q, fp16_kv, page_ids, page_formats = small_fixture
        # page 0: FP16, page 1: INT8, rest: SKIP
        page_formats[0] = int(AdaptivePageFormat.FP16)
        page_formats[1] = int(AdaptivePageFormat.INT8)
        page_formats[2] = int(AdaptivePageFormat.SKIP)
        page_formats[3] = int(AdaptivePageFormat.SKIP)
        # Only page 0 and 1 selected
        page_ids[0, 0, 0] = 0
        page_ids[0, 0, 1] = 1
        int8_kv = (fp16_kv * 10).to(torch.int8)
        int8_scales = torch.full((NP,), 0.1, dtype=torch.float32)
        config = AdaptiveFormatKernelConfig(page_size=PS, head_dim=D, max_selected_pages=MAX_SEL)
        out = adaptive_format_decode_attention_reference_dispatch(
            q, page_ids, page_formats, fp16_kv, fp16_kv,
            int8_k_pages=int8_kv, int8_v_pages=int8_kv,
            int8_k_scales=int8_scales, int8_v_scales=int8_scales,
            config=config,
        )
        assert out.shape == (B, H, D)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    def test_sparse_pages_through_reference(self, small_fixture):
        B, H, D, PS, NP, MAX_SEL, q, fp16_kv, page_ids, page_formats = small_fixture
        page_formats[0] = int(AdaptivePageFormat.SPARSE)
        page_formats[1] = int(AdaptivePageFormat.SPARSE)
        # Build sparse indices/values
        sp_max = 4
        sp_indices = torch.zeros(NP, H, sp_max, 2, dtype=torch.int32)
        sp_values = torch.zeros(NP, H, sp_max, dtype=torch.float16)
        sp_indices[0, 0, 0, 0] = 2
        sp_indices[0, 0, 0, 1] = 0
        sp_values[0, 0, 0] = 1.5
        sp_nnz = torch.zeros(NP, H, dtype=torch.int32)
        sp_nnz[0, 0] = 1
        sp_nnz[1, 0] = 0
        config = AdaptiveFormatKernelConfig(
            page_size=PS, head_dim=D, max_selected_pages=MAX_SEL, sparse_max_nnz=sp_max,
        )
        out = adaptive_format_decode_attention_reference_dispatch(
            q, page_ids, page_formats, fp16_kv, fp16_kv,
            sparse_k_indices=sp_indices, sparse_k_values=sp_values,
            sparse_v_indices=sp_indices, sparse_v_values=sp_values,
            sparse_nnz=sp_nnz, config=config,
        )
        assert out.shape == (B, H, D)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    def test_zero_selected_pages(self, small_fixture):
        B, H, D, PS, NP, MAX_SEL, q, fp16_kv, page_ids, page_formats = small_fixture
        page_ids[:] = -1  # no selected pages
        config = AdaptiveFormatKernelConfig(page_size=PS, head_dim=D, max_selected_pages=MAX_SEL)
        out = adaptive_format_decode_attention_reference_dispatch(
            q, page_ids, page_formats, fp16_kv, fp16_kv, config=config,
        )
        assert out.shape == (B, H, D)
        assert torch.allclose(out, torch.zeros_like(out), atol=1e-6)


class TestTritonSmoke:
    @pytest.mark.skipif(not is_cuda_available(), reason="CUDA not available")
    @pytest.mark.skipif(not is_triton_available(), reason="Triton not available")
    def test_gpu_output_shape(self):
        B, H, D = 1, 2, 32
        PS = 8
        NP = 4
        MAX_SEL = 2
        q = torch.randn(B, H, D, dtype=torch.float16, device="cuda")
        fp16_kv = torch.randn(NP, PS, D, dtype=torch.float16, device="cuda")
        int8_kv = (fp16_kv * 10).to(torch.int8)
        int8_scales = torch.full((NP,), 0.1, dtype=torch.float32, device="cuda")
        page_ids = torch.zeros(B, H, MAX_SEL, dtype=torch.int32, device="cuda")
        page_ids[0, 0, 0] = 0
        page_ids[0, 0, 1] = 1
        page_formats = torch.full((NP,), int(AdaptivePageFormat.FP16), dtype=torch.int32, device="cuda")
        config = AdaptiveFormatKernelConfig(page_size=PS, head_dim=D, max_selected_pages=MAX_SEL)
        out = adaptive_format_decode_attention_triton(
            q, page_ids, page_formats, fp16_kv, fp16_kv,
            int8_k_pages=int8_kv, int8_v_pages=int8_kv,
            int8_k_scales=int8_scales, int8_v_scales=int8_scales,
            config=config,
        )
        assert out.shape == (B, H, D)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    @pytest.mark.skipif(not is_cuda_available(), reason="CUDA not available")
    @pytest.mark.skipif(not is_triton_available(), reason="Triton not available")
    def test_gpu_mixed_fp16_int8(self):
        B, H, D = 1, 1, 16
        PS = 4
        NP = 3
        MAX_SEL = 3
        q = torch.randn(B, H, D, dtype=torch.float16, device="cuda")
        fp16_kv = torch.randn(NP, PS, D, dtype=torch.float16, device="cuda")
        int8_kv = (fp16_kv * 8).to(torch.int8)
        int8_scales = torch.full((NP,), 0.125, dtype=torch.float32, device="cuda")
        page_ids = torch.zeros(B, H, MAX_SEL, dtype=torch.int32, device="cuda")
        page_formats = torch.zeros(NP, dtype=torch.int32, device="cuda")
        for i in range(MAX_SEL):
            page_ids[0, 0, i] = i
        page_formats[0] = int(AdaptivePageFormat.FP16)
        page_formats[1] = int(AdaptivePageFormat.INT8)
        page_formats[2] = int(AdaptivePageFormat.FP16)
        config = AdaptiveFormatKernelConfig(page_size=PS, head_dim=D, max_selected_pages=MAX_SEL)
        out = adaptive_format_decode_attention_triton(
            q, page_ids, page_formats, fp16_kv, fp16_kv,
            int8_k_pages=int8_kv, int8_v_pages=int8_kv,
            int8_k_scales=int8_scales, int8_v_scales=int8_scales,
            config=config,
        )
        assert out.shape == (B, H, D)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()
