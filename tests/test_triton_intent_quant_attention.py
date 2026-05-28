import pytest
import torch

from intent_attention.triton_intent_quant_attention import (
    IntentQuantKernelConfig,
    TritonKVPrecision,
    fake_int8_pages_from_fp16,
    intent_quant_decode_attention_triton,
    is_cuda_available,
    is_triton_available,
    make_page_tables_from_selected_pages,
    make_precision_tensor,
)

_has_triton = is_triton_available()
_has_cuda = is_cuda_available()
_has_gpu = _has_triton and _has_cuda


def test_is_triton_available_returns_bool():
    assert isinstance(is_triton_available(), bool)


def test_is_cuda_available_returns_bool():
    assert isinstance(is_cuda_available(), bool)


def test_triton_kv_precision_values():
    assert TritonKVPrecision.FP16 == 0
    assert TritonKVPrecision.FP8 == 1
    assert TritonKVPrecision.INT8 == 2
    assert TritonKVPrecision.INT4 == 3
    assert TritonKVPrecision.INT4_RESIDUAL == 4
    assert TritonKVPrecision.SKIP == 5


def test_make_page_tables_from_1d():
    pages = torch.tensor([0, 2, 5], dtype=torch.int32)
    page_ids, page_count = make_page_tables_from_selected_pages(
        pages, batch=2, heads=3, max_selected_pages=8
    )
    assert page_ids.shape == (2, 3, 8)
    assert page_count.shape == (2, 3)
    assert (page_count == 3).all()
    assert (page_ids[:, :, :3] == pages.view(1, 1, 3)).all()
    assert (page_ids[:, :, 3:] == -1).all()


def test_make_page_tables_from_3d():
    pages = torch.arange(12, dtype=torch.int32).reshape(2, 3, 2)
    page_ids, page_count = make_page_tables_from_selected_pages(
        pages, batch=2, heads=3, max_selected_pages=4
    )
    assert page_ids.shape == (2, 3, 4)
    assert (page_count == 2).all()
    assert (page_ids[:, :, :2] == pages).all()
    assert (page_ids[:, :, 2:] == -1).all()


def test_make_page_tables_exceeds_max_raises():
    pages = torch.arange(10, dtype=torch.int32)
    with pytest.raises(ValueError, match="exceeds max_selected_pages"):
        make_page_tables_from_selected_pages(pages, batch=1, heads=1, max_selected_pages=5)


def test_make_page_tables_bad_ndim_raises():
    pages = torch.arange(24, dtype=torch.int32).reshape(2, 3, 2, 2)
    with pytest.raises(ValueError, match="must have shape"):
        make_page_tables_from_selected_pages(pages, batch=2, heads=3)


def test_make_page_tables_3d_mismatch_raises():
    pages = torch.arange(6, dtype=torch.int32).reshape(1, 3, 2)
    with pytest.raises(ValueError, match="must match batch and heads"):
        make_page_tables_from_selected_pages(pages, batch=2, heads=3)


def test_fake_int8_pages_from_fp16():
    pages = torch.randn(4, 16, 32)
    q, scales = fake_int8_pages_from_fp16(pages)
    assert q.shape == pages.shape
    assert q.dtype == torch.int8
    assert scales.shape == (4,)
    assert scales.dtype == torch.float32


def test_make_precision_tensor_defaults():
    prec = make_precision_tensor(10)
    assert prec.shape == (10,)
    assert prec.dtype == torch.int32
    assert (prec == int(TritonKVPrecision.INT8)).all()


def test_make_precision_tensor_with_fp16():
    fp16 = torch.tensor([0, 1, 2])
    int8 = torch.tensor([3, 4])
    prec = make_precision_tensor(10, fp16_pages=fp16, int8_pages=int8)
    assert (prec[:3] == int(TritonKVPrecision.FP16)).all()
    assert (prec[3:5] == int(TritonKVPrecision.INT8)).all()
    assert (prec[5:] == int(TritonKVPrecision.INT8)).all()


def test_validate_decode_inputs_validation():
    """Input validation should reject obviously wrong shapes without GPU."""
    B, H, D = 2, 4, 64
    q = torch.randn(B, H, D)
    page_ids = torch.randint(0, 10, (B, H, 8), dtype=torch.int32)
    page_count = torch.full((B, H), 3, dtype=torch.int32)
    page_precision = torch.full((20,), 0, dtype=torch.int32)

    k_fp16 = torch.randn(20, 16, D)
    v_fp16 = torch.randn(20, 16, D)
    k_i8 = torch.randint(-127, 127, (20, 16, D), dtype=torch.int8)
    v_i8 = torch.randint(-127, 127, (20, 16, D), dtype=torch.int8)
    k_scales = torch.randn(20)
    v_scales = torch.randn(20)

    with pytest.raises(RuntimeError, match="Triton is not installed|CUDA is not available"):
        intent_quant_decode_attention_triton(
            q, k_fp16, v_fp16, k_i8, v_i8, k_scales, v_scales,
            page_ids, page_count, page_precision,
        )


def test_intent_quant_kernel_config_defaults():
    cfg = IntentQuantKernelConfig()
    assert cfg.page_size == 16
    assert cfg.head_dim == 64
    assert cfg.max_selected_pages == 64
    assert cfg.block_d == 64


@pytest.mark.skipif(not _has_gpu, reason="Triton/CUDA not available")
def test_kernel_smoke():
    """GPU smoke test: run the kernel on a tiny case and check output shape."""
    B, H, D = 1, 1, 64
    num_pages = 4
    page_size = 16

    q = torch.randn(B, H, D, device="cuda", dtype=torch.float16)
    k_fp16 = torch.randn(num_pages, page_size, D, device="cuda", dtype=torch.float16)
    v_fp16 = torch.randn(num_pages, page_size, D, device="cuda", dtype=torch.float16)

    k_i8 = torch.randint(-127, 127, (num_pages, page_size, D), device="cuda", dtype=torch.int8)
    v_i8 = torch.randint(-127, 127, (num_pages, page_size, D), device="cuda", dtype=torch.int8)
    k_scales = torch.randn(num_pages, device="cuda", dtype=torch.float32)
    v_scales = torch.randn(num_pages, device="cuda", dtype=torch.float32)

    selected_pages = torch.arange(num_pages, device="cuda")
    page_ids, page_count = make_page_tables_from_selected_pages(selected_pages, B, H, max_selected_pages=num_pages)
    page_precision = make_precision_tensor(num_pages, fp16_pages=selected_pages, device="cuda")

    config = IntentQuantKernelConfig(page_size=page_size, head_dim=D, max_selected_pages=num_pages, block_d=64)

    out = intent_quant_decode_attention_triton(
        q, k_fp16, v_fp16, k_i8, v_i8, k_scales, v_scales,
        page_ids, page_count, page_precision,
        config=config,
    )
    assert out.shape == (B, H, D)
    assert out.dtype == q.dtype
    assert torch.isfinite(out).all()
