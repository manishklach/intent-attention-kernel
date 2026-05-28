"""Tests for fused selected-quant decode attention kernel.

Covers:
- CPU reference correctness vs dense attention on selected pages
- FP16 precision path
- INT8 precision path  
- SKIP precision path
- Mixed precision (FP16 + INT8 + SKIP)
- Edge cases: zero pages, single page, all SKIP
- Metadata conversion from BlockRouter output
- Triton kernel smoke test (requires CUDA + Triton)
"""

from __future__ import annotations

import math
import pytest
import torch

from intent_attention.fused_selected_quant_decode import (
    FusedDecodeConfig,
    FusedKVPrecision,
    fake_int8_pages_from_fp16,
    fused_selected_quant_decode,
    fused_selected_quant_decode_reference,
    is_triton_available,
    metadata_to_kernel_tensors,
)


# ------------------------------------------------------------------ #
#  Fixtures
# ------------------------------------------------------------------ #

_BATCH = 2
_HEADS = 4
_HEAD_DIM = 64
_PAGE_SIZE = 16
_NUM_PAGES = 8
_MAX_SELECTED = 4
_DEVICE = torch.device("cpu")


@pytest.fixture
def config():
    return FusedDecodeConfig(
        page_size=_PAGE_SIZE,
        head_dim=_HEAD_DIM,
        max_selected_pages=_MAX_SELECTED,
        block_d=64,
    )


@pytest.fixture
def tensors(config):
    B, H, D, PS = _BATCH, _HEADS, _HEAD_DIM, _PAGE_SIZE
    NP = _NUM_PAGES

    # Full KV pages
    k_fp16 = torch.randn(NP, PS, D, dtype=torch.float16)
    v_fp16 = torch.randn(NP, PS, D, dtype=torch.float16)
    k_i8, k_sc = fake_int8_pages_from_fp16(k_fp16)
    v_i8, v_sc = fake_int8_pages_from_fp16(v_fp16)

    # Query
    q = torch.randn(B, H, 1, D, dtype=torch.float16)

    # Page table: select first _MAX_SELECTED pages
    pt = torch.zeros(B, H, _MAX_SELECTED, dtype=torch.int32)
    pc = torch.zeros(B, H, dtype=torch.int32)
    for b_idx in range(B):
        for h_idx in range(H):
            n_sel = min(_MAX_SELECTED, NP)
            for p in range(n_sel):
                pt[b_idx, h_idx, p] = p
            pc[b_idx, h_idx] = n_sel

    # Precision: all FP16 by default
    prec = torch.full((NP,), FusedKVPrecision.FP16, dtype=torch.int32)

    return {
        "query": q,
        "k_fp16": k_fp16,
        "v_fp16": v_fp16,
        "k_i8": k_i8,
        "v_i8": v_i8,
        "k_sc": k_sc,
        "v_sc": v_sc,
        "page_table": pt,
        "page_precision": prec,
        "page_counts": pc,
    }


# ------------------------------------------------------------------ #
#  Helpers
# ------------------------------------------------------------------ #


def _dense_selected_attention(q, k, v, page_table, page_counts, config):
    """Dense attention over selected pages (ground truth)."""
    B, H, _, D = q.shape
    out = torch.zeros(B, H, 1, D, dtype=torch.float16)
    for b_idx in range(B):
        for h_idx in range(H):
            n = int(page_counts[b_idx, h_idx].item())
            selected_k = []
            selected_v = []
            for p in range(n):
                pid = int(page_table[b_idx, h_idx, p].item())
                selected_k.append(k[pid])
                selected_v.append(v[pid])
            if not selected_k:
                continue
            k_sel = torch.cat(selected_k, dim=0).unsqueeze(0).unsqueeze(0)
            v_sel = torch.cat(selected_v, dim=0).unsqueeze(0).unsqueeze(0)
            q_cur = q[b_idx:b_idx + 1, h_idx:h_idx + 1, :, :]
            # Manual attention for float16 compatibility
            scores = (q_cur @ k_sel.transpose(-2, -1)) / math.sqrt(D)
            attn = torch.softmax(scores.float(), dim=-1).to(torch.float16)
            out[b_idx, h_idx] = (attn @ v_sel)
    return out


# ------------------------------------------------------------------ #
#  CPU reference tests
# ------------------------------------------------------------------ #


class TestFusedSelectedQuantDecodeReference:
    def test_output_shape(self, tensors, config):
        out = fused_selected_quant_decode_reference(
            tensors["query"], tensors["k_fp16"], tensors["v_fp16"],
            tensors["k_i8"], tensors["v_i8"],
            tensors["k_sc"], tensors["v_sc"],
            tensors["page_table"], tensors["page_precision"],
            tensors["page_counts"], config,
        )
        assert out.shape == (_BATCH, _HEADS, 1, _HEAD_DIM)
        assert out.dtype == torch.float16

    def test_fp16_matches_dense_selected(self, tensors, config):
        out = fused_selected_quant_decode_reference(
            tensors["query"], tensors["k_fp16"], tensors["v_fp16"],
            tensors["k_i8"], tensors["v_i8"],
            tensors["k_sc"], tensors["v_sc"],
            tensors["page_table"], tensors["page_precision"],
            tensors["page_counts"], config,
        )
        expected = _dense_selected_attention(
            tensors["query"], tensors["k_fp16"], tensors["v_fp16"],
            tensors["page_table"], tensors["page_counts"], config,
        )
        # FP16 reference should closely match dense selected attention
        diff = (out.float() - expected.float()).abs().max().item()
        assert diff < 1e-2, f"Max diff: {diff}"

    def test_int8_path(self, tensors, config):
        """Mark all pages as INT8, verify kernel runs."""
        prec = torch.full(
            (_NUM_PAGES,), FusedKVPrecision.INT8, dtype=torch.int32
        )
        out = fused_selected_quant_decode_reference(
            tensors["query"], tensors["k_fp16"], tensors["v_fp16"],
            tensors["k_i8"], tensors["v_i8"],
            tensors["k_sc"], tensors["v_sc"],
            tensors["page_table"], prec,
            tensors["page_counts"], config,
        )
        assert out.shape == (_BATCH, _HEADS, 1, _HEAD_DIM)

    def test_skip_path(self, tensors, config):
        """Mark all pages as SKIP -> output should be zero."""
        prec = torch.full(
            (_NUM_PAGES,), FusedKVPrecision.SKIP, dtype=torch.int32
        )
        out = fused_selected_quant_decode_reference(
            tensors["query"], tensors["k_fp16"], tensors["v_fp16"],
            tensors["k_i8"], tensors["v_i8"],
            tensors["k_sc"], tensors["v_sc"],
            tensors["page_table"], prec,
            tensors["page_counts"], config,
        )
        assert out.abs().max().item() < 1e-6

    def test_mixed_precision(self, tensors, config):
        """Half FP16, half INT8."""
        prec = torch.full((_NUM_PAGES,), FusedKVPrecision.FP16, dtype=torch.int32)
        prec[_NUM_PAGES // 2:] = FusedKVPrecision.INT8
        out = fused_selected_quant_decode_reference(
            tensors["query"], tensors["k_fp16"], tensors["v_fp16"],
            tensors["k_i8"], tensors["v_i8"],
            tensors["k_sc"], tensors["v_sc"],
            tensors["page_table"], prec,
            tensors["page_counts"], config,
        )
        assert out.shape == (_BATCH, _HEADS, 1, _HEAD_DIM)

    def test_mixed_with_skip(self, tensors, config):
        """Mix of FP16, INT8, SKIP."""
        prec = torch.full((_NUM_PAGES,), FusedKVPrecision.FP16, dtype=torch.int32)
        if _NUM_PAGES >= 3:
            prec[1] = FusedKVPrecision.INT8
            prec[2] = FusedKVPrecision.SKIP
        out = fused_selected_quant_decode_reference(
            tensors["query"], tensors["k_fp16"], tensors["v_fp16"],
            tensors["k_i8"], tensors["v_i8"],
            tensors["k_sc"], tensors["v_sc"],
            tensors["page_table"], prec,
            tensors["page_counts"], config,
        )
        assert out.shape == (_BATCH, _HEADS, 1, _HEAD_DIM)

    def test_zero_selected_pages(self, tensors, config):
        """Zero selected pages -> output should be zero."""
        pc = torch.zeros(_BATCH, _HEADS, dtype=torch.int32)
        out = fused_selected_quant_decode_reference(
            tensors["query"], tensors["k_fp16"], tensors["v_fp16"],
            tensors["k_i8"], tensors["v_i8"],
            tensors["k_sc"], tensors["v_sc"],
            tensors["page_table"], tensors["page_precision"],
            pc, config,
        )
        assert out.abs().max().item() < 1e-6

    def test_single_selected_page(self, tensors, config):
        """Single selected page -> verify shape."""
        pc = torch.ones(_BATCH, _HEADS, dtype=torch.int32)
        out = fused_selected_quant_decode_reference(
            tensors["query"], tensors["k_fp16"], tensors["v_fp16"],
            tensors["k_i8"], tensors["v_i8"],
            tensors["k_sc"], tensors["v_sc"],
            tensors["page_table"], tensors["page_precision"],
            pc, config,
        )
        assert out.shape == (_BATCH, _HEADS, 1, _HEAD_DIM)

    def test_different_page_counts_per_head(self, tensors, config):
        """Different batch/head have different selected pages."""
        pc = torch.full((_BATCH, _HEADS), 2, dtype=torch.int32)
        pc[0, 0] = 1
        pc[_BATCH - 1, _HEADS - 1] = 3
        out = fused_selected_quant_decode_reference(
            tensors["query"], tensors["k_fp16"], tensors["v_fp16"],
            tensors["k_i8"], tensors["v_i8"],
            tensors["k_sc"], tensors["v_sc"],
            tensors["page_table"], tensors["page_precision"],
            pc, config,
        )
        assert out.shape == (_BATCH, _HEADS, 1, _HEAD_DIM)


# ------------------------------------------------------------------ #
#  Metadata conversion tests
# ------------------------------------------------------------------ #


class TestMetadataToKernelTensors:
    def test_basic_conversion(self):
        meta = {
            "selected_page_ids": [0, 3, 5],
            "block_precision_by_page": {
                "0": "FP16",
                "3": "INT8",
                "5": "SKIP",
            },
        }
        pt, prec, pc = metadata_to_kernel_tensors(meta, num_pages=8)
        assert pt.shape == (1, 1, 64)
        assert prec.shape == (8,)
        assert pc.shape == (1, 1)
        assert prec[0] == FusedKVPrecision.FP16
        assert prec[3] == FusedKVPrecision.INT8
        assert prec[5] == FusedKVPrecision.SKIP
        assert pc[0, 0].item() == 3

    def test_empty_selection(self):
        meta = {
            "selected_page_ids": [],
            "block_precision_by_page": {},
        }
        pt, prec, pc = metadata_to_kernel_tensors(meta, num_pages=4)
        assert pt.sum().item() == 0
        assert pc.sum().item() == 0
        # Default precision should be FP16
        assert all(prec == FusedKVPrecision.FP16)


# ------------------------------------------------------------------ #
#  Public API dispatch test
# ------------------------------------------------------------------ #


class TestFusedSelectedQuantDecode:
    def test_cpu_fallback(self, tensors, config):
        """On CPU, dispatch should go to reference."""
        out = fused_selected_quant_decode(
            tensors["query"], tensors["k_fp16"], tensors["v_fp16"],
            tensors["k_i8"], tensors["v_i8"],
            tensors["k_sc"], tensors["v_sc"],
            tensors["page_table"], tensors["page_precision"],
            tensors["page_counts"], config,
        )
        assert out.shape == (_BATCH, _HEADS, 1, _HEAD_DIM)

    def test_head_dim_mismatch_raises(self, tensors):
        with pytest.raises(ValueError, match="head_dim"):
            bad_config = FusedDecodeConfig(head_dim=128)
            fused_selected_quant_decode(
                tensors["query"], tensors["k_fp16"], tensors["v_fp16"],
                tensors["k_i8"], tensors["v_i8"],
                tensors["k_sc"], tensors["v_sc"],
                tensors["page_table"], tensors["page_precision"],
                tensors["page_counts"], bad_config,
            )


# ------------------------------------------------------------------ #
#  GPU smoke test (only with CUDA + Triton)
# ------------------------------------------------------------------ #


@pytest.mark.skipif(
    not is_triton_available(),
    reason="Triton not available; skip GPU kernel test",
)
class TestFusedSelectedQuantDecodeGPU:
    def test_gpu_output_shape(self, tensors, config):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device("cuda")
        gpu_tensors = {k: v.to(device) for k, v in tensors.items()}
        out = fused_selected_quant_decode(
            gpu_tensors["query"], gpu_tensors["k_fp16"],
            gpu_tensors["v_fp16"],
            gpu_tensors["k_i8"], gpu_tensors["v_i8"],
            gpu_tensors["k_sc"], gpu_tensors["v_sc"],
            gpu_tensors["page_table"], gpu_tensors["page_precision"],
            gpu_tensors["page_counts"], config,
        )
        assert out.shape == (_BATCH, _HEADS, 1, _HEAD_DIM)
        assert out.is_cuda

    def test_gpu_vs_reference(self, tensors, config):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device("cuda")
        gpu_tensors = {k: v.to(device) for k, v in tensors.items()}

        ref = fused_selected_quant_decode_reference(
            tensors["query"], tensors["k_fp16"], tensors["v_fp16"],
            tensors["k_i8"], tensors["v_i8"],
            tensors["k_sc"], tensors["v_sc"],
            tensors["page_table"], tensors["page_precision"],
            tensors["page_counts"], config,
        )
        gpu = fused_selected_quant_decode(
            gpu_tensors["query"], gpu_tensors["k_fp16"],
            gpu_tensors["v_fp16"],
            gpu_tensors["k_i8"], gpu_tensors["v_i8"],
            gpu_tensors["k_sc"], gpu_tensors["v_sc"],
            gpu_tensors["page_table"], gpu_tensors["page_precision"],
            gpu_tensors["page_counts"], config,
        )

        diff = (ref.float() - gpu.cpu().float()).abs().max().item()
        assert diff < 1e-1, f"GPU vs reference max diff: {diff}"
