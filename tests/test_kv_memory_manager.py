"""Tests for the CPU Adaptive KV Runtime (KVMemoryManager)."""

from __future__ import annotations

import pytest
import torch

from intent_attention.kv_memory_manager import (
    KVMemoryManager,
    PageState,
    PageStorageFormat,
    PageFormatPolicy,
)
from intent_attention.block_metadata import BlockLayout, SemanticBlock, BlockPolicy


class TestPageStorageFormat:
    def test_enum_values_are_stable(self):
        assert PageStorageFormat.FP16.value == 0
        assert PageStorageFormat.INT8.value == 1
        assert PageStorageFormat.SPARSE.value == 2
        assert PageStorageFormat.SKIP.value == 3

    def test_enum_members(self):
        assert len(PageStorageFormat) == 4


class TestPageState:
    def test_default_format(self):
        s = PageState(page_id=0)
        assert s.format == PageStorageFormat.FP16

    def test_custom_format(self):
        s = PageState(page_id=0, format=PageStorageFormat.INT8)
        assert s.format == PageStorageFormat.INT8

    def test_format_coercion(self):
        s = PageState(page_id=0, format=2)
        assert s.format == PageStorageFormat.SPARSE


class TestPageFormatPolicy:
    def test_defaults(self):
        p = PageFormatPolicy()
        assert p.always_format == PageStorageFormat.FP16
        assert p.skip_format == PageStorageFormat.SKIP
        assert p.demote_cold_after_steps == 10
        assert p.promote_hot_after_accesses == 3


class TestKVMemoryManager:
    @pytest.fixture
    def small_manager(self):
        return KVMemoryManager(num_pages=8, page_size=4, head_dim=16)

    def test_init(self, small_manager):
        mgr = small_manager
        assert mgr.num_pages == 8
        assert mgr.page_size == 4
        assert mgr.head_dim == 16
        assert len(mgr.pages) == 0
        assert mgr.step_count == 0

    def test_allocate_pages(self, small_manager):
        mgr = small_manager
        ids = mgr.allocate_pages("system", "ALWAYS", 2)
        assert ids == [0, 1]
        assert mgr.pages[0].block_name == "system"
        assert mgr.pages[0].block_policy == "ALWAYS"
        assert mgr.pages[0].format == PageStorageFormat.FP16

    def test_allocate_skip(self, small_manager):
        mgr = small_manager
        ids = mgr.allocate_pages("scratch", "SKIP", 3, score=0.0)
        assert len(ids) == 3
        for pid in ids:
            assert mgr.pages[pid].format == PageStorageFormat.SKIP

    def test_write_and_read_fp16(self, small_manager):
        mgr = small_manager
        mgr.allocate_pages("block", "ALWAYS", 1)
        kv = torch.randn(4, 16, dtype=torch.float16)
        mgr.write_page(0, kv)
        assert mgr.pages[0].kv_fp16 is not None
        assert torch.allclose(mgr.pages[0].kv_fp16, kv)

    def test_write_and_read_int8(self, small_manager):
        mgr = small_manager
        mgr.policy.attend_low_format = PageStorageFormat.INT8
        mgr.allocate_pages("block", "ATTEND", 1, score=0.1)
        kv = torch.randn(4, 16, dtype=torch.float16)
        mgr.write_page(0, kv)
        assert mgr.pages[0].kv_int8 is not None
        assert mgr.pages[0].kv_int8_scale > 0
        # Read back should be close
        readback = mgr._read_page_data(0)
        assert readback is not None
        assert readback.shape == (4, 16)

    def test_write_skip_does_nothing(self, small_manager):
        mgr = small_manager
        mgr.allocate_pages("skip", "SKIP", 1)
        kv = torch.randn(4, 16)
        mgr.write_page(0, kv)
        s = mgr.pages[0]
        assert s.kv_fp16 is None
        assert s.kv_int8 is None

    def test_set_page_format(self, small_manager):
        mgr = small_manager
        mgr.allocate_pages("block", "ALWAYS", 1)
        kv = torch.randn(4, 16, dtype=torch.float16)
        mgr.write_page(0, kv)
        mgr.set_page_format(0, PageStorageFormat.INT8)
        assert mgr.pages[0].format == PageStorageFormat.INT8
        assert mgr.pages[0].kv_fp16 is None
        assert mgr.pages[0].kv_int8 is not None

    def test_demote_cold_pages(self, small_manager):
        mgr = small_manager
        mgr.allocate_pages("block", "ALWAYS", 2)
        for pid in range(2):
            mgr.write_page(pid, torch.randn(4, 16, dtype=torch.float16))
            mgr.pages[pid].last_access_step = 0
        mgr.step_count = 15  # well past threshold
        demoted = mgr.demote_cold_pages()
        assert len(demoted) == 2
        assert mgr.pages[0].format == PageStorageFormat.INT8
        assert mgr.pages[1].format == PageStorageFormat.INT8

    def test_promote_hot_pages(self, small_manager):
        mgr = small_manager
        mgr.policy.attend_low_format = PageStorageFormat.INT8
        mgr.allocate_pages("block", "ATTEND", 1, score=0.1)
        mgr.write_page(0, torch.randn(4, 16, dtype=torch.float16))
        mgr.pages[0].format = PageStorageFormat.INT8
        mgr.pages[0].access_count = 5  # above threshold
        promoted = mgr.promote_hot_pages()
        assert len(promoted) == 1
        assert mgr.pages[0].format == PageStorageFormat.FP16

    def test_select_pages_excludes_skip(self, small_manager):
        mgr = small_manager
        mgr.allocate_pages("keep", "ALWAYS", 2)
        mgr.allocate_pages("skip", "SKIP", 2)
        selected, _, page_ids_t = mgr.select_pages()
        assert len(selected) == 2
        assert 0 in selected
        assert 1 in selected
        assert all(pid not in selected for pid in (2, 3))

    def test_step_output_shape(self, small_manager):
        mgr = small_manager
        mgr.allocate_pages("system", "ALWAYS", 4)
        for pid in range(4):
            mgr.write_page(pid, torch.randn(4, 16, dtype=torch.float16))
        q = torch.randn(1, 1, 16, dtype=torch.float16)
        out = mgr.step(q)
        assert out.shape == (1, 1, 16)

    def test_step_updates_access_counts(self, small_manager):
        mgr = small_manager
        mgr.allocate_pages("system", "ALWAYS", 2)
        for pid in range(2):
            mgr.write_page(pid, torch.randn(4, 16, dtype=torch.float16))
        q = torch.randn(1, 1, 16, dtype=torch.float16)
        mgr.step(q)
        assert mgr.pages[0].access_count == 1
        assert mgr.pages[0].last_access_step == 1
        mgr.step(q)
        assert mgr.pages[0].access_count == 2

    def test_step_with_mixed_formats(self, small_manager):
        mgr = small_manager
        mgr.allocate_pages("fp16_block", "ALWAYS", 1)
        mgr.policy.attend_low_format = PageStorageFormat.INT8
        mgr.allocate_pages("int8_block", "ATTEND", 1, score=0.1)
        mgr.write_page(0, torch.randn(4, 16, dtype=torch.float16))
        mgr.write_page(1, torch.randn(4, 16, dtype=torch.float16))
        q = torch.randn(1, 1, 16, dtype=torch.float16)
        out = mgr.step(q)
        assert out.shape == (1, 1, 16)

    def test_register_layout(self, small_manager):
        mgr = small_manager
        layout = BlockLayout([
            SemanticBlock("sys", 0, 32, BlockPolicy.ALWAYS),
            SemanticBlock("doc", 32, 64, BlockPolicy.ATTEND, score=0.8),
        ])
        mgr.register_layout(layout)
        assert len(mgr.pages) > 0
        assert "sys" in mgr.block_name_to_page_ids
        assert "doc" in mgr.block_name_to_page_ids

    def test_page_summary(self, small_manager):
        mgr = small_manager
        mgr.allocate_pages("sys", "ALWAYS", 4)
        mgr.allocate_pages("doc", "ATTEND", 4, score=0.9)
        summary = mgr.page_summary()
        assert summary["num_pages"] == 8
        assert summary["format_distribution"]["FP16"] == 8
        assert summary["step"] == 0

    def test_prefetch_predictions(self, small_manager):
        mgr = small_manager
        mgr.allocate_pages("sys", "ALWAYS", 4)
        for pid in range(4):
            mgr.write_page(pid, torch.randn(4, 16, dtype=torch.float16))
        q = torch.randn(1, 1, 16, dtype=torch.float16)
        mgr.step(q)
        preds = mgr.predict_prefetch()
        # After a single step with history_size=4, min_frequency=3, no prediction
        assert isinstance(preds, list)

    def test_empty_manager_output(self):
        mgr = KVMemoryManager(num_pages=4, page_size=4, head_dim=16)
        q = torch.randn(1, 1, 16, dtype=torch.float16)
        out = mgr.step(q)
        # No pages selected -> zero output
        assert torch.allclose(out, torch.zeros_like(out), atol=1e-6)

    def test_demote_promote_cycle(self, small_manager):
        mgr = small_manager
        mgr.allocate_pages("block", "ALWAYS", 1)
        kv = torch.randn(4, 16, dtype=torch.float16)
        mgr.write_page(0, kv)

        # Force demotion
        mgr.pages[0].last_access_step = 0
        mgr.step_count = 15
        mgr.demote_cold_pages()
        assert mgr.pages[0].format == PageStorageFormat.INT8

        # Simulate many accesses to trigger promotion
        mgr.pages[0].access_count = 10
        mgr.promote_hot_pages()
        assert mgr.pages[0].format == PageStorageFormat.FP16
