"""
CPU Adaptive KV Runtime — smart memory layer for KV cache.

Orchestrates per-page format assignment, self-tuning demotion/promotion,
partial-page mask support, prefetch warmup, and adaptive-format attention.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, List, Optional, Tuple

import torch

from .block_metadata import BlockLayout, BlockPolicy
from .adaptive_format_attention import adaptive_format_attention_reference
from .prefetch import BlockPrefetcher


class PageStorageFormat(IntEnum):
    FP16 = 0
    INT8 = 1
    SPARSE = 2
    SKIP = 3


@dataclass
class PageState:
    page_id: int
    format: PageStorageFormat = PageStorageFormat.FP16
    access_count: int = 0
    last_access_step: int = -1
    block_name: str = ""
    block_policy: str = ""
    score: float = 0.0
    numa_hint: int = 0

    token_start: int = 0
    token_end: int = 0

    kv_fp16: Optional[torch.Tensor] = None
    kv_int8: Optional[torch.Tensor] = None
    kv_int8_scale: float = 1.0

    sp_k_indices: Optional[torch.Tensor] = None
    sp_k_values: Optional[torch.Tensor] = None
    sp_v_indices: Optional[torch.Tensor] = None
    sp_v_values: Optional[torch.Tensor] = None
    sp_nnz: int = 0

    partial_page_mask: Optional[torch.Tensor] = None

    def __post_init__(self):
        self.format = PageStorageFormat(self.format)


@dataclass
class PageFormatPolicy:
    """Maps block policies and access patterns to storage formats.

    Thresholds may be tuned automatically when
    ``KVMemoryManager.adapt_policy_every`` > 0.
    """

    always_format: PageStorageFormat = PageStorageFormat.FP16
    global_format: PageStorageFormat = PageStorageFormat.FP16
    recent_format: PageStorageFormat = PageStorageFormat.FP16
    attend_high_format: PageStorageFormat = PageStorageFormat.FP16
    attend_low_format: PageStorageFormat = PageStorageFormat.INT8
    skip_format: PageStorageFormat = PageStorageFormat.SKIP
    score_threshold: float = 0.5
    demote_cold_after_steps: int = 10
    demote_cold_to: PageStorageFormat = PageStorageFormat.INT8
    promote_hot_after_accesses: int = 3
    promote_hot_to: PageStorageFormat = PageStorageFormat.FP16

    demote_step_min: int = 2
    demote_step_max: int = 50
    promote_access_min: int = 1
    promote_access_max: int = 20


def _format_for_block(policy: str, score: float, policy_cfg: PageFormatPolicy) -> PageStorageFormat:
    if policy == BlockPolicy.ALWAYS.value:
        return policy_cfg.always_format
    if policy == BlockPolicy.GLOBAL.value:
        return policy_cfg.global_format
    if policy == BlockPolicy.RECENT.value:
        return policy_cfg.recent_format
    if policy == BlockPolicy.SKIP.value:
        return policy_cfg.skip_format
    if policy == BlockPolicy.ATTEND.value:
        return policy_cfg.attend_high_format if score >= policy_cfg.score_threshold else policy_cfg.attend_low_format
    return policy_cfg.attend_low_format


def _page_page_mask(page_size: int, token_start: int, token_end: int) -> torch.Tensor:
    """Build a boolean mask for valid tokens within a single partial page."""
    mask = torch.zeros(page_size, dtype=torch.bool)
    lo = token_start % page_size
    hi = lo + (token_end - token_start)
    hi = min(hi, page_size)
    mask[lo:hi] = True
    return mask


class KVMemoryManager:
    """Orchestrates smart KV cache memory management on CPU.

    Features:
    - Per-page storage format assignment (FP16/INT8/SPARSE/SKIP)
    - Partial-page token masks for precise block boundaries
    - Self-tuning demotion/promotion thresholds
    - Prefetch-aware warmup (promote predicted pages)
    - Adaptive-format attention dispatch
    """

    def __init__(
        self,
        num_pages: int,
        page_size: int,
        head_dim: int,
        policy: Optional[PageFormatPolicy] = None,
        device: str = "cpu",
        sparse_max_nnz: int = 8,
        adapt_policy_every: int = 5,
    ):
        self.num_pages = num_pages
        self.page_size = page_size
        self.head_dim = head_dim
        self.policy = policy or PageFormatPolicy()
        self.device = torch.device(device)
        self.sparse_max_nnz = sparse_max_nnz
        self.adapt_policy_every = adapt_policy_every
        self.step_count = 0

        self.pages: Dict[int, PageState] = {}
        self.block_name_to_page_ids: Dict[str, List[int]] = {}

        self._prefetcher = BlockPrefetcher()
        self._prefetch_predictions: List[int] = []

        total_elements = num_pages * page_size * head_dim
        self._fp16_pool = torch.empty(total_elements, dtype=torch.float16, device=self.device)
        self._int8_pool = torch.empty(total_elements, dtype=torch.int8, device=self.device)

        self._demote_history: deque[bool] = deque(maxlen=50)
        self._hot_access_history: List[int] = []

    def _offset(self, page_id: int) -> int:
        return page_id * self.page_size * self.head_dim

    def _grow_pool(self, needed: int) -> None:
        current_pages = self._fp16_pool.numel() // (self.page_size * self.head_dim)
        if needed <= current_pages:
            return
        grow_to = max(needed, current_pages * 2)
        new_fp16 = torch.empty(grow_to * self.page_size * self.head_dim, dtype=torch.float16, device=self.device)
        new_fp16[:self._fp16_pool.numel()] = self._fp16_pool
        self._fp16_pool = new_fp16
        new_i8 = torch.empty(grow_to * self.page_size * self.head_dim, dtype=torch.int8, device=self.device)
        new_i8[:self._int8_pool.numel()] = self._int8_pool
        self._int8_pool = new_i8

    def _fp16_slice(self, page_id: int, grow: bool = False) -> torch.Tensor:
        self._grow_pool(page_id + 1) if grow else None
        o = self._offset(page_id)
        return self._fp16_pool[o: o + self.page_size * self.head_dim].view(self.page_size, self.head_dim)

    def _int8_slice(self, page_id: int, grow: bool = False) -> torch.Tensor:
        self._grow_pool(page_id + 1) if grow else None
        o = self._offset(page_id)
        return self._int8_pool[o: o + self.page_size * self.head_dim].view(self.page_size, self.head_dim)

    def register_layout(self, layout: BlockLayout) -> None:
        """Register a semantic block layout, assign page formats and token bounds."""
        self.block_name_to_page_ids.clear()
        next_page = 0
        for block in layout.blocks:
            n_tokens = block.end - block.start
            n_pages = (n_tokens + self.page_size - 1) // self.page_size
            ids = []
            for i in range(n_pages):
                pid = next_page + i
                policy_str = str(block.policy.value) if hasattr(block.policy, 'value') else str(block.policy)
                fmt = _format_for_block(policy_str, block.score or 0.0, self.policy)

                page_lo = block.start + i * self.page_size
                page_hi = min(block.start + (i + 1) * self.page_size, block.end)
                full_page = (page_lo >= block.start and page_hi <= block.end and
                             (page_hi - page_lo) == self.page_size)
                partial_mask = None if full_page else _page_page_mask(self.page_size, page_lo, page_hi)

                self.pages[pid] = PageState(
                    page_id=pid,
                    format=fmt,
                    block_name=block.name,
                    block_policy=policy_str,
                    score=block.score or 0.0,
                    token_start=page_lo,
                    token_end=page_hi,
                    partial_page_mask=partial_mask,
                )
                ids.append(pid)
            self.block_name_to_page_ids[block.name] = ids
            next_page += n_pages

    def allocate_pages(
        self,
        block_name: str,
        block_policy: str,
        num_pages: int,
        score: float = 0.0,
        token_start: int = 0,
        token_end: int = 0,
    ) -> List[int]:
        pid_start = len(self.pages)
        ids = []
        for i in range(num_pages):
            pid = pid_start + i
            fmt = _format_for_block(block_policy, score, self.policy)
            self.pages[pid] = PageState(
                page_id=pid, format=fmt, block_name=block_name,
                block_policy=block_policy, score=score,
                token_start=token_start, token_end=token_end,
            )
            ids.append(pid)
        self.block_name_to_page_ids.setdefault(block_name, []).extend(ids)
        return ids

    def write_page(self, page_id: int, kv: torch.Tensor) -> None:
        if page_id not in self.pages:
            raise ValueError(f"page_id {page_id} not allocated")
        state = self.pages[page_id]

        if state.format == PageStorageFormat.SKIP:
            return

        if state.format == PageStorageFormat.FP16:
            self._fp16_slice(page_id, grow=True).copy_(kv.to(torch.float16))
            state.kv_fp16 = self._fp16_slice(page_id)

        elif state.format == PageStorageFormat.INT8:
            kv_fp16 = kv.to(torch.float16)
            max_abs = kv_fp16.abs().max().item()
            scale = max_abs / 127.0 if max_abs > 0 else 1.0
            i8 = (kv_fp16 / scale).round().clamp(-128, 127).to(torch.int8)
            self._int8_slice(page_id, grow=True).copy_(i8)
            state.kv_int8 = self._int8_slice(page_id)
            state.kv_int8_scale = scale
            state.kv_fp16 = None

        elif state.format == PageStorageFormat.SPARSE:
            kv_fp16 = kv.to(torch.float16)
            flat = kv_fp16.flatten()
            k = min(self.sparse_max_nnz, flat.numel())
            vals, idxs = flat.abs().topk(k)
            sp_vals = flat[idxs].to(torch.float16)
            sp_idxs = idxs.to(torch.int64)
            state.sp_k_indices = sp_idxs
            state.sp_k_values = sp_vals
            state.sp_v_indices = sp_idxs
            state.sp_v_values = sp_vals
            state.sp_nnz = k

    def set_page_format(self, page_id: int, new_fmt: PageStorageFormat) -> None:
        if page_id not in self.pages:
            return
        state = self.pages[page_id]
        if state.format == new_fmt:
            return
        kv_data = self._read_page_data(page_id)
        state.format = new_fmt
        state.kv_fp16 = None
        state.kv_int8 = None
        state.sp_k_indices = None
        state.sp_k_values = None
        state.sp_v_indices = None
        state.sp_v_values = None
        state.sp_nnz = 0
        if kv_data is not None:
            self.write_page(page_id, kv_data)

    def _read_page_data(self, page_id: int) -> Optional[torch.Tensor]:
        state = self.pages.get(page_id)
        if state is None:
            return None
        if state.format == PageStorageFormat.FP16 and state.kv_fp16 is not None:
            return state.kv_fp16.clone()
        if state.format == PageStorageFormat.INT8 and state.kv_int8 is not None:
            return (state.kv_int8.to(torch.float32) * state.kv_int8_scale).to(torch.float16)
        if state.format == PageStorageFormat.SPARSE and state.sp_k_values is not None:
            out = torch.zeros(self.page_size * self.head_dim, dtype=torch.float16)
            out[state.sp_k_indices] = state.sp_k_values
            return out.view(self.page_size, self.head_dim)
        if state.format == PageStorageFormat.SKIP:
            return torch.zeros(self.page_size, self.head_dim, dtype=torch.float16)
        return None

    # ──────────────────────────────────────────────
    #  Self-tuning policy
    # ──────────────────────────────────────────────

    def _adapt_policy(self) -> None:
        """Tune demotion/promotion thresholds based on access history."""
        if self.step_count % self.adapt_policy_every != 0:
            return

        recent_hits = sum(self._demote_history)
        recent_total = len(self._demote_history)
        hit_rate = recent_hits / max(recent_total, 1)

        cold_fp16 = 0
        total_fp16 = 0
        for s in self.pages.values():
            if s.format not in (PageStorageFormat.FP16,):
                continue
            total_fp16 += 1
            steps_since = self.step_count - s.last_access_step if s.last_access_step >= 0 else self.step_count
            if steps_since >= self.policy.demote_cold_after_steps:
                cold_fp16 += 1

        cold_rate = cold_fp16 / max(total_fp16, 1) if total_fp16 > 0 else 0.0

        if hit_rate > 0.30 and self.policy.demote_cold_after_steps < self.policy.demote_step_max:
            self.policy.demote_cold_after_steps += 1
        if cold_rate > 0.50 and self.policy.demote_cold_after_steps > self.policy.demote_step_min:
            self.policy.demote_cold_after_steps -= 1

        hot_count = sum(
            1 for s in self.pages.values()
            if s.format == PageStorageFormat.INT8 and s.access_count >= self.policy.promote_hot_after_accesses
        )
        if hot_count == 0 and total_fp16 > 0 and cold_rate < 0.2:
            if self.policy.promote_hot_after_accesses < self.policy.promote_access_max:
                self.policy.promote_hot_after_accesses += 1

    def _prefetch_warmup(self) -> List[int]:
        """Pre-emptively promote prefetch-predicted INT8 pages to FP16."""
        warmed = []
        for pid in self._prefetch_predictions:
            s = self.pages.get(pid)
            if s is not None and s.format == PageStorageFormat.INT8:
                self.set_page_format(pid, self.policy.promote_hot_to)
                warmed.append(pid)
        return warmed

    def demote_cold_pages(self) -> List[int]:
        """Demote infrequently accessed pages to INT8."""
        demoted = []
        for pid, state in list(self.pages.items()):
            if state.format in (PageStorageFormat.SKIP, PageStorageFormat.INT8, PageStorageFormat.SPARSE):
                continue
            steps_since_access = self.step_count - state.last_access_step
            if state.last_access_step >= 0 and steps_since_access >= self.policy.demote_cold_after_steps:
                self.set_page_format(pid, self.policy.demote_cold_to)
                demoted.append(pid)
                was_reaccessed = steps_since_access < self.policy.demote_cold_after_steps * 2
                self._demote_history.append(was_reaccessed)
        return demoted

    def promote_hot_pages(self) -> List[int]:
        """Promote frequently-accessed INT8 pages back to FP16."""
        promoted = []
        for pid, state in list(self.pages.items()):
            if state.format != PageStorageFormat.INT8:
                continue
            if state.access_count >= self.policy.promote_hot_after_accesses:
                self.set_page_format(pid, self.policy.promote_hot_to)
                promoted.append(pid)
        return promoted

    # ──────────────────────────────────────────────
    #  Page selection with masks
    # ──────────────────────────────────────────────

    def select_pages(
        self,
    ) -> Tuple[List[int], torch.Tensor, torch.Tensor, Optional[Dict[int, torch.Tensor]]]:
        """Return selected pages, metadata, and optional per-page token masks.

        Returns:
            selected_pids: list of page IDs to attend to.
            page_formats_t: int32 tensor [num_pages] of format tags.
            page_ids_t: int32 tensor [1, 1, max_sel] of selected page IDs.
            page_masks: dict of {pid: bool mask [page_size]} for partial pages, or None.
        """
        selected = [
            pid for pid, s in self.pages.items()
            if s.format != PageStorageFormat.SKIP
            and s.block_policy != BlockPolicy.SKIP.value
        ]
        num_pages = len(self.pages) or 1
        page_formats_t = torch.zeros(num_pages, dtype=torch.int32)
        for pid, s in self.pages.items():
            page_formats_t[pid] = int(s.format)

        max_sel = max(len(selected), 1)
        page_ids_t = torch.full((1, 1, max_sel), fill_value=-1, dtype=torch.int32)
        for i, pid in enumerate(selected):
            page_ids_t[0, 0, i] = pid

        page_masks = None
        for pid in selected:
            s = self.pages.get(pid)
            if s is not None and s.partial_page_mask is not None:
                if page_masks is None:
                    page_masks = {}
                page_masks[pid] = s.partial_page_mask

        return selected, page_formats_t, page_ids_t, page_masks

    def _apply_token_masks(
        self,
        fp16_kv: torch.Tensor,
        int8_kv: torch.Tensor,
        page_ids_t: torch.Tensor,
        page_masks: Optional[Dict[int, torch.Tensor]],
    ) -> None:
        """Zero out masked token rows within each partial page, in-place."""
        if page_masks is None:
            return
        B, H, W = page_ids_t.shape
        for bi in range(B):
            for hi in range(H):
                for wi in range(W):
                    pid = page_ids_t[bi, hi, wi].item()
                    if pid < 0:
                        continue
                    pm = page_masks.get(pid)
                    if pm is None:
                        continue
                    if pid < fp16_kv.shape[0]:
                        fp16_kv[pid, ~pm] = 0.0
                    if pid < int8_kv.shape[0]:
                        int8_kv[pid, ~pm] = 0

    # ──────────────────────────────────────────────
    #  Attention
    # ──────────────────────────────────────────────

    def _build_attention_tensors(
        self,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        NP = len(self.pages) or 1
        D = self.head_dim
        PS = self.page_size
        _d = torch.device(self.device)

        fp16_kv = torch.zeros(NP, PS, D, dtype=torch.float16, device=_d)
        int8_kv = torch.zeros(NP, PS, D, dtype=torch.int8, device=_d)
        int8_scales = torch.ones(NP, dtype=torch.float16, device=_d)
        sp_k_idx = torch.zeros(NP, self.sparse_max_nnz, dtype=torch.int64, device=_d)
        sp_k_val = torch.zeros(NP, self.sparse_max_nnz, dtype=torch.float16, device=_d)
        sp_v_idx = torch.zeros(NP, self.sparse_max_nnz, dtype=torch.int64, device=_d)
        sp_v_val = torch.zeros(NP, self.sparse_max_nnz, dtype=torch.float16, device=_d)
        sp_nnz = torch.zeros(NP, dtype=torch.int32, device=_d)

        for pid, s in self.pages.items():
            if s.format == PageStorageFormat.FP16 and s.kv_fp16 is not None:
                fp16_kv[pid] = s.kv_fp16
            elif s.format == PageStorageFormat.INT8 and s.kv_int8 is not None:
                int8_kv[pid] = s.kv_int8
                int8_scales[pid] = s.kv_int8_scale
            elif s.format == PageStorageFormat.SPARSE and s.sp_k_values is not None:
                k = min(s.sp_nnz, self.sparse_max_nnz)
                sp_k_idx[pid, :k] = s.sp_k_indices[:k]
                sp_k_val[pid, :k] = s.sp_k_values[:k]
                sp_v_idx[pid, :k] = s.sp_v_indices[:k] if s.sp_v_indices is not None else s.sp_k_indices[:k]
                sp_v_val[pid, :k] = s.sp_v_values[:k] if s.sp_v_values is not None else s.sp_k_values[:k]
                sp_nnz[pid] = k

        class _Cfg:
            page_size = PS
            head_dim = D

        return fp16_kv, int8_kv, int8_scales, sp_k_idx, sp_k_val, sp_v_idx, sp_v_val, sp_nnz, _Cfg()

    def step(
        self,
        query: torch.Tensor,
        demote: bool = False,
        promote: bool = False,
        warmup: bool = False,
        adapt: bool = False,
        mask: bool = False,
    ) -> torch.Tensor:
        """Execute one decode step.

        Args:
            query: [B, H, D] query tensor.
            demote: run cold-page demotion before this step.
            promote: run hot-page promotion before this step.
            warmup: run prefetch warmup before this step.
            adapt: run policy self-tuning after this step.
            mask: apply partial-page token masks.

        Returns:
            out: [B, H, D] attention output.
        """
        self.step_count += 1

        if warmup:
            self._prefetch_warmup()
        if demote:
            self.demote_cold_pages()
        if promote:
            self.promote_hot_pages()

        selected, page_formats_t, page_ids_t, page_masks = self.select_pages()

        for pid in selected:
            s = self.pages.get(pid)
            if s is not None:
                s.access_count += 1
                s.last_access_step = self.step_count

        B, H, D = query.shape
        q_4d = query.unsqueeze(2)

        (
            fp16_kv, int8_kv, int8_scales,
            sp_k_idx, sp_k_val, sp_v_idx, sp_v_val, sp_nnz, cfg,
        ) = self._build_attention_tensors()

        if mask:
            self._apply_token_masks(fp16_kv, int8_kv, page_ids_t, page_masks)

        page_counts_t = torch.full((1, 1), fill_value=len(selected), dtype=torch.int32)

        out_4d = adaptive_format_attention_reference(
            q_4d, fp16_kv, int8_kv, int8_scales,
            sp_k_idx, sp_k_val, page_formats_t, page_ids_t, page_counts_t, cfg,
        )

        self._prefetcher.record(selected)
        self._prefetch_predictions = self._prefetcher.predict_next(selected)

        if adapt:
            self._adapt_policy()

        return out_4d.squeeze(2)

    def predict_prefetch(self) -> List[int]:
        return self._prefetch_predictions

    def page_summary(self) -> Dict:
        fmt_counts = {f.name: 0 for f in PageStorageFormat}
        total_accesses = 0
        cold_count = 0

        for s in self.pages.values():
            fmt_counts[PageStorageFormat(s.format).name] += 1
            total_accesses += s.access_count
            steps_since = self.step_count - s.last_access_step if s.last_access_step >= 0 else self.step_count
            if steps_since >= self.policy.demote_cold_after_steps:
                cold_count += 1

        return {
            "num_pages": len(self.pages),
            "step": self.step_count,
            "format_distribution": fmt_counts,
            "total_accesses": total_accesses,
            "cold_pages": cold_count,
            "prefetch_predictions": len(self._prefetch_predictions),
            "demote_after_steps": self.policy.demote_cold_after_steps,
            "promote_after_accesses": self.policy.promote_hot_after_accesses,
        }
