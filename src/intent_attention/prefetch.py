from __future__ import annotations

from collections import Counter, deque
from typing import List, Optional

import torch

__all__ = [
    "BlockPrefetcher",
    "get_prefetcher",
    "reset_prefetcher",
    "get_prefetch_stream",
    "launch_prefetch_pages",
]

# ------------------------------------------------------------------ #
#  BlockPrefetcher — frequency-based prediction of next-step blocks
# ------------------------------------------------------------------ #


class BlockPrefetcher:
    """Predicts which KV blocks will be attended at the next decode step
    by tracking recent selection history.

    The prediction uses a simple frequency heuristic: a block is predicted
    if it appears in at least ``min_frequency`` of the last ``history_size``
    completed steps *plus* the current step (passed to ``predict_next``).
    """

    def __init__(self, history_size: int = 4, min_frequency: int = 3) -> None:
        self.history_size = history_size
        self.min_frequency = min_frequency
        self._history: deque[frozenset[int]] = deque(maxlen=history_size)

    def predict_next(self, current_selected: List[int]) -> List[int]:
        """Return sorted list of block IDs predicted for the next step.

        The prediction window is ``list(self._history) + [current_selected]``.
        An ID is predicted iff it appears in at least ``min_frequency`` of
        those sets.
        """
        window = list(self._history) + [frozenset(current_selected)]
        if len(window) < self.min_frequency:
            return []
        counter: Counter[int] = Counter()
        for s in window:
            counter.update(s)
        predicted = [bid for bid, cnt in counter.items() if cnt >= self.min_frequency]
        return sorted(predicted)

    def record(self, selected: List[int]) -> None:
        """Store *selected* (the step just completed) into history."""
        self._history.append(frozenset(selected))

    def reset(self) -> None:
        """Clear history (call when the layout changes)."""
        self._history.clear()


# ------------------------------------------------------------------ #
#  Module-level globals — persistent prefetcher + CUDA stream
# ------------------------------------------------------------------ #

_prefetcher: Optional[BlockPrefetcher] = None
_prefetch_stream: Optional["torch.cuda.Stream"] = None


def get_prefetcher() -> BlockPrefetcher:
    global _prefetcher
    if _prefetcher is None:
        _prefetcher = BlockPrefetcher()
    return _prefetcher


def reset_prefetcher() -> None:
    global _prefetcher
    if _prefetcher is not None:
        _prefetcher.reset()


def get_prefetch_stream() -> Optional["torch.cuda.Stream"]:
    global _prefetch_stream
    if _prefetch_stream is None and torch.cuda.is_available():
        _prefetch_stream = torch.cuda.Stream()
    return _prefetch_stream


# ------------------------------------------------------------------ #
#  Triton prefetch kernel — load KV pages into L2 cache
# ------------------------------------------------------------------ #

_triton_available: bool = False


def _probe_triton() -> bool:
    try:
        import triton  # noqa: F401
        import triton.language as tl  # noqa: F401

        return True
    except ImportError:
        return False


_triton_available = _probe_triton()

if _triton_available:
    import triton
    import triton.language as tl

    @triton.jit
    def _prefetch_kv_pages_kernel(
        K,
        V,
        prefetch_buf,
        page_ids,
        stride_kb,
        stride_kh,
        stride_kn,
        stride_kd,
        stride_vb,
        stride_vh,
        stride_vn,
        stride_vd,
        stride_pb,
        stride_ph,
        stride_pp,
        stride_ps,
        stride_pd,
        kv_len: tl.int32,
        PAGE_SIZE: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        pid_b = tl.program_id(0)
        pid_h = tl.program_id(1)
        pid_p = tl.program_id(2)

        page_id = tl.load(page_ids + pid_p)

        offs_n = page_id * PAGE_SIZE + tl.arange(0, PAGE_SIZE)
        offs_d = tl.arange(0, BLOCK_D)
        n_mask = offs_n[:, None] < kv_len

        k_ptrs = (
            K
            + pid_b * stride_kb
            + pid_h * stride_kh
            + offs_n[:, None] * stride_kn
            + offs_d[None, :] * stride_kd
        )
        k_val = tl.load(k_ptrs, mask=n_mask, other=0.0, cache_modifier=".cg")

        v_ptrs = (
            V
            + pid_b * stride_vb
            + pid_h * stride_vh
            + offs_n[:, None] * stride_vn
            + offs_d[None, :] * stride_vd
        )
        tl.load(v_ptrs, mask=n_mask, other=0.0, cache_modifier=".cg")

        buf_ptrs = (
            prefetch_buf
            + pid_b * stride_pb
            + pid_h * stride_ph
            + pid_p * stride_pp
            + offs_n[:, None] * stride_ps
            + offs_d[None, :] * stride_pd
        )
        tl.store(buf_ptrs, k_val.to(tl.float16), mask=n_mask)

else:
    # Stub so that import-time access does not error on CPU-only boxes.
    pass


# ------------------------------------------------------------------ #
#  Host launch helper
# ------------------------------------------------------------------ #


def launch_prefetch_pages(
    k: torch.Tensor,
    v: torch.Tensor,
    page_ids: torch.Tensor,
    stream: Optional["torch.cuda.Stream"] = None,
    page_size: int = 128,
) -> None:
    """Issue a Triton kernel that loads the given *page_ids* into L2 cache.

    The kernel uses ``cache_modifier=".cg"`` (L2-cache global hint).
    Correctness is unaffected if the kernel is skipped (no-op on CPU).
    """
    if not _triton_available or not torch.cuda.is_available():
        return

    num_pages = page_ids.shape[0]
    if num_pages == 0:
        return

    batch, heads, kv_len, d_head = k.shape

    prefetch_buf = torch.empty(
        batch,
        heads,
        num_pages,
        page_size,
        d_head,
        dtype=torch.float16,
        device=k.device,
    )

    grid = (batch, heads, num_pages)

    if stream is not None:
        with torch.cuda.stream(stream):
            _prefetch_kv_pages_kernel[grid](
                k,
                v,
                prefetch_buf,
                page_ids,
                k.stride(0),
                k.stride(1),
                k.stride(2),
                k.stride(3),
                v.stride(0),
                v.stride(1),
                v.stride(2),
                v.stride(3),
                prefetch_buf.stride(0),
                prefetch_buf.stride(1),
                prefetch_buf.stride(2),
                prefetch_buf.stride(3),
                prefetch_buf.stride(4),
                kv_len,
                PAGE_SIZE=page_size,
                BLOCK_D=d_head,
            )
    else:
        _prefetch_kv_pages_kernel[grid](
            k,
            v,
            prefetch_buf,
            page_ids,
            k.stride(0),
            k.stride(1),
            k.stride(2),
            k.stride(3),
            v.stride(0),
            v.stride(1),
            v.stride(2),
            v.stride(3),
            prefetch_buf.stride(0),
            prefetch_buf.stride(1),
            prefetch_buf.stride(2),
            prefetch_buf.stride(3),
            prefetch_buf.stride(4),
            kv_len,
            PAGE_SIZE=page_size,
            BLOCK_D=d_head,
        )
