from __future__ import annotations

import torch

_triton_available: bool = False
_cuda_available: bool = torch.cuda.is_available()


def _probe_triton() -> bool:
    try:
        import triton  # noqa: F401
        import triton.language as tl  # noqa: F401
        return True
    except ImportError:
        return False


_triton_available = _probe_triton()


def is_triton_available() -> bool:
    return _triton_available


def is_cuda_available() -> bool:
    return _cuda_available


def semantic_block_attention_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    layout,
) -> torch.Tensor:
    """Placeholder for a future Triton-accelerated semantic block attention.

    Currently a stub:
    - On CPU-only machines: falls back to the PyTorch reference.
    - When Triton+CUDA are present: raises NotImplementedError (kernel not written).
    """
    if not is_triton_available() or not is_cuda_available():
        from .reference import semantic_block_attention as _fallback
        return _fallback(q, k, v, layout)

    raise NotImplementedError(
        "GPU kernel not yet implemented. This function is a stub that proves "
        "the interface only. The real Triton/CUDA kernel requires hardware validation."
    )
