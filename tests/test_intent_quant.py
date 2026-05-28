from __future__ import annotations

import torch

from intent_attention.block_metadata import BlockLayout, BlockPolicy, SemanticBlock
from intent_attention.intent_quant import (
    KVPrecision,
    IntentQuantizer,
    QuantPolicy,
    compute_quant_error,
    fake_quantize_tensor,
)


def test_always_global_high_precision() -> None:
    quant = IntentQuantizer(memory_pressure=0.0)
    always = SemanticBlock("sys", 0, 128, BlockPolicy.ALWAYS)
    global_ = SemanticBlock("glb", 128, 256, BlockPolicy.GLOBAL)

    p_always = quant.assign_block_precision(always)
    p_global = quant.assign_block_precision(global_)

    assert p_always.precision == KVPrecision.FP16
    assert p_global.precision == KVPrecision.FP16


def test_always_global_not_below_int8() -> None:
    quant = IntentQuantizer(memory_pressure=0.95)
    always = SemanticBlock("sys", 0, 128, BlockPolicy.ALWAYS)
    global_ = SemanticBlock("glb", 128, 256, BlockPolicy.GLOBAL)

    p_always = quant.assign_block_precision(always)
    p_global = quant.assign_block_precision(global_)

    assert p_always.precision in (KVPrecision.FP16, KVPrecision.FP8, KVPrecision.INT8)
    assert p_global.precision in (KVPrecision.FP16, KVPrecision.FP8, KVPrecision.INT8)


def test_skip_gets_skip_or_int4() -> None:
    quant = IntentQuantizer(memory_pressure=0.0)
    skip = SemanticBlock("ignored", 0, 128, BlockPolicy.SKIP)
    p = quant.assign_block_precision(skip)
    assert p.precision == KVPrecision.SKIP


def test_low_score_gets_int4_or_skip() -> None:
    quant = IntentQuantizer(memory_pressure=0.0)
    low = SemanticBlock("low", 0, 128, BlockPolicy.ATTEND, score=0.1)
    p = quant.assign_block_precision(low)
    assert p.precision in (KVPrecision.INT4, KVPrecision.INT4_RESIDUAL, KVPrecision.SKIP)


def test_high_score_gets_int8_or_fp8() -> None:
    quant = IntentQuantizer(memory_pressure=0.0)
    high = SemanticBlock("high", 0, 128, BlockPolicy.ATTEND, score=0.9)
    p = quant.assign_block_precision(high)
    assert p.precision in (KVPrecision.FP16, KVPrecision.FP8, KVPrecision.INT8)


def test_memory_pressure_downgrades() -> None:
    low_quant = IntentQuantizer(memory_pressure=0.0)
    high_quant = IntentQuantizer(memory_pressure=0.9)

    attend = SemanticBlock("doc", 0, 128, BlockPolicy.ATTEND, score=0.8)
    recent = SemanticBlock("rec", 128, 256, BlockPolicy.RECENT)

    p_low = low_quant.assign_block_precision(attend)
    p_high = high_quant.assign_block_precision(attend)
    bytes_low = p_low.estimated_bytes_per_value
    bytes_high = p_high.estimated_bytes_per_value
    assert bytes_high <= bytes_low, "high pressure should downgrade or keep same"

    r_low = low_quant.assign_block_precision(recent)
    r_high = high_quant.assign_block_precision(recent)
    r_bytes_low = r_low.estimated_bytes_per_value
    r_bytes_high = r_high.estimated_bytes_per_value
    assert r_bytes_high <= r_bytes_low, "recent should downgrade under pressure"


def test_byte_savings_positive() -> None:
    quant = IntentQuantizer(memory_pressure=0.5)
    layout = BlockLayout(
        [
            SemanticBlock("sys", 0, 128, BlockPolicy.ALWAYS),
            SemanticBlock("doc1", 128, 640, BlockPolicy.ATTEND, score=0.9),
            SemanticBlock("doc2", 640, 1152, BlockPolicy.ATTEND, score=0.3),
            SemanticBlock("recent", 1152, 1280, BlockPolicy.RECENT),
            SemanticBlock("skip", 1280, 1408, BlockPolicy.SKIP),
        ]
    )
    est = quant.estimate_layout_bytes(layout, heads=32, head_dim=128)
    assert est["bytes_saved"] > 0
    assert est["bytes_saved_pct"] > 0.0
    assert est["total_tokens"] == 1408


def test_fake_quant_returns_same_shape() -> None:
    x = torch.randn(2, 4, 128, 64)
    for prec in (KVPrecision.FP16, KVPrecision.FP8, KVPrecision.INT8, KVPrecision.INT4):
        recon, meta = fake_quantize_tensor(x, prec)
        assert recon.shape == x.shape, f"shape mismatch for {prec}"
        assert isinstance(meta, dict)


def test_error_metrics_contain_keys() -> None:
    x = torch.randn(1, 2, 16, 32)
    for prec in (KVPrecision.FP8, KVPrecision.INT8, KVPrecision.INT4, KVPrecision.INT4_RESIDUAL):
        recon, meta = fake_quantize_tensor(x, prec)
        for key in ("mse", "max_abs_error", "cosine_similarity"):
            assert key in meta, f"missing {key} in {prec}"
        assert isinstance(meta["mse"], float)
        assert isinstance(meta["max_abs_error"], float)
        assert isinstance(meta["cosine_similarity"], float)


def test_compute_quant_error() -> None:
    orig = torch.randn(2, 4, 16, 64)
    recon = orig + torch.randn_like(orig) * 0.01
    err = compute_quant_error(orig, recon)
    for key in ("mse", "max_abs_error", "cosine_similarity"):
        assert key in err


def test_summary_table() -> None:
    quant = IntentQuantizer(memory_pressure=0.3)
    layout = BlockLayout(
        [
            SemanticBlock("sys", 0, 64, BlockPolicy.ALWAYS),
            SemanticBlock("doc", 64, 192, BlockPolicy.ATTEND, score=0.8),
            SemanticBlock("skip", 192, 256, BlockPolicy.SKIP),
        ]
    )
    rows = quant.summary_table(layout, heads=8, head_dim=64)
    assert len(rows) == 3
    assert rows[0]["block"] == "sys"
    assert rows[2]["block"] == "skip"
    for r in rows:
        assert "precision" in r
        assert "dense_fp16_bytes" in r
        assert "quant_bytes" in r


def test_assign_layout_precision() -> None:
    quant = IntentQuantizer()
    layout = BlockLayout(
        [
            SemanticBlock("a", 0, 64, BlockPolicy.ALWAYS),
            SemanticBlock("b", 64, 128, BlockPolicy.RECENT),
            SemanticBlock("c", 128, 192, BlockPolicy.SKIP),
        ]
    )
    policies = quant.assign_layout_precision(layout)
    assert len(policies) == 3
    assert "a" in policies
    assert "b" in policies
    assert "c" in policies


def test_precision_distribution() -> None:
    quant = IntentQuantizer(memory_pressure=0.5)
    layout = BlockLayout(
        [
            SemanticBlock("sys", 0, 64, BlockPolicy.ALWAYS),
            SemanticBlock("doc", 64, 192, BlockPolicy.ATTEND, score=0.9),
            SemanticBlock("ldoc", 192, 320, BlockPolicy.ATTEND, score=0.2),
            SemanticBlock("recent", 320, 384, BlockPolicy.RECENT),
            SemanticBlock("skip", 384, 448, BlockPolicy.SKIP),
        ]
    )
    est = quant.estimate_layout_bytes(layout, heads=1, head_dim=64)
    dist = est["precision_distribution"]
    assert isinstance(dist, dict)
    assert sum(dist.values()) == 448


def test_quant_policy_dataclass() -> None:
    p = QuantPolicy(KVPrecision.INT4_RESIDUAL, "test")
    assert p.estimated_bytes_per_value == 1.0
    assert p.requires_scale is True
    assert p.uses_residual is True

    p2 = QuantPolicy(KVPrecision.SKIP)
    assert p2.estimated_bytes_per_value == 0.0
    assert p2.requires_scale is False
    assert p2.uses_residual is False


def test_benchmark_import_no_cuda() -> None:
    import importlib

    spec = importlib.util.find_spec("benchmarks.bench_intent_quant")
    if spec is not None:
        import benchmarks.bench_intent_quant  # noqa: F401


def test_fp16_fake_quant_preserves_shape() -> None:
    x = torch.randn(1, 2, 8, 32)
    recon, meta = fake_quantize_tensor(x, KVPrecision.FP16)
    assert recon.shape == x.shape
    assert meta["precision"] == "fp16"


def test_skip_fake_quant_returns_zeros() -> None:
    x = torch.randn(1, 2, 8, 32)
    recon, _ = fake_quantize_tensor(x, KVPrecision.SKIP)
    assert (recon == 0.0).all()


def test_memory_pressure_clamp() -> None:
    try:
        IntentQuantizer(memory_pressure=-0.1)
        assert False, "should have raised"
    except ValueError:
        pass

    try:
        IntentQuantizer(memory_pressure=1.5)
        assert False, "should have raised"
    except ValueError:
        pass
