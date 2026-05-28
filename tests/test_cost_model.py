from intent_attention.cost_model import (
    attention_flops,
    kv_read_bytes,
    semantic_attention_cost,
    savings_report,
)


class TestAttentionFlops:
    def test_basic(self):
        flops = attention_flops(1, 1, 128, 1024, 64)
        # 2 * 1 * 1 * 128 * 1024 * 64 (QK) + 2 * 1 * 1 * 128 * 1024 * 64 (PV)
        expected = 2 * 128 * 1024 * 64 + 2 * 128 * 1024 * 64
        assert flops == expected


class TestKvReadBytes:
    def test_basic(self):
        bytes_ = kv_read_bytes(1, 1, 1024, 64, dtype_bytes=2)
        # 2 * 1 * 1 * 1024 * 64 * 2
        assert bytes_ == 2 * 1024 * 64 * 2


class TestSemanticAttentionCost:
    def test_basic(self):
        cost = semantic_attention_cost(1, 1, 128, 1024, 512, 64)
        assert cost["flops"] == attention_flops(1, 1, 128, 512, 64)
        assert cost["kv_bytes"] == kv_read_bytes(1, 1, 512, 64, 2)


class TestSavingsReport:
    def test_half_selected(self):
        report = savings_report(1, 1, 128, 1024, 512, 64)
        assert report["selected_fraction"] == 0.5
        assert report["flops_saved_pct"] == 50.0
        assert report["kv_bytes_saved_pct"] == 50.0
        assert report["semantic_flops"] < report["dense_flops"]

    def test_all_selected(self):
        report = savings_report(1, 1, 128, 1024, 1024, 64)
        assert report["selected_fraction"] == 1.0
        assert report["flops_saved_pct"] == 0.0
        assert report["kv_bytes_saved_pct"] == 0.0

    def test_none_selected(self):
        report = savings_report(1, 1, 128, 1024, 0, 64)
        assert report["selected_fraction"] == 0.0
        assert report["flops_saved_pct"] == 100.0
        assert report["kv_bytes_saved_pct"] == 100.0

    def test_zero_total_tokens(self):
        report = savings_report(1, 1, 128, 0, 0, 64)
        assert report["selected_fraction"] == 0.0
        assert report["flops_saved_pct"] == 0.0

    def test_report_keys(self):
        report = savings_report(2, 4, 128, 4096, 1024, 128)
        expected_keys = {
            "dense_flops",
            "semantic_flops",
            "dense_kv_bytes",
            "semantic_kv_bytes",
            "flops_saved_pct",
            "kv_bytes_saved_pct",
            "selected_fraction",
        }
        assert set(report.keys()) == expected_keys
