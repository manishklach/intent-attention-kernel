from intent_attention.cost_model import savings_report

def test_savings_report():
    report = savings_report(1, 1, 128, 1024, 512, 64)
    assert report["selected_fraction"] == 0.5
    assert report["flops_saved_pct"] == 50.0
    assert report["kv_bytes_saved_pct"] == 50.0
    assert report["semantic_flops"] < report["dense_flops"]
