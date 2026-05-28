"""Import tests for experiment modules.

These verify that the experiment scripts can be imported without their
heavy dependencies (transformers, datasets, CUDA, etc.).
"""

from __future__ import annotations

import importlib
import sys

import pytest


class TestLLMQualityValidationImports:
    def test_import_modules(self):
        """Import the validation module (without transformers/datasets)."""
        import experiments.llm_quality_validation  # noqa: F401

    def test_dry_run_parse(self, monkeypatch):
        """Simulate --dry-run to ensure argument parsing and early-exit work."""
        monkeypatch.setattr(sys, "argv", ["llm_quality_validation.py", "--dry-run"])
        import experiments.llm_quality_validation  # noqa: F401

        mod = importlib.import_module("experiments.llm_quality_validation")
        with pytest.raises(SystemExit):
            mod.main()

    def test_dry_run_parse_no_transformers(self, monkeypatch):
        """Simulate --dry-run without transformers/datasets installed."""
        monkeypatch.setattr(sys, "argv", ["llm_quality_validation.py", "--dry-run"])
        # Remove any cached import
        for mname in list(sys.modules.keys()):
            if "transformers" in mname or "datasets" in mname:
                del sys.modules[mname]

        import experiments.llm_quality_validation  # noqa: F401

        mod = importlib.import_module("experiments.llm_quality_validation")
        with pytest.raises(SystemExit):
            mod.main()


class TestGPUDecodeBenchmarkImports:
    def test_import_modules(self):
        """Import the GPU benchmark module (without CUDA)."""
        import experiments.gpu_decode_benchmark  # noqa: F401

    def test_dry_run_parse(self, monkeypatch):
        """Simulate --dry-run to ensure argument parsing and early-exit work."""
        monkeypatch.setattr(sys, "argv", ["gpu_decode_benchmark.py", "--dry-run"])
        import experiments.gpu_decode_benchmark  # noqa: F401

        mod = importlib.import_module("experiments.gpu_decode_benchmark")
        with pytest.raises(SystemExit):
            mod.main()

    def test_dry_run_parse_no_cuda(self, monkeypatch):
        """Simulate --dry-run even when CUDA is not available."""
        monkeypatch.setattr(sys, "argv", ["gpu_decode_benchmark.py", "--dry-run"])
        monkeypatch.setattr("torch.cuda.is_available", lambda: False)

        import experiments.gpu_decode_benchmark  # noqa: F401

        mod = importlib.import_module("experiments.gpu_decode_benchmark")
        with pytest.raises(SystemExit):
            mod.main()
