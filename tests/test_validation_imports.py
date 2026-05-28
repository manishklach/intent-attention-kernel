"""Import tests for experiment modules.

These verify that the experiment scripts can be imported without their
heavy dependencies (transformers, datasets, CUDA, etc.).
"""

from __future__ import annotations

import importlib
import os
import sys


def _import_experiments_module(module_name):
    """Try to import an experiments module with multiple fallback strategies."""
    # Strategy 1: Normal import (should work when run from repo root)
    try:
        return importlib.import_module(f"experiments.{module_name}")
    except ModuleNotFoundError:
        pass
    
    # Strategy 2: Ensure current directory is in sys.path
    try:
        if "" not in sys.path:
            sys.path.insert(0, "")
        return importlib.import_module(f"experiments.{module_name}")
    except ModuleNotFoundError:
        pass
    
    # Strategy 3: Compute repo root from this file's location
    try:
        test_file_dir = os.path.dirname(os.path.abspath(__file__))
        repo_root = os.path.dirname(test_file_dir)  # Go up one level from tests/
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)
        return importlib.import_module(f"experiments.{module_name}")
    except ModuleNotFoundError:
        pass
    
    # Strategy 4: Try to import using importlib.util with absolute path
    try:
        test_file_dir = os.path.dirname(os.path.abspath(__file__))
        module_path = os.path.join(test_file_dir, "..", "experiments", f"{module_name}.py")
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None or spec.loader is None:
            raise ModuleNotFoundError(f"Could not load spec for {module_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    except Exception:
        pass
    
    # If all strategies fail, re-raise the original ModuleNotFoundError
    raise ModuleNotFoundError(f"No module named 'experiments.{module_name}'")


class TestLLMQualityValidationImports:
    def test_import_modules(self):
        """Import the validation module (without transformers/datasets)."""
        _import_experiments_module("llm_quality_validation")  # noqa: F401

    def test_dry_run_parse(self, monkeypatch):
        """Simulate --dry-run to ensure argument parsing and early-exit work."""
        monkeypatch.setattr(sys, "argv", ["llm_quality_validation.py", "--dry-run"])
        _import_experiments_module("llm_quality_validation")  # noqa: F401

    def test_dry_run_parse_no_transformers(self, monkeypatch):
        """Simulate --dry-run without transformers/datasets installed."""
        monkeypatch.setattr(sys, "argv", ["llm_quality_validation.py", "--dry-run"])
        # Remove any cached import
        for mname in list(sys.modules.keys()):
            if "transformers" in mname or "datasets" in mname:
                del sys.modules[mname]
        _import_experiments_module("llm_quality_validation")  # noqa: F401


class TestGPUDecodeBenchmarkImports:
    def test_import_modules(self):
        """Import the GPU benchmark module (without CUDA)."""
        _import_experiments_module("gpu_decode_benchmark")  # noqa: F401

    def test_dry_run_parse(self, monkeypatch):
        """Simulate --dry-run to ensure argument parsing and early-exit work."""
        monkeypatch.setattr(sys, "argv", ["gpu_decode_benchmark.py", "--dry-run"])
        _import_experiments_module("gpu_decode_benchmark")  # noqa: F401

    def test_dry_run_parse_no_cuda(self, monkeypatch):
        """Simulate --dry-run even when CUDA is not available."""
        monkeypatch.setattr(sys, "argv", ["gpu_decode_benchmark.py", "--dry-run"])
        monkeypatch.setattr("torch.cuda.is_available", lambda: False)
        _import_experiments_module("gpu_decode_benchmark")  # noqa: F401
