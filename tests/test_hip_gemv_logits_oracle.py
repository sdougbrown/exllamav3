"""Pytest entry point for the isolated ROCm GEMV full-logit oracle."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
_ORACLE = _REPO_ROOT / "tests" / "hip_gemv_logits_oracle.py"
_DEFAULT_MODEL = Path("~/Models/Qwen3.8-27B-exl3").expanduser()
_MODEL = Path(os.environ.get("EXL3_TEST_MODEL", _DEFAULT_MODEL))


def test_hip_gemv_logits_oracle_in_isolated_workers():
    if not (torch.version.hip and torch.cuda.is_available()):
        pytest.skip("ROCm build / device not available")
    if not _MODEL.is_dir():
        pytest.skip(f"Test model not found: {_MODEL} (set EXL3_TEST_MODEL)")

    env = os.environ.copy()
    env["PYTHONPATH"] = str(_REPO_ROOT) if not env.get("PYTHONPATH") else str(_REPO_ROOT) + os.pathsep + env["PYTHONPATH"]
    subprocess.run([sys.executable, str(_ORACLE)], check = True, env = env, timeout = 1300)
