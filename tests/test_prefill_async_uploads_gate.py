import torch

from exllamav3.generator.job import Job
from exllamav3.generator.pagetable import (
    _block_index_pin_enabled,
    _prefill_async_uploads_enabled,
)


def test_gate_default_off(monkeypatch):
    monkeypatch.delenv("EXL3_PREFILL_ASYNC_UPLOADS", raising=False)
    assert _prefill_async_uploads_enabled() is False


def test_gate_env_on(monkeypatch):
    monkeypatch.setenv("EXL3_PREFILL_ASYNC_UPLOADS", "1")
    assert _prefill_async_uploads_enabled() is True


def test_gate_env_off(monkeypatch):
    monkeypatch.setenv("EXL3_PREFILL_ASYNC_UPLOADS", "0")
    assert _prefill_async_uploads_enabled() is False


def test_block_index_pin_requires_gate_and_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setenv("EXL3_PREFILL_ASYNC_UPLOADS", "1")
    assert _block_index_pin_enabled() is True
    monkeypatch.setenv("EXL3_PREFILL_ASYNC_UPLOADS", "0")
    assert _block_index_pin_enabled() is False
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setenv("EXL3_PREFILL_ASYNC_UPLOADS", "1")
    assert _block_index_pin_enabled() is False


def test_job_staging_default_off(monkeypatch):
    monkeypatch.delenv("EXL3_PREFILL_ASYNC_UPLOADS", raising=False)
    job = Job(torch.tensor([[1, 2]]), max_new_tokens = 2)
    assert job._prefill_staging_enabled is False


def test_job_staging_env_on(monkeypatch):
    monkeypatch.setenv("EXL3_PREFILL_ASYNC_UPLOADS", "1")
    job = Job(torch.tensor([[1, 2]]), max_new_tokens = 2)
    assert job._prefill_staging_enabled is True

def test_staged_cache_seqlens_disabled_returns_fresh_tensor(monkeypatch):
    # With the gate off, _prefill_staged_cache_seqlens must return a fresh pageable
    # tensor (old path) and never touch the staging slots.
    monkeypatch.delenv("EXL3_PREFILL_ASYNC_UPLOADS", raising=False)
    job = Job(torch.tensor([[1, 2]]), max_new_tokens = 2)
    t = job._prefill_staged_cache_seqlens(7)
    assert t.item() == 7
    assert t.dtype == torch.int32
    assert job._prefill_cs_staging is None
