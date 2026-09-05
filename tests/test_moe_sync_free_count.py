import pytest
import time

import torch
from exllamav3.modules.block_sparse_mlp import (
    _HIP_PREFILL_MAX_EXPERT_ROWS,
    _moe_sync_free_count,
    _scatter_expert_count,
)

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_scatter_matches_bincount_random():
    E = 512
    ids = torch.randint(0, E + 1, (5120,), device=DEV)
    result = _scatter_expert_count(ids, E + 1)
    expected = torch.bincount(ids, minlength=E + 1)
    assert torch.equal(result, expected)


def test_scatter_matches_bincount_sentinel_all():
    E = 512
    ids = torch.full((5120,), E, dtype=torch.long, device=DEV)
    result = _scatter_expert_count(ids, E + 1)
    expected = torch.bincount(ids, minlength=E + 1)
    assert torch.equal(result, expected)
    assert result[E].item() == 5120


def test_scatter_matches_bincount_empty_experts():
    E = 512
    ids = torch.randint(0, 8, (5120,), device=DEV)
    ids[:100] = torch.randint(200, 500, (100,), device=DEV)
    result = _scatter_expert_count(ids, E + 1)
    expected = torch.bincount(ids, minlength=E + 1)
    assert torch.equal(result, expected)
    assert (result > 0).sum().item() == (expected > 0).sum().item()
    assert (result == 0).sum().item() >= 100


def test_scatter_long_index():
    E = 512
    ids = torch.randint(0, E + 1, (10240,), device=DEV)
    assert ids.shape[0] > _HIP_PREFILL_MAX_EXPERT_ROWS
    result = _scatter_expert_count(ids, E + 1)
    expected = torch.bincount(ids, minlength=E + 1)
    assert torch.equal(result, expected)


def test_scatter_empty_index():
    E = 512
    ids = torch.empty((0,), dtype=torch.long, device=DEV)
    result = _scatter_expert_count(ids, E + 1)
    expected = torch.bincount(ids, minlength=E + 1)
    assert torch.equal(result, expected)
    assert result.dtype == torch.long
    assert result.shape == (E + 1,)


def test_scatter_dtype_and_shape():
    E = 512
    ids = torch.randint(0, E + 1, (1024,), device=DEV)
    result = _scatter_expert_count(ids, E + 1)
    assert result.dtype == torch.long
    assert result.shape == (E + 1,)
    assert result.device == ids.device


def test_env_gate_default_on(monkeypatch):
    monkeypatch.delenv("EXL3_MOE_SYNC_FREE_COUNT", raising=False)
    assert _moe_sync_free_count() is True
    monkeypatch.setenv("EXL3_MOE_SYNC_FREE_COUNT", "0")
    assert _moe_sync_free_count() is False
    monkeypatch.setenv("EXL3_MOE_SYNC_FREE_COUNT", "1")
    assert _moe_sync_free_count() is True


def test_helpers_have_expected_signature():
    assert callable(_scatter_expert_count)
    ids = torch.tensor([0, 1, 2], dtype=torch.long, device=DEV)
    result = _scatter_expert_count(ids, 3)
    assert len(result.shape) == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU only")
def test_scatter_host_sync_free_with_pending_work():
    # Pending-work probe: hold the device with a long on-stream kernel, then time the
    # HOST wall time of each counting call. torch.bincount on a device tensor blocks
    # the host until the queued kernel drains; the scatter histogram must return
    # immediately (device-only). The bincount leg validates that the probe detects a
    # host sync — without it, a fast call would be indistinguishable from an idle
    # device.
    E = 512
    ids = torch.randint(0, E + 1, (5120,), device=DEV)
    big = torch.randint(0, E + 1, (20480,), device=DEV)  # > persistent-ones boundary
    for _ in range(3):                                   # warm allocations/JIT
        torch.bincount(ids, minlength=E + 1)
        _scatter_expert_count(ids, E + 1)
        _scatter_expert_count(big, E + 1)
    torch.cuda.synchronize()

    torch.cuda._sleep(int(4e8))                          # ~hundreds of ms on gfx12
    t0 = time.perf_counter()
    ref = torch.bincount(ids, minlength=E + 1)           # probe validation leg
    bincount_host = time.perf_counter() - t0
    torch.cuda.synchronize()

    torch.cuda._sleep(int(4e8))
    t0 = time.perf_counter()
    got = _scatter_expert_count(ids, E + 1)
    scatter_host = time.perf_counter() - t0
    torch.cuda.synchronize()
    assert torch.equal(got, ref)

    torch.cuda._sleep(int(4e8))
    t0 = time.perf_counter()
    got_big = _scatter_expert_count(big, E + 1)          # fresh-ones path, also device-only
    scatter_big_host = time.perf_counter() - t0
    torch.cuda.synchronize()
    assert torch.equal(got_big, torch.bincount(big, minlength=E + 1))

    assert bincount_host > 0.02, \
        f"probe invalid: bincount returned in {bincount_host*1e3:.3f} ms with work pending"
    assert scatter_host < bincount_host / 10, \
        f"scatter blocked host {scatter_host*1e3:.3f} ms vs bincount {bincount_host*1e3:.3f} ms"
    assert scatter_big_host < bincount_host / 10, \
        f"fresh-ones scatter blocked host {scatter_big_host*1e3:.3f} ms"


def test_scatter_512_rows_sentinel_and_empty():
    # 512 rows x top-k 10 = 5120 assignments: the persistent ones-buffer boundary.
    E = 512
    ids = torch.randint(0, 8, (5120,), device=DEV)            # narrow range -> bins 8..E empty
    ids[:64] = E                                              # sentinel present
    ids[64:128] = torch.randint(200, 500, (64,), device=DEV)  # scattered highs
    result = _scatter_expert_count(ids, E + 1)
    expected = torch.bincount(ids, minlength=E + 1)
    assert torch.equal(result, expected)
    assert result[E].item() == 64
    assert (result == 0).sum().item() >= 200  # bins 8..199 + 500..511 = 204 empty


def test_scatter_1024_rows_sentinel_and_empty():
    # 1024 rows x top-k 10 = 10240 assignments.
    E = 512
    ids = torch.randint(0, 8, (10240,), device=DEV)           # narrow range -> bins 8..E empty
    ids[:128] = E                                              # sentinel present
    ids[128:256] = torch.randint(200, 500, (128,), device=DEV)  # scattered highs
    result = _scatter_expert_count(ids, E + 1)
    expected = torch.bincount(ids, minlength=E + 1)
    assert torch.equal(result, expected)
    assert result[E].item() == 128
    assert (result == 0).sum().item() >= 200  # bins 8..199 + 500..511 = 204 empty


def test_scatter_2048_rows_sentinel_and_empty():
    # 2048 rows x top-k 10 = 20480 assignments: beyond the persistent buffer, so the
    # fresh-ones path runs (counting precedes route eligibility; P3b must reuse this
    # helper without bincount).
    E = 512
    ids = torch.randint(0, 8, (20480,), device=DEV)           # narrow range -> bins 8..E empty
    ids[:256] = E                                              # sentinel present
    ids[256:512] = torch.randint(200, 500, (256,), device=DEV)  # scattered highs
    result = _scatter_expert_count(ids, E + 1)
    expected = torch.bincount(ids, minlength=E + 1)
    assert torch.equal(result, expected)
    assert result[E].item() == 256
    assert (result == 0).sum().item() >= 200  # bins 8..199 + 500..511 = 204 empty