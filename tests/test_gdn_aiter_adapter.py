#!/usr/bin/env python3
"""CPU tests for the optional AITER chunked GDN prefill adapter (Stage 3).

Covers the adapter's selection predicate, state K,V <-> V,K contract (with
nonsymmetric sentinel values that make a missing transpose visible), pool
preservation, and fail-before-mutation behavior. torch runs on CPU; no GPU and
no aiter import happens unless a test explicitly requires the lazy import.

Run:  python -m pytest tests/test_gdn_aiter_adapter.py -q
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

HERE = Path(__file__).resolve().parent


def _load(name: str, fname: str):
    spec = importlib.util.spec_from_file_location(name, HERE / fname)
    mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(name, mod)
    spec.loader.exec_module(mod)
    return mod


adapter = _load("gdn_aiter_adapter", str(HERE.parent / "exllamav3" / "modules" / "gated_delta_net_fn" / "gdn_aiter_adapter.py"))


# --- selection predicate ------------------------------------------------------

def test_eligible_case():
    assert adapter.aiter_eligible(
        seqlen=512, history=False, channelwise_g=False,
        k_head_dim=128, v_head_dim=128,
        num_v_heads=48, num_k_heads=16,
    )


def test_tp_shard_head_counts_eligible():
    assert adapter.aiter_eligible(
        seqlen=512, history=False, channelwise_g=False,
        k_head_dim=128, v_head_dim=128,
        num_v_heads=24, num_k_heads=8,
    )


def test_history_not_eligible():
    assert not adapter.aiter_eligible(
        seqlen=512, history=True, channelwise_g=False,
        k_head_dim=128, v_head_dim=128, num_v_heads=48, num_k_heads=16,
    )


def test_kda_not_eligible():
    assert not adapter.aiter_eligible(
        seqlen=512, history=False, channelwise_g=True,
        k_head_dim=128, v_head_dim=128, num_v_heads=48, num_k_heads=16,
    )


def test_non_128_head_dims_not_eligible():
    assert not adapter.aiter_eligible(
        seqlen=512, history=False, channelwise_g=False,
        k_head_dim=64, v_head_dim=128, num_v_heads=48, num_k_heads=16,
    )
    assert not adapter.aiter_eligible(
        seqlen=512, history=False, channelwise_g=False,
        k_head_dim=128, v_head_dim=96, num_v_heads=48, num_k_heads=16,
    )


def test_ungrouped_heads_not_eligible():
    assert not adapter.aiter_eligible(
        seqlen=512, history=False, channelwise_g=False,
        k_head_dim=128, v_head_dim=128, num_v_heads=47, num_k_heads=16,
    )


# --- state K,V <-> V,K contract (nonsymmetric sentinels) ----------------------

def make_pool(n_slots=4, h=2, k=4, v=8, seed=0):
    torch.manual_seed(seed)
    # nonsymmetric K != V so a missing transpose cannot hide in shape checks
    return torch.randn(n_slots, 1, h, k, v, dtype=torch.float32)


def test_gather_transposes_selected_slots_only():
    pool = make_pool()
    slots = torch.tensor([2], dtype=torch.int32)
    vk = adapter.gather_states_vk(pool, slots)
    assert vk.shape == (1, 2, 8, 4)
    assert vk.dtype == torch.float32
    assert vk.is_contiguous()
    # sentinel check: vk[n, h, v, k] must equal pool[slot, 0, h, k, v]
    assert torch.equal(vk[0, 1, 5, 3], pool[2, 0, 1, 3, 5])


def test_gather_leaves_pool_untouched():
    pool = make_pool()
    before = pool.clone()
    slots = torch.tensor([0, 3], dtype=torch.int32)
    adapter.gather_states_vk(pool, slots)
    assert torch.equal(pool, make_pool())


def test_scatter_writes_back_transposed_and_preserves_others():
    pool = make_pool()
    slots = torch.tensor([1], dtype=torch.int32)
    vk = adapter.gather_states_vk(pool, slots)
    vk = vk + 100.0  # simulate AITER's final state
    adapter.scatter_states_vk(vk, pool, slots)
    ref = make_pool()
    assert torch.equal(pool[0, 0], ref[0, 0])          # untouched slot 0
    assert torch.equal(pool[2, 0], ref[2, 0])          # untouched slot 2
    assert torch.equal(pool[3, 0], ref[3, 0])          # untouched slot 3
    assert torch.allclose(pool[1, 0], ref[1, 0] + 100.0)  # written slot 1


def test_scatter_roundtrip_matches_original():
    pool = make_pool()
    slots = torch.tensor([1, 2], dtype=torch.int32)
    vk = adapter.gather_states_vk(pool, slots)
    adapter.scatter_states_vk(vk, pool, slots)
    ref = make_pool()
    assert torch.equal(pool[1, 0], ref[1, 0])
    assert torch.equal(pool[2, 0], ref[2, 0])


def test_none_state_roundtrip():
    assert adapter.gather_states_vk(None, None) is None
    adapter.scatter_states_vk(None, torch.empty(0), torch.empty(0, dtype=torch.int32))  # no-op


# --- sparse/swapped slots ------------------------------------------------------

def test_swapped_sparse_slots():
    pool = make_pool()
    slots = torch.tensor([3, 1], dtype=torch.int32)  # swapped order
    vk = adapter.gather_states_vk(pool, slots)
    assert torch.equal(vk[0], pool[3, 0].transpose(-1, -2))
    assert torch.equal(vk[1], pool[1, 0].transpose(-1, -2))
    vk = vk + 7.0
    adapter.scatter_states_vk(vk, pool, slots)
    ref = make_pool()
    assert torch.allclose(pool[3, 0], ref[3, 0] + 7.0)
    assert torch.allclose(pool[1, 0], ref[1, 0] + 7.0)
    assert torch.equal(pool[0, 0], ref[0, 0])
    assert torch.equal(pool[2, 0], ref[2, 0])


# --- eligibility gate drives wrapper routing (monkeypatched import) ------------

def test_optin_env_gates_selection(monkeypatch):
    monkeypatch.delenv("EXL3_GDN_AITER_PREFILL", raising=False)
    assert not adapter.optin_enabled()
    monkeypatch.setenv("EXL3_GDN_AITER_PREFILL", "1")
    assert adapter.optin_enabled()


def test_lazy_import_failure_is_clear(monkeypatch):
    monkeypatch.setenv("EXL3_GDN_AITER_PREFILL", "1")
    import builtins
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("aiter"):
            raise ModuleNotFoundError(f"No module named {name!r}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(RuntimeError, match="aiter"):
        adapter.load_aiter_vk()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))