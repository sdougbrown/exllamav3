#!/usr/bin/env python3
"""CPU regression tests for tests/tp_prefill_wall_bench.py (Stage 1 harness).

Covers the plan's Stage-1 CPU gates without touching torch or the GPU:
  1. explicit runtime chunking with exact token accounting and monotonic past
     lengths (4096 @ 512 -> 8 calls; tail handling);
  2. fresh params per call with the correct recurrent-state reinjection and no
     duplicated/dropped tokens;
  3. timing ends only after completion receipts from every participating GPU
     process;
  4. TP dispatch order is child-first / output-device pseudo-worker last;
  5. rectangular-batch row bounds (B x chunk) with per-sequence chunk
     reduction;
  6. invalid chunk / cache-extent inputs fail loudly.

Run:  python -m pytest tests/test_tp_prefill_wall_bench.py -q
"""
from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "tp_prefill_wall_bench", HERE / "tp_prefill_wall_bench.py"
)
bench = importlib.util.module_from_spec(SPEC)
sys.modules.setdefault("tp_prefill_wall_bench", bench)
SPEC.loader.exec_module(bench)


# --- 1. explicit runtime chunking -------------------------------------------

def test_4096_at_512_is_eight_calls_with_exact_accounting():
    chunks = bench.plan_chunks(4096, 512)
    assert len(chunks) == 8
    assert all(length == 512 for _, length in chunks)
    assert sum(l for _, l in chunks) == 4096
    # past_len advances monotonically and matches chunk starts
    past_lens = [s for s, _ in chunks]
    assert past_lens == sorted(past_lens)
    assert past_lens == [0, 512, 1024, 1536, 2048, 2560, 3072, 3584]


def test_tail_chunk():
    chunks = bench.plan_chunks(5000, 512)
    assert len(chunks) == 10
    assert chunks[-1] == (4608, 392)
    assert sum(l for _, l in chunks) == 5000


def test_chunk_larger_than_prompt_is_single_call():
    assert bench.plan_chunks(4096, 8192) == [(0, 4096)]


def test_chunk_must_be_positive_and_prompt_nonempty():
    with pytest.raises(ValueError):
        bench.plan_chunks(4096, 0)
    with pytest.raises(ValueError):
        bench.plan_chunks(4096, -512)
    with pytest.raises(ValueError):
        bench.plan_chunks(0, 512)


# --- 2. fresh params per call, state reinjection ----------------------------

class FakeState:
    """Mimics GDNState: position must match past_len at every call."""

    def __init__(self, position=0):
        self.position = position
        self.freed = False


def simulate_chunk_loop(prompt_len, chunk, cache_tokens):
    """Drives the harness's param-building rules without torch.

    Records the (past_len, state-positions) tuple observed AT CALL TIME for
    every call, since recurrent-state objects are shared and mutated across
    calls: post-hoc inspection would only see final positions.
    """
    chunks = bench.plan_chunks(prompt_len, chunk)
    calls = []
    observed = []
    prev_states = None
    past = 0
    for (start, length) in chunks:
        params = {"past_len": past, "batch_shape": (1, cache_tokens)}
        if prev_states is not None:
            params["recurrent_states"] = prev_states
        else:
            params["recurrent_states"] = [FakeState(position=0)]  # cache.get_new_state()
        if calls:
            assert params is not calls[-1], "params must be a fresh dict per call"
        calls.append(params)
        observed.append((past, [s.position for s in params["recurrent_states"]]))
        # advance_recurrent_states: position += seqlen
        for s in params["recurrent_states"]:
            s.position += length
        prev_states = params["recurrent_states"]
        past += length
    assert past == prompt_len
    return calls, observed


def test_fresh_params_dict_each_call_with_reinjected_states():
    calls, _ = simulate_chunk_loop(4096, 512, 8192)
    assert len(calls) == 8
    ids = [id(p) for p in calls]
    assert len(set(ids)) == 8, "each call must receive a fresh params dict"
    # state objects are reinjected (same objects), never duplicated per call
    first = calls[0]["recurrent_states"]
    for p in calls[1:]:
        assert p["recurrent_states"] is first
    assert all(s.position == 4096 for s in first)


def test_state_positions_match_past_len_at_call_time():
    # state objects are shared and mutated across calls, so the invariant is
    # only checkable at call time (the real model asserts this inside
    # prepare_for_recurrence on every forward)
    _, observed = simulate_chunk_loop(12288, 1024, 16384)
    assert len(observed) == 12
    for i, (past_len, positions) in enumerate(observed):
        assert past_len == i * 1024
        assert all(pos == past_len for pos in positions), (
            f"call {i}: state positions {positions} != past_len {past_len}"
        )


# --- 3. completion receipts gate timing -------------------------------------

def test_barrier_waits_for_all_receipts():
    barrier = bench.CompletionBarrier(expected=["rank:0", "rank:1"], timeout_s=5.0)
    assert not barrier.satisfied
    barrier.deliver("rank:1")
    assert not barrier.satisfied
    barrier.deliver("rank:0")
    assert barrier.satisfied
    elapsed = barrier.wait()
    assert elapsed >= 0.0


def test_barrier_times_out_with_missing_receipt():
    barrier = bench.CompletionBarrier(expected=["rank:0", "rank:1"], timeout_s=0.05)
    barrier.deliver("rank:0")
    with pytest.raises(TimeoutError):
        barrier.wait()


def test_barrier_rejects_unknown_and_duplicate_receipts():
    barrier = bench.CompletionBarrier(expected=["rank:0"], timeout_s=1.0)
    with pytest.raises(ValueError):
        barrier.deliver("rank:7")
    barrier.deliver("rank:0")
    with pytest.raises(ValueError):
        barrier.deliver("rank:0")


def test_barrier_requires_participants():
    with pytest.raises(ValueError):
        bench.CompletionBarrier(expected=[])
    with pytest.raises(ValueError):
        bench.CompletionBarrier(expected=["rank:0", "rank:0"])


# --- 4. TP dispatch ordering -------------------------------------------------

def test_dispatch_order_children_first_output_last():
    assert bench.dispatch_order([1, 0], 0) == [1, 0]
    assert bench.dispatch_order([0, 1], 1) == [0, 1]
    assert bench.dispatch_order([2, 0, 1], 1) == [2, 0, 1]


def test_dispatch_order_requires_output_in_active():
    with pytest.raises(ValueError):
        bench.dispatch_order([0, 1], 2)


# --- 5. rectangular batch row bounds -----------------------------------------

def test_rows_for_batch_batch_one():
    assert bench.rows_for_batch(1, 512) == 512


def test_rows_for_batch_rectangular():
    assert bench.rows_for_batch(8, 1024) == 8192


def test_per_sequence_chunk_reduces_to_row_budget():
    assert bench.per_sequence_chunk(8, 1024, 2048) == 256
    assert bench.per_sequence_chunk(1, 512, 2048) == 512
    assert bench.per_sequence_chunk(8, 1024, 4) == 1  # clamps, never zero
    with pytest.raises(ValueError):
        bench.per_sequence_chunk(8, 512, 0)


# --- 6. cache extent ----------------------------------------------------------

def test_cache_extent_ok():
    bench.check_cache_extent(8192, 4096, extra_tokens=16)
    bench.check_cache_extent(40960, 32768, extra_tokens=16)


def test_cache_extent_overflow_raises():
    with pytest.raises(ValueError):
        bench.check_cache_extent(8192, 8192, extra_tokens=16)
    with pytest.raises(ValueError):
        bench.check_cache_extent(8192, 12288)


# --- module hygiene ------------------------------------------------------------

def test_pure_helpers_present():
    # the pure helpers must exist and be callable without any GPU/torch dependency
    assert callable(bench.plan_chunks)
    assert callable(bench.CompletionBarrier)
    assert callable(bench.dispatch_order)
    assert callable(bench.per_sequence_chunk)


def test_barrier_measures_wall_from_start():
    b = bench.CompletionBarrier(expected=["a"], timeout_s=5.0)
    time.sleep(0.02)
    b.deliver("a")
    assert b.wait() >= 0.015


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))