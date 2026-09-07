"""Regression tests for the P4/P5 perf-harness metrics (Stage 0 of the P5 plan).

These pin the timing boundary and arm-classification semantics that earlier
perf results silently violated: decode-wall sampled after the trial, and arm
grouping by the per-trial flag rather than index parity or cumulative counters.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from p4_perf_metrics import arms_by_flag, decode_wall_s


def test_decode_wall_positive():
    # TTFT at t=100.0, trial ends at t=164.0 -> 64 s of decode wall.
    assert decode_wall_s(100.0, 164.0) == 64.0


def test_decode_wall_sampled_after_trial():
    # Degenerate regression guard: if t_end were sampled before the trial, it
    # would be <= ttft and the clamp would hide it as 0. The helper clamps, but
    # the harness contract is that t_end >= ttft for a real decode trial; this
    # documents the clamp behavior for prefill-only trials.
    assert decode_wall_s(164.0, 100.0) == 0.0
    assert decode_wall_s(0.0, 0.0) == 0.0


def test_arms_by_flag_under_counterbalanced_order():
    # Counterbalanced order flips arms every rep; classification must follow the
    # recorded flag, not index parity (rep0: eager,graph; rep1: graph,eager).
    trials = [
        {"flag": "0", "wall_s": 10.0},
        {"flag": "1", "wall_s": 11.0},
        {"flag": "1", "wall_s": 12.0},
        {"flag": "0", "wall_s": 13.0},
    ]
    arms = arms_by_flag(trials)
    assert [t["wall_s"] for t in arms["0"]] == [10.0, 13.0]
    assert [t["wall_s"] for t in arms["1"]] == [11.0, 12.0]


def test_arms_by_flag_not_fooled_by_cumulative_counters():
    # Historical bug: classifying on graph_replays > 0 mislabeled every trial
    # after the first capture. Flags are authoritative regardless of counters.
    trials = [
        {"flag": "0", "wall_s": 10.0, "graph_replays": 720},
        {"flag": "1", "wall_s": 11.0, "graph_replays": 1440},
        {"flag": "0", "wall_s": 12.0, "graph_replays": 2160},
    ]
    arms = arms_by_flag(trials)
    assert len(arms["0"]) == 2 and len(arms["1"]) == 1
    assert arms["0"][0]["graph_replays"] == 720


def test_arms_by_flag_empty():
    assert arms_by_flag([]) == {}