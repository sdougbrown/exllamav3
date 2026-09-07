"""Pure metrics/classification helpers for the P4/P5 perf harnesses.

Kept free of heavy imports so unit tests can exercise them without loading a
model or the validated-prefill harness.
"""


def decode_wall_s(ttft_s, t_end):
    """Decode-only wall clock for one trial.

    `ttft_s` is absolute time.time at first token; `t_end` is absolute
    time.time sampled after the trial completes. Clamped at zero for
    degenerate trials (prefill-only or clock anomalies).
    """
    return max(0.0, t_end - ttft_s)


def arms_by_flag(trials):
    """Group trial dicts by their recorded arm flag.

    The flag recorded at trial run time is the source of truth; index parity
    and cumulative graph counters are not reliable under counterbalanced
    ordering once runners persist across trials.
    """
    arms = {}
    for t in trials:
        arms.setdefault(t["flag"], []).append(t)
    return arms