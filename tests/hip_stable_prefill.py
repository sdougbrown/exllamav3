"""Diagnostic entry point for the existing forced-control prefill harness.

--stable-router-ties is off by default and replaces only the torch fallback.
--fixed-order-gdn is also off by default; it is set before model loading and
kept constant for the lifetime of captured graphs. Correctness captures,
prefixes, vocabulary clipping and cache assertions remain owned by
bench-prefill-validated.py. Bench mode requires --qualification directories
and measures synchronized prefill-only host wall, not device-event spans.
"""
import json
import os
from pathlib import Path
import sys

from hip_moe_isolation import load_harness


def reject_overlapping_out(out, qualifications):
    """Out must never alias, contain, or sit inside any declared qualification directory."""
    resolved_out = Path(out).expanduser().resolve()
    for qualification in qualifications:
        resolved_qual = Path(qualification).expanduser().resolve()
        if resolved_out == resolved_qual or resolved_qual in resolved_out.parents \
                or resolved_out in resolved_qual.parents:
            raise SystemExit(f'--out {out} overlaps qualification directory {qualification}')


def reject_bench_capture_gdn(mode, capture_gdn):
    """Bench measures uninstrumented runs; capture hooks (warmup CPU copies) are correctness-only."""
    if mode == 'bench' and capture_gdn:
        raise SystemExit('--capture-gdn is not allowed in bench mode: capture hooks violate the '
                         'uninstrumented benchmark contract')


def parse_wrapper_args(argv):
    args = list(argv[1:])
    mode = args[0] if args and not args[0].startswith('-') else 'correctness'
    if '--out' not in args or '--out' not in argv:
        raise SystemExit('--out is required')
    out = Path(argv[argv.index('--out') + 1]).expanduser()
    qualifications = []
    rest = list(argv)
    while '--qualification' in rest:
        index = rest.index('--qualification')
        qualifications.append(Path(rest[index + 1]).expanduser())
        del rest[index:index + 2]
    return mode, out, qualifications, argv


def main():
    mode, out, qualifications, argv = parse_wrapper_args(sys.argv)
    reject_overlapping_out(out, qualifications)
    capture_gdn = '--capture-gdn' in argv
    reject_bench_capture_gdn(mode, capture_gdn)
    stable = '--stable-router-ties' in argv
    if stable:
        argv.remove('--stable-router-ties')
    fixed_gdn = '--fixed-order-gdn' in argv
    if fixed_gdn:
        argv.remove('--fixed-order-gdn')
    os.environ['EXL3_DIAGNOSTIC_GDN_FIXED_ORDER'] = '1' if fixed_gdn else '0'
    os.environ['EXL3_DIAGNOSTIC_GDN_TRACE'] = '0'
    if capture_gdn:
        argv.remove('--capture-gdn')
    while '--qualification' in argv:
        index = argv.index('--qualification')
        del argv[index:index + 2]
    harness = load_harness()
    restore = None
    if stable:
        from hip_stable_router_control import install
        restore = install()
    out = Path(argv[argv.index('--out') + 1])
    out.mkdir(parents=True, exist_ok=True)

    original_memory = harness.memory_snapshot

    def memory_snapshot():
        peaks = {str(d): harness.torch.cuda.max_memory_reserved(d) for d in (0, 1)}
        result = original_memory()
        for d, peak in peaks.items():
            result[d]['torch_peak_reserved_b'] = peak
        return result

    harness.memory_snapshot = memory_snapshot
    telemetry = []
    original_prefill = harness.run_prefill

    def recorded_prefill(generator, job, timed=True):
        before = harness.kfd_evicted_ms(os.getpid())
        result = original_prefill(generator, job, timed)
        telemetry.append({'gates': {k: os.environ[k] for k in ('EXL3_MOE_SYNC_FREE_COUNT', 'EXL3_PREFILL_ASYNC_UPLOADS')},
                          'kfd_before_prefill': before, 'kfd_after_prefill': harness.kfd_evicted_ms(os.getpid()),
                          'memory_after_prefill': memory_snapshot()})
        (out / 'prefill-telemetry.json').write_text(json.dumps(telemetry, indent=2))
        return result

    harness.run_prefill = recorded_prefill
    restore_gdn = None
    if capture_gdn:
        from hip_gdn_isolation import install_capture
        original_load = harness.load_model

        def load_then_capture(*args, **kwargs):
            nonlocal restore_gdn
            loaded = original_load(*args, **kwargs)
            restore_gdn = install_capture(out)
            return loaded

        harness.load_model = load_then_capture
    controls = {
        'stable_router_ties': stable,
        'fixed_order_gdn': fixed_gdn,
        'capture_gdn': capture_gdn,
        'policy': 'score descending; lower expert ID for exact ties only',
        'scope': '_routing_std_torch only; native small-row router unchanged',
        'harness': str(Path(harness.__file__).resolve()),
        'argv': argv,
        'chunk': int(argv[argv.index('--chunk') + 1]) if '--chunk' in argv else 512,
    }
    (out / 'diagnostic-controls.json').write_text(json.dumps(controls, indent=2))
    from hip_prefill_wall_bench import bench
    harness.cmd_bench = lambda args: bench(harness, args, qualifications, controls)
    try:
        return harness.main()
    finally:
        if restore_gdn is not None:
            restore_gdn()
        if restore is not None:
            restore()


if __name__ == '__main__':
    raise SystemExit(main())
