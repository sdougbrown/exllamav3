"""P5 Phase 2 Stage P2-1: grouped MoE decode kernel resource inspection driver.

Replays frozen real-routing bundles (Stage 2 routing.pt) through the grouped
binding only (no prefill arm, no model load) with roctx markers around each
class's measured window. Device-event medians are printed per class as the
untraced equivalent; run under rocprofv3 for kernel traces and PMC counters:

  # untraced (sanity timing)
  python tests/hip_p5_p2_inspect.py --routing <routing.pt> --out <dir>

  # kernel + marker trace
  /opt/rocm/bin/rocprofv3 --hip-runtime-trace --kernel-trace --marker-trace \
      --output-format csv -d <dir> -- python tests/hip_p5_p2_inspect.py ...

  # PMC counters (grouped kernel family only)
  /opt/rocm/bin/rocprofv3 --pmc SQ_WAVES SQ_WAVE_CYCLES ... \
      --kernel-include-regex 'moe_grouped_gemv_k3_kernel' \
      --output-format csv -d <dir> -- python tests/hip_p5_p2_inspect.py ...

Zero eviction required. Rows 3/5 do not occur in production (MTP3 -> rows=4,
MTP1 -> rows=2); frozen routing contains rows 1, 2, 4 only.
"""

import argparse
import ctypes
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
_HARNESS = Path(os.environ.get(
    "EXL3_VALIDATED_PREFILL_HARNESS",
    str(Path.home() / "Serve/hosts/rocky/bench-prefill-validated.py"))).expanduser()
import importlib.util  # noqa: E402
_spec = importlib.util.spec_from_file_location("validated", _HARNESS)
_b = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_b)

from hip_p5_isolated_ab import (  # noqa: E402
    alloc_scratch, run_grouped)
import hip_p5_isolated_ab as _ab  # noqa: E402

PMC_COUNTERS = [
    # waves / occupancy / issue
    "SQ_WAVES", "SQ_WAVE_CYCLES", "SQ_BUSY_CYCLES", "GRBM_COUNT", "GRBM_GUI_ACTIVE",
    # stalls
    "SQ_WAIT_ANY", "SQ_WAIT_INST_ANY",
    # instruction mix
    "SQ_INSTS_VALU", "SQ_INSTS_LDS", "SQ_INSTS_FLAT",
    "SQ_INST_CYCLES_VALU", "SQ_INST_CYCLES_VMEM",
    # LDS / icache
    "SQC_LDS_BANK_CONFLICT", "SQC_ICACHE_MISSES", "SQC_ICACHE_REQ",
    # L1/L2 traffic and hits
    "FETCH_SIZE", "L2CacheHit",
    "GL2C_HIT", "GL2C_MISS",
    # DRAM (external access) request sizes
    "GL2C_EA_RDREQ", "GL2C_EA_RDREQ_32B", "GL2C_EA_RDREQ_64B", "GL2C_EA_RDREQ_128B",
    "GL2C_EA_WRREQ", "GL2C_EA_WRREQ_64B",
    # L1 (TCP) requests
    "TCP_REQ", "TCP_REQ_MISS",
]


def _roctx():
    try:
        lib = ctypes.CDLL('/opt/rocm/lib/librocprofiler-sdk-roctx.so')
        lib.roctxRangePushA.argtypes = [ctypes.c_char_p]
        lib.roctxProfilerPause(0)
        return lib
    except OSError:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--routing", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--reps", type=int, default=100)
    ap.add_argument("--rows", type=str, default="",
                    help="comma list of row counts to include (default: all classes)")
    ap.add_argument("--skip-untraced", action="store_true",
                    help="skip the device-event timing loop (use for profiled runs; "
                         "event timing under counter replay is serialized)")
    ap.add_argument("--force-device", type=int, default=-1,
                    help="remap every class to this CUDA device (profiled runs are "
                         "single-GPU: the PMC collector breaks multi-GPU allocation)")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("EXL3_HIP_GROUPED_MAX_ROWS", "5")
    os.environ.setdefault("EXL3_MOE_SYNC_FREE_COUNT", "1")
    os.environ.setdefault("EXL3_QC_STAGING", "1")
    os.environ.setdefault("EXL3_PREFILL_ASYNC_UPLOADS", "0")

    import torch  # noqa: E402
    from exllamav3.ext import exllamav3_ext as ext  # noqa: E402

    torch.set_num_threads(8)
    rows_filter = {int(r) for r in args.rows.split(",") if r.strip()} or None

    if args.force_device >= 0:
        _ab._device_for = lambda tag, layer: torch.device(f"cuda:{args.force_device}")
    classes = _ab.prepare_bundles(args.routing, max_per_class=1)
    if rows_filter:
        classes = [c for c in classes if c["rows"] in rows_filter]
    for c in classes:
        gt = c["proj"]["gate_trellis"][0]
        intermediate = gt.numel() * 16 // (3 * 2560)
        c["I"] = intermediate
        c["scratch"] = alloc_scratch(c["dev"], c["rows"], intermediate)
    print(f"[inspect] classes: {[c['key'] for c in classes]}", flush=True)

    lib = _roctx()

    # warmup (profiler paused if roctx present)
    for c in classes:
        s = c["scratch"]
        with torch.cuda.device(c["dev"]):
            for _ in range(args.warmup):
                run_grouped(c, s)
            torch.cuda.synchronize()

    # untraced device-event timing per class (the untraced equivalent)
    untraced = {}
    skip_untraced = args.skip_untraced
    for c in classes if not skip_untraced else []:
        s, dev = c["scratch"], c["dev"]
        times = []
        with torch.cuda.device(dev):
            for _ in range(args.reps):
                torch.cuda.synchronize()
                e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
                e0.record()
                run_grouped(c, s)
                e1.record()
                torch.cuda.synchronize()
                times.append(e0.elapsed_time(e1))
        times.sort()
        n = len(times)
        untraced[str(c["key"])] = {
            "median_ms": times[n // 2] if n % 2 else (times[n // 2 - 1] + times[n // 2]) / 2,
            "p25_ms": times[n // 4], "p75_ms": times[(3 * n) // 4], "n": n}
        print(f"[inspect untraced] {c['key']}: {untraced[str(c['key'])]}", flush=True)

    # marked measurement windows (one per class), profiler resumed inside the marker
    if lib is not None and not skip_untraced:
        lib.roctxProfilerResume(0)
    for c in classes:
        s = c["scratch"]
        tag = f"p2_{c['key'][0]}_{c['key'][1]}_r{c['rows']}".replace(" ", "")
        with torch.cuda.device(c["dev"]):
            torch.cuda.synchronize()
            if lib is not None:
                lib.roctxRangePushA(tag.encode())
            for _ in range(args.reps):
                run_grouped(c, s)
            torch.cuda.synchronize()
            if lib is not None:
                lib.roctxRangePop()
    if lib is not None and not skip_untraced:
        lib.roctxProfilerPause(0)

    kfd1 = _b.kfd_evicted_ms(os.getpid())
    result = {"date": datetime.now(timezone.utc).isoformat(),
              "warmup": args.warmup, "reps": args.reps,
              "untraced": untraced, "kfd_delta": kfd1}
    (args.out / "inspect-untraced.json").write_text(json.dumps(result, indent=2))
    print("[inspect] done", flush=True)


if __name__ == "__main__":
    main()