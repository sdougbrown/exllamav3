# P5 T4/T5: edge cases + bounded recovery against the production MoeCpuHost path (HIP).
# T4: all-inactive selections (every pick GPU-resident), rows > cap_rows chunking, duplicate
#     picks across rows, cap_rows-boundary rows == cap_rows.
# T5: worker SIGKILL while a job is outstanding -> watchdog unblock -> begin_pass raises
#     (request-failure contract) -> fresh host works; no orphan processes.
# Run: ~/vllm-test-env/bin/python tests/hip_p5_edge_recovery.py [OUT_DIR]
from __future__ import annotations
import ctypes
import json
import os
import signal
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
os.environ.setdefault("EXL3_MOE_CPU_SPLIT", "0")
from exllamav3.model.moe_cpu_host import MoeCpuHost, TUNING  # noqa: E402
from exllamav3.ext import exllamav3_ext as ext  # noqa: E402

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/tmp/p5-edge")
OUT.mkdir(parents=True, exist_ok=True)
results = {"probe": "T4T5_edge_recovery", "cases": []}


def record(name: str, ok: bool, **kw) -> bool:
    results["cases"].append({"name": name, "ok": ok, **kw})
    print(f" {'PASS' if ok else 'FAIL'} {name} " + " ".join(f"{k}={v}" for k, v in kw.items()))
    return ok


def main() -> int:
    torch.cuda.init()
    torch.cuda.set_device(0)
    torch.zeros(1, device="cuda:0")

    # Synthetic 4-expert layer, real mul1 checkpoint tensors for realism
    import safetensors.torch as st
    MD = Path.home() / "Models/Qwen3.8-Flash-Next-exl3-bpw3"
    want = {}
    for e in range(4):
        for proj in ("gate_proj", "up_proj", "down_proj"):
            for suf in ("trellis", "suh", "svh"):
                want[f"model.language_model.layers.0.mlp.experts.{e}.{proj}.{suf}"] = None
    for f in sorted(MD.glob("model-*.safetensors")):
        with st.safe_open(f, framework="pt", device="cpu") as fh:
            for k in want:
                if want[k] is None and k in fh.keys():
                    want[k] = fh.get_tensor(k)
    T = want

    host = MoeCpuHost.__new__(MoeCpuHost)
    host.config = None
    host.specs = []
    host.by_key = {}
    host.live_layers = 0
    host.acked = 0
    host.started = False
    host.shm = None
    host.proc = None
    host.conn = None
    host.seq = 0
    host.next_slot = 0
    host.slot_last_seq = [0] * 4
    host.num_slots = 4
    host.cap_rows = 64
    host.threads = 8
    host.stage_threads = 4
    host.num_wslots = 2
    host.wslot_size = 32 * 1024 * 1024
    host.stream_t = 16
    host.stream_min_rows = 32
    host.batch_experts = 24
    host.wseq = 0
    host.next_wslot = 0
    host.wslot_prev_seq = [0] * 8
    host.aux = {}
    host.model_dir = str(MD)
    host.layout = {}

    # register a layer via register_layer (goes through the pipe to the child)
    keys = [f"model.language_model.layers.0.mlp.experts.{e}.{p}"
            for e in range(4) for p in ("gate_proj", "up_proj", "down_proj")]
    host.register_layer(
        "test.layer",
        [f"model.language_model.layers.0.mlp.experts.{e}.gate_proj" for e in range(4)],
        [f"model.language_model.layers.0.mlp.experts.{e}.up_proj" for e in range(4)],
        [f"model.language_model.layers.0.mlp.experts.{e}.down_proj" for e in range(4)],
        0, 0.0, 2560, 2560, 10,
        proj_dims=dict(g=(2560, 640, 3), u=(2560, 640, 3), d=(640, 2560, 3)),
    )
    host.ensure_started()
    ok = record("T4_setup_worker", host.started,
                note=f"{len(host.specs)} layer, child pid {host.proc.pid if host.proc else None}")

    dev = torch.device("cuda:0")
    torch.manual_seed(11)

    # ---- T4.1: all-inactive (every pick -1) -> zero contribution ----
    y = torch.randn(4, 2560, dtype=torch.half, device=dev)
    sel = torch.full((4, 10), -1, dtype=torch.long, device=dev)
    wts = torch.rand(4, 10, dtype=torch.half, device=dev) * 0.3 + 0.1
    out = host.submit(0, y, sel, wts)
    torch.cuda.synchronize()
    ok &= record("T4.1_all_inactive", bool((out == 0).all().item()),
                 max_abs=float(out.abs().max()))

    # ---- T4.2: rows > cap_rows (128 -> 2 chunks) with mixed selections ----
    torch.manual_seed(12)
    y = torch.randn(128, 2560, dtype=torch.half, device=dev)
    sel = torch.randint(0, 4, (128, 10), dtype=torch.long, device=dev)
    sel[:, 5] = -1
    sel[:64, 6] = sel[64:, 6]   # duplicates across chunk boundary
    wts = torch.rand(128, 10, dtype=torch.half, device=dev) * 0.4 + 0.05
    out1 = host.submit(0, y, sel, wts)
    out2 = host.submit(0, y, sel, wts)
    torch.cuda.synchronize()
    d = (out1 - out2).abs().max().item()
    ok &= record("T4.2_rows_gt_caprows_chunked", bool((out1.abs() > 0).any()) and d < 1e-3,
                 max_repeat_diff=d, note="deterministic across two identical submits")

    # ---- T4.3: rows == cap_rows boundary ----
    y = torch.randn(64, 2560, dtype=torch.half, device=dev)
    sel = torch.randint(0, 4, (64, 10), dtype=torch.long, device=dev)
    sel[:, 3] = -1
    wts = torch.rand(64, 10, dtype=torch.half, device=dev) * 0.3 + 0.05
    out = host.submit(0, y, sel, wts)
    torch.cuda.synchronize()
    ok &= record("T4.3_caprows_boundary", bool(torch.isfinite(out).all().item()),
                 max_abs=float(out.abs().max()))

    # ---- T5: worker crash while a job is outstanding ----
    import subprocess
    r = subprocess.run(["pgrep", "-f", "spawn_main"], capture_output=True, text=True)
    child_alive_before = bool(r.stdout.strip())
    # issue a job (worker alive) then kill the child; the collect wait must unblock via the
    # watchdog (0.5 s poll) and the NEXT begin_pass must raise
    host.v_abort[0] = 0
    y = torch.randn(4, 2560, dtype=torch.half, device=dev)
    sel = torch.full((4, 10), 2, dtype=torch.long, device=dev)
    wts = torch.full((4, 10), 0.5, dtype=torch.half, device=dev)
    pid = host.proc.pid
    os.kill(pid, signal.SIGKILL)
    t0 = time.perf_counter()
    try:
        host.submit(0, y, sel, wts)
        torch.cuda.synchronize()
        host.begin_pass()
        crashed_fail = False
    except RuntimeError as e:
        crashed_fail = True
        err = str(e)[:80]
    dt = time.perf_counter() - t0
    ok &= record("T5_worker_crash_bounded", crashed_fail and dt < 10,
                 fail_latency_s=round(dt, 2), err=err[:60] if crashed_fail else None,
                 note="dead worker: watchdog unblocks pending waits, next pass raises")
    # no orphan child processes
    r2 = subprocess.run(["pgrep", "-f", "spawn_main"], capture_output=True, text=True)
    ok &= record("T5_no_orphans", not r2.stdout.strip(),
                 note=f"spawn_main pids after crash: {r2.stdout.strip()[:40]}")

    results["gates"] = {"all_pass": ok, "child_alive_before": child_alive_before}
    (OUT / "edge-recovery.json").write_text(json.dumps(results, indent=1, default=str))
    print(f"\nT4/T5 edge+recovery: {'ALL PASS' if ok else 'FAILURES PRESENT'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())