"""P5 Phase 2: untraced-context in-loop kernel duration probe (torch.profiler, c1 MTP3).

One model load. Per arm (cfg_off / cfg_c12): warm decode, then a torch.profiler
(CUDA activity) window over 24 decode iterations. Reports per-iteration wall,
moe_grouped kernel-duration sums, and merged per-device busy union. Torch
profiler's per-event host overhead is ~1-2 us vs rocprofv3's ~8 us, so the
wall stays close to untraced while kernel durations are device-side truth.

Run: python tests/hip_p5_p2_torchprobe.py --out <dir>
"""

import argparse
import json
import os
import sys
import time
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

os.environ.setdefault("EXL3_HIP_GROUPED_MAX_ROWS", "5")

import torch  # noqa: E402
from exllamav3 import Cache, Config, Generator, GreedySampler, Job, Model, Tokenizer  # noqa: E402
from exllamav3.cache import CacheLayer_quant  # noqa: E402

sys.path.insert(0, "/home/douglasbrown/Code/_wt/exllamav3-hip-upstream/tests")
from hip_p5_p2_e2e import ARMS, build  # noqa: E402  (same model/cache build)

WARM_ITER = 40
MEASURE_ITER = 24
PROMPT = ("Write a detailed technical explanation of how a modern GPU command processor "
          "submits work, including queues, doorbells and fences. ")


def busy_union_ms(intervals_ns):
    if not intervals_ns:
        return 0.0
    intervals_ns.sort()
    total = 0
    cs, ce = intervals_ns[0]
    for s, e in intervals_ns[1:]:
        if s > ce:
            total += ce - cs
            cs, ce = s, e
        else:
            ce = max(ce, e)
    total += ce - cs
    return total / 1e6  # ns -> ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    results = {}

    for arm, envs in ARMS.items():
        saved = {k: os.environ.get(k) for k in list(ARMS["cfg_off"]) + list(ARMS[arm])}
        for k in ARMS["cfg_off"]:
            os.environ.pop(k, None)
        for k, v in envs.items():
            os.environ[k] = v
        target, draft, cache, draft_cache, tokenizer, gen = build(batch=1)
        job = Job(input_ids=tokenizer.encode(PROMPT, add_bos=True),
                  max_new_tokens=WARM_ITER + 8 + 2 * MEASURE_ITER,
                  stop_conditions=[], sampler=GreedySampler())
        gen.enqueue(job)
        it = 0
        wall0 = wall1 = 0.0
        prof = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(wait=WARM_ITER, warmup=0, active=MEASURE_ITER))
        prof.start()
        while gen.num_remaining_jobs():
            for _r in gen.iterate():
                it += 1
                if it == WARM_ITER:
                    wall0 = time.perf_counter()
                if it > WARM_ITER:
                    prof.step()
                if it == WARM_ITER + MEASURE_ITER:
                    wall1 = time.perf_counter()
        prof.stop()
        wall_ms = (wall1 - wall0) * 1000.0
        moe = {"gu": 0.0, "down": 0.0}
        busy = {}
        for ev in prof.events():
            if ev.device_type != torch.autograd.DeviceType.CUDA or ev.time_range is None:
                continue
            dev = getattr(ev, "device_index", None)
            if dev is None:
                continue
            s, e = ev.time_range.start, ev.time_range.end
            if "moe_grouped_gemv_k3_kernel" in ev.name:
                moe["gu" if "<false" in ev.name else "down"] += (e - s) / 1e3
            key = f"cuda{dev}"
            busy.setdefault(key, []).append((s, e))
        per_iter = {k: busy_union_ms(v) / MEASURE_ITER for k, v in busy.items()}
        results[arm] = {
            "wall_per_iter_ms": wall_ms / MEASURE_ITER,
            "moe_gu_sum_per_iter_ms": moe["gu"] / MEASURE_ITER,
            "moe_down_sum_per_iter_ms": moe["down"] / MEASURE_ITER,
            "busy_union_per_iter_ms": per_iter,
            "kernels_counted": sum(len(v) for v in busy.values()),
        }
        print(f"[torchprobe {arm}] {json.dumps(results[arm], indent=1)}", flush=True)
        for k in saved:
            if saved[k] is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = saved[k]
        target.unload()
        draft.unload()
        del target, draft, gen
        import gc
        gc.collect()
        torch.cuda.empty_cache()

    args.out.joinpath("torchprobe.json").write_text(
        json.dumps({"date": datetime.now(timezone.utc).isoformat(),
                    "measure_iter": MEASURE_ITER, "arms": results}, indent=2))


if __name__ == "__main__":
    main()