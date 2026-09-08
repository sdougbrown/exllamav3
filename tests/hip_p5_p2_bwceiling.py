"""P5 Phase 2: DRAM bandwidth ceiling microbench (event-based, no profiler).

Measures achieved device bandwidth for read-dominated and copy patterns at the
grouped-MoE resident-footprint sizes, plus an expert-image read pattern
(40 disjoint 1.86 MB blocks = the rows=4 logical footprint, read row-wise).

Run: python tests/hip_p5_p2_bwceiling.py --out <dir>
Zero eviction required.
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import torch


def timed(dev, fn, reps=50):
    torch.cuda.synchronize(dev)
    times = []
    for _ in range(reps):
        e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
        e0.record()
        fn()
        e1.record()
        torch.cuda.synchronize(dev)
        times.append(e0.elapsed_time(e1))
    times.sort()
    return times[len(times) // 2]  # ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    result = {"date": datetime.now(timezone.utc).isoformat(), "devices": {}}
    for dev_i in (0, 1):
        dev = torch.device(f"cuda:{dev_i}")
        torch.cuda.set_device(dev)
        res = {}
        # sizes in MB: rows=1 18.6, rows=2 37.2, rows=4 74.4 resident; plus 32/148
        for mb in (18.6, 32.0, 37.2, 74.4, 148.0):
            n = int(mb * 1024 * 1024 // 2)  # fp16 elements
            src = torch.randn(n, dtype=torch.float16, device=dev)
            dst = torch.empty_like(src)
            # copy: read + write = 2x traffic; MB/ms == GB/s
            t_copy = timed(dev, lambda: dst.copy_(src))
            # sum: read-only traffic (small scalar write)
            t_read = timed(dev, lambda: src.sum(dtype=torch.float32))
            res[f"{mb}MB"] = {
                "copy_ms": t_copy,
                "copy_GBps": 2 * mb / t_copy,
                "read_GBps": mb / t_read,
                "read_ms": t_read,
            }
            print(f"[gpu{dev_i}] {mb}MB: copy {t_copy:.3f} ms "
                  f"({res[f'{mb}MB']['copy_GBps']:.0f} GB/s eff), "
                  f"read {t_read:.3f} ms ({res[f'{mb}MB']['read_GBps']:.0f} GB/s)", flush=True)
        # expert-image read: 40 disjoint 1.86 MB rows read in one reduction
        E = 40
        per = int(1.8624 * 1024 * 1024 // 2)
        pool = torch.randn(E * per, dtype=torch.float16, device=dev).view(E, per)
        dst40 = torch.empty(E, dtype=torch.float32, device=dev)
        t_g = timed(dev, lambda: torch.sum(pool, dim=1, dtype=torch.float32, out=dst40), reps=20)
        res["expert_rows_read"] = {
            "bytes_MB": E * 1.8624, "ms": t_g, "GBps": E * 1.8624 / t_g}
        print(f"[gpu{dev_i}] expert-rows-read {E * 1.8624:.1f} MB: {t_g:.3f} ms "
              f"({res['expert_rows_read']['GBps']:.0f} GB/s)", flush=True)
        result["devices"][f"gpu{dev_i}"] = res
    args.out.joinpath("bw-ceiling.json").write_text(json.dumps(result, indent=2))
    print("[bw] done", flush=True)


if __name__ == "__main__":
    main()