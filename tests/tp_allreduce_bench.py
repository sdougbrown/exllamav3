#!/usr/bin/env python3
"""TP2 microbench: RCCL all_reduce at the real prefill payload shapes
(rows x 5120 bf16) — establishes the per-call collective cost that an AITER
custom all-reduce would have to beat.

Spawned as two rank processes:
  RANK=0 WORLD_SIZE=2 HIP_VISIBLE_DEVICES=0,1 python tests/tp_allreduce_bench.py --out-dir <dir> &
  RANK=1 WORLD_SIZE=2 HIP_VISIBLE_DEVICES=0,1 python tests/tp_allreduce_bench.py --out-dir <dir>
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path

import torch
import torch.distributed as dist


def timed(fn, warmup=5, n=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts) * 1e3


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--iters", type=int, default=20)
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl",
        init_method="tcp://127.0.0.1:29617",
        rank=rank, world_size=world,
    )
    dev = torch.device(f"cuda:{rank}")
    results = []
    for rows in (512, 1024, 2048, 4096):
        x = torch.randn(rows, 5120, dtype=torch.bfloat16, device=dev)
        payload_mb = x.numel() * x.element_size() / 1e6
        ms = timed(lambda: dist.all_reduce(x, op=dist.ReduceOp.SUM),
                   n=args.iters)
        rec = {
            "rank": rank, "rows": rows, "payload_mb": round(payload_mb, 1),
            "rccl_ms": round(ms, 3),
            "gbps_effective": round(payload_mb / (ms / 1e3), 1),
        }
        results.append(rec)
        if rank == 0:
            print(json.dumps(rec), flush=True)
    dist.barrier()
    dist.destroy_process_group()
    if rank == 0:
        (out_dir / "allreduce-bench.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()