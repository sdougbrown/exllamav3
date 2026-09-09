#!/usr/bin/env python3
"""GEMM attribution microbenchmark for dense 27B EXL3 prefill (Stage 2 follow-on).

Attribution questions:
  1. Which projections own the ~756 ms hipBLASLt time, at which (M, N, K)?
  2. What is the achieved GEMM efficiency vs raw bf16 references (torch.mm and the
     installed AITER a16w16 GEMM) at the same shapes — i.e. the headroom?

Run (single GPU, approved window):
  HIP_VISIBLE_DEVICES=0 python tests/gemm_attribution_bench.py --out-dir <dir>
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/home/douglasbrown/Models/Qwen3.8-27B-exl3")
    ap.add_argument("--prompt-tokens", type=int, default=4096)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    from exllamav3 import Cache, Config, Model

    dev = torch.device("cuda:0")
    config = Config.from_directory(args.model)
    model = Model.from_config(config)
    cache = Cache(model=model, max_num_tokens=8192, max_batch_size=1)
    loaded = False
    try:
        model.load(device="cuda:0", max_chunk_size=4096, max_batch_size=1)
        loaded = True

        # --- profile one 4096-token prefill with shapes ---
        ids = torch.randint(0, 200000, (1, args.prompt_tokens), dtype=torch.long)
        from torch.profiler import ProfilerActivity, profile

        with torch.inference_mode():
            params = {
                "attn_mode": "flash_attn", "cache": cache, "past_len": 0,
                "batch_shape": (1, 8192),
            }
            prof = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                           record_shapes=True)
            prof.start()
            t0 = time.perf_counter()
            model.prefill(input_ids=ids, params=params)
            torch.cuda.synchronize()
            wall = time.perf_counter() - t0
            prof.stop()

        rows = []
        for evt in prof.key_averages():
            cuda_us = getattr(evt, "self_device_time_total", 0) or 0
            if cuda_us <= 0:
                continue
            rows.append({
                "op": evt.key,
                "cuda_ms": round(cuda_us / 1000, 2),
                "count": evt.count,
                "shapes": [str(s) for s in (getattr(evt, "input_shapes", None) or [])][:4],
            })
        rows.sort(key=lambda r: -r["cuda_ms"])
        (out_dir / "op-averages.json").write_text(json.dumps(rows[:150], indent=1))
        print(f"prefill wall {wall:.3f}s; top ops:")
        for r in rows[:12]:
            print(f"  {r['cuda_ms']:9.2f} ms {r['count']:6d}x {r['op'][:60]} {r['shapes'][0] if r['shapes'] else ''}")

        # --- raw GEMM references at the model's dominant shapes ---
        # config: hidden 5120, intermediate 17408, vocab ~248k; GDN qkv fused 2H+V=10240
        H = 5120
        I = 17408
        V = config.vocab_size
        candidates = [
            ("hidden->qkv_like", H, 10240),
            ("hidden->gateup", H, 2 * I),
            ("inter->down", I, H),
            ("hidden->lmhead", H, V),
        ]
        micro = []
        aiter_gemm_op = None
        for attr in ("gemm_op", "gemm_a16w16"):
            try:
                from aiter.ops.gemm_op_a16w16 import getattr as _g  # never taken
            except Exception:
                pass
            try:
                mod = __import__("aiter.ops.gemm_op_a16w16", fromlist=["*"])
                aiter_gemm_op = getattr(mod, attr, None)
                if aiter_gemm_op is not None:
                    break
            except Exception:
                aiter_gemm_op = None
        print(f"AITER a16w16 reference: {aiter_gemm_op}")
        for label, K, N in candidates:
            for M in (512, 1024, 2048, 4096):
                a = torch.randn(M, K, dtype=torch.bfloat16, device=dev)
                b = torch.randn(N, K, dtype=torch.bfloat16, device=dev)
                for _ in range(3):
                    torch.mm(a, b.t())
                torch.cuda.synchronize()
                ts = []
                for _ in range(5):
                    t0 = time.perf_counter()
                    torch.mm(a, b.t())
                    torch.cuda.synchronize()
                    ts.append(time.perf_counter() - t0)
                ms = statistics.median(ts) * 1e3
                rec = {
                    "shape": label, "M": M, "K": K, "N": N,
                    "torch_mm_ms": round(ms, 3),
                    "torch_mm_tflops": round(2 * M * N * K / (ms * 1e-3) / 1e12, 1),
                }
                if aiter_gemm_op is not None:
                    try:
                        for _ in range(3):
                            aiter_gemm_op(a, b)
                        torch.cuda.synchronize()
                        ts = []
                        for _ in range(5):
                            t0 = time.perf_counter()
                            aiter_gemm_op(a, b)
                            torch.cuda.synchronize()
                            ts.append(time.perf_counter() - t0)
                        ams = statistics.median(ts) * 1e3
                        rec["aiter_ms"] = round(ams, 3)
                        rec["aiter_tflops"] = round(2 * M * N * K / (ams * 1e-3) / 1e12, 1)
                    except Exception as e:
                        rec["aiter_error"] = repr(e)[:200]
                micro.append(rec)
                print(json.dumps(rec))
        (out_dir / "gemm-micro.json").write_text(json.dumps(micro, indent=1))
    finally:
        if loaded:
            model.unload()


if __name__ == "__main__":
    main()