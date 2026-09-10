#!/usr/bin/env python3
"""Per-projection GEMM attribution: time real EXL3 quantized Linear modules at
prefill row counts, splitting Hadamard/reconstruct vs hipBLASLt (ext.hgemm) via
profiler, with raw torch.mm as the bf16 reference ceiling.

  HIP_VISIBLE_DEVICES=0 python tests/gemm_projection_bench.py --out-dir <dir>
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


def main() -> None:  # (no per-call profiler helper needed; timing only)
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/home/douglasbrown/Models/Qwen3.8-27B-exl3")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    from exllamav3 import Config, Model
    from exllamav3.ext import exllamav3_ext as ext

    dev = torch.device("cuda:0")
    config = Config.from_directory(args.model)
    model = Model.from_config(config)
    loaded = False
    try:
        model.load(device="cuda:0", max_chunk_size=4096, max_batch_size=1)
        loaded = True

        # inventory distinct Linear shapes
        shapes = defaultdict(lambda: {"count": 0, "module": None})
        linears = []
        stack = [model]
        while stack:
            m = stack.pop()
            if type(m).__name__ == "Linear":
                linears.append(m)
            stack.extend(getattr(m, "modules", []))
        for lin in linears:
            shapes[(lin.in_features, lin.out_features)]["count"] += 1
            shapes[(lin.in_features, lin.out_features)]["module"] = lin
        inv = {f"{k[0]}->{k[1]}": v["count"] for k, v in sorted(shapes.items())}
        print("Linear shapes:", json.dumps(inv))
        (out_dir / "linear-inventory.json").write_text(json.dumps(inv, indent=1))

        results = []
        for (kin, nout), v in sorted(shapes.items()):
            lin = v["module"]
            inner_mod = type(lin.inner).__module__
            if not inner_mod.endswith("exl3"):
                print(f"skip non-EXL3 linear {kin}->{nout} ({inner_mod})")
                continue
            for M in (512, 1024, 2048, 4096):
                x = torch.randn(M, kin, dtype=torch.half, device=dev)
                # full quantized path (had/reconstruct/hgemm per EXL3 routing)
                with torch.inference_mode():
                    for _ in range(3):
                        lin.forward(x, {})
                    torch.cuda.synchronize()
                    ts = []
                    for _ in range(5):
                        t0 = time.perf_counter()
                        lin.forward(x, {})
                        torch.cuda.synchronize()
                        ts.append(time.perf_counter() - t0)
                    full_ms = statistics.median(ts) * 1e3

                    # isolated hgemm with a prebuilt fp16 weight
                    inner = lin.inner
                    if nout <= 32768:
                        w = torch.empty((kin, nout), dtype=torch.half, device=dev)
                        ext.reconstruct(w, inner.trellis, inner.K, inner.mcg, inner.mul1)
                    else:
                        w = None  # lm_head not used in prefill; skip isolated hgemm
                    hgemm_ms = None
                    hgemm_tflops = None
                    torch_mm_ms = None
                    if w is not None:
                        xh = torch.empty_like(x, dtype=torch.half)
                        y = torch.empty((M, nout), dtype=torch.half, device=dev)
                        for _ in range(3):
                            ext.hgemm(xh, w, y)
                        torch.cuda.synchronize()
                        ts = []
                        for _ in range(5):
                            t0 = time.perf_counter()
                            ext.hgemm(xh, w, y)
                            torch.cuda.synchronize()
                            ts.append(time.perf_counter() - t0)
                        hgemm_ms = statistics.median(ts) * 1e3
                        hgemm_tflops = 2 * M * nout * kin / (hgemm_ms * 1e-3) / 1e12
                        wb = torch.randn(nout, kin, dtype=torch.half, device=dev)
                        for _ in range(3):
                            torch.mm(x, wb.t())
                        torch.cuda.synchronize()
                        ts = []
                        for _ in range(5):
                            t0 = time.perf_counter()
                            torch.mm(x, wb.t())
                            torch.cuda.synchronize()
                            ts.append(time.perf_counter() - t0)
                        torch_mm_ms = statistics.median(ts) * 1e3
                rec = {
                    "in": kin, "out": nout, "M": M, "layer_count": v["count"],
                    "exl3_full_ms": round(full_ms, 3),
                    "hgemm_ms": round(hgemm_ms, 3) if hgemm_ms else None,
                    "hgemm_tflops": round(hgemm_tflops, 1) if hgemm_tflops else None,
                    "torch_mm_ms": round(torch_mm_ms, 3) if torch_mm_ms else None,
                    "torch_mm_tflops": round(2 * M * nout * kin / (torch_mm_ms * 1e-3) / 1e12, 1) if torch_mm_ms else None,
                }
                results.append(rec)
                print(json.dumps(rec))
        (out_dir / "projection-bench.json").write_text(json.dumps(results, indent=1))
    finally:
        if loaded:
            model.unload()


if __name__ == "__main__":
    main()