#!/usr/bin/env python3
"""Stage B control 1: per-call decomposition of the HIP exl3 GEMV on real 27B shapes.

Loads the 27B EXL3 model on one GPU, picks one representative LinearEXL3 per
shape class from the inventory, and times the raw ext.exl3_gemv binding at
M = 1, 2, 4, 8, 16 with preallocated workspaces (binding call only: includes the
kernel's input-Hadamard stage, excludes Python-side allocs/bias). Controls per
shape: the module's reconstruct_hgemm path (what the 96 ba_proj linears do) and
a dense FP16 torch.mm matvec on the reconstructed weight (bandwidth-bound
reference). Reports per-call time, implied compressed-byte bandwidth, and a
t(M) = t0 + M*t_row least-squares fit to split fixed vs per-row cost.

  PYTORCH_ROCM_ARCH=gfx1201 HIP_VISIBLE_DEVICES=0 \
  /home/douglasbrown/vllm-test-env/bin/python tests/hip_gemv_msweep.py \
      --out-dir <dir>
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = os.path.expanduser("~/Models/Qwen3.8-27B-exl3")

# shape class -> representative substring of the module key (in->out)
SHAPES = [
    ("gdn_qkv_5120_6144", (5120, 6144)),
    ("gdn_qkvz_5120_10240", (5120, 10240)),
    ("attn_qkv_5120_12288", (5120, 12288)),
    ("attn_o_6144_5120", (6144, 5120)),
    ("gate_up_5120_17408", (5120, 17408)),
    ("down_17408_5120", (17408, 5120)),
    ("lm_head_5120_248320_k6", (5120, 248320)),
    ("ba_proj_5120_48", (5120, 48)),
]
MS = [1, 2, 4, 8, 16]
ITERS = 100
WARMUP = 8


def discover_shapes(model, max_classes=8):
    """Pick representative EXL3 linear shape classes (in, out), largest-N first."""
    classes = {}
    for module in model:
        inner = getattr(module, "inner", None)
        if inner is None or not hasattr(inner, "trellis"):
            continue
        key = (inner.in_features, inner.out_features)
        classes.setdefault(key, []).append(inner)
    picked = sorted(classes, key=lambda k: -classes[k][0].out_features)[:max_classes]
    return [(f"{i}->{o}", (i, o)) for i, o in sorted(picked, key=lambda k: -k[1])]


def time_callable(fn, iters=ITERS, warmup=WARMUP):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts) * 1e6  # us


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--auto-shapes", type=int, default=0,
                    help="discover the N largest EXL3 linear shape classes from the model instead of the hardcoded 27B list")
    ap.add_argument("--use-dev", default=None,
                    help="per-device load budgets, e.g. '30,30' (default: all on device 0)")
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(REPO_ROOT))
    from exllamav3 import Config, Model
    from exllamav3.ext import exllamav3_ext as ext_mod
    from exllamav3.util.tensor import g_tensor_cache

    torch.manual_seed(0)
    config = Config.from_directory(args.model)
    model = Model.from_config(config)
    loaded = False
    try:
        load_budgets = [float(x) for x in args.use_dev.split(",")] if args.use_dev else [31.0, 0.0]
        model.load(use_per_device=load_budgets, max_chunk_size=512, max_batch_size=1)
        loaded = True

        shape_classes = discover_shapes(model, args.auto_shapes) if args.auto_shapes else SHAPES

        # one representative linear per (in,out) shape (Linear.load_exl3 stores the
        # EXL3 implementation in .inner; plain FP16 linears have no .inner)
        picked = {}
        for module in model:
            inner = getattr(module, "inner", None)
            if inner is None or not hasattr(inner, "trellis"):
                continue
            key = (inner.in_features, inner.out_features)
            if key in [s[1] for s in shape_classes] and key not in picked:
                picked[key] = inner
            if len(picked) == len(shape_classes):
                break

        rows = []
        for name, (inp, outp) in shape_classes:
            lin = picked.get((inp, outp))
            if lin is None:
                print(f"skip {name}: no linear found")
                continue
            dev = lin.trellis.device
            bytes_compressed = lin.trellis.numel() * 2 + lin.suh.numel() * 2 + lin.svh.numel() * 2
            row = {
                "shape": f"{inp}->{outp}",
                "class": name,
                "K": int(lin.K),
                "mul1": bool(lin.mul1),
                "compressed_MB": round(bytes_compressed / 1e6, 2),
                "bw_bound_us_at_550GBs": round(bytes_compressed / 550e9 * 1e6, 1),
                "gemv_us": {},
                "reconstruct_us": {},
                "dense_fp16_us": {},
            }
            for m in MS:
                x = (torch.randn(m, inp, device=dev, dtype=torch.half) * 0.1).contiguous()
                y = torch.empty(m, outp, device=dev, dtype=torch.half)
                a_had = g_tensor_cache.get(dev, (m, inp), torch.half, "msweep_a_had")
                def gemv_call(x=x, y=y, a_had=a_had, lin=lin):
                    ext_mod.exl3_gemv(x, lin.trellis, y, lin.suh, a_had, lin.svh,
                                      lin.mcg, lin.mul1)
                row["gemv_us"][m] = round(time_callable(gemv_call), 2)

                xr = x.to(torch.half).contiguous()
                def recon_call(xr=xr, lin=lin):
                    lin.reconstruct_hgemm(xr, None)
                try:
                    row["reconstruct_us"][m] = round(time_callable(recon_call, iters=30), 2)
                except Exception as exc:
                    row["reconstruct_us"][m] = f"err: {exc}"

                if m == 1:
                    try:
                        w = lin.get_weight_tensor()  # dense [in,out] per torch.mm probe
                        wt = w if w.dtype == torch.half else w.to(torch.half)
                        def mv_call(wt=wt, xr=xr):
                            torch.mm(xr, wt)
                        row["dense_fp16_us"][m] = round(time_callable(mv_call, iters=30), 2)
                    except Exception as exc:
                        row["dense_fp16_us"][m] = f"err: {exc}"
            rows.append(row)
            print(json.dumps(row), flush=True)

        # fixed-vs-row fit for the gemv data
        import numpy as np
        for row in rows:
            ms = [m for m in MS if isinstance(row["gemv_us"].get(m), (int, float))]
            ts = [row["gemv_us"][m] for m in ms]
            if len(ms) >= 2:
                A = np.vstack([np.ones(len(ms)), ms]).T
                t0, t_row = np.linalg.lstsq(A, np.array(ts), rcond=None)[0]
                row["fit_t0_us"] = round(float(t0), 2)
                row["fit_t_per_row_us"] = round(float(t_row), 2)
                row["fit"] = [round(t0 + t_row * m, 2) for m in ms]

        (out_dir / "gemv-msweep.json").write_text(json.dumps(rows, indent=1))
        print(json.dumps({"summary": "done", "rows": len(rows)}))
    finally:
        if loaded:
            model.unload()


if __name__ == "__main__":
    main()