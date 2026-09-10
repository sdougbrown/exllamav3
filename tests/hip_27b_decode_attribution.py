#!/usr/bin/env python3
"""Stage A: Qwen3.8-27B dense EXL3 c1 decode attribution (single process).

Loads the 27B EXL3 model (dense, mul1, 4.0 bpw, head_bits 6) in-process in the
requested topology, prefills one sequence, then times target-only decode steps
with a torch.cuda.synchronize bracket. During the timed phase it counts EXL3
dispatch routes by spying on the ext bindings (hip gemv / reconstruct / hgemm).
With --profile-steps it additionally captures a torch.profiler window and
classifies CUDA kernel time into attribution buckets (linears, Hadamard, GDN,
attention, elementwise, copies, other), reporting device-busy union per step
and top kernels.

From the repository root (layer-split control over two GPUs):
  PYTORCH_ROCM_ARCH=gfx1201 HIP_VISIBLE_DEVICES=0,1 \
  TORCH_EXTENSIONS_DIR=$HOME/.cache/qwen38-flash-exl3-rocm/torch_extensions \
  TRITON_CACHE_DIR=$HOME/.cache/qwen38-flash-exl3-rocm/triton \
  /home/douglasbrown/vllm-test-env/bin/python tests/hip_27b_decode_attribution.py \
      --mode ls --context 128 --decode-steps 256 --out-dir <dir>

TP2 (matched against a TP2 vLLM reference):
  ... tests/hip_27b_decode_attribution.py --mode tp --context 4096 ...
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = os.path.expanduser("~/Models/Qwen3.8-27B-exl3")
PROMPT = (
    "Once upon a time, a small robot named Sparky discovered a mysterious glowing door "
    "in the middle of the forest. When Sparky opened it, he found a library of every "
    "book that had ever been written and every book that ever would be written."
)

# Kernel-name bucket regexes (checked in order; first match wins)
BUCKETS = [
    ("exl3_gemv", ("exl3_gemv", "gemv_k", "gemvkernel")),
    ("exl3_moe", ("moe_", "grouped_gemv")),
    ("reconstruct_dequant", ("dequant", "reconstruct", "trellis", "unpack")),
    ("hgemm_blas", ("cijk", "hgemm", "hipblaslt", "gemm")),
    ("hadamard", ("had",)),
    ("gdn_recurrent", ("gated_delta", "recurrent_", "gdn", "conv1d", "causal_conv")),
    ("attention", ("flash", "fmha", "attn_", "attention", "paged", "topk", "dsa", "qsa")),
    ("norm_rope", ("rmsnorm", "rms_", "rope", "norm", "layer_norm")),
    ("elementwise", ("elementwise", "vectorized", "silu", "sigmoid", "mul", "add", "div", "clamp", "softmax")),
    ("copy_cat", ("copy", "cat_", "concat", "nonzero", "index", "gather", "scatter", "sort", "unique", "bincount")),
]

UNKNOWN = "other"


def classify(name: str) -> str:
    n = name.lower()
    for bucket, keys in BUCKETS:
        if any(k in n for k in keys):
            return bucket
    return UNKNOWN


def round_up_256(x: int) -> int:
    return (x + 255) // 256 * 256


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", required=True, choices=("ls", "tp"))
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--context", type=int, default=128)
    ap.add_argument("--decode-steps", type=int, default=256)
    ap.add_argument("--warmup-steps", type=int, default=16)
    ap.add_argument("--prefill-chunk", type=int, default=512)
    ap.add_argument("--use-dev", default="30,30")
    ap.add_argument("--profile-steps", type=int, default=0,
                    help="capture a torch.profiler window over N mid-loop decode steps")
    ap.add_argument("--profile-wait-steps", type=int, default=8,
                    help="timed steps to run (unprofiled) before the profiler window")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    use_dev = [float(x) for x in args.use_dev.split(",")]

    sys.path.insert(0, str(REPO_ROOT))
    import torch

    # Route spies must be installed before exllamav3.modules.quant.exl3 is first
    # imported (module-level name binding would bypass later patches).
    from exllamav3 import ext as ext_mod
    counts = {"gemv": 0, "reconstruct": 0, "reconstruct_slice": 0,
              "reconstruct_had_slice": 0, "hgemm": 0}
    real = {
        "gemv": getattr(ext_mod, "exl3_gemv", None),
        "reconstruct": getattr(ext_mod, "reconstruct", None),
        "reconstruct_slice": getattr(ext_mod, "reconstruct_slice", None),
        "reconstruct_had_slice": getattr(ext_mod, "reconstruct_had_slice", None),
        "hgemm": getattr(ext_mod, "hgemm", None),
    }

    def spy(key, fn):
        def wrapper(*a, **k):
            counts[key] += 1
            return fn(*a, **k)
        return wrapper

    for key, attr in (("gemv", "exl3_gemv"), ("reconstruct", "reconstruct"),
                      ("reconstruct_slice", "reconstruct_slice"),
                      ("reconstruct_had_slice", "reconstruct_had_slice"),
                      ("hgemm", "hgemm")):
        if real[key] is not None:
            setattr(ext_mod, attr, spy(key, real[key]))

    from exllamav3 import Cache, Config, Model, Tokenizer

    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise SystemExit("bench requires a visible HIP device")

    torch.manual_seed(0)
    config = Config.from_directory(args.model)
    model = Model.from_config(config)
    tokenizer = Tokenizer.from_config(config)
    per_seq_len = round_up_256(args.context + args.warmup_steps + args.decode_steps)
    cache = Cache(model=model, max_num_tokens=per_seq_len, max_batch_size=1)
    loaded = False
    rs = None
    try:
        load_args = {
            "tensor_p": args.mode == "tp",
            "use_per_device": use_dev,
            "max_chunk_size": args.prefill_chunk,
            "max_batch_size": 1,
        }
        if args.mode == "tp":
            load_args["tp_backend"] = "nccl"
        t_load = time.time()
        model.load(**load_args)
        loaded = True
        load_s = time.time() - t_load

        ids = tokenizer.encode(PROMPT, add_bos=True).cpu()
        while ids.shape[1] < args.context:
            ids = torch.cat((ids, ids), dim=1)
        ids = ids[:, -args.context:].contiguous()

        prefill_params = {
            "attn_mode": "flash_attn",
            "cache": cache,
            "past_len": 0,
            "batch_shape": (1, per_seq_len),
        }
        with torch.inference_mode():
            model.prefill(input_ids=ids, params=prefill_params)
        rs = prefill_params.get("recurrent_states")
        past_len = args.context

        def step_params() -> dict:
            p = {"attn_mode": "flash_attn", "cache": cache, "past_len": past_len,
                 "batch_shape": (1, per_seq_len)}
            if rs:
                p["recurrent_states"] = rs
            return p

        in_ids = ids[:, -1:].contiguous()
        for _ in range(args.warmup_steps):
            with torch.inference_mode():
                model.forward(input_ids=in_ids, params=step_params())
            torch.cuda.synchronize()
            in_ids = in_ids.new_zeros((1, 1), dtype=torch.int32)
            past_len += 1

        # Route spies already installed above (pre-import); counts cover the
        # whole process run: warmup + timed phase.
        prof = None
        lat = []
        profiled = 0
        if args.profile_steps > 0:
            from torch.profiler import ProfilerActivity, profile
            prof = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA])
            prof.__enter__()
        try:
            for step in range(args.decode_steps):
                torch.cuda.synchronize()
                t0 = time.time()
                with torch.inference_mode():
                    model.forward(input_ids=in_ids, params=step_params())
                torch.cuda.synchronize()
                lat.append(time.time() - t0)
                in_ids = in_ids.new_zeros((1, 1), dtype=torch.int32)
                past_len += 1
                if args.profile_steps > 0 and step >= args.warmup_steps + args.profile_wait_steps:
                    profiled += 1
                    if profiled == args.profile_steps:
                        try:
                            prof.export_chrome_trace(str(out_dir / f"profile-{args.mode}-ctx{args.context}{args.tag}.json"))
                        except Exception as exc:
                            print(f"profiler export skipped: {exc}", file=sys.stderr)
                        prof.__exit__(None, None, None)
                        prof = None
        finally:
            if prof is not None:
                try:
                    prof.export_chrome_trace(str(out_dir / f"profile-{args.mode}-ctx{args.context}{args.tag}.json"))
                except Exception:
                    pass
                prof.__exit__(None, None, None)
            for key, attr in (("gemv", "exl3_gemv"), ("reconstruct", "reconstruct"),
                              ("reconstruct_slice", "reconstruct_slice"),
                              ("reconstruct_had_slice", "reconstruct_had_slice"),
                              ("hgemm", "hgemm")):
                if real[key] is not None:
                    setattr(ext_mod, attr, real[key])

        lat_ms = [x * 1000.0 for x in lat]
        n = len(lat_ms)
        report = {
            "mode": args.mode,
            "model": args.model,
            "context_len": args.context,
            "load_s": round(load_s, 1),
            "warmup_steps": args.warmup_steps,
            "decode_steps": n,
            "per_step_ms_mean": round(statistics.mean(lat_ms), 3),
            "per_step_ms_median": round(statistics.median(lat_ms), 3),
            "per_step_ms_p95": round(sorted(lat_ms)[max(0, min(n - 1, -(-int(0.95 * n * 100) // 100) - 1))], 3),
            "per_step_ms_min": round(min(lat_ms), 3),
            "per_step_ms_max": round(max(lat_ms), 3),
            "tok_s_median": round(1000.0 / statistics.median(lat_ms), 2),
            "tok_s_mean": round(1000.0 / statistics.mean(lat_ms), 2),
            "route_counts": counts,
            "profile_steps": profiled,
        }

        if args.profile_steps > 0:
            trace_path = out_dir / f"profile-{args.mode}-ctx{args.context}{args.tag}.json"
            if Path(trace_path).exists():
                kern_evts = [e for e in json.loads(trace_path.read_text())["traceEvents"]
                             if e.get("cat") == "kernel" and "dur" in e]
                span_us = (max(e["ts"] + e["dur"] for e in kern_evts)
                           - min(e["ts"] for e in kern_evts)) if kern_evts else 0
                busy_us = 0.0
                per_bucket_us = defaultdict(float)
                per_bucket_calls = defaultdict(int)
                kernel_rows = defaultdict(lambda: [0.0, 0])
                for e in kern_evts:
                    b = classify(e["name"])
                    per_bucket_us[b] += e["dur"]
                    per_bucket_calls[b] += 1
                    nm = e["name"]
                    row = kernel_rows[nm if len(nm) < 160 else nm[:157] + "..."]
                    row[0] += e["dur"]
                    row[1] += 1
                # union of device intervals (single device if pid splits absent)
                iv = sorted((e["ts"], e["ts"] + e["dur"]) for e in kern_evts)
                cur = None
                for s, en in iv:
                    if cur is None:
                        cur = [s, en]
                    elif s <= cur[1]:
                        cur[1] = max(cur[1], en)
                    else:
                        busy_us += cur[1] - cur[0]
                        cur = [s, en]
                if cur:
                    busy_us += cur[1] - cur[0]
                # attribute the window to the steps it covers (>=1; count via span/median)
                steps_in_window = max(1, round(span_us / 1000.0 / statistics.median(lat_ms)))
                report["profile_span_ms"] = round(span_us / 1000.0, 2)
                report["profile_steps_captured"] = steps_in_window
                report["device_busy_ms_per_step"] = round(busy_us / 1000.0 / steps_in_window, 3)
                report["bucket_ms_per_step"] = {
                    k: round(v / 1000.0 / steps_in_window, 3)
                    for k, v in sorted(per_bucket_us.items(), key=lambda kv: -kv[1])
                }
                report["bucket_calls_per_step"] = {
                    k: round(v / steps_in_window, 1)
                    for k, v in sorted(per_bucket_calls.items(), key=lambda kv: -kv[1])
                }
                report["top_kernels"] = [
                    {"name": name, "cuda_ms_per_step": round(v[0] / 1000.0 / steps_in_window, 3),
                     "calls_per_step": round(v[1] / steps_in_window, 1)}
                    for name, v in sorted(kernel_rows.items(), key=lambda kv: -kv[1][0])[:20]
                ]
                report["trace"] = str(trace_path)
                profiled = steps_in_window

        tag = args.tag or f"{args.mode}-ctx{args.context}"
        (out_dir / f"report-{tag}.json").write_text(json.dumps(report, indent=1))
        print(json.dumps(report, sort_keys=True))
        sys.stdout.flush()
    finally:
        for state in (rs or []):
            try:
                state.free()
            except Exception:
                pass
        if loaded:
            model.unload()


if __name__ == "__main__":
    main()