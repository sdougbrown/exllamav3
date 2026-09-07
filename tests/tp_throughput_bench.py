#!/usr/bin/env python3
"""Direct-load decode/prefill throughput benchmark: TP vs layer-split, one harness.

Loads the model in-process (the same direct path used by tp_logits_control.py and the
Stage 5 smoke) and times single-token decode forwards over N steps plus chunked prefill.
Run each mode in an exclusive two-GPU window (sequential processes), then compare the
reported tok/s. This is a raw model-kernel A/B; it does not include tabby/serving or MTP.

From the repository root (TP mode):
  TORCH_EXTENSIONS_DIR=$HOME/.cache/qwen38-flash-exl3-rocm/torch_extensions \
  TRITON_CACHE_DIR=$HOME/.cache/qwen38-flash-exl3-rocm/triton \
  PYTORCH_ROCM_ARCH=gfx1201 HIP_VISIBLE_DEVICES=0,1 EXL3_DIR=$PWD \
  /home/douglasbrown/vllm-test-env/bin/python tests/tp_throughput_bench.py --mode tp --decode-steps 64

Layer-split control (same window, after the TP process exits):
  ... HIP_VISIBLE_DEVICES=0,1 ... tests/tp_throughput_bench.py --mode ls --decode-steps 64
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = "/home/douglasbrown/Models/Qwen3.8-Flash-Next-exl3-bpw3"
PROMPT = (
    "Once upon a time, a small robot named Sparky discovered a mysterious glowing door "
    "in the middle of the forest. When Sparky opened it, he found a library of every "
    "book that had ever been written and every book that ever would be written."
)


def _warmup(model: Any, cache: Any, tokenizer: Any, n: int = 4) -> None:
    """One quick warmup forward so kernels/CUDA graphs are resident before timing."""
    ids = tokenizer.encode("warm", add_bos=True).cpu()
    params = {
        "attn_mode": "flash_attn",
        "cache": cache,
        "past_len": 0,
        "batch_shape": (1, ids.shape[1]),
    }
    with torch.inference_mode():
        model.prefill(input_ids=ids, params=params)
        model.forward(
            input_ids=ids[:, -1:].contiguous(),
            params={"attn_mode": "flash_attn", "cache": cache, "past_len": ids.shape[1] - 1,
                    "batch_shape": (1, 1)},
        )


def parse_dev(value: str) -> list[float]:
    try:
        d = [float(x) for x in value.split(",")]
    except ValueError:
        raise argparse.ArgumentTypeError("--use-dev comma GiB") from None
    if len(d) != 2 or any(x <= 0 for x in d):
        raise argparse.ArgumentTypeError("--use-dev needs two positive GiB")
    return d


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", required=True, choices=("tp", "ls"))
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--cache-tokens", type=int, default=8192)
    ap.add_argument("--use-dev", type=parse_dev, default=parse_dev("30,30"))
    ap.add_argument("--decode-steps", type=int, default=48)
    ap.add_argument("--warmup-steps", type=int, default=8)
    ap.add_argument("--context", type=int, default=720)
    ap.add_argument("--prefill-chunk", type=int, default=512)
    args = ap.parse_args()

    sys.path.insert(0, str(REPO_ROOT))
    import torch
    from exllamav3 import Cache, Config, Model, Tokenizer

    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        raise SystemExit("bench requires two visible HIP devices")

    torch.manual_seed(0)
    config = Config.from_directory(args.model)
    model = Model.from_config(config)
    tokenizer = Tokenizer.from_config(config)
    cache = Cache(model=model, max_num_tokens=args.cache_tokens, max_batch_size=1)
    loaded = False
    try:
        load_args = {
            "tensor_p": args.mode == "tp",
            "use_per_device": args.use_dev,
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
        # Build a context of length ~args.context (+ prompt length if shorter).
        while ids.shape[1] < args.context:
            ids = torch.cat((ids, ids[:, : args.context - ids.shape[1]]), dim=1)

        params = {
        "attn_mode": "flash_attn",
        "cache": cache,
        "past_len": 0,
        "batch_shape": (1, cache_tokens),
    }
        t_pre = time.time()
        with torch.inference_mode():
            model.prefill(input_ids=ids, params=params)
        prefill_s = time.time() - t_pre
        prefill_tps = ids.shape[1] / prefill_s
        past_len = ids.shape[1]

        # warmup decode tokens
        in_ids = ids[:, -1:].contiguous()
        for _ in range(args.warmup_steps):
            with torch.inference_mode():
                model.forward(input_ids=in_ids, params=params)
            in_ids = in_ids.new_zeros((1, 1), dtype=torch.int32)
            params["past_len"] += 1

        # timed decode tokens
        lat = []
        for _ in range(args.decode_steps):
            t0 = time.time()
            with torch.inference_mode():
                model.forward(input_ids=in_ids, params=params)
            lat.append(time.time() - t0)
            in_ids = in_ids.new_zeros((1, 1), dtype=torch.int32)
            params["past_len"] += 1

        lat_ms = [x * 1000.0 for x in lat]
        tps = [1000.0 / x for x in lat_ms]
        report = {
            "mode": args.mode,
            "load_s": round(load_s, 1),
            "context_len": past_len - args.decode_steps - args.warmup_steps,
            "prefill_tps": round(prefill_tps, 1),
            "prefill_chunk": args.prefill_chunk,
            "decode_steps": len(lat),
            "decode_tps_mean": round(statistics.mean(tps), 2),
            "decode_tps_median": round(statistics.median(tps), 2),
            "decode_tps_min": round(min(tps), 2),
            "decode_tps_max": round(max(tps), 2),
            "decode_ms_mean": round(statistics.mean(lat_ms), 3),
            "decode_ms_p50": round(statistics.median(lat_ms), 3),
            "decode_ms_p95": round(sorted(lat_ms)[int(0.95 * len(lat_ms)) - 1], 3),
            "decode_ms_p99": round(sorted(lat_ms)[int(0.99 * len(lat_ms)) - 1], 3),
        }
        print(json.dumps(report, sort_keys=True))
    finally:
        if loaded:
            model.unload()


if __name__ == "__main__":
    main()
