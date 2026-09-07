#!/usr/bin/env python3
"""Direct-load BATCHED decode throughput benchmark: TP vs layer-split at concurrency > 1.

Loads the model in-process (same direct path as tp_throughput_bench.py) with
max_batch_size = B, prefills B independent (offset-replicated) sequences of identical
length so each occupies a distinct cache row, then decodes one next-token for all B
sequences in a single forward per step. Reports per-step latency and aggregate tok/s
(= B tokens / step) so TP and layer-split can be compared at the batch sizes where
TP's all-reduce overhead may be amortized.

Timed forwards are bracketed by torch.cuda.synchronize() (the bsz=1 harness does not
sync, but LS forwards return before GPU completion, so syncing is required for a
fair TP-vs-LS comparison). Recurrent (GDN) states: prefill allocates one state slot
per sequence (cache.max_batch_size = B); the captured state list is re-injected into
a fresh params dict every forward, since the TP path strips params["cache"] and tensor
params to int handles after the first send.

From the repository root (TP mode):
  TORCH_EXTENSIONS_DIR=$HOME/.cache/qwen38-flash-exl3-rocm/torch_extensions \
  TRITON_CACHE_DIR=$HOME/.cache/qwen38-flash-exl3-rocm/triton \
  PYTORCH_ROCM_ARCH=gfx1201 HIP_VISIBLE_DEVICES=0,1 EXL3_DIR=$PWD \
  /home/douglasbrown/vllm-test-env/bin/python tests/tp_batch_decode_bench.py \
      --mode tp --batch 8 --context 4096 --decode-steps 40

Layer-split control (same window, after the TP process exits):
  ... HIP_VISIBLE_DEVICES=0,1 ... tests/tp_batch_decode_bench.py \
      --mode ls --batch 8 --context 4096 --decode-steps 40
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = "/home/douglasbrown/Models/Qwen3.8-Flash-Next-exl3-bpw3"
PROMPT = (
    "Once upon a time, a small robot named Sparky discovered a mysterious glowing door "
    "in the middle of the forest. When Sparky opened it, he found a library of every "
    "book that had ever been written and every book that ever would be written."
)


def parse_dev(value: str) -> list[float]:
    try:
        d = [float(x) for x in value.split(",")]
    except ValueError:
        raise argparse.ArgumentTypeError("--use-dev comma GiB") from None
    if len(d) != 2 or any(x <= 0 for x in d):
        raise argparse.ArgumentTypeError("--use-dev needs two positive GiB")
    return d


def round_up_256(x: int) -> int:
    return (x + 255) // 256 * 256


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", required=True, choices=("tp", "ls"))
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--context", type=int, default=4096)
    ap.add_argument("--cache-tokens", type=int, default=None,
                    help="total cache tokens, multiple of 256 (default: B * per-seq rounded capacity)")
    ap.add_argument("--decode-steps", type=int, default=40)
    ap.add_argument("--warmup-steps", type=int, default=8)
    ap.add_argument("--prefill-chunk", type=int, default=512)
    ap.add_argument("--use-dev", type=parse_dev, default=parse_dev("30,30"))
    args = ap.parse_args()

    batch = args.batch
    context = args.context
    # Per-sequence KV extent must cover prefill + every warmup/timed decode token, page-aligned
    per_seq_len = round_up_256(context + args.warmup_steps + args.decode_steps)
    cache_tokens = args.cache_tokens
    if cache_tokens is None:
        cache_tokens = batch * per_seq_len
    if cache_tokens % 256 != 0:
        raise SystemExit("--cache-tokens must be a multiple of 256")
    if batch * per_seq_len > cache_tokens:
        raise SystemExit(
            f"--cache-tokens too small: need {batch} * {per_seq_len} = {batch * per_seq_len} "
            f"tokens for batch {batch} x {context}-token contexts plus decode steps")

    sys.path.insert(0, str(REPO_ROOT))
    import torch
    from exllamav3 import Cache, Config, Model, Tokenizer

    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        raise SystemExit("bench requires two visible HIP devices")

    torch.manual_seed(0)
    config = Config.from_directory(args.model)
    model = Model.from_config(config)
    tokenizer = Tokenizer.from_config(config)
    cache = Cache(model=model, max_num_tokens=cache_tokens, max_batch_size=batch)
    loaded = False
    rs: list | None = None
    try:
        load_args = {
            "tensor_p": args.mode == "tp",
            "use_per_device": args.use_dev,
            "max_chunk_size": args.prefill_chunk,
            "max_batch_size": batch,
        }
        if args.mode == "tp":
            load_args["tp_backend"] = "nccl"
        t_load = time.time()
        model.load(**load_args)
        loaded = True
        load_s = time.time() - t_load

        # Build B distinct rows of identical length: an offset slice of a repeated context,
        # so row i starts i tokens into the base stream
        base = tokenizer.encode(PROMPT, add_bos=True).cpu()
        while base.shape[1] < context + batch:
            base = torch.cat((base, base), dim=1)
        ids = torch.stack([base[0, i:i + context] for i in range(batch)], dim=0).contiguous()

        prefill_params = {
            "attn_mode": "flash_attn",
            "cache": cache,
            "past_len": 0,
            "batch_shape": (batch, per_seq_len),
        }
        t_pre = time.time()
        with torch.inference_mode():
            model.prefill(input_ids=ids, params=prefill_params)
        prefill_s = time.time() - t_pre
        # One recurrent state slot per sequence, advanced to position = context by prefill
        rs = prefill_params.get("recurrent_states")
        past_len = context

        def step_params() -> dict:
            # Fresh dict every forward: the TP path strips params["cache"] and tensor params
            # to int ids after the first send, so nothing here may be reused
            p = {"attn_mode": "flash_attn", "cache": cache, "past_len": past_len,
                 "batch_shape": (batch, per_seq_len)}
            if rs:
                p["recurrent_states"] = rs
            return p

        # warmup decode tokens (one forward per step, B tokens per step)
        in_ids = ids[:, -1:].contiguous()
        for _ in range(args.warmup_steps):
            with torch.inference_mode():
                model.forward(input_ids=in_ids, params=step_params())
            torch.cuda.synchronize()
            in_ids = in_ids.new_zeros((batch, 1), dtype=torch.int32)
            past_len += 1

        # timed batched decode tokens
        lat = []
        for _ in range(args.decode_steps):
            torch.cuda.synchronize()
            t0 = time.time()
            with torch.inference_mode():
                model.forward(input_ids=in_ids, params=step_params())
            torch.cuda.synchronize()
            lat.append(time.time() - t0)
            in_ids = in_ids.new_zeros((batch, 1), dtype=torch.int32)
            past_len += 1

        lat_ms = [x * 1000.0 for x in lat]
        n = len(lat_ms)
        p95_ms = sorted(lat_ms)[max(0, min(n - 1, -(-int(0.95 * n * 100) // 100) - 1))]
        step_s = lat  # seconds per step
        agg_tps = [batch / s for s in step_s]
        req_tps = [1.0 / s for s in step_s]
        report = {
            "mode": args.mode,
            "batch": batch,
            "context_len": context,
            "load_s": round(load_s, 1),
            "prefill_s": round(prefill_s, 3),
            "cache_tokens": cache_tokens,
            "warmup_steps": args.warmup_steps,
            "decode_steps": n,
            "per_step_ms_mean": round(statistics.mean(lat_ms), 3),
            "per_step_ms_median": round(statistics.median(lat_ms), 3),
            "per_step_ms_p95": round(p95_ms, 3),
            "per_step_ms_min": round(min(lat_ms), 3),
            "per_step_ms_max": round(max(lat_ms), 3),
            "aggregate_tok_s_mean": round(statistics.mean(agg_tps), 2),
            "aggregate_tok_s_median": round(statistics.median(agg_tps), 2),
            "per_req_tok_s_mean": round(statistics.mean(req_tps), 2),
        }
        print(json.dumps(report, sort_keys=True))
    finally:
        # Return the recurrent state slots to the cache pool before the cache loses its
        # layer tensors with the model (state.free() is bookkeeping-only)
        for state in (rs or []):
            try:
                state.free()
            except Exception:
                pass
        if loaded:
            model.unload()


if __name__ == "__main__":
    main()