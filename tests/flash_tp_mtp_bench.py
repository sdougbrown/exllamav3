#!/usr/bin/env python3
"""Flash TP2 + MTP3 prefill+decode bench (offline Generator, production profile).

Adapted from tests/hip_p4_perf.py's build/run_trial with tensor-parallel target
loading. One process per pool size (VRAM-clean). Measures TTFT (12K prompt),
decode wall/tok-s with MTP3, draft acceptance, per-rank memory via rank-local
sync receipts, and KFD evicted_ms deltas.

  HIP_VISIBLE_DEVICES=0,1 python tests/flash_tp_mtp_bench.py \
      --cache-tokens 393216 --k-bits 8 --v-bits 8 --out-dir <dir>
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import statistics
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).parent))

MODEL = os.path.expanduser("~/Models/Qwen3.8-Flash-Next-exl3-bpw3")

LONG_PROMPT = (
    "The old lighthouse keeper counted the waves each night, and every wave "
    "carried a name he had given it long ago. "
)


def read_evicted_ms(pid: int) -> dict:
    out = {}
    for e in sorted(Path(f"/sys/class/kfd/kfd/proc/{pid}").glob("stats_*/evicted_ms")):
        try:
            out[e.parent.name] = int(e.read_text().strip())
        except OSError:
            out[e.parent.name] = None
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-tokens", type=int, default=393216)
    ap.add_argument("--k-bits", type=int, default=8)
    ap.add_argument("--v-bits", type=int, default=8)
    ap.add_argument("--prompt-tokens", type=int, default=12288)
    ap.add_argument("--new-tokens", type=int, default=255)
    ap.add_argument("--draft-tokens", type=int, default=3)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--mtp", action="store_true", default=True)
    ap.add_argument("--no-tp", action="store_true")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    import torch
    from exllamav3 import Cache, Config, Model, Tokenizer
    from exllamav3.cache import CacheLayer_quant
    from exllamav3.generator import Generator, Job
    from exllamav3.generator.sampler.presets import DefaultSampler
    from tp_prefill_wall_bench import mp_rank_sync  # rank receipts via worker machinery

    # production profile
    os.environ["EXL3_MOE_SYNC_FREE_COUNT"] = "1"
    os.environ["EXL3_QC_STAGING"] = "1"
    os.environ["EXL3_PREFILL_ASYNC_UPLOADS"] = "0"
    os.environ["EXL3_BLOCK_GRAPH"] = "0"

    config = Config.from_directory(args.model)
    config.infer_params.ngram_stream_from_disk = True
    target = Model.from_config(config, component="text")
    tokenizer = Tokenizer.from_config(config)

    cache = Cache(target, max_num_tokens=args.cache_tokens,
                  layer_type=CacheLayer_quant, k_bits=args.k_bits, v_bits=args.v_bits,
                  max_batch_size=1, max_history=3)

    draft = None
    draft_cache = None
    gen_kwargs = {}
    loaded = False
    try:
        # capture per-rank pids after children spawn
        target.load(tensor_p=not args.no_tp, tp_backend="nccl",
                    use_per_device=[30.0, 30.0], max_chunk_size=512,
                    max_output_size=32, max_batch_size=1, verbose=False)
        loaded = True
        # draft last: it is small; loading it first fragments dev0 and OOMs the
        # TP importer (2.8 GiB reserved-but-unallocated observed at 393K pool)
        if args.mtp:
            for idx in range(2):
                free, _ = torch.cuda.mem_get_info(idx)
                print(f"before draft.load dev{idx}: allocated {torch.cuda.memory_allocated(idx)/2**30:.2f} GiB, "
                      f"reserved {torch.cuda.memory_reserved(idx)/2**30:.2f} GiB, free {free/2**30:.2f} GiB", flush=True)
            torch.cuda.empty_cache()
            for idx in range(2):
                free, _ = torch.cuda.mem_get_info(idx)
                print(f"after empty_cache dev{idx}: free {free/2**30:.2f} GiB", flush=True)
            draft = Model.from_config(config, component="mtp")
            draft_cache = Cache(draft, max_num_tokens=args.cache_tokens, max_batch_size=1, max_history=3)
            draft.load(use_per_device=[4.0, 0.0], max_chunk_size=256, max_output_size=4,
                       max_batch_size=1, verbose=False)
            gen_kwargs = dict(draft_model=draft, draft_cache=draft_cache,
                              num_draft_tokens=args.draft_tokens, record_draft_stats=True)
        rank_pids = [os.getpid()] + [c.pid for c in (target.mp_children or []) if getattr(c, "pid", None)]

        gen = Generator(model=target, cache=cache, tokenizer=tokenizer,
                        max_batch_size=1, max_chunk_size=512,
                        recurrent_cache_size=512 * 1024 ** 2, cpu_cache_size=0,
                        **gen_kwargs)

        ids = tokenizer.encode(LONG_PROMPT, add_bos=True)
        while len(ids) < args.prompt_tokens:
            ids = ids + ids[: args.prompt_tokens - len(ids)]
        ids = ids[:args.prompt_tokens]

        # prefill TTFT: enqueue one job, wall until first token
        job = Job(input_ids=ids, max_new_tokens=args.new_tokens,
                  stop_conditions=[], sampler=DefaultSampler())
        ev_before = {p: read_evicted_ms(p) for p in rank_pids}
        gen.enqueue(job)
        t0 = time.perf_counter()
        first = None
        n_gen = 0
        for _r in gen.iterate():
            if job.time_first_token is not None and first is None:
                first = time.perf_counter() - t0
            n_gen += 1
        wall = time.perf_counter() - t0
        torch.cuda.synchronize()
        ev_after = {p: read_evicted_ms(p) for p in rank_pids}

        accepted = sum(sum(s[2] for s in (job.draft_stats or [])) for job in [job])
        rejected = sum(sum(s[1] - s[2] for s in (job.draft_stats or [])) for job in [job])
        mem = {}
        for idx in range(2):
            free, total = torch.cuda.mem_get_info(idx)
            mem[f"cuda:{idx}"] = {
                "allocated_gib": round(torch.cuda.memory_allocated(idx) / 2**30, 2),
                "reserved_gib": round(torch.cuda.memory_reserved(idx) / 2**30, 2),
                "free_gib": round(free / 2**30, 2),
                "peak_gib": round(torch.cuda.max_memory_allocated(idx) / 2**30, 2),
            }
        ev_delta = {}
        for p in rank_pids:
            d = {}
            for k in ev_after.get(p, {}):
                try:
                    d[k] = (ev_after[p][k] or 0) - (ev_before[p][k] or 0)
                except TypeError:
                    d[k] = None
            ev_delta[str(p)] = d

        rec = {
            "tag": args.tag or f"{'mtp3' if args.mtp else 'target'}-k{args.k_bits}v{args.v_bits}-pool{args.cache_tokens}",
            "pool": args.cache_tokens, "kv_bits": f"{args.k_bits}/{args.v_bits}",
            "mtp": args.mtp,
            "ttft_s": round(first, 3) if first else None,
            "prefill_tok_s": round(args.prompt_tokens / first, 1) if first else None,
            "decode_wall_s": round(wall - (first or 0), 3),
            "decode_tok_s": round((args.new_tokens) / max(wall - (first or 0), 1e-9), 2),
            "n_iter_chunks": n_gen,
            "draft_accepted": accepted, "draft_rejected": rejected,
            "acceptance_pct": round(100 * accepted / max(accepted + rejected, 1), 1),
            "memory": mem,
            "evicted_ms_delta": ev_delta,
            "error": None,
        }
        (out_dir / f"mtp-{rec['tag']}.json").write_text(json.dumps(rec, indent=1))
        print(json.dumps({k: rec[k] for k in ("tag", "ttft_s", "prefill_tok_s", "decode_tok_s",
                                              "acceptance_pct", "memory")}, indent=1))
    except Exception as e:
        import traceback
        rec = {"tag": args.tag, "error": repr(e), "traceback": traceback.format_exc()}
        (out_dir / "mtp-error.json").write_text(json.dumps(rec, indent=1))
        print("ERROR:", repr(e))
        # bounded teardown: after a rank exception the TP close barrier can hang
        # (known defect); kill children and hard-exit instead of unloading
        for c in (target.mp_children or []):
            try:
                c.terminate()
            except Exception:
                pass
        os._exit(1)
    finally:
        if loaded:
            target.unload()
        if draft is not None:
            draft.unload()


if __name__ == "__main__":
    main()