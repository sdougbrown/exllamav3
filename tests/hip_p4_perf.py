"""P4 Stage 4 phase B: paired eager/graph decode throughput (offline, production-representative).

One process per concurrency level (VRAM-clean): loads the flash model with MTP3 (draft on
GPU0 as in serving), chunk 512, Q8 pool, and runs `trials` paired eager/graph decode trials
(counterbalanced: eager, graph, eager, graph...) with 64 new tokens per job. Reports
per-request tok/s, aggregate tok/s, TTFT, first-capture cost (graph runs), per-device KFD
eviction deltas, and kernel-journal warnings. Nothing committed; no server operations.
"""

import importlib
import importlib.util
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

_HARNESS = Path(os.environ.get(
    "EXL3_VALIDATED_PREFILL_HARNESS",
    str(Path.home() / "Serve/hosts/rocky/bench-prefill-validated.py"))).expanduser()
_spec = importlib.util.spec_from_file_location("validated", _HARNESS)
_b = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_b)

import torch  # noqa: E402
from exllamav3 import Cache, Config, Generator, GreedySampler, Job, Model, Tokenizer  # noqa: E402
from exllamav3.cache import CacheLayer_quant  # noqa: E402
from exllamav3.modules import block_graph  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from p4_perf_metrics import arms_by_flag, decode_wall_s  # noqa: E402

MODEL = os.path.expanduser("~/Models/Qwen3.8-Flash-Next-exl3-bpw3")
OUT_ROOT = Path(os.path.expanduser(
    "~/Serve/hosts/rocky/serving/qwen38-flash-exl3/benchmarks"))
CACHE_NUM_TOKENS = 65536
STEPS = 256

PROMPTS = [
    "The quick brown fox jumps over the lazy dog. " * 8,
    "Explain the difference between a mutex and a semaphore in concurrent programming.",
    "Write a short recipe for pancakes including ingredients and steps.",
    "Summarize the plot of Romeo and Juliet in three sentences.",
]


def journal_warnings_since(cursor):
    out = subprocess.run(
        ["journalctl", "-k", "--after-cursor", cursor, "--no-pager", "-q"],
        capture_output=True, text=True)
    return [l for l in out.stdout.splitlines()
            if "restore" in l.lower() or "svm" in l.lower() or "userptr" in l.lower()]


def get_journal_cursor():
    out = subprocess.run(["journalctl", "-k", "--show-cursor", "-n", "1", "-q"],
                         capture_output=True, text=True)
    for line in out.stdout.splitlines():
        if line.startswith("-- cursor:"):
            return line.split("-- cursor:", 1)[1].strip()
    return None


def build(flag_value, mtp, batch):
    # production profile: sync-free MoE histogram, quant-direct staging, no async uploads
    os.environ["EXL3_MOE_SYNC_FREE_COUNT"] = "1"
    os.environ["EXL3_QC_STAGING"] = "1"
    os.environ["EXL3_PREFILL_ASYNC_UPLOADS"] = "0"
    os.environ["EXL3_BLOCK_GRAPH"] = flag_value
    # No module reload: no runners exist before the model loads, and the initial
    # enabled state is set directly (a reload would reset _registry semantics).
    block_graph.BLOCK_GRAPH_ENABLED = flag_value == "1"
    config = Config.from_directory(MODEL)
    config.infer_params.ngram_stream_from_disk = True
    target = Model.from_config(config, component="text")
    draft = Model.from_config(config, component="mtp")
    draft_cache = Cache(draft, max_num_tokens=CACHE_NUM_TOKENS, max_batch_size=batch)
    draft.load(use_per_device=[3.0, 0.0], max_chunk_size=256, max_output_size=4,
               max_batch_size=batch, verbose=False)
    cache = Cache(target, max_num_tokens=CACHE_NUM_TOKENS,
                  layer_type=CacheLayer_quant, k_bits=8, v_bits=8,
                  max_batch_size=batch, max_history=3)
    target.load(use_per_device=[30.0, 30.0], max_chunk_size=512, max_output_size=32,
                max_batch_size=batch, verbose=False)
    tokenizer = Tokenizer.from_config(config)
    gen = Generator(model=target, cache=cache, tokenizer=tokenizer,
                    draft_model=draft, draft_cache=draft_cache,
                    num_draft_tokens=3, record_draft_stats=True,
                    max_batch_size=batch, max_chunk_size=512,
                    recurrent_cache_size=512 * 1024 ** 2, cpu_cache_size=0)
    return target, draft, cache, draft_cache, tokenizer, gen


def run_trial(gen, tokenizer, prompts):
    jobs = [Job(input_ids=tokenizer.encode(p, add_bos=True),
                max_new_tokens=STEPS, stop_conditions=[], sampler=GreedySampler())
            for p in prompts]
    t0 = time.perf_counter()
    for job in jobs:
        gen.enqueue(job)
    while gen.num_remaining_jobs():
        for _r in gen.iterate():
            pass
    wall = time.perf_counter() - t0
    ttft = max((job.time_first_token for job in jobs
                if job.time_first_token is not None), default=0.0)
    accepted = sum(sum(s[2] for s in (job.draft_stats or [])) for job in jobs)
    rejected = sum(sum(s[1] - s[2] for s in (job.draft_stats or [])) for job in jobs)
    return {"wall_s": wall, "ttft_s": ttft, "accepted": accepted, "rejected": rejected,
            "n_jobs": len(jobs)}


def main():
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = OUT_ROOT / f"p4-perf-{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    pid = os.getpid()
    print(f"[phase B] output: {out_dir}", flush=True)

    for conc in (1, 2, 3, 4):
        target, draft, cache, draft_cache, tokenizer, gen = build("0", mtp=True, batch=4)
        trials = []
        pre_kfd = _b.kfd_evicted_ms(os.getpid())
        pre_cursor = None
        for rep in range(3):  # 3 paired trials per concurrency, order counterbalanced
            order = ("0", "1") if rep % 2 == 0 else ("1", "0")
            for flag in order:
                os.environ["EXL3_BLOCK_GRAPH"] = flag
                block_graph.BLOCK_GRAPH_ENABLED = flag == "1"  # dynamic toggle, no reload
                cursor = get_journal_cursor()
                kfd0 = _b.kfd_evicted_ms(os.getpid())
                t0 = time.perf_counter()
                r = run_trial(gen, tokenizer, PROMPTS[:conc])
                t_end = time.time()
                wall = time.perf_counter() - t0
                # time_first_token is absolute wall-clock (time.time); decode wall uses
                # the same clock sampled after the trial completes
                r["decode_wall_s"] = decode_wall_s(r["ttft_s"], t_end)
                kfd1 = _b.kfd_evicted_ms(os.getpid())
                kfd_delta = {k: v1 - kfd0[k] for k, v1 in kfd1.items()
                             if k.startswith("stats_") and isinstance(v1, int)
                             and isinstance(kfd0.get(k), int)}
                warns = journal_warnings_since(cursor)
                stats = block_graph.global_stats()
                trials.append({
                    "flag": flag, "rep": rep, "wall_s": wall,
                    "decode_wall_s": r["decode_wall_s"],
                    "per_req_tok_s": STEPS * conc / wall,
                    "decode_tok_s": STEPS * conc / max(r["decode_wall_s"], 1e-9),
                    "ttft_s": r["ttft_s"],
                    "accepted": r["accepted"], "rejected": r["rejected"],
                    "kfd_delta": kfd_delta,
                    "journal_warnings": warns,
                    "graph_captures": stats["captures"],
                    "graph_replays": stats["replays"],
                })
                print(f"[c{conc} {flag} rep{rep}] wall={wall:.2f}s decode={r['decode_wall_s']:.2f}s "
                      f"agg={STEPS * conc / wall:.1f} dec={STEPS * conc / max(r['decode_wall_s'], 1e-9):.1f} tok/s "
                      f"kfd={kfd_delta} acc={r['accepted']}", flush=True)
        arms = arms_by_flag(trials)
        summary = {
            "concurrency": conc,
            "eager_wall_s": [t["wall_s"] for t in arms.get("0", [])],
            "graph_wall_s": [t["wall_s"] for t in arms.get("1", [])],
            "eager_decode_wall_s": [t["decode_wall_s"] for t in arms.get("0", [])],
            "graph_decode_wall_s": [t["decode_wall_s"] for t in arms.get("1", [])],
            "journal_warning_count": sum(len(t["journal_warnings"]) for t in trials),
            "trials": trials,
        }
        # Arm classification uses the per-trial flag recorded at run time; cumulative
        # counters and index parity are unreliable under counterbalanced ordering.
        (out_dir / f"perf-c{conc}.json").write_text(json.dumps(summary, indent=2))
        e_w = [t["wall_s"] for t in arms.get("0", [])]
        g_w = [t["wall_s"] for t in arms.get("1", [])]
        e_dec = [t["decode_wall_s"] for t in arms.get("0", [])]
        g_dec = [t["decode_wall_s"] for t in arms.get("1", [])]
        print(f"[c{conc}] eager_wall={e_w} graph_wall={g_w} "
              f"eager_decode={e_dec} graph_decode={g_dec} "
              f"warns={summary['journal_warning_count']}", flush=True)
        block_graph.purge()
        target.unload()
        draft.unload()
        del target, draft, gen
        import gc
        gc.collect()
        torch.cuda.empty_cache()
    print(f"[phase B] output: {out_dir}", flush=True)


if __name__ == "__main__":
    main()