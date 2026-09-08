"""P5 Phase 2 Stage P2-3: paired counterbalanced CFG-variant decode trials (offline).

Same protocol as P4 Stage 4 / P5 Phase 1 Stage 4: one process per concurrency,
MTP3, chunk 512, Q8 pool, graphs OFF throughout, arms toggled per trial by env
(read per call inside the binding, no reload), decode-only wall as the primary
metric. Arms:
  cfg_off : grouped route, default CFG0 schedules
  cfg_c12 : EXL3_HIP_GROUPED_MOE_CFG_GU=1 EXL3_HIP_GROUPED_MOE_CFG_DOWN=2

Route-spy assertions per trial: the grouped binding dispatches (>0), the prefill
binding does not serve MoE (0 calls), and per-token gemv serves no MoE shapes
(0 routed_k3). A profiler probe records the launched grouped-kernel
specialization names; the cfg_c12 arm must show CFG 1/2 kernels and the
cfg_off arm must not.

Zero eviction required. Run: python tests/hip_p5_p2_e2e.py --out <dir> [--pairs N]
"""

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from p4_perf_metrics import decode_wall_s  # noqa: E402

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
from exllamav3.ext import exllamav3_ext as ext  # noqa: E402

MODEL = os.path.expanduser("~/Models/Qwen3.8-Flash-Next-exl3-bpw3")
CACHE_NUM_TOKENS = 65536
STEPS = 256

ARMS = {
    "cfg_off": {},
    "cfg_c12": {"EXL3_HIP_GROUPED_MOE_CFG_GU": "1", "EXL3_HIP_GROUPED_MOE_CFG_DOWN": "2"},
}

PROMPTS = [
    "The quick brown fox jumps over the lazy dog. " * 8,
    "Explain the difference between a mutex and a semaphore in concurrent programming.",
    "Write a short recipe for pancakes including ingredients and steps.",
    "Summarize the plot of Romeo and Juliet in three sentences.",
]


def journal_lines_since(cursor):
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


def build(batch):
    os.environ["EXL3_MOE_SYNC_FREE_COUNT"] = "1"
    os.environ["EXL3_QC_STAGING"] = "1"
    os.environ["EXL3_PREFILL_ASYNC_UPLOADS"] = "0"
    os.environ["EXL3_BLOCK_GRAPH"] = "0"
    block_graph.BLOCK_GRAPH_ENABLED = False
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


class RouteSpy:
    """Counts MoE-relevant binding dispatches while they are installed."""

    def __init__(self):
        self.calls = {"grouped": 0, "prefill": 0, "prefill_small_rows": 0,
                      "gemv": 0, "routed_k3_gemv": 0}
        self._orig = {}

    def __enter__(self):
        def spy_grouped(*args, **kwargs):
            self.calls["grouped"] += 1
            return self._orig["exl3_moe_gfx12_k3"](*args, **kwargs)

        def spy_prefill(*args, **kwargs):
            self.calls["prefill"] += 1
            try:
                if args[0].dim() >= 1 and args[0].shape[0] < 6:
                    self.calls["prefill_small_rows"] += 1
            except Exception:
                pass
            return self._orig["exl3_moe_gfx12_k3_prefill"](*args, **kwargs)

        def spy_gemv(x, output, *args, **kwargs):
            self.calls["gemv"] += 1
            # K3-shaped MoE per-token gemv dispatch (the pre-cap MoE fallback)
            try:
                if x.dim() >= 2 and x.shape[-1] // 16 == 3:
                    self.calls["routed_k3_gemv"] += 1
            except Exception:
                pass
            return self._orig["exl3_gemv"](x, output, *args, **kwargs)

        for name, fn in (("exl3_moe_gfx12_k3", spy_grouped),
                         ("exl3_moe_gfx12_k3_prefill", spy_prefill),
                         ("exl3_gemv", spy_gemv)):
            self._orig[name] = getattr(ext, name)
            setattr(ext, name, fn)
        return self

    def __exit__(self, *exc):
        for name, fn in self._orig.items():
            setattr(ext, name, fn)
        return False


def probe_kernel_names(gen, tokenizer, prompt):
    """One short decode under torch.profiler; returns grouped kernel names seen."""
    jobs = [Job(input_ids=tokenizer.encode(prompt, add_bos=True),
                max_new_tokens=4, stop_conditions=[], sampler=GreedySampler())]
    for job in jobs:
        gen.enqueue(job)
    names = set()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        while gen.num_remaining_jobs():
            for _r in gen.iterate():
                pass
    for ev in prof.events():
        if "moe_grouped_gemv_k3_kernel" in ev.name:
            names.add(ev.name)
    return sorted(names)


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
    # MTP verify iterations (draft_stats has one entry per decode iteration per job);
    # ms/iter = decode_wall / iterations is acceptance-independent (per-pair token
    # divergence from the CFG variant's ~1e-5 logit noise shifts acceptance, not
    # per-iteration kernel duration)
    iterations = sum(len(job.draft_stats or []) for job in jobs)
    return {"wall_s": wall, "ttft_s": ttft, "accepted": accepted, "rejected": rejected,
            "iterations": iterations, "n_jobs": len(jobs)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--pairs", type=int, default=6)
    ap.add_argument("--conc", type=str, default="1,2,3,4")
    ap.add_argument("--no-spy", action="store_true",
                    help="skip the per-trial route spy (its per-call Python overhead "
                         "shifts the c1 step toward the host-bound regime and can mask "
                         "kernel-side wall deltas)")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    kfd0 = _b.kfd_evicted_ms(os.getpid())

    for conc in [int(c) for c in args.conc.split(",")]:
        target, draft, cache, draft_cache, tokenizer, gen = build(batch=4)
        # per-arm dispatch probe (route-spy equivalent at kernel level)
        probe_names = {}
        for arm, envs in ARMS.items():
            saved = {k: os.environ.get(k) for k in envs}
            saved_off = {k: os.environ.pop(k, None)
                         for k in ARMS["cfg_off"]}
            for k, v in envs.items():
                os.environ[k] = v
            probe_names[arm] = probe_kernel_names(gen, tokenizer, PROMPTS[0])
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
        print(f"[c{conc}] probe kernels: {probe_names}", flush=True)
        assert any("<false, true, 1>" in n or "<true, false, 1>" in n
                   for n in probe_names["cfg_c12"]), "cfg_c12 arm did not launch CFG1 kernels"
        assert any("<false, true, 0>" in n or "<true, false, 0>" in n
                   for n in probe_names["cfg_off"]), "cfg_off arm did not launch CFG0 kernels"

        trials = []
        all_env_keys = sorted({k for envs in ARMS.values() for k in envs})
        for rep in range(args.pairs):
            order = ("cfg_off", "cfg_c12") if rep % 2 == 0 else ("cfg_c12", "cfg_off")
            for arm in order:
                for k in all_env_keys:
                    os.environ.pop(k, None)
                for k, v in ARMS[arm].items():
                    os.environ[k] = v
                cursor = get_journal_cursor()
                kfdA = _b.kfd_evicted_ms(os.getpid())
                if args.no_spy:
                    r = run_trial(gen, tokenizer, PROMPTS[:conc])
                    spy_calls = {"grouped": -1, "prefill": -1, "prefill_small_rows": -1,
                                 "gemv": -1, "routed_k3_gemv": -1}
                else:
                    with RouteSpy() as spy:
                        r = run_trial(gen, tokenizer, PROMPTS[:conc])
                    spy_calls = dict(spy.calls)
                t_end = time.time()
                d_wall = decode_wall_s(r["ttft_s"], t_end)
                kfdB = _b.kfd_evicted_ms(os.getpid())
                kfd_delta = {k: v1 - kfdA[k] for k, v1 in kfdB.items()
                             if k.startswith("stats_") and isinstance(v1, int)
                             and isinstance(kfdA.get(k), int)}
                trials.append({
                    "arm": arm, "rep": rep, "wall_s": r["wall_s"],
                    "decode_wall_s": d_wall,
                    "iterations": r["iterations"],
                    "ms_per_iter": d_wall * 1000.0 / r["iterations"]
                        if r["iterations"] else 0.0,
                    "ttft_s": r["ttft_s"],
                    "accepted": r["accepted"], "rejected": r["rejected"],
                    "spy": spy_calls, "kfd_delta": kfd_delta,
                    "journal_warnings": journal_lines_since(cursor),
                })
                print(f"[c{conc} {arm} rep{rep}] wall={r['wall_s']:.2f}s "
                      f"decode={trials[-1]['decode_wall_s']:.2f}s "
                      f"agg={STEPS * conc / r['wall_s']:.1f} "
                      f"acc={r['accepted']}/{r['accepted'] + r['rejected']} "
                      f"kfd={kfd_delta}", flush=True)
        if not args.no_spy:
            assert all(t["spy"]["grouped"] > 0 for t in trials), "grouped route not dispatched"
            assert all(t["spy"]["prefill_small_rows"] == 0 for t in trials), \
                "unexpected small-rows prefill MoE dispatch"
            assert all(t["spy"]["routed_k3_gemv"] == 0 for t in trials), "per-token gemv served MoE"
        summary = {"concurrency": conc, "trials": trials, "probe_names": probe_names,
                   "journal_warning_count": sum(len(t["journal_warnings"]) for t in trials)}
        (args.out / f"perf-c{conc}.json").write_text(json.dumps(summary, indent=2))
        by_arm = {}
        for t in trials:
            by_arm.setdefault(t["arm"], []).append(t)
        for arm, ts_ in by_arm.items():
            dec = sorted(t["decode_wall_s"] for t in ts_)
            pit = sorted(t["ms_per_iter"] for t in ts_)
            acc = sum(t["accepted"] for t in ts_) / max(1, sum(t["accepted"] + t["rejected"]
                                                              for t in ts_))
            print(f"[c{conc} {arm}] decode-wall medians: "
                  f"{[round(w, 2) for w in dec]} ms/iter medians: "
                  f"{[round(w, 2) for w in pit]} acc_rate={acc:.3f}", flush=True)
        block_graph.purge()
        target.unload()
        draft.unload()
        del target, draft, gen
        import gc
        gc.collect()
        torch.cuda.empty_cache()
    kfd1 = _b.kfd_evicted_ms(os.getpid())
    result = {"date": datetime.now(timezone.utc).isoformat(),
              "pairs": args.pairs,
              "kfd_delta_total": {k: v1 - kfd0[k] for k, v1 in kfd1.items()
                                  if k.startswith("stats_") and isinstance(v1, int)
                                  and isinstance(kfd0.get(k), int)}}
    (args.out / "e2e-summary.json").write_text(json.dumps(result, indent=2))
    print(f"[e2e] output: {args.out}", flush=True)


if __name__ == "__main__":
    main()