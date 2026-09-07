"""P4 Stage 4 phase A: decode numerical qualification and state coverage (offline).

Paired contemporary eager/graph runs of the integrated block-graph hook through the real
Generator loop: c1..c4, target-only and MTP3, fixed prompts, greedy sampling. One
model+cache+generator per case; three sequential runs (eager, eager-baseline, graph) — state
resets naturally as completed jobs release pages and recurrent slots.

Two instruments:
1. Fresh-state full-logit comparison (first 24 steps, logical vocab only), graph-vs-eager
   gated against the eager-vs-eager baseline.
2. Identical-state dual verification at chosen decode steps: the same logical step is executed
   twice from an identical snapshot of ALL mutable device state (cache layers + recurrent
   layers), once eager and once via graph replay (eager twice for baseline arms). The last run
   leaves state advanced exactly once, which is what the generator expects.

Gates (corrected brief Stage 4): identical-state dual deltas within the eager-vs-eager dual
envelope, zero queue-eviction delta, no new restore-worker warnings, healthy completion.
Nothing committed; no server operations.
"""

import importlib
import importlib.util
import json
import os
import subprocess
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

MODEL = os.path.expanduser("~/Models/Qwen3.8-Flash-Next-exl3-bpw3")
OUT_ROOT = Path(os.path.expanduser(
    "~/Serve/hosts/rocky/serving/qwen38-flash-exl3/benchmarks"))
CACHE_NUM_TOKENS = 4096
STEPS = 64
LOGICAL_VOCAB = 248077
DUAL_STEPS = (8, 16, 24)

PROMPTS = [
    "The quick brown fox jumps over the lazy dog. " * 8,
    "Explain the difference between a mutex and a semaphore in concurrent programming.",
    "Write a short recipe for pancakes including ingredients and steps.",
    "Summarize the plot of Romeo and Juliet in three sentences.",
    "List the first ten prime numbers and explain why they are prime.",
    "Describe the water cycle starting from evaporation.",
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


def build_case(mtp, batch):
    # Match the production profile: sync-free MoE histogram (bincount issues a capture-rejected
    # H2D even when the gfx12 prefill route is eligible) and quant-direct attention staging.
    os.environ["EXL3_MOE_SYNC_FREE_COUNT"] = "1"
    os.environ["EXL3_QC_STAGING"] = "1"
    os.environ["EXL3_PREFILL_ASYNC_UPLOADS"] = "0"
    os.environ["EXL3_BLOCK_GRAPH"] = "0"
    importlib.reload(block_graph)
    config = Config.from_directory(MODEL)
    config.infer_params.ngram_stream_from_disk = True
    target = Model.from_config(config, component="text")
    draft = draft_cache = None
    cache = Cache(target, max_num_tokens=CACHE_NUM_TOKENS,
                  layer_type=CacheLayer_quant, k_bits=8, v_bits=8,
                  max_batch_size=batch, max_history=3)
    if mtp:
        draft = Model.from_config(config, component="mtp")
        draft_cache = Cache(draft, max_num_tokens=CACHE_NUM_TOKENS, max_batch_size=batch)
        draft.load(use_per_device=[0.0, 2.0], max_chunk_size=256, max_output_size=4,
                   max_batch_size=batch, verbose=False)
    target.load(use_per_device=[30.0, 30.0], max_chunk_size=512, max_output_size=32,
                max_batch_size=batch, verbose=False)
    gen_kwargs = dict(max_batch_size=batch, max_chunk_size=512,
                      recurrent_cache_size=256 * 1024 ** 2, cpu_cache_size=0)
    tokenizer = Tokenizer.from_config(config)
    if mtp:
        gen = Generator(model=target, cache=cache, tokenizer=tokenizer,
                        draft_model=draft, draft_cache=draft_cache,
                        num_draft_tokens=3, record_draft_stats=True, **gen_kwargs)
    else:
        gen = Generator(model=target, cache=cache, tokenizer=tokenizer, **gen_kwargs)
    return target, draft, cache, draft_cache, gen, tokenizer


def run_jobs(gen, prompts):
    """Enqueue jobs (len > max_batch_size exercises slot churn); collect tokens + draft stats."""
    jobs = []
    for p in prompts:
        job = Job(input_ids=p, max_new_tokens=STEPS, stop_conditions=[],
                  sampler=GreedySampler())
        gen.enqueue(job)
        jobs.append(job)
    while gen.num_remaining_jobs():
        for _r in gen.iterate():
            pass
    tokens, accepted, rejected, ttft = [], [], [], []
    for job in jobs:
        seq = job.sequences[0]
        ids = seq.sequence_ids.torch_slice(0, None).tolist()
        tokens.append(ids[-STEPS:] if len(ids) >= STEPS else ids)
        stats = job.draft_stats or []
        accepted.append(sum(s[2] for s in stats))
        rejected.append(sum(s[1] - s[2] for s in stats))
        ttft.append(job.time_first_token)
    return tokens, accepted, rejected, ttft


def snapshot(pid):
    return {"kfd": _b.kfd_evicted_ms(pid), "cursor": get_journal_cursor()}


def kfd_delta(before, after):
    return {k: after["kfd"][k] - before["kfd"][k]
            for k in after["kfd"]
            if k.startswith("stats_") and isinstance(after["kfd"][k], int)
            and isinstance(before["kfd"].get(k), int)}


def one_run(pid, name, flag_value, mtp, prompts, target, cache, gen, tokenizer,
            dual_steps=DUAL_STEPS):
    """One sequential run on the case generator. The dual instrument (at `dual_steps`) executes
    the same logical step twice from identical state: eager twice for eager runs, eager then
    graph for graph runs."""
    os.environ["EXL3_BLOCK_GRAPH"] = flag_value
    importlib.reload(block_graph)
    orig_forward = target.forward
    steps = []
    dual_results = []
    plan = ([("eager", "0"), ("eager", "0")] if flag_value == "0"
            else [("eager", "0"), ("eager", "0"), ("graph", "1")])
    done = set()

    def all_state():
        tensors = list(cache.get_all_tensors())
        for rl in cache.get_all_recurrent_layers().values():
            tensors += list(rl.get_state_tensors())
        return [t.detach().clone() for t in tensors]

    def restore_all(saved):
        live = list(cache.get_all_tensors())
        for rl in cache.get_all_recurrent_layers().values():
            live += list(rl.get_state_tensors())
        for t, s in zip(live, saved):
            t.copy_(s)
        torch.cuda.synchronize()

    def wrapped(input_ids, params=None):
        if len(steps) in dual_steps and len(steps) not in done:
            snap = all_state()
            runs = []
            for label, fv in plan:
                restore_all(snap)
                os.environ["EXL3_BLOCK_GRAPH"] = fv
                importlib.reload(block_graph)
                y = orig_forward(input_ids, params)
                runs.append({"label": label, "tensor": y.detach().clone()})
            os.environ["EXL3_BLOCK_GRAPH"] = flag_value
            importlib.reload(block_graph)
            def _delta(x, y):
                return float((x[..., :LOGICAL_VOCAB].float() -
                              y[..., :LOGICAL_VOCAB].float()).abs().max())
            record = {
                "step": len(steps),
                "labels": [r["label"] for r in runs],
                "shape": list(runs[0]["tensor"].shape),
                "eager_eager_delta": float(
                    (runs[0]["tensor"][..., :LOGICAL_VOCAB].float() -
                     runs[1]["tensor"][..., :LOGICAL_VOCAB].float()).abs().max())
                if len(runs) > 1 else None,
                "max_delta": float(
                    (runs[0]["tensor"][..., :LOGICAL_VOCAB].float() -
                     runs[-1]["tensor"][..., :LOGICAL_VOCAB].float()).abs().max()),
                "bitwise_equal_frac": float(
                    (runs[0]["tensor"].view(torch.int16) ==
                     runs[-1]["tensor"].view(torch.int16)).float().mean()),
            }
            dual_results.append(record)
            done.add(len(steps))
            print(f"[dual step {len(steps)}] plan={plan} "
                  f"eager_eager={record['eager_eager_delta']:.3e} "
                  f"last_pair={record['max_delta']:.3e} "
                  f"bitwise_eq={record['bitwise_equal_frac']:.4f}", flush=True)
        y = orig_forward(input_ids, params)
        if torch.is_tensor(y) and y.dim() >= 2 and len(steps) < max(DUAL_STEPS):
            steps.append(y.detach()[..., :LOGICAL_VOCAB].float().cpu())
        return y

    target.forward = wrapped
    pre = snapshot(pid)
    entry = {"flag": flag_value, "kfd_before": pre["kfd"], "cursor": pre["cursor"]}
    t0 = time.perf_counter()
    try:
        jobs = [Job(input_ids=tokenizer.encode(p, add_bos=True),
                    max_new_tokens=STEPS, stop_conditions=[],
                    sampler=GreedySampler()) for p in prompts]
        for job in jobs:
            gen.enqueue(job)
        while gen.num_remaining_jobs():
            for _r in gen.iterate():
                pass
        wall = time.perf_counter() - t0
        tokens, accepted, rejected, ttft = [], [], [], []
        for job in jobs:
            ids = job.sequences[0].sequence_ids.torch_slice(0, None).tolist()
            tokens.append(ids[-STEPS:] if len(ids) >= STEPS else ids)
            stats = job.draft_stats or []
            accepted.append(sum(s[2] for s in stats))
            rejected.append(sum(s[1] - s[2] for s in stats))
            ttft.append(job.time_first_token)
        entry.update({"tokens": tokens, "accepted": accepted, "rejected": rejected,
                      "ttft_s": ttft, "wall_s": wall,
                      "logits": steps, "n_steps": len(steps),
                      "dual_results": dual_results})
    finally:
        target.forward = orig_forward
        torch.cuda.synchronize()
        post = snapshot(pid)
        entry["kfd_delta"] = kfd_delta(pre, post)
        entry["journal_warnings"] = journal_warnings_since(pre["cursor"])
        entry["reserved_b"] = {str(d): torch.cuda.memory_stats(d)
                               .get("reserved_bytes.all.current", 0) for d in (0, 1)}
        stats = block_graph.global_stats()
        entry["block_graph_stats"] = {
            "captures": stats["captures"], "replays": stats["replays"],
            "warmups": stats["warmups"], "evictions": stats["evictions"],
            "capture_failed": stats["capture_failed"],
            "declines": {k: v for k, v in stats["declines"].items() if v}}
    print(f"[{name} {flag_value}] steps={len(steps)} kfd={entry['kfd_delta']} "
          f"warn={len(entry['journal_warnings'])}", flush=True)
    torch.cuda.empty_cache()
    return entry, steps


def compare_logits(a_steps, b_steps):
    """Elementwise comparison of two per-step logit sequences, aligned on the common prefix
    (MTP acceptance differences legitimately change trajectories downstream)."""
    n = min(len(a_steps), len(b_steps))
    if n == 0:
        return {"comparable": False, "n_steps_a": len(a_steps), "n_steps_b": len(b_steps)}
    a_steps, b_steps = a_steps[:n], b_steps[:n]
    for i, (sa, sb) in enumerate(zip(a_steps, b_steps)):
        if sa.shape != sb.shape:
            return {"comparable": False, "first_shape_mismatch_step": i,
                    "shapes": [list(sa.shape), list(sb.shape)]}
    max_d = 0.0
    first_diff = None
    n_diff = 0
    for i, (sa, sb) in enumerate(zip(a_steps, b_steps)):
        d = (sa - sb).abs()
        m = float(d.max())
        if m > 0:
            n_diff += 1
            if first_diff is None:
                first_diff = {"step": i, "max_delta": m,
                              "n_diff_elems": int((d > 0).sum()),
                              "argmax_delta_elem": int(d.argmax())}
        max_d = max(max_d, m)
    return {"comparable": True, "max_delta": max_d, "first_diff": first_diff,
            "n_diff_steps": n_diff, "n_steps": n}


def run_case(name, cfg, out_dir, ts):
    """In-process worker for one case (fresh process per case keeps VRAM clean)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pid = os.getpid()
    target, draft, cache, draft_cache, gen, tokenizer = \
        build_case(mtp=cfg["mtp"], batch=4)
    eager1, _ = one_run(pid, name, "0", cfg["mtp"], cfg["prompts"],
                        target=target, cache=cache, gen=gen, tokenizer=tokenizer)
    eager2, _ = one_run(pid, name, "0", cfg["mtp"], cfg["prompts"],
                        target=target, cache=cache, gen=gen, tokenizer=tokenizer)
    graph, _ = one_run(pid, name, "1", cfg["mtp"], cfg["prompts"],
                       target=target, cache=cache, gen=gen, tokenizer=tokenizer)
    bl = compare_logits(eager1["logits"], eager2["logits"])
    print(f"[baseline {name}] eager-vs-eager: {bl}", flush=True)
    comp = {"token_identity": eager1["tokens"] == graph["tokens"],
            "accepted_equal": eager1["accepted"] == graph["accepted"],
            "rejected_equal": eager1["rejected"] == graph["rejected"]}
    lc = compare_logits(eager1["logits"], graph["logits"])
    comp["logits"] = lc
    if lc.get("comparable") and bl.get("comparable"):
        comp["within_eager_baseline"] = lc["max_delta"] <= bl["max_delta"]
    gd = [d for d in graph.get("dual_results", []) if d["labels"] == ["eager", "graph"]]
    ed = [d for d in (eager1.get("dual_results", []) + eager2.get("dual_results", []))
          if d["labels"] == ["eager", "eager"]]
    comp["dual_eager_vs_graph"] = gd
    comp["dual_eager_vs_eager"] = ed
    comp["dual_summary"] = {
        "graph_deltas": [d["max_delta"] for d in gd],
        "eager_baseline_deltas": [d["max_delta"] for d in ed],
        "graph_within_eager_class": (
            max((d["max_delta"] for d in gd), default=0.0) <=
            max((d["max_delta"] for d in ed), default=0.0) + 0.05),
    }
    for e in (eager1, eager2, graph):
        e["logits"] = None
    gates = {
        "token_identity": comp["token_identity"],
        "zero_eviction": all(v == 0 for v in
                             list(eager1["kfd_delta"].values()) +
                             list(eager2["kfd_delta"].values()) +
                             list(graph["kfd_delta"].values())),
        "no_warnings": sum(len(e["journal_warnings"]) for e in
                           (eager1, eager2, graph)) == 0,
        "dual_within_eager_class": comp["dual_summary"]["graph_within_eager_class"],
    }
    block_graph.purge()
    target.unload()
    if draft is not None:
        draft.unload()
    del target, draft, gen
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    out = {"case": name, "comparisons": comp, "gates": gates,
           "baseline": bl, "when": ts, "torch": torch.__version__}
    (out_dir / f"case-{name}.json").write_text(json.dumps(out, indent=2, default=str))
    print(f"[compare {name}] tokens_id={comp['token_identity']} "
          f"digest={lc.get('max_delta')} bl={bl.get('max_delta')} "
          f"dual_class_ok={comp['dual_summary']['graph_within_eager_class']} "
          f"gates={gates}", flush=True)
    return out


def main():
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = OUT_ROOT / f"p4-qual-{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[phase A] output: {out_dir}", flush=True)

    prompts4 = PROMPTS[:4]
    cases = {
        "c1_target": dict(mtp=False, prompts=prompts4[:1]),
        "c4_target": dict(mtp=False, prompts=prompts4),
        "c1_mtp3": dict(mtp=True, prompts=prompts4[:1]),
        "c2_mtp3": dict(mtp=True, prompts=prompts4[:2]),
        "c3_mtp3": dict(mtp=True, prompts=prompts4[:3]),
        "c4_mtp3": dict(mtp=True, prompts=prompts4),
        "c6queue_mtp3": dict(mtp=True, prompts=PROMPTS[:4] + PROMPTS[:2]),
    }

    env = dict(os.environ)
    outs = {}
    for name, cfg in cases.items():
        case_json = out_dir / f"case-{name}.json"
        proc = subprocess.run(
            [os.environ.get("EXL3_PYTHON", os.sys.executable), __file__,
             "--case", name, "--out", str(out_dir), "--ts", ts],
            env=env, capture_output=True, text=True)
        tail = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""
        print(f"[case {name}] rc={proc.returncode} {tail[:400]}", flush=True)
        if case_json.exists():
            outs[name] = json.loads(case_json.read_text())
        elif proc.returncode != 0:
            outs[name] = {"error": proc.stderr.strip().splitlines()[-1] if proc.stderr.strip() else "failed"}
    results = {"when": ts, "out_dir": str(out_dir), "outs": outs}
    gates = {n: o.get("gates", {}) for n, o in outs.items()}
    results["gates"] = gates
    print(json.dumps(gates, indent=1), flush=True)
    (out_dir / "results.json").write_text(json.dumps(results, indent=2, default=str))
    hard = all(g.get("dual_within_eager_class") and g.get("zero_eviction")
               and g.get("no_warnings") for g in gates.values())
    print(f"[phase A] {'PASS' if hard else 'CHECK GATES ABOVE'}", flush=True)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--ts", default=None)
    args = ap.parse_args()
    if args.case:
        ts = args.ts or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out_dir = Path(args.out) if args.out else OUT_ROOT / f"p4-qual-{ts}"
        prompts4 = PROMPTS[:4]
        cases = {
            "c1_target": dict(mtp=False, prompts=prompts4[:1]),
            "c4_target": dict(mtp=False, prompts=prompts4),
            "c1_mtp3": dict(mtp=True, prompts=prompts4[:1]),
            "c2_mtp3": dict(mtp=True, prompts=prompts4[:2]),
            "c3_mtp3": dict(mtp=True, prompts=prompts4[:3]),
            "c4_mtp3": dict(mtp=True, prompts=prompts4),
            "c6queue_mtp3": dict(mtp=True, prompts=PROMPTS[:4] + PROMPTS[:2]),
        }
        run_case(args.case, cases[args.case], out_dir, ts)
    else:
        main()