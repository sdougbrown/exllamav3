"""P5 Stage 2: capture real decode-time grouped-MoE routing (offline, c1).

Loads the model once (target + MTP draft), runs a decode job with a wrapper on
`ext.exl3_moe_gfx12_k3`, and records:
  - routing tensors (A, selected, weights) for every grouped call after warmup,
    attributed to target layer / draft layer by installation point (wrapped
    mlp.forward sets the current context; decode is single-threaded);
  - full frozen replay bundles (projections + inputs) for a few representative
    layers, one per GPU and one draft, for the Stage 3 isolated A/B;
  - per-layer duplicate-expert distribution (replaces the uniform-routing
    assumption).

Projection trellises are per-layer constants and dominate capture size, so they
are cloned once per sampled layer (~1.4 GB each on CPU), not per call.

Env expected before model load (production profile):
  EXL3_MOE_SYNC_FREE_COUNT=1 EXL3_QC_STAGING=1 EXL3_PREFILL_ASYNC_UPLOADS=0
EXL3_BLOCK_GRAPH must be unset/0. Zero eviction required.
"""

import argparse
import importlib.util
import json
import os
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
from exllamav3.ext import exllamav3_ext as ext  # noqa: E402

MODEL = os.path.expanduser("~/Models/Qwen3.8-Flash-Next-exl3-bpw3")
CACHE_NUM_TOKENS = 65536
WARM_ITER = 32
PROMPT = ("Write a detailed technical explanation of how a modern GPU command processor "
          "submits work, including queues, doorbells and fences. ")
PROJ_NAMES = ("gate_trellis", "gate_suh", "gate_svh",
              "up_trellis", "up_suh", "up_svh",
              "down_trellis", "down_suh", "down_svh")
INPUT_NAMES = ("A", "selected", "weights")

_current = {"ctx": None}  # ('target', layer_idx) | ('draft', layer_idx) | None


def install_context_wrappers(model, tag):
    """Wrap every block mlp forward to tag the current attribution context."""
    wraps = []
    for block, _instance, idx in model.fwd_modules:
        mlp = getattr(block, "mlp", None)
        if mlp is None:
            continue  # embedding / head modules have no MoE
        original = mlp.forward

        def wrapped(*a, _orig=original, _idx=idx, _tag=tag, **kw):
            _current["ctx"] = (_tag, _idx)
            try:
                return _orig(*a, **kw)
            finally:
                _current["ctx"] = None

        mlp.forward = wrapped
        wraps.append((mlp, original))
    return wraps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--warm-iter", type=int, default=WARM_ITER)
    ap.add_argument("--full-per-class", type=int, default=6,
                    help="full replay captures per (attribution, rows) class")
    ap.add_argument("--full-layers", type=str, default="target:5,target:40,draft:1",
                    help="layerIdx list for full projection clones, comma separated")
    ap.add_argument("--mtp-sweep", type=str, default="3,1",
                    help="draft-token counts to sweep (rows = 1 + mtp)")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    # production profile, set after harness import (it pins its own env)
    os.environ["EXL3_MOE_SYNC_FREE_COUNT"] = "1"
    os.environ["EXL3_QC_STAGING"] = "1"
    os.environ["EXL3_PREFILL_ASYNC_UPLOADS"] = "0"
    os.environ.pop("EXL3_BLOCK_GRAPH", None)

    full_layers = {}
    for part in args.full_layers.split(","):
        tag, idx = part.split(":")
        full_layers.setdefault(tag, set()).add(int(idx))

    kfd0 = _b.kfd_evicted_ms(os.getpid())
    config = Config.from_directory(MODEL)
    config.infer_params.ngram_stream_from_disk = True
    target = Model.from_config(config, component="text")
    draft = Model.from_config(config, component="mtp")

    gen_holder = {}

    records = []          # per-call routing records (small)
    full_bundles = {}     # (tag, layer_idx) -> dict of cpu tensors
    full_counts = {}      # (tag, layer_idx, rows) -> captures so far
    proj_cache = {}       # (tag, layer_idx) -> dict of projection clones

    original_binding = ext.exl3_moe_gfx12_k3

    def capture_binding(A, output, selected, weights, *proj_and_scratch):
        ctx = _current["ctx"] or ("unknown", -1)
        tag, layer_idx = ctx
        rows = A.shape[0]
        rec = {"tag": tag, "layer": layer_idx, "rows": rows,
               "selected": selected.detach().cpu().clone(),
               "weights": weights.detach().cpu().clone()}
        records.append(rec)
        key = (tag, layer_idx, rows)
        want_full = layer_idx in full_layers.get(tag, set())
        if want_full and full_counts.get(key, 0) < args.full_per_class:
            full_counts[key] = full_counts.get(key, 0) + 1
            if (tag, layer_idx) not in proj_cache:
                mlp = _mlp_for(tag, layer_idx)
                lines = {"gate": mlp.gates, "up": mlp.ups, "down": mlp.downs}
                proj = {}
                for pname, linears in lines.items():
                    for attr in ("trellis", "suh", "svh"):
                        proj[f"{pname}_{attr}"] = [getattr(l.inner, attr).detach().cpu().clone()
                                                   for l in linears]
                proj_cache[(tag, layer_idx)] = proj
            bundle = {n: t.detach().cpu().clone() for n, t in zip(INPUT_NAMES, (A, selected, weights))}
            bundle["projections"] = proj_cache[(tag, layer_idx)]
            full_bundles.setdefault(key, []).append(bundle)
        return original_binding(A, output, selected, weights, *proj_and_scratch)

    _mlp_map = {}

    def _mlp_for(tag, layer_idx):
        return _mlp_map[(tag, layer_idx)]

    ext.exl3_moe_gfx12_k3 = capture_binding
    wraps = install_context_wrappers(target, "target") + install_context_wrappers(draft, "draft")
    for tag, model in (("target", target), ("draft", draft)):
        for block, _instance, idx in model.fwd_modules:
            if hasattr(block, "mlp"):
                _mlp_map[(tag, idx)] = block.mlp

    kfd_base = None
    try:
        for mtp in (int(x) for x in args.mtp_sweep.split(",")):
            draft_cache = Cache(draft, max_num_tokens=CACHE_NUM_TOKENS, max_batch_size=1)
            draft.load(use_per_device=[3.0, 0.0], max_chunk_size=256, max_output_size=4,
                       max_batch_size=1, verbose=False)
            cache = Cache(target, max_num_tokens=CACHE_NUM_TOKENS,
                          layer_type=CacheLayer_quant, k_bits=8, v_bits=8,
                          max_batch_size=1, max_history=3)
            target.load(use_per_device=[30.0, 30.0], max_chunk_size=512, max_output_size=32,
                        max_batch_size=1, verbose=False)
            tokenizer = Tokenizer.from_config(config)
            gen = Generator(model=target, cache=cache, tokenizer=tokenizer,
                            draft_model=draft, draft_cache=draft_cache,
                            num_draft_tokens=mtp, record_draft_stats=True,
                            max_batch_size=1, max_chunk_size=512,
                            recurrent_cache_size=512 * 1024 ** 2, cpu_cache_size=0)
            gen_holder["gen"] = gen
            max_tokens = (args.warm_iter + 16) * (mtp + 1) + 16
            job = Job(input_ids=tokenizer.encode(PROMPT, add_bos=True),
                      max_new_tokens=max_tokens, stop_conditions=[], sampler=GreedySampler())
            gen.enqueue(job)
            seq = job.sequences[0]
            prompt_pos = seq.kv_position
            di = 0
            kfd_base = _b.kfd_evicted_ms(os.getpid())
            n_before = len(records)
            while gen.num_remaining_jobs():
                for _r in gen.iterate():
                    pass
                if seq.kv_position >= prompt_pos:
                    di += 1
                if di == args.warm_iter:
                    n_before = len(records)  # warmup calls end here
            torch.cuda.synchronize()
            print(f"[capture] mtp={mtp}: {len(records) - n_before} calls after warmup "
                  f"(rows=1+{mtp} expected)", flush=True)
            target.unload()
            draft.unload()
            del cache, draft_cache, gen
            torch.cuda.empty_cache()
    finally:
        ext.exl3_moe_gfx12_k3 = original_binding
        for mlp, original in wraps:
            mlp.forward = original

    kfd1 = _b.kfd_evicted_ms(os.getpid())
    kfd_delta = {k: v1 - kfd_base[k] for k, v1 in kfd1.items()
                 if k.startswith("stats_") and isinstance(v1, int)
                 and isinstance(kfd_base.get(k), int)}

    dist = {}
    for rec in records:
        sel = rec["selected"]
        dup = sel.shape[0] * sel.shape[1] - len(set(sel.flatten().tolist()))
        key = f"{rec['tag']}/L{rec['layer']}/rows{rec['rows']}"
        d = dist.setdefault(key, {"calls": 0, "assignments": 0, "duplicate_assignments": 0})
        d["calls"] += 1
        d["assignments"] += sel.numel()
        d["duplicate_assignments"] += dup

    payload = {"records": records, "dist": dist, "bundles": full_bundles,
               "arg_names": ["A", "selected", "weights"], "proj_names": PROJ_NAMES}
    torch.save(payload, args.out / "routing.pt")
    manifest = {
        "date": datetime.now(timezone.utc).isoformat(),
        "model": MODEL, "torch": torch.__version__, "hip": torch.version.hip,
        "mtp_sweep": args.mtp_sweep, "warm_iter": args.warm_iter,
        "full_per_class": args.full_per_class,
        "full_bundle_keys": {f"{t}/L{l}/rows{r}": n for (t, l, r), n in sorted(full_counts.items())},
        "dist": dist,
        "kfd_delta": kfd_delta,
        "binding": "exl3_moe_gfx12_k3",
        "arg_names": ["A", "output", "selected", "weights", *PROJ_NAMES,
                      "gu_had", "gu_out", "down_had", "down_out"],
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[capture] wrote {args.out/'routing.pt'} ({len(records)} calls), manifest")
    print(f"[capture] full bundles: {manifest['full_bundle_keys']}")
    print(f"[capture] kfd_delta: {kfd_delta}")


if __name__ == "__main__":
    main()