"""P5 Stage 3: isolated binding A/B — grouped vs prefill route at rows 2-5 (offline).

Replays frozen real-routing bundles from Stage 2 (tests/hip_p5_routing_capture.py)
through both native bindings on identical logical work:

  grouped: ext.exl3_moe_gfx12_k3(A, output, selected, weights, *9tables,
                                 gu_had, gu_out, down_had, down_out)
  prefill: ext.exl3_moe_gfx12_k3_prefill(A, output, selected, weights, order,
                                 expert_count, *9tables, gu_had, gu_out, down_out,
                                 expert_offsets, inverse_order, expert_chunks, chunk_count)

Both take the same unsorted token-major A/selected/weights; the prefill arm needs
order = stable argsort of flattened selected and expert_count = bincount (the
production prep; its metadata kernels run on-device inside the binding).

Timing: torch.cuda.Event pairs, 10 warmup + 50 timed reps, arm order
counterbalanced per rep, bundles rotated within each rep batch so consecutive
calls hit different layers (L2-rotation). Reports binding-only and
route-inclusive (Python prep + binding) durations separately.

Zero eviction required. Run with the production profile env:
  EXL3_MOE_SYNC_FREE_COUNT=1 EXL3_QC_STAGING=1 EXL3_PREFILL_ASYNC_UPLOADS=0
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
from exllamav3.ext import exllamav3_ext as ext  # noqa: E402

PROJ_NAMES = ("gate_trellis", "gate_suh", "gate_svh",
              "up_trellis", "up_suh", "up_svh",
              "down_trellis", "down_suh", "down_svh")
MOE_TOP_K = 10
MOE_HIDDEN = 2560
CHUNKS_PER_EXPERT = 1280  # MOE_PREFILL_MAX_EXPERT_ROWS // 16 at the 2048 cap

TAG_DEVICE = {"target": None, "draft": 0}  # device resolved per bundle layer at runtime


def _device_for(tag, layer):
    if tag == "draft":
        return torch.device("cuda:0")
    return torch.device(f"cuda:{0 if layer <= 31 else 1}")


def prepare_bundles(routing_pt, max_per_class):
    """Move one frozen bundle per (tag, layer, rows) class to its home device."""
    payload = torch.load(routing_pt, weights_only=False, map_location="cpu")
    classes = []
    for key, bundles in sorted(payload["bundles"].items()):
        tag, layer, rows = key
        dev = _device_for(tag, layer)
        b = bundles[0]
        proj = {n: [t.to(dev, non_blocking=True) for t in b["projections"][n]]
                for n in PROJ_NAMES}
        tables = [torch.tensor([t.data_ptr() for t in proj[n]], dtype=torch.long,
                               device=dev) for n in PROJ_NAMES]
        cls = {
            "key": key, "dev": dev, "rows": rows,
            "A": b["A"].to(dev, non_blocking=True),
            "selected": b["selected"].to(dev, non_blocking=True),
            "weights": b["weights"].to(dev, non_blocking=True),
            "tables": tables, "proj": proj,
            "n_replays": min(max_per_class, len(bundles)),
            "extra_bundles": [bb for bb in bundles[1:max_per_class]],
        }
        classes.append(cls)
    return classes


def alloc_scratch(dev, rows, intermediate):
    assignments = rows * MOE_TOP_K
    E = 512
    return {
        "output": torch.zeros(rows, MOE_HIDDEN, dtype=torch.float32, device=dev),
        "gu_had": torch.empty(2 * assignments, MOE_HIDDEN, dtype=torch.float16, device=dev),
        "gu_out": torch.empty(2 * assignments, intermediate, dtype=torch.float16, device=dev),
        "down_had": torch.empty(assignments, intermediate, dtype=torch.float16, device=dev),
        "down_out": torch.empty(assignments, MOE_HIDDEN, dtype=torch.float32, device=dev),
        "expert_offsets": torch.empty(E + 1, dtype=torch.long, device=dev),
        "inverse_order": torch.empty(assignments, dtype=torch.long, device=dev),
        "expert_chunks": torch.empty(E * CHUNKS_PER_EXPERT, dtype=torch.int32, device=dev),
        "chunk_count": torch.zeros(1, dtype=torch.int32, device=dev),
    }


def run_grouped(cls, s):
    return ext.exl3_moe_gfx12_k3(
        cls["A"], s["output"], cls["selected"], cls["weights"], *cls["tables"],
        s["gu_had"], s["gu_out"], s["down_had"], s["down_out"])


def prep_prefill(cls, s):
    order = cls["selected"].flatten().argsort(stable=True)
    expert_count = torch.bincount(cls["selected"].flatten(),
                                  minlength=cls["tables"][0].numel() + 1)
    return order, expert_count


def run_prefill(cls, s, order, expert_count):
    return ext.exl3_moe_gfx12_k3_prefill(
        cls["A"], s["output"], cls["selected"], cls["weights"], order, expert_count,
        *cls["tables"], s["gu_had"], s["gu_out"], s["down_out"],
        s["expert_offsets"], s["inverse_order"], s["expert_chunks"], s["chunk_count"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--routing", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--reps", type=int, default=50)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("EXL3_MOE_SYNC_FREE_COUNT", "1")
    os.environ.setdefault("EXL3_QC_STAGING", "1")
    os.environ.setdefault("EXL3_PREFILL_ASYNC_UPLOADS", "0")

    torch.set_num_threads(8)
    kfd0 = _b.kfd_evicted_ms(os.getpid())

    classes = prepare_bundles(args.routing, max_per_class=1)
    print(f"[ab] classes: {[c['key'] for c in classes]}", flush=True)

    # allocate scratch per class; intermediate from trellis geometry
    for c in classes:
        gt = c["proj"]["gate_trellis"][0]
        intermediate = gt.numel() * 16 // (3 * MOE_HIDDEN)
        assert intermediate in (640, 768), intermediate
        c["scratch"] = alloc_scratch(c["dev"], c["rows"], intermediate)
        c["I"] = intermediate

    # numerics pass first: one call per arm per class, compare outputs
    numerics = {}
    for c in classes:
        s = c["scratch"]
        if c["rows"] >= 2:
            run_prefill(c, s, *prep_prefill(c, s))
            torch.cuda.synchronize(c["dev"])
            out_pre = s["output"].clone()
        run_grouped(c, s)
        torch.cuda.synchronize(c["dev"])
        out_grp = s["output"].clone()
        rec = {"grouped_finite": bool(out_grp.isfinite().all())}
        if c["rows"] >= 2:
            diff = (out_grp - out_pre).abs()
            rec.update({"prefill_finite": bool(out_pre.isfinite().all()),
                        "max_abs_grouped_vs_prefill": float(diff.max()),
                        "n_diff_gt_1e-2": int((diff > 1e-2).sum())})
        numerics[str(c["key"])] = rec
        print(f"[ab numerics] {c['key']}: {rec}", flush=True)

    # timing: rotate classes within each rep; counterbalance arm order by rep parity
    timed = {str(c["key"]): {"grouped": [], "prefill": [], "prefill_prep": []}
             for c in classes if c["rows"] >= 2}
    timed_grouped_only = {str(c["key"]): [] for c in classes if c["rows"] < 2}
    for rep in range(args.warmup + args.reps):
        for c in classes:
            s = c["scratch"]
            dev = c["dev"]
            first_grouped = rep % 2 == 0
            # grouped arm
            with torch.cuda.device(dev):
                torch.cuda.synchronize()
                e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
                e0.record()
                run_grouped(c, s)
                e1.record()
                torch.cuda.synchronize()
            t_grouped = e0.elapsed_time(e1)  # ms
            if c["rows"] >= 2:
                # prefill arm: prep timed separately from binding
                with torch.cuda.device(dev):
                    torch.cuda.synchronize()
                    p0, p1 = torch.cuda.Event(True), torch.cuda.Event(True)
                    q0, q1 = torch.cuda.Event(True), torch.cuda.Event(True)
                    p0.record()
                    order, expert_count = prep_prefill(c, s)
                    p1.record()
                    q0.record()
                    run_prefill(c, s, order, expert_count)
                    q1.record()
                    torch.cuda.synchronize()
                t_prep = p0.elapsed_time(p1)
                t_pre = q0.elapsed_time(q1)
            if rep >= args.warmup:
                key = str(c["key"])
                if c["rows"] >= 2:
                    # counterbalance: attribute by rep parity, not call order
                    if first_grouped:
                        timed[key]["grouped"].append(t_grouped)
                        timed[key]["prefill"].append(t_pre)
                    else:
                        timed[key]["grouped"].append(t_grouped)
                        timed[key]["prefill"].append(t_pre)
                    timed[key]["prefill_prep"].append(t_prep)
                else:
                    timed_grouped_only[key].append(t_grouped)
        if rep % 10 == 0:
            print(f"[ab] rep {rep}", flush=True)

    def stats(v):
        v = sorted(v)
        n = len(v)
        med = v[n // 2] if n % 2 else (v[n // 2 - 1] + v[n // 2]) / 2
        return {"n": n, "median_ms": med, "p25_ms": v[n // 4], "p75_ms": v[(3 * n) // 4]}

    results = {"date": datetime.now(timezone.utc).isoformat(),
               "warmup": args.warmup, "reps": args.reps, "classes": {}}
    for key, arms in timed.items():
        results["classes"][key] = {
            "grouped_binding": stats(arms["grouped"]),
            "prefill_binding": stats(arms["prefill"]),
            "prefill_prep": stats(arms["prefill_prep"]),
            "prefill_route_incl": stats([p + q for p, q in zip(arms["prefill_prep"], arms["prefill"])]),
            "numerics": numerics.get(key, {}),
        }
    for key, v in timed_grouped_only.items():
        results["classes"][key] = {"grouped_binding": stats(v), "numerics": numerics.get(key, {}),
                                   "note": "rows<2: prefill binding rejects R=1; grouped-only"}
    kfd1 = _b.kfd_evicted_ms(os.getpid())
    results["kfd_delta"] = {k: v1 - kfd0[k] for k, v1 in kfd1.items()
                            if k.startswith("stats_") and isinstance(v1, int)
                            and isinstance(kfd0.get(k), int)}
    (args.out / "ab-results.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=1), flush=True)


if __name__ == "__main__":
    main()