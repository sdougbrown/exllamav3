"""P4 feasibility probes: single-block and contiguous-run HIP graph capture (read-only wrt serving).

Stage 2 of the corrected P4 brief. Offline only: loads EXL3_FLASH_TEST_MODEL with a small cache,
no server, no generator, no draft model. Captures one non-QSA GDN+MoE block (and a bounded
contiguous two-block run) under torch.cuda.CUDAGraph from isolated state, and verifies
replay-vs-eager parity from identical restored state.

Protocol (per brief Stage 2):
- Warmups run on isolated inputs; conv/recurrent state is snapshotted before warmups and
  restored (copy_ + sync) before every measured run. Each logical forward commits exactly one
  state transition; warmups never touch a live token's state.
- Capture records on a side stream without executing; replay is the committed step.
- Capture-failure cleanup is exercised with an injected host sync, then the allocator and eager
  path are verified intact after the failed capture.
- No server is touched; nothing is committed. Probe artifacts land under
  ~/Serve/hosts/rocky/serving/qwen38-flash-exl3/benchmarks/p4-block-capture-<ts>/.
"""
import argparse
import importlib.util
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

# Reuse the validated harness's launcher-matching environment (sets HIP/torch env before torch import).
_HARNESS = Path(os.environ.get(
    "EXL3_VALIDATED_PREFILL_HARNESS",
    str(Path.home() / "Serve/hosts/rocky/bench-prefill-validated.py"))).expanduser()
_spec = importlib.util.spec_from_file_location("validated", _HARNESS)
_b = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_b)

MODEL_DIR = os.environ.get("EXL3_FLASH_TEST_MODEL",
                           os.path.expanduser("~/Models/Qwen3.8-Flash-Next-exl3-bpw3"))
OUT_ROOT = Path(os.path.expanduser(
    "~/Serve/hosts/rocky/serving/qwen38-flash-exl3/benchmarks"))


def kfd_delta(b, pid):
    try:
        return b.kfd_evicted_ms(pid)
    except Exception as e:
        return {"error": str(e)}


def load(model_dir, chunk=512):
    from exllamav3 import Config, Model, Cache, Tokenizer
    from exllamav3.cache import CacheLayer_quant
    config = Config.from_directory(model_dir)
    tokenizer = Tokenizer.from_config(config)
    model = Model.from_config(config)
    cache = Cache(model,
                  max_num_tokens=40960,
                  layer_type=CacheLayer_quant,
                  k_bits=8, v_bits=8,
                  max_batch_size=1,
                  max_history=3)
    model.load(use_per_device=[30.0, 30.0],
               max_batch_size=1,
               max_chunk_size=chunk,
               max_output_size=32,
               verbose=True)
    return config, model, cache, tokenizer


def module_route_flags(module):
    """Read-only route-relevant attributes for the P0 dump."""
    from exllamav3.modules import TransformerBlock, BlockSparseMLP, GatedDeltaNet, Attention
    out = {"cls": type(module).__name__}
    if isinstance(module, TransformerBlock):
        attn = module.attn
        out["attn"] = type(attn).__name__
        out["layer_idx"] = module.layer_idx
        if isinstance(attn, GatedDeltaNet):
            out["kda"] = bool(getattr(attn, "kda", False))
            out["bc_split"] = bool(getattr(attn, "bc_split", False))
        elif isinstance(attn, Attention):
            out["qsa_indexer"] = attn.qsa_indexer is not None
            if attn.qsa_indexer is not None:
                out["sparse_threshold"] = attn.qsa_indexer.sparse_threshold()
        mlp = module.mlp
        if isinstance(mlp, BlockSparseMLP):
            out.update({
                "num_experts": mlp.num_experts,
                "top_k": mlp.num_experts_per_tok,
                "f_threshold": mlp.f_threshold,
                "support_hip_grouped": bool(getattr(mlp, "support_hip_grouped", False)),
                "support_hip_prefill": bool(getattr(mlp, "support_hip_prefill", False)),
                "support_quant_paths": bool(getattr(mlp, "support_quant_paths", False)),
                "grouped_buffers": mlp.hip_grouped_buffers is not None,
                "is_quantized": bool(mlp.is_quantized),
                "gated": bool(mlp.gated),
                "shared_experts": mlp.shared_experts is not None,
            })
    return out


def dump_placement(model, out_dir):
    from exllamav3.modules import TransformerBlock
    fwd = []
    for module, instance, idx in model.fwd_modules:
        dev = getattr(module, "device", None)
        entry = {"idx": idx, "device": str(getattr(dev, "index", dev)),
                 **module_route_flags(module)}
        if isinstance(module, TransformerBlock):
            entry["key"] = module.key
        fwd.append(entry)
    placement = {
        "model_dir": MODEL_DIR,
        "fwd_modules": fwd,
        "logit_layer_idx": model.logit_layer_idx,
        "last_kv_module_idx": model.last_kv_module_idx,
        "active_devices": [str(d) for d in model.active_devices],
    }
    (out_dir / "placement.json").write_text(json.dumps(placement, indent=2))
    blocks = [e for e in fwd if e["cls"] == "TransformerBlock"]
    n_gdn = sum(1 for e in blocks if e.get("attn") == "GatedDeltaNet")
    n_qsa = sum(1 for e in blocks if e.get("qsa_indexer"))
    n_moe_ok = sum(1 for e in blocks if e.get("support_hip_grouped"))
    print(f"[P0] modules={len(fwd)} blocks={len(blocks)} gdn={n_gdn} qsa={n_qsa} "
          f"hip_grouped_ok={n_moe_ok}", flush=True)
    for e in blocks[:6]:
        print(f"[P0]   idx={e['idx']} dev={e['device']} attn={e.get('attn')} "
              f"qsa={e.get('qsa_indexer')} grouped={e.get('support_hip_grouped')}", flush=True)
    return placement


def find_gdn_runs(model, count):
    """First run of `count` consecutive non-QSA GDN blocks in fwd order."""
    from exllamav3.modules import TransformerBlock, GatedDeltaNet
    entries = [e for e in model.fwd_modules
               if isinstance(e[0], TransformerBlock) and isinstance(e[0].attn, GatedDeltaNet)]
    runs = []
    i = 0
    while i + count <= len(entries):
        idxs = [e[2] for e in entries[i:i + count]]
        if idxs == list(range(idxs[0], idxs[0] + count)):
            runs.append(entries[i:i + count])
        i += 1
    return runs


def layer_instances_for(cache, blocks):
    """(layer_idx, 0) instance tuple per GDN block, as keyed in cache.recurrent_layers."""
    insts = []
    for block, inst, idx in blocks:
        li = (block.attn.layer_idx, 0)
        assert li in cache.recurrent_layers, f"{li} not in recurrent layers"
        insts.append(li)
    return insts


class StateSnapshot:
    """Snapshot/restore of recurrent state tensors for specific GDN layer instances."""

    def __init__(self, cache, layer_instances):
        self.tensors = {}
        self.saved = {}
        for li in layer_instances:
            conv, rec = cache.get_recurrent_layer(li).get_state_tensors()
            self.tensors[li] = (conv, rec)
            self.saved[li] = (conv.detach().clone(), rec.detach().clone())

    def restore(self):
        import torch
        for li, (conv, rec) in self.tensors.items():
            sconv, srec = self.saved[li]
            conv.copy_(sconv)
            rec.copy_(srec)
        torch.cuda.synchronize()

    def current(self):
        return {li: tuple(t.detach().clone() for t in ts) for li, ts in self.tensors.items()}


def tensor_diff(a, b):
    import torch
    if a.shape != b.shape:
        return {"error": f"shape {tuple(a.shape)} vs {tuple(b.shape)}"}
    d = (a.float() - b.float()).abs()
    if a.dtype == torch.float32:
        eq = float((a.view(torch.int32) == b.view(torch.int32)).float().mean())
    elif a.dtype == torch.float16:
        eq = float((a.view(torch.int16) == b.view(torch.int16)).float().mean())
    elif a.dtype == torch.bfloat16:
        eq = float((a.view(torch.int16) == b.view(torch.int16)).float().mean())
    else:
        eq = float((a == b).float().mean())
    return {"max_abs": float(d.max()), "mean_abs": float(d.mean()), "bitwise_equal_frac": eq}


def mem_stats(devices=(0, 1)):
    import torch
    out = {}
    for d in devices:
        s = torch.cuda.memory_stats(d)
        out[str(d)] = {
            "allocated_b": s.get("allocated_bytes.all.current", 0),
            "reserved_b": s.get("reserved_bytes.all.current", 0),
        }
    return out


def capture_and_verify(name, blocks, layer_instances, cache, x_shape, out_dir, trials=3):
    """Capture `blocks` as one graph with static stream-stack I/O; verify vs eager from restored state."""
    import torch
    from exllamav3.util.tensor import get_for_device
    dev = blocks[0][0].device
    report = {"name": name, "device": str(dev), "block_idxs": [b[2] for b in blocks],
              "layer_instances": [list(li) for li in layer_instances]}

    slot = 0
    state = cache.get_new_state()  # slot 0, cleared
    params = {
        "layer_instance": 0,
        "recurrent_states": [state],
        "recurrent_slots": torch.tensor([slot], dtype=torch.int32),
    }
    try:
        return _capture_and_verify_inner(name, blocks, layer_instances, cache, x_shape,
                                         out_dir, trials, params, state)
    finally:
        state.free()


def _capture_and_verify_inner(name, blocks, layer_instances, cache, x_shape, out_dir,
                              trials, params, state):
    import torch
    from exllamav3.util.tensor import get_for_device
    dev = blocks[0][0].device
    report = {"name": name, "device": str(dev), "block_idxs": [b[2] for b in blocks],
              "layer_instances": [list(li) for li in layer_instances]}
    # Pre-create the static per-device copy of recurrent_slots outside capture
    # (get_for_device caches persistent copies for _static_dev_cache tensors).
    slots_dev = get_for_device(params, "recurrent_slots", dev)
    assert slots_dev.device == dev
    report["recurrent_slots_static"] = True

    x_static = torch.zeros(x_shape, dtype=torch.float32, device=dev)

    def fwd(x):
        for block, inst, idx in blocks:
            x = block.forward(x, params)
        return x

    snap = StateSnapshot(cache, layer_instances)

    # Warmups on the default stream; each is a completed logical call on isolated state.
    with torch.inference_mode():
        for i in range(3):
            x_static.normal_(0, 0.5)
            fwd(x_static)
    torch.cuda.synchronize()
    import gc
    gc.collect()
    torch.cuda.empty_cache()  # drop warmup garbage so the reserved delta isolates the graph pool

    before_mem = mem_stats()
    t0 = time.perf_counter()
    g = None
    try:
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            with torch.inference_mode():
                fwd(x_static)
        capture_ok, capture_err = True, None
    except Exception as e:
        capture_ok, capture_err = False, repr(e)
        g = None
    t_capture = time.perf_counter() - t0
    after_mem = mem_stats()
    report["capture"] = {
        "ok": capture_ok, "error": capture_err, "wall_s": t_capture,
        "reserved_delta_b": {d: after_mem[d]["reserved_b"] - before_mem[d]["reserved_b"]
                             for d in before_mem},
        "allocated_delta_b": {d: after_mem[d]["allocated_b"] - before_mem[d]["allocated_b"]
                              for d in before_mem},
    }
    if not capture_ok:
        (out_dir / f"{name}.json").write_text(json.dumps(report, indent=2))
        print(f"[{name}] CAPTURE FAILED: {capture_err}", flush=True)
        return report

    # Parity trials: restore state, eager from a fresh input; restore again, replay from the
    # same input; compare in-place stream-stack output and post-forward state.
    parity = []
    for trial in range(trials):
        x0 = torch.randn(x_shape, dtype=torch.float32, device=dev) * 0.5
        torch.cuda.synchronize()

        snap.restore()
        x_e = x0.clone()
        t0 = time.perf_counter()
        with torch.inference_mode():
            y_eager = fwd(x_e)
        torch.cuda.synchronize()
        t_eager = time.perf_counter() - t0
        post_eager = snap.current()

        snap.restore()
        x_static.copy_(x0)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        g.replay()
        torch.cuda.synchronize()
        t_replay = time.perf_counter() - t0
        post_graph = snap.current()

        row = {"trial": trial, "t_eager_s": t_eager, "t_replay_s": t_replay,
               "x_out": tensor_diff(y_eager, x_static)}
        for li in layer_instances:
            row[f"conv_state@{li[0]}"] = tensor_diff(post_eager[li][0], post_graph[li][0])
            row[f"recurrent_state@{li[0]}"] = tensor_diff(post_eager[li][1], post_graph[li][1])
        parity.append(row)
        li0 = layer_instances[0][0]
        print(f"[{name}] trial {trial}: x_max={row['x_out']['max_abs']:.3e} "
              f"rec_max={row[f'recurrent_state@{li0}']['max_abs']:.3e} "
              f"eager={t_eager*1e3:.2f}ms replay={t_replay*1e3:.2f}ms", flush=True)

    # Chained replay: the in-place output becomes the next replay's input; two committed
    # logical steps, compared against two chained eager forwards from the same start state.
    x0 = torch.randn(x_shape, dtype=torch.float32, device=dev) * 0.5
    snap.restore()
    x_static.copy_(x0)
    torch.cuda.synchronize()
    g.replay(); torch.cuda.synchronize()
    g.replay(); torch.cuda.synchronize()
    out_graph_chain = x_static.detach().clone()
    post_graph_chain = snap.current()

    snap.restore()
    with torch.inference_mode():
        xc = x0.clone()
        xc = fwd(xc)
        xc = fwd(xc)
    torch.cuda.synchronize()
    post_eager_chain = snap.current()
    chain = {"x_out": tensor_diff(xc, out_graph_chain)}
    for li in layer_instances:
        chain[f"recurrent_state@{li[0]}"] = tensor_diff(post_eager_chain[li][1],
                                                        post_graph_chain[li][1])
    print(f"[{name}] chained: x_max={chain['x_out']['max_abs']:.3e}", flush=True)

    # Capture-failure cleanup: capture attempt whose grouped-MoE call injects a host sync must
    # fail; afterwards the graph is dropped, the allocator is sane and eager still advances
    # state exactly once from a restored snapshot.
    cleanup = test_capture_failure(blocks, params, snap, dev)
    print(f"[{name}] capture-failure cleanup: failed={cleanup.get('capture_failed')} "
          f"eager_ok={cleanup.get('eager_after_failure')} "
          f"reserved_delta={cleanup.get('reserved_delta_b')}", flush=True)

    report["parity"] = parity
    report["chained"] = chain
    report["capture_failure_cleanup"] = cleanup
    (out_dir / f"{name}.json").write_text(json.dumps(report, indent=2))
    return report


def test_capture_failure(blocks, params, snap, dev):
    """Inject a .item() into the grouped MoE ext call during a capture attempt."""
    import torch
    from exllamav3.ext import exllamav3_ext as ext
    result = {"attempted": True}
    x_static = torch.zeros(1, 1, 4, 2560, dtype=torch.float32, device=dev)
    snap.restore()
    with torch.inference_mode():
        for i in range(2):
            x_static.normal_(0, 0.5)
            for block, inst, idx in blocks:
                block.forward(x_static, params)
    torch.cuda.synchronize()
    mem_before = torch.cuda.memory_stats(dev)["reserved_bytes.all.current"]

    original = ext.exl3_moe_gfx12_k3

    def syncy(*args, **kwargs):
        torch.zeros(1, device=dev).item()  # host sync: illegal during capture
        return original(*args, **kwargs)

    g = None
    try:
        ext.exl3_moe_gfx12_k3 = syncy
        try:
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                with torch.inference_mode():
                    for block, inst, idx in blocks:
                        block.forward(x_static, params)
            result["capture_failed"] = False
        except Exception as e:
            result["capture_failed"] = True
            result["error"] = repr(e)[:400]
        finally:
            ext.exl3_moe_gfx12_k3 = original
    finally:
        g = None  # drop any partially-captured graph; its pool frees with it
        torch.cuda.synchronize()
        snap.restore()
        with torch.inference_mode():
            y = x_static.clone()
            for block, inst, idx in blocks:
                y = block.forward(y, params)
        torch.cuda.synchronize()
        result["eager_after_failure"] = bool(torch.isfinite(y.float()).all())
        result["reserved_b_after"] = torch.cuda.memory_stats(dev)["reserved_bytes.all.current"]
        result["reserved_delta_b"] = result["reserved_b_after"] - mem_before
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--blocks", type=int, default=1, help="blocks per captured run (P1=1, P2=2)")
    ap.add_argument("--trials", type=int, default=3)
    args = ap.parse_args()

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(args.out) if args.out else OUT_ROOT / f"p4-block-capture-{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[probe] output: {out_dir}", flush=True)

    import torch
    pid = os.getpid()
    kfd_before = kfd_delta(_b, pid)

    t0 = time.perf_counter()
    config, model, cache, tokenizer = load(MODEL_DIR)
    t_load = time.perf_counter() - t0
    print(f"[load] {t_load:.1f}s", flush=True)

    placement = dump_placement(model, out_dir)
    results = {"model": MODEL_DIR, "load_s": t_load, "kfd_before": kfd_before, "pid": pid}

    p1 = p2 = None
    try:
        # The loader allocates recurrent state under inference_mode; every probe-side touch
        # of those tensors (clear, snapshot, restore, forward) must run in the same mode.
        with torch.inference_mode():
            runs = find_gdn_runs(model, 1)
            assert runs, "no GDN block found"
            p1 = capture_and_verify("p1_single_block", runs[0],
                                    layer_instances_for(cache, runs[0]), cache,
                                    (1, 1, 4, 2560), out_dir, trials=args.trials)

            runs2 = find_gdn_runs(model, 2)
            if runs2:
                p2 = capture_and_verify("p2_two_blocks", runs2[0],
                                        layer_instances_for(cache, runs2[0]), cache,
                                        (1, 1, 4, 2560), out_dir, trials=args.trials)
    finally:
        kfd_after = kfd_delta(_b, pid)

    def max_abs(p):
        if not p or not p.get("capture", {}).get("ok"):
            return None
        m = max(r["x_out"]["max_abs"] for r in p["parity"])
        m = max(m, p["chained"]["x_out"]["max_abs"])
        return m

    summary = {
        "when": ts, "torch": torch.__version__, "hip": torch.version.hip,
        "kfd_before": kfd_before, "kfd_after": kfd_after,
        "p1": {"capture_ok": p1["capture"]["ok"],
               "capture_err": p1["capture"]["error"],
               "max_abs": max_abs(p1),
               "reserved_delta_b": p1["capture"]["reserved_delta_b"]} if p1 else None,
        "p2": ({"capture_ok": p2["capture"]["ok"],
                "capture_err": p2["capture"]["error"],
                "max_abs": max_abs(p2),
                "reserved_delta_b": p2["capture"]["reserved_delta_b"]} if p2 else None),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()