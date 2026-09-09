#!/usr/bin/env python3
"""Synchronized TP/LS prefill wall-time benchmark with explicit runtime chunking.

Unlike tests/tp_throughput_bench.py, this harness:
  * actually splits the prompt at runtime into `--chunk`-token calls (the old
    harness only passed --prefill-chunk to model.load, sizing load-time work);
  * stops timing only after completion receipts from every participating GPU
    process (TP child ranks synchronize their own CUDA context via the existing
    worker machinery; the parent cannot synchronize a child's context);
  * uses time.perf_counter and fences only at region boundaries, never inside
    the chunk loop;
  * records actual device placement, per-rank memory peaks, KFD evicted_ms
    deltas, and journal evidence instead of reporting host-return time.

Run each cell as a separate exclusive process, e.g.:

  TORCH_EXTENSIONS_DIR=$HOME/.cache/qwen38-flash-exl3-rocm/torch_extensions \
  TRITON_CACHE_DIR=$HOME/.cache/qwen38-flash-exl3-rocm/triton \
  PYTORCH_ROCM_ARCH=gfx1201 HIP_VISIBLE_DEVICES=0,1 \
  /home/douglasbrown/vllm-test-env/bin/python tests/tp_prefill_wall_bench.py \
      --mode tp --model /home/douglasbrown/Models/Qwen3.8-27B-exl3 \
      --prompt-tokens 4096 --chunk 512 --cache-tokens 8192 --reps 3 \
      --out-dir /tmp/tprefill

Pure-logic helpers (plan_chunks, CompletionBarrier, ...) live at module level
and import no torch, so tests/test_tp_prefill_wall_bench.py can exercise the
chunking/receipt/ordering logic on CPU.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DISPATCH_TIMEOUT_S = 600.0


# ---------------------------------------------------------------------------
# Pure logic (no torch)
# ---------------------------------------------------------------------------

def plan_chunks(prompt_len: int, chunk: int) -> list[tuple[int, int]]:
    """Split prompt_len tokens into (start, length) chunks of at most `chunk`.

    The union of chunks covers the prompt exactly once: no token is dropped and
    no final token is duplicated.
    """
    if chunk <= 0:
        raise ValueError(f"chunk must be positive, got {chunk}")
    if prompt_len <= 0:
        raise ValueError(f"prompt_len must be positive, got {prompt_len}")
    chunks = []
    start = 0
    while start < prompt_len:
        length = min(chunk, prompt_len - start)
        chunks.append((start, length))
        start += length
    return chunks


def check_cache_extent(cache_tokens: int, prompt_len: int, extra_tokens: int = 0) -> None:
    """Raise if prompt (+ continuation) cannot fit the fixed cache pool."""
    needed = prompt_len + extra_tokens
    if needed > cache_tokens:
        raise ValueError(
            f"cache pool {cache_tokens} tokens is smaller than prompt {prompt_len}"
            f" + continuation {extra_tokens} = {needed}"
        )


def rows_for_batch(batch_size: int, chunk_len: int) -> int:
    """Projection rows a rectangular batch submits in one call."""
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    return batch_size * chunk_len


def per_sequence_chunk(batch_size: int, chunk: int, max_rows: int) -> int:
    """Reduce the per-sequence chunk so B x chunk stays within max_rows."""
    if max_rows <= 0:
        raise ValueError(f"max_rows must be positive, got {max_rows}")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    return max(1, min(chunk, max_rows // batch_size))


def dispatch_order(active_devices: list[int], output_device: int) -> list[int]:
    """Child ranks first, in-process output-device pseudo-worker last.

    The pseudo-worker executes commands synchronously and can block inside TP
    collectives; all spawned ranks must have received the same command first.
    """
    if output_device not in active_devices:
        raise ValueError(f"output device {output_device} not in {active_devices}")
    children = [d for d in active_devices if d != output_device]
    return children + [output_device]


class CompletionBarrier:
    """Collects completion receipts from every participating GPU process.

    wait() returns only after every expected participant delivered a receipt;
    a deferred Python acknowledgment does not count as a receipt.
    """

    def __init__(self, expected: list[str], timeout_s: float = DISPATCH_TIMEOUT_S):
        if not expected:
            raise ValueError("CompletionBarrier needs at least one participant")
        dupes = {e for e in expected if expected.count(e) > 1}
        if dupes:
            raise ValueError(f"duplicate participants: {sorted(dupes)}")
        self.expected = list(expected)
        self.received: dict[str, float] = {}
        self.timeout_s = timeout_s
        self.t_start = time.perf_counter()

    def deliver(self, participant: str, payload=None):
        if participant not in self.expected:
            raise ValueError(f"unexpected participant {participant!r}")
        if participant in self.received:
            raise ValueError(f"duplicate receipt from {participant!r}")
        self.received[participant] = time.perf_counter()
        return payload

    @property
    def satisfied(self) -> bool:
        return all(p in self.received for p in self.expected)

    def wait(self) -> float:
        """Block until all receipts are in; returns seconds since barrier start."""
        while not self.satisfied:
            if time.perf_counter() - self.t_start > self.timeout_s:
                missing = [p for p in self.expected if p not in self.received]
                raise TimeoutError(f"missing completion receipts from {missing}")
            time.sleep(0.001)
        return time.perf_counter() - self.t_start


# ---------------------------------------------------------------------------
# Child-rank callback (executed inside TP worker processes)
# ---------------------------------------------------------------------------

def _proc_rss_kb() -> int:
    try:
        with open("/proc/self/statm") as f:
            return int(f.readline().split()[1]) * (os.sysconf("SC_PAGE_SIZE") // 1024)
    except OSError:
        return -1


def _read_evicted_ms(pid: int) -> dict:
    """Per-GPU evicted_ms counters exposed by KFD for this process."""
    out = {}
    base = Path(f"/sys/class/kfd/kfd/proc/{pid}")
    try:
        for entry in sorted(base.glob("stats_*/evicted_ms")):
            out[entry.parent.name] = int(entry.read_text().strip())
    except OSError:
        pass
    return out


def mp_rank_sync(local_context: dict, reset_peak: bool = False) -> dict:
    """Rank-local CUDA synchronization and stats, dispatched through the
    existing worker machinery (child-first, output device last). This is the
    GPU-completion receipt of a TP child; torch.cuda.synchronize() in the
    parent cannot synchronize a child's context."""
    import torch

    device = local_context["device"]
    torch.cuda.synchronize()
    if reset_peak:
        torch.cuda.reset_peak_memory_stats()
    free, total = torch.cuda.mem_get_info()
    return {
        "device": device,
        "pid": os.getpid(),
        "allocated_b": torch.cuda.memory_allocated(),
        "reserved_b": torch.cuda.memory_reserved(),
        "peak_b": torch.cuda.max_memory_allocated(),
        "free_vram_b": free,
        "total_vram_b": total,
        "rss_kb": _proc_rss_kb(),
        "evicted_ms": _read_evicted_ms(os.getpid()),
    }


def mp_rank_placement(local_context: dict) -> list:
    """Actual device of every module loaded on this rank (TP children own
    their weights; the parent's module objects are unset)."""
    out = []
    for module in local_context["modules"]:
        name = module.get_name() if hasattr(module, "get_name") else type(module).__name__
        device = getattr(module, "device", None)
        out.append((name, str(device) if device is not None else "unset"))
    return out


def mp_prof_start(local_context: dict) -> None:
    """Start a CPU+CUDA profiler on this rank. The traced region spans every
    forward the parent dispatches until mp_prof_stop; keep it inference-only
    and short (profiler wall is perturbed and excluded from headline timing)."""
    import torch
    from torch.profiler import ProfilerActivity, profile

    prof = profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=False,
        with_stack=False,
    )
    prof.start()
    local_context["_prof"] = prof
    return None


def mp_prof_stop(local_context: dict, out_dir: str) -> dict:
    """Stop this rank's profiler, export a chrome trace, and return a compact
    per-kernel-name duration summary (device time only)."""
    import torch

    prof = local_context.pop("_prof")
    prof.stop()
    device = local_context["device"]
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"trace-rank{device}-pid{os.getpid()}.json"
    prof.export_chrome_trace(str(path))
    # compact summary: total device kernel time per kernel name
    summary = {}
    for evt in prof.key_averages():
        cuda_us = getattr(evt, "self_device_time_total", 0) or 0
        if cuda_us > 0:
            summary[evt.key] = {
                "cuda_us": cuda_us,
                "count": evt.count,
            }
    return {"device": device, "pid": os.getpid(), "trace": str(path),
            "kernels": summary}


# ---------------------------------------------------------------------------
# GPU benchmark
# ---------------------------------------------------------------------------

def _walk_placement(module, out: list):
    name = module.get_name() if hasattr(module, "get_name") else type(module).__name__
    device = getattr(module, "device", None)
    out.append((name, str(device) if device is not None else "cpu/unset"))
    for child in getattr(module, "modules", []):
        _walk_placement(child, out)


def _git(*args) -> str:
    try:
        return subprocess.run(
            ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:
        return "unavailable"


def _journal_cursor() -> str | None:
    try:
        r = subprocess.run(
            ["journalctl", "-k", "-n", "0", "--show-cursor"],
            capture_output=True, text=True, timeout=10,
        )
        for line in r.stdout.splitlines():
            if line.startswith("-- cursor:"):
                return line.split("cursor:")[1].strip()
    except Exception:
        pass
    return None


def _journal_since(cursor: str | None) -> list[str]:
    if cursor is None:
        return []
    try:
        r = subprocess.run(
            ["journalctl", "-k", f"--after-cursor={cursor}"],
            capture_output=True, text=True, timeout=10,
        )
        return [
            ln for ln in r.stdout.splitlines()
            if any(k in ln.lower() for k in ("amdgpu", "svm_range", "restore", "kfd"))
        ]
    except Exception:
        return ["<journal read failed>"]


def _collect_rank_pids(model) -> list[int]:
    pids = [os.getpid()]
    children = getattr(model, "mp_children", None) or []
    if isinstance(children, dict):
        children = children.values()
    for child in children:
        pid = getattr(child, "pid", None)
        if pid is not None:
            pids.append(pid)
    return pids


def _sync_devices(model, mode: str, used_devices: list[str]) -> tuple[CompletionBarrier, list]:
    """Fence every participating GPU process and collect receipts."""
    if mode == "tp":
        participants = [f"rank:{d}" for d in model.active_devices]
    else:
        participants = [f"rank:{int(d.split(':')[-1])}" for d in used_devices]
    receipts = CompletionBarrier(expected=participants)
    if mode == "tp":
        # Existing worker machinery: dispatch to child ranks first, output
        # pseudo-worker last, then collect every rank's sync receipt.
        results = model.tp_worker_dispatch_wait_multi(
            model.active_devices, mp_rank_sync, ()
        )
        for dev, res in zip(model.active_devices, results):
            receipts.deliver(f"rank:{dev}", res)
        return receipts, results
    import torch
    results = []
    for d in sorted(used_devices):
        idx = int(d.split(":")[-1])
        with torch.cuda.device(idx):
            torch.cuda.synchronize()
        free, total = torch.cuda.mem_get_info(idx)
        res = {
            "device": idx,
            "pid": os.getpid(),
            "allocated_b": torch.cuda.memory_allocated(idx),
            "reserved_b": torch.cuda.memory_reserved(idx),
            "peak_b": torch.cuda.max_memory_allocated(idx),
            "free_vram_b": free,
            "total_vram_b": total,
            "rss_kb": _proc_rss_kb(),
            "evicted_ms": _read_evicted_ms(os.getpid()),
        }
        results.append(res)
        receipts.deliver(f"rank:{idx}", res)
    return receipts, results


def run_bench(args) -> dict:
    sys.path.insert(0, str(REPO_ROOT))
    import torch
    from exllamav3 import Cache, Config, Model, Tokenizer

    if not torch.cuda.is_available():
        raise SystemExit("no HIP devices visible")
    torch.manual_seed(0)

    report: dict = {
        "mode": args.mode,
        "model": args.model,
        "prompt_tokens": args.prompt_tokens,
        "chunk": args.chunk,
        "cache_tokens": args.cache_tokens,
        "reps": args.reps,
        "repo_head": _git("rev-parse", "HEAD"),
        "repo_dirty": _git("status", "--porcelain"),
        "visible_devices": os.environ.get("HIP_VISIBLE_DEVICES", ""),
        "journal_cursor": _journal_cursor(),
    }

    config = Config.from_directory(args.model)
    model = Model.from_config(config)
    tokenizer = Tokenizer.from_config(config)
    cache = Cache(model=model, max_num_tokens=args.cache_tokens, max_batch_size=1)
    check_cache_extent(args.cache_tokens, args.prompt_tokens, extra_tokens=16)

    load_args = {"max_chunk_size": args.chunk, "max_batch_size": 1}
    if args.mode == "tp":
        load_args.update(tensor_p=True, tp_backend="nccl", use_per_device=args.use_dev)
    elif args.mode == "ls":
        load_args.update(use_per_device=args.use_dev)
    elif args.mode == "single":
        load_args.update(device="cuda:0", max_chunk_size=args.chunk)
    loaded = False
    try:
        t0 = time.perf_counter()
        model.load(**load_args)
        report["load_s"] = round(time.perf_counter() - t0, 2)
        loaded = True

        placement = []
        if args.mode == "tp":
            rank_placements = model.tp_worker_dispatch_wait_multi(
                model.active_devices, mp_rank_placement, ()
            )
            report["rank_placements"] = {
                str(dev): placements
                for dev, placements in zip(model.active_devices, rank_placements)
            }
            placement = [
                (f"rank{dev}:{name}", dev)
                for dev, placements in zip(model.active_devices, rank_placements)
                for (name, _) in placements
            ]
            used_devices = [f"cuda:{d}" for d in model.active_devices]
        else:
            _walk_placement(model, placement)
            used_devices = sorted({d for _, d in placement if d.startswith("cuda")})
        report["active_devices"] = getattr(model, "active_devices", None)
        report["tp_output_device"] = getattr(model, "tp_output_device", None)
        report["used_devices"] = used_devices
        report["placement"] = placement
        report["rank_pids"] = _collect_rank_pids(model)

        # Warmup: same chunk shape, short prompt, excluded from timing. Peak
        # counters reset after warmup so timed reps report their own peaks.
        warm_ids = tokenizer.encode("warm", add_bos=True).cpu()
        while warm_ids.shape[1] < min(512, args.prompt_tokens):
            warm_ids = torch.cat([warm_ids, warm_ids[:, : min(512, args.prompt_tokens) - warm_ids.shape[1]]], dim=1)
        with torch.inference_mode():
            params = {
                "attn_mode": "flash_attn", "cache": cache, "past_len": 0,
                "batch_shape": (1, args.cache_tokens),
            }
            model.prefill(input_ids=warm_ids, params=params)
        states = params.get("recurrent_states") or []
        for s in states:
            s.free()
        receipts, warm_stats = _sync_devices(model, args.mode, used_devices)
        receipts.wait()
        report["warmup_stats"] = warm_stats
        # reset per-rank peaks so timed reps report their own peaks
        if args.mode == "tp":
            model.tp_worker_dispatch_wait_multi(
                model.active_devices, mp_rank_sync, (True,)
            )
        else:
            for d in used_devices:
                torch.cuda.reset_peak_memory_stats(int(d.split(":")[-1]))
        report["evicted_ms_warmup"] = {
            res["pid"]: res["evicted_ms"] for res in warm_stats
        }

        ids = tokenizer.encode("x", add_bos=True).cpu()
        while ids.shape[1] < args.prompt_tokens:
            ids = torch.cat([ids, ids[:, : args.prompt_tokens - ids.shape[1]]], dim=1)
        assert ids.shape[1] == args.prompt_tokens
        ids = ids[:, :args.prompt_tokens].contiguous()

        chunks = plan_chunks(args.prompt_tokens, args.chunk)
        report["chunk_plan"] = [
            {"start": s, "len": l} for s, l in chunks
        ]
        assert sum(l for _, l in chunks) == args.prompt_tokens, "token accounting"

        rows_this_call = rows_for_batch(1, args.chunk)
        max_rows = args.chunk  # batch 1: never exceed the dispatched chunk
        assert rows_this_call <= max_rows

        reps = []
        parent_prof = None
        if args.trace:
            # Inference-only trace: profile every rank across ONE uncached
            # region (excluded from headline timing). Real children keep full
            # chrome traces; the output-device pseudo-worker lives in the
            # parent process and is summarized only — profiling artifacts
            # (chrome export) segfault in the parent on this ROCm/kineto stack.
            trace_dir = Path(args.out_dir or "/tmp") / "traces"
            tp_children = [d for d in model.active_devices if d != model.tp_output_device]
            model.tp_worker_dispatch_multi(tp_children, mp_prof_start, ())
            from torch.profiler import ProfilerActivity, profile
            parent_prof = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA])
            parent_prof.start()

        for rep in range(args.reps):
            t0 = time.perf_counter()
            past = 0
            prev_states = None
            with torch.inference_mode():
                for (start, length) in chunks:
                    # Fresh params dict per call; reinject the same recurrent
                    # state objects (not copies); past_len must match state
                    # positions exactly (prepare_for_recurrence asserts this).
                    params = {
                        "attn_mode": "flash_attn", "cache": cache,
                        "past_len": past, "batch_shape": (1, args.cache_tokens),
                    }
                    if prev_states is not None:
                        params["recurrent_states"] = prev_states
                    model.prefill(input_ids=ids[:, start:start + length].contiguous(), params=params)
                    prev_states = params.get("recurrent_states")
                    past += length
            assert past == args.prompt_tokens
            if prev_states:
                for s in prev_states:
                    s.free()
            # Fence every participating GPU process; the timer stops here,
            # after receipts from all ranks, never at Python-call return.
            receipts, rank_stats = _sync_devices(model, args.mode, used_devices)
            receipts.wait()
            wall = time.perf_counter() - t0
            if args.trace and rep == 0 and parent_prof is not None:
                try:
                    parent_prof.stop()
                    tp_children = [d for d in model.active_devices if d != model.tp_output_device]
                    stop_results = model.tp_worker_dispatch_wait_multi(
                        tp_children, mp_prof_stop, (str(trace_dir),)
                    )
                    trace_dir.mkdir(parents=True, exist_ok=True)
                    # parent (output-device rank): summary only, no chrome export
                    parent_summary = {}
                    for evt in parent_prof.key_averages():
                        cuda_us = getattr(evt, "self_device_time_total", 0) or 0
                        if cuda_us > 0:
                            parent_summary[evt.key] = {
                                "cuda_us": cuda_us, "count": evt.count,
                            }
                    report["trace"] = {
                        "dir": str(trace_dir),
                        "ranks": [r for r in stop_results],
                        "parent_device": model.tp_output_device,
                        "parent_kernels": parent_summary,
                        "note": "parent rank summarized via key_averages; chrome export segfaults in the profiled parent process on this stack",
                    }
                except Exception:
                    report["trace_error"] = traceback.format_exc()
            reps.append({
                "rep": rep,
                "wall_s": round(wall, 4),
                "tok_s": round(args.prompt_tokens / wall, 1),
                "rank_stats": rank_stats,
            })
            print(f"[{args.mode}] rep {rep}: {wall:.3f}s  {args.prompt_tokens / wall:.1f} tok/s", flush=True)

        report["reps"] = reps
        walls = sorted(r["wall_s"] for r in reps)
        med = walls[len(walls) // 2] if len(walls) % 2 else (walls[len(walls)//2 - 1] + walls[len(walls)//2]) / 2
        report["median_wall_s"] = round(med, 4)
        report["median_tok_s"] = round(args.prompt_tokens / med, 1)
        report["evicted_ms_final"] = {
            pid: _read_evicted_ms(pid) for pid in report["rank_pids"]
        }
        report["journal_lines"] = _journal_since(report["journal_cursor"])

    except Exception as e:
        report["error"] = repr(e)
        report["traceback"] = traceback.format_exc()
        raise
    finally:
        if loaded:
            try:
                model.unload()
            except Exception:
                report["unload_error"] = traceback.format_exc()

    if args.out_dir:
        out = Path(args.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        fname = out / f"bench-{args.mode}-p{args.prompt_tokens}-c{args.chunk}-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.json"
        fname.write_text(json.dumps(report, indent=2, default=str))
        print(f"wrote {fname}")
    print(json.dumps({k: report[k] for k in ("mode", "median_wall_s", "median_tok_s", "used_devices") if k in report}))
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", required=True, choices=("tp", "ls", "single"))
    ap.add_argument("--model", default="/home/douglasbrown/Models/Qwen3.8-27B-exl3")
    ap.add_argument("--cache-tokens", type=int, default=8192)
    ap.add_argument("--use-dev", type=lambda s: [float(x) for x in s.split(",")], default=[30.0, 30.0])
    ap.add_argument("--prompt-tokens", type=int, default=4096)
    ap.add_argument("--chunk", type=int, default=512)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--trace", action="store_true",
                    help="profile one prefill region on every rank (inference-only, not timed)")
    args = ap.parse_args()
    run_bench(args)


if __name__ == "__main__":
    main()