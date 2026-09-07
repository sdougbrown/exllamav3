"""P4 smoke: integrated block-graph hook inside the real generator loop (offline, quiescent).

Runs a short greedy target-only generation with EXL3_BLOCK_GRAPH enabled and compares token
IDs against the eager configuration from the same restored starting point. Reports runner
stats and graph-pool growth. Not a qualification run: no MTP, no concurrency, no serving."""

import importlib.util
import os
import sys
from pathlib import Path

_HARNESS = Path(os.environ.get(
    "EXL3_VALIDATED_PREFILL_HARNESS",
    str(Path.home() / "Serve/hosts/rocky/bench-prefill-validated.py"))).expanduser()
_spec = importlib.util.spec_from_file_location("validated", _HARNESS)
_b = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_b)

import torch  # noqa: E402
from collections import Counter  # noqa: E402
from exllamav3 import Cache, Config, Generator, GreedySampler, Job, Model, Tokenizer  # noqa: E402
from exllamav3.cache import CacheLayer_quant  # noqa: E402
from exllamav3.modules import block_graph  # noqa: E402

MODEL = os.path.expanduser("~/Models/Qwen3.8-Flash-Next-exl3-bpw3")
PROMPT = "The capital of France is"
STEPS = 24


def build(env_flag):
    os.environ["EXL3_BLOCK_GRAPH"] = "1" if env_flag else "0"
    import importlib
    importlib.reload(block_graph)
    config = Config.from_directory(MODEL)
    tokenizer = Tokenizer.from_config(config)
    model = Model.from_config(config)
    cache = Cache(model, max_num_tokens=40960, layer_type=CacheLayer_quant,
                  k_bits=8, v_bits=8, max_batch_size=1, max_history=3)
    model.load(use_per_device=[30.0, 30.0], max_batch_size=1, max_chunk_size=512,
               max_output_size=32, verbose=False)
    gen = Generator(model, cache, tokenizer, max_batch_size=1, max_chunk_size=512,
                    recurrent_cache_size=1 * 1024 ** 3, cpu_cache_size=0)
    return model, cache, tokenizer, gen


def run(gen):
    job = Job(input_ids=gen.tokenizer.encode(PROMPT, add_bos=True),
              max_new_tokens=STEPS, stop_conditions=[], sampler=GreedySampler())
    gen.enqueue(job)
    out = []
    while gen.num_remaining_jobs():
        for r in gen.iterate():
            if "token_ids" in r:
                out += list(r["token_ids"].cpu().tolist())
    return out


def main():
    res0 = None
    # Eager control
    model, cache, tok, gen = build(False)
    eager_ids = run(gen)
    print("eager tokens:", eager_ids[:8], "...", flush=True)
    del gen, model, cache

    # Graph-enabled run
    model, cache, tok, gen = build(True)
    res_before = torch.cuda.memory_stats(0)["reserved_bytes.all.current"]
    graph_ids = run(gen)
    res_after = torch.cuda.memory_stats(0)["reserved_bytes.all.current"]
    stats = block_graph.global_stats()
    declines = dict(stats["declines"])
    print("graph tokens:", graph_ids, flush=True)
    print(f"token identity: {graph_ids == eager_ids}")
    print(f"stats: captures={stats['captures']} replays={stats['replays']} "
          f"warmups={stats['warmups']} capture_failed={stats['capture_failed']} "
          f"pool_delta≈{res_after - res_before} bytes", flush=True)
    print("declines:", {k: v for k, v in declines.items() if v}, flush=True)
    assert graph_ids == eager_ids, "token identity failed"
    assert stats["captures"] > 0 and stats["replays"] > 0, "no graph replays occurred"
    assert stats["capture_failed"] == 0
    print("SMOKE PASS", flush=True)


if __name__ == "__main__":
    main()