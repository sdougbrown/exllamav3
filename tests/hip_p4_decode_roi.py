"""P4 decode ROI profiler driver (marked rocprofv3 regions, offline, c1 MTP3).

Loads the model once, warms eager and graph capture with the profiler PAUSED, then opens
marked regions (roctx) around two windows of decode iterations: eager and graph (flag toggled
dynamically, no module reload). Devices are synchronized only at region boundaries. Run under:
  /opt/rocm/bin/rocprofv3 --hip-runtime-trace --kernel-trace --memory-copy-trace \
      --marker-trace --output-format csv --output-directory <dir> -- python hip_p4_decode_roi.py
"""

import ctypes
import importlib
import importlib.util
import json
import os
import time
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
CACHE_NUM_TOKENS = 65536
WARM_ITER = 40          # decode iterations before the measured window
MEASURE_ITER = 24       # measured decode iterations per arm
PROMPT = ("Write a detailed technical explanation of how a modern GPU command processor "
          "submits work, including queues, doorbells and fences. ")
MAX_TOKENS = int(2.5 * (WARM_ITER + 8 + 2 * MEASURE_ITER)) + 48


def main():
    # production profile: sync-free MoE histogram required for capture eligibility
    os.environ["EXL3_MOE_SYNC_FREE_COUNT"] = "1"
    os.environ["EXL3_QC_STAGING"] = "1"
    os.environ["EXL3_PREFILL_ASYNC_UPLOADS"] = "0"
    out_dir = Path(os.environ.get("P4_ROI_DIR", os.getcwd()))
    lib = ctypes.CDLL('/opt/rocm/lib/librocprofiler-sdk-roctx.so')
    lib.roctxRangePushA.argtypes = [ctypes.c_char_p]
    lib.roctxProfilerPause(0)  # warmup and capture stay out of the trace

    config = Config.from_directory(MODEL)
    config.infer_params.ngram_stream_from_disk = True
    target = Model.from_config(config, component="text")
    draft = Model.from_config(config, component="mtp")
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
                    num_draft_tokens=3, record_draft_stats=True,
                    max_batch_size=1, max_chunk_size=512,
                    recurrent_cache_size=512 * 1024 ** 2, cpu_cache_size=0)

    # schedule: [label, open_at, close_at, flag]; flag must be set at the arm's warm start
    # so graph capture/warmup happens under the paused profiler, before its measured window.
    arms = [("eager", WARM_ITER, WARM_ITER + MEASURE_ITER, "0"),
            ("graph", WARM_ITER + 8 + MEASURE_ITER,
             WARM_ITER + 8 + 2 * MEASURE_ITER, "1")]
    arm_flag = {"value": "0"}
    arm_idx = 0
    results = {}
    job = Job(input_ids=tokenizer.encode(PROMPT, add_bos=True),
              max_new_tokens=MAX_TOKENS, stop_conditions=[], sampler=GreedySampler())
    gen.enqueue(job)
    seq = job.sequences[0]
    prompt_pos = seq.kv_position
    di = 0
    region_open = False
    t0 = t1 = None
    counters0 = {}
    accepted = rejected = 0
    while gen.num_remaining_jobs():
        label, open_at, close_at, flag = arms[arm_idx]
        if arm_flag["value"] != flag:
            arm_flag["value"] = flag
            os.environ["EXL3_BLOCK_GRAPH"] = flag
            block_graph.BLOCK_GRAPH_ENABLED = (flag == "1")  # dynamic toggle, no reload
        if region_open and di >= close_at:
            lib.roctxRangePop()
            lib.roctxProfilerPause(0)
            t1 = time.perf_counter()
            counters1 = block_graph.global_stats()
            results[label] = {"wall_s": t1 - t0, "measured_iters": open_at and MEASURE_ITER,
                              "captures_delta": counters1["captures"] - counters0[label]["captures"],
                              "replays_delta": counters1["replays"] - counters0[label]["replays"],
                              "open_at": open_at, "close_at": close_at}
            region_open = False
            arm_idx = min(arm_idx + 1, len(arms) - 1)
        for _r in gen.iterate():
            pass
        if seq.kv_position >= prompt_pos:
            di += 1
        if not region_open and di >= open_at and label not in results:
            print(f"[roi dbg] open {label}: flag_env={os.environ.get('EXL3_BLOCK_GRAPH')} "
                  f"module_flag={block_graph.BLOCK_GRAPH_ENABLED} "
                  f"stats={block_graph.global_stats()['captures']}/"
                  f"{block_graph.global_stats()['replays']} "
                  f"declines={dict(block_graph.global_stats()['declines'])}", flush=True)
            if label not in counters0:
                s0 = block_graph.global_stats()
                counters0[label] = {"captures": s0["captures"], "replays": s0["replays"]}
            lib.roctxProfilerResume(0)
            lib.roctxRangePushA(f'p4_{label}'.encode())
            region_open = True
            t0 = time.perf_counter()
        if arm_flag["value"] != arms[arm_idx][3]:
            arm_flag["value"] = arms[arm_idx][3]
            os.environ["EXL3_BLOCK_GRAPH"] = arms[arm_idx][3]
            block_graph.BLOCK_GRAPH_ENABLED = (arms[arm_idx][3] == "1")
    torch.cuda.synchronize()
    accepted = sum(s[2] for s in (job.draft_stats or []))
    rejected = sum(s[1] - s[2] for s in (job.draft_stats or []))
    results["job"] = {"accepted": accepted, "rejected": rejected,
                      "max_tokens": MAX_TOKENS}
    print(f"[roi] {json.dumps(results, indent=1)}", flush=True)

    (out_dir / "p4-decode-roi.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=1), flush=True)
    block_graph.purge()
    target.unload()
    draft.unload()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()