"""Stage 5 TP-load smoke: load Qwen3.8-Flash-Next target-only under EXL3 tensor
parallel (tp_backend='nccl' -> RCCL) on 2 GPUs, run a short generate, and report
per-device memory + output. Host-only script; run only inside the exclusive-GPU
window. Target-only (no MTP draft; MTP attach is a separate stage).

Run (from repo root, 2 visible GPUs, isolated caches, roundup of production env):
  env PYTHONPATH=$PWD \\
      TORCH_EXTENSIONS_DIR=$HOME/.cache/qwen38-flash-exl3-rocm/torch_extensions \\
      TRITON_CACHE_DIR=$HOME/.cache/qwen38-flash-exl3-rocm/triton \\
      PYTORCH_ROCM_ARCH=gfx1201 \\
      EXL3_DIR=$PWD \\
      /home/douglasbrown/vllm-test-env/bin/python tests/tp_load_smoke.py
"""
import os
import sys
import time
import argparse
import torch

def _mem_gib():  # per visible device
    return [(i, torch.cuda.memory_allocated(i) / 2**30, torch.cuda.memory_reserved(i) / 2**30)
            for i in range(torch.cuda.device_count())]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/home/douglasbrown/Models/Qwen3.8-Flash-Next-exl3-bpw3")
    ap.add_argument("--cache-tokens", type=int, default=16384)
    ap.add_argument("--use-dev", default="30,30")
    ap.add_argument("--chunk", type=int, default=512)
    ap.add_argument("--prompt", default="Once upon a time, a small robot named Sparky discovered a mysterious glowing door in the middle of the forest. When Sparky opened it,")
    ap.add_argument("--new-tokens", type=int, default=16)
    ap.add_argument("--max-seq", type=int, default=32768)
    ap.add_argument("--tp-backend", default="nccl")
    ap.add_argument("--no-tp", action="store_true", help="load layer-split (control) instead of TP")
    args = ap.parse_args()

    use_per_device = [float(x) for x in args.use_dev.split(",")]
    assert torch.cuda.device_count() >= 2, "need 2 visible HIP devices"
    print("devices:", torch.cuda.device_count(), "use_per_device:", use_per_device)

    from exllamav3 import Config, Model, Cache, Tokenizer
    from exllamav3.generator import Generator
    from exllamav3.generator.sampler.presets import DefaultSampler

    config = Config.from_directory(args.model)
    model = Model.from_config(config)
    tokenizer = Tokenizer.from_config(config)

    # Build the cache BEFORE model.load (CaChe must be created before load).
    # Small pool for the smoke; target-only.
    cache = Cache(
        model=model,
        max_num_tokens=args.cache_tokens,
    )

    t0 = time.time()
    model.load(
        tensor_p=not args.no_tp,
        tp_backend=args.tp_backend,
        use_per_device=use_per_device,
        max_chunk_size=args.chunk,
        max_batch_size=1,
    )
    print(f"LOAD ok in {time.time()-t0:.1f}s")
    print("active_devices:", model.active_devices, "output_device:", getattr(model, "output_device", None))
    print("after load mem(GiB):", _mem_gib())

    gen = Generator(
        model, cache, tokenizer,
        max_batch_size=1,
        max_chunk_size=args.chunk,
        # target-only
        # recurrent cache small for host; fine at default
    )

    settings = DefaultSampler()
    tgen = time.time()
    out = ""
    try:
        result = gen.generate(prompt=args.prompt, max_new_tokens=args.new_tokens,
                              sampler=settings, streaming=False)
        out = result.get("text") if isinstance(result, dict) else result
    except Exception as e:
        print("GEN ERROR:", type(e).__name__, e)
        torch.cuda.synchronize()
        print("mem at error:", _mem_gib())
        raise
    print(f"GEN ok in {time.time()-tgen:.1f}s")
    print("transcribed:", repr(out))
    torch.cuda.synchronize()
    print("after gen mem(GiB):", _mem_gib())
    print("SMOKE OK")

if __name__ == "__main__":
    main()
