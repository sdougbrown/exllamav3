"""End-to-end 128K QSA selection A/B on the Flash-Next model's real indexer shape."""
from __future__ import annotations

from pathlib import Path

import torch

from exllamav3 import Config, Model
from exllamav3.ext import exllamav3_ext as ext
from exllamav3 import ext_fallbacks

MODEL = Path("/home/douglasbrown/Models/Qwen3.8-Flash-Next-exl3")
DEVICE = torch.device("cuda", 0)
ROWS = 512
CONTEXT = 128 * 1024


def elapsed(call, iterations: int) -> float:
    call()
    torch.cuda.synchronize(DEVICE)
    start = torch.cuda.Event(enable_timing = True)
    end = torch.cuda.Event(enable_timing = True)
    start.record()
    for _ in range(iterations):
        call()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / iterations


def main():
    if not (torch.version.hip and torch.cuda.is_available()):
        raise SystemExit("requires ROCm with an available GPU")
    props = torch.cuda.get_device_properties(DEVICE)
    arch = getattr(props, "gcnArchName", "").split(":", 1)[0]
    if arch not in ("gfx1200", "gfx1201") or getattr(props, "warp_size", 0) != 32:
        raise SystemExit("requires gfx1200/gfx1201 with wave32")
    if not hasattr(ext, "dsa_topk_gfx12"):
        raise SystemExit("dsa_topk_gfx12 binding is unavailable")
    if not MODEL.is_dir():
        raise SystemExit(f"model not found: {MODEL}")
    model = Model.from_config(Config.from_directory(str(MODEL)))
    attention = next(module for module in model if getattr(module, "qsa_indexer", None) is not None)
    indexer = attention.qsa_indexer
    assert indexer.block_topk == 512 and indexer.compress_ratio == 4
    try:
        generator = torch.Generator(device = DEVICE).manual_seed(1201)
        pooled = torch.randn((CONTEXT // indexer.compress_ratio, indexer.head_dim),
                             generator = generator, device = DEVICE, dtype = torch.half)
        q = torch.randn((ROWS, indexer.n_heads, indexer.head_dim), generator = generator,
                        device = DEVICE, dtype = torch.half)
        out_native = torch.empty((ROWS, indexer.k_pad()), device = DEVICE, dtype = torch.int32)
        out_fallback = torch.empty_like(out_native)
        pos0 = CONTEXT - ROWS

        calls = {"native": 0}
        native_topk = ext.dsa_topk_gfx12
        fallback_topk = ext_fallbacks.dsa_topk

        def native_spy(*args):
            calls["native"] += 1
            return native_topk(*args)

        def unexpected_fallback(*_args, **_kwargs):
            raise AssertionError("eligible QSA benchmark route used the fallback")

        ext.dsa_topk_gfx12 = native_spy
        ext_fallbacks.dsa_topk = unexpected_fallback
        try:
            native_ms = elapsed(
                lambda: indexer._select_rows(q, pooled, pos0, pooled.shape[0], out_native), 3,
            )
        finally:
            ext.dsa_topk_gfx12 = native_topk
            ext_fallbacks.dsa_topk = fallback_topk
        if calls["native"] == 0:
            raise AssertionError("QSA benchmark did not dispatch dsa_topk_gfx12")

        original = ext.dsa_topk
        ext.dsa_topk = fallback_topk
        try:
            fallback_ms = elapsed(
                lambda: indexer._select_rows(q, pooled, pos0, pooled.shape[0], out_fallback), 2,
            )
        finally:
            ext.dsa_topk = original
        assert torch.equal(out_native, out_fallback)
        print(
            f"Flash-Next QSA R={ROWS} context={CONTEXT}: native={native_ms:.3f} ms "
            f"fallback={fallback_ms:.3f} ms speedup={fallback_ms / native_ms:.2f}x"
        )
    finally:
        model.unload()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
