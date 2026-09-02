"""A/B benchmark for the gfx12 fixed-K QSA top-k route."""
from __future__ import annotations

import torch

from exllamav3.ext import exllamav3_ext as ext
from exllamav3 import ext_fallbacks

DEVICE = torch.device("cuda", 0)
K = 512
ROWS = 512


def elapsed(call, iterations: int) -> float:
    for _ in range(3):
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
    if (getattr(props, "gcnArchName", "").split(":", 1)[0] not in ("gfx1200", "gfx1201") or
            getattr(props, "warp_size", 0) != 32):
        raise SystemExit("requires gfx1200/gfx1201 with wave32")
    if not hasattr(ext, "dsa_topk_gfx12"):
        raise SystemExit("dsa_topk_gfx12 binding is unavailable")

    for width in (3072, 32768):  # QSA pools at 12K and 128K token context, respectively.
        stride = -(-width // 128) * 128
        scores = torch.randn((ROWS, stride), device = DEVICE, dtype = torch.half)[:, :width]
        native = torch.empty((ROWS, K), device = DEVICE, dtype = torch.int32)
        fallback = torch.empty_like(native)
        native_ms = elapsed(lambda: ext.dsa_topk_gfx12(scores, native), 12)
        fallback_ms = elapsed(lambda: ext_fallbacks.dsa_topk(scores, fallback, K), 5)
        torch.cuda.synchronize(DEVICE)
        assert torch.equal(native, fallback)
        print(
            f"R={ROWS} T={width}: native={native_ms:.3f} ms "
            f"fallback={fallback_ms:.3f} ms speedup={fallback_ms / native_ms:.2f}x"
        )


if __name__ == "__main__":
    main()
