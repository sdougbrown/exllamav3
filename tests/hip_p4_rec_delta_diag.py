"""P4 diagnostic: localize the recurrent-state delta between integrated block-graph replay
and eager on one GDN block of the flash test model. Reports eager-vs-eager, replay-vs-replay
and eager-vs-replay deltas with element coordinates."""

import importlib.util
import os
from pathlib import Path

_HARNESS = Path(os.environ.get(
    "EXL3_VALIDATED_PREFILL_HARNESS",
    str(Path.home() / "Serve/hosts/rocky/bench-prefill-validated.py"))).expanduser()
_spec = importlib.util.spec_from_file_location("validated", _HARNESS)
_b = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_b)

from collections import Counter
import torch  # noqa: E402
from exllamav3 import Cache, Config, Model  # noqa: E402
from exllamav3.cache import CacheLayer_quant  # noqa: E402
from exllamav3.modules import block_graph  # noqa: E402
from exllamav3.modules.block_graph import BlockGraphRunner  # noqa: E402
from exllamav3.util.tensor import get_for_device  # noqa: E402

MODEL = os.path.expanduser("~/Models/Qwen3.8-Flash-Next-exl3-bpw3")


def describe(delta, rec):
    flat = int(delta.argmax())
    ring = flat // (rec.numel() // rec.shape[0]) % rec.shape[1]
    return float(delta.max()), flat, ring


def main():
    config = Config.from_directory(MODEL)
    model = Model.from_config(config)
    cache = Cache(model, max_num_tokens=40960, layer_type=CacheLayer_quant,
                  k_bits=8, v_bits=8, max_batch_size=1, max_history=3)
    model.load(use_per_device=[30.0, 30.0], max_batch_size=1, max_chunk_size=512,
               max_output_size=32, verbose=False)

    from exllamav3.modules import GatedDeltaNet, TransformerBlock
    block = next(e[0] for e in model.fwd_modules
                 if isinstance(e[0], TransformerBlock) and isinstance(e[0].attn, GatedDeltaNet))
    dev = block.device
    li = (block.attn.layer_idx, 0)

    with torch.inference_mode():
        state = cache.get_new_state()
        params = {
            "layer_instance": 0,
            "recurrent_states": [state],
            "recurrent_slots": torch.tensor([state.slot], dtype=torch.int32),
        }
        get_for_device(params, "recurrent_slots", dev)
        conv, rec = cache.get_recurrent_layer(li).get_state_tensors()
        print("rec shape:", tuple(rec.shape), "dtype", rec.dtype, flush=True)

        runner = BlockGraphRunner(block)
        shape = (1, 1, 4, 2560)
        x0 = torch.randn(shape, dtype=torch.float32, device=dev) * 0.5

        def snap():
            return conv.detach().clone(), rec.detach().clone()

        def restore(s):
            conv.copy_(s[0])
            rec.copy_(s[1])
            torch.cuda.synchronize(dev)

        block_graph.BLOCK_GRAPH_ENABLED = True
        for _ in range(3):
            block.forward(torch.randn(shape, dtype=torch.float32, device=dev) * 0.5, params)
        torch.cuda.synchronize(dev)

        # eager-vs-eager envelope over 5 samples
        s0 = snap()
        env = 0.0
        prev = None
        for i in range(5):
            restore(s0)
            block.forward(x0.clone(), params)
            s_i = snap()
            if prev is not None:
                d = (s_i[1].float() - prev[1].float()).abs()
                env = max(env, float(d.max()))
                print(f"eager#{i}: rec delta max={float(d.max()):.3e} "
                      f"flat={int(d.argmax())}", flush=True)
            prev = s_i
        restore(s0)
        print(f"eager envelope: {env:.3e}", flush=True)

        # capture on the 4th call through the integrated hook
        for i in range(3):
            block.forward(x0.clone(), params)
        torch.cuda.synchronize(dev)
        s_cap = snap()
        block.forward(x0.clone(), params)  # 4th call -> capture + replay
        runner = getattr(block, "block_graph_runner", None)
        if runner is None or not runner.slots:
            import json
            stats = {k: (dict(v) if hasattr(v, "items") and not isinstance(v, str) else v)
                     for k, v in (runner.stats.items() if runner else [])}
            print("RUNNER STATS:", stats, flush=True)
        assert runner is not None and runner.slots, "no slot captured"
        g = next(iter(runner.slots.values())).graph
        torch.cuda.synchronize(dev)
        s_cap = snap()
        block.forward(x0.clone(), params)  # 4th call -> capture + replay
        runner = getattr(block, "block_graph_runner", None)
        if runner is None or not runner.slots:
            import json
            stats = {k: (dict(v) if hasattr(v, "items") and not isinstance(v, str) else v)
                     for k, v in (runner.stats.items() if runner else [])}
            print("RUNNER STATS:", stats, flush=True)
        assert runner is not None and runner.slots, "no slot captured"
        g = next(iter(runner.slots.values())).graph

        # replay vs replay
        restore(s_cap)
        g1 = block.forward(x0, params)
        r1 = snap()
        restore(s_cap)
        g2 = block.forward(x0, params)
        r2 = snap()
        drr = (r1[1].float() - r2[1].float()).abs()
        print(f"replay-vs-replay rec delta max={float(drr.max()):.3e} "
              f"flat={int(drr.argmax())}", flush=True)
        dout = float((g1.float() - g2.float()).abs().max())
        print(f"replay-vs-replay output delta max={dout:.3e}", flush=True)

        # eager vs replay
        restore(s_cap)
        y_e = block.forward(x0.clone(), params)
        r_e = snap()
        restore(s_cap)
        g3 = block.forward(x0, params)
        r_g = snap()
        de = (r_e[1].float() - r_g[1].float()).abs()
        print(f"eager-vs-replay rec delta max={float(de.max()):.3e} "
              f"flat={int(de.argmax())}", flush=True)
        print(f"eager-vs-replay output delta max="
              f"{float((y_e.float() - g3.float()).abs().max()):.3e}", flush=True)


if __name__ == "__main__":
    main()