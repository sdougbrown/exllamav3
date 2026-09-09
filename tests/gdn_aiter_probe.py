#!/usr/bin/env python3
"""GPU probe for the optional AITER chunked GDN prefill adapter (Stage 3).

On frozen random inputs, batch 1, history disabled, headwise 128x128 GDN,
compares three backends on identical inputs and states:
  * current route: HIP sequence-serial recurrence (ext.cuda_recurrent_gated_delta_rule)
  * AITER adapter (chunk_gated_delta_rule_opt_vk, conversions included)
  * independent torch recurrence reference (torch_recurrent_gated_delta_rule)

Covers lengths 1/5/24/48/127/128/129/512/1024/2048, unsplit (16/48) and TP-shard
(8/24) head counts, nonsymmetric nonzero initial states, sparse slots,
save_state=False, untouched slots, and split-vs-whole prompt equivalence.
Timing at prefill lengths is adapter-inclusive vs the HIP route.

Run from repo root (single GPU):
  HIP_VISIBLE_DEVICES=0 python tests/gdn_aiter_probe.py --out-dir /tmp/gdnprobe
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from exllamav3.ext import exllamav3_ext as ext
from exllamav3.modules.gated_delta_net_fn.gated_delta_rule import (
    torch_recurrent_gated_delta_rule,
)
from exllamav3.modules.gated_delta_net_fn import gdn_aiter_adapter as adapter

LENGTHS = [1, 5, 24, 48, 127, 128, 129, 512, 1024, 2048]
HEAD_CONFIGS = [(16, 48, "unsplit"), (8, 24, "tp_shard")]
RTOL = 2e-2
ATOL = 2e-2


def make_inputs(bsz, seqlen, nk, nv, kd, vd, device, seed=0):
    torch.manual_seed(seed)
    fdim = 2 * nk * kd + nv * vd
    mixed_qkv = torch.randn(bsz, seqlen, fdim, dtype=torch.bfloat16, device=device)
    beta = torch.rand(bsz, seqlen, nv, dtype=torch.bfloat16, device=device)
    # g <= 0, natural-log decay (as produced by -softplus * exp(a_log))
    g = -torch.rand(bsz, seqlen, nv, dtype=torch.float32, device=device) * 0.5
    return mixed_qkv, beta, g


def make_pool(n_slots, nv, kd, vd, device, seed=1):
    torch.manual_seed(seed)
    # nonsymmetric kd != vd would catch transposes; 128x128 is nonsymmetric in
    # VALUES here (state[...,:kd,:vd] full of distinct values)
    return torch.randn(n_slots, 1, nv, kd, vd, dtype=torch.float32, device=device)


def run_hip(mixed_qkv, beta, g, pool, slots, save_state, nk, nv, kd, vd):
    bsz, seqlen, _ = mixed_qkv.shape
    out = torch.empty(
        (bsz, seqlen, nv, vd), dtype=torch.bfloat16, device=mixed_qkv.device
    )
    ext.cuda_recurrent_gated_delta_rule(
        mixed_qkv.contiguous(), g.contiguous(), beta.contiguous(),
        pool, out, nk, nv, kd, vd, slots, False,
    )
    return out, pool.clone() if save_state else None


def run_aiter(mixed_qkv, beta, g, pool, slots, save_state, nk, nv, kd, vd):
    out = adapter.aiter_chunked_gdn_prefill(
        mixed_qkv, beta, g, pool, slots, save_state, nk, nv, kd, vd,
    )
    return out, pool.clone() if save_state else None


def torch_ref_grouped(mixed_qkv, beta, g, pool, slots, save_state, nk, nv, kd, vd):
    """Independent FP32 recurrence reference with grouped heads: value head j
    uses key head j // r (the convention the HIP kernel implements). State and
    output are per value head."""
    bsz, seqlen, _ = mixed_qkv.shape
    r = nv // nk
    k_dim, v_dim = nk * kd, nv * vd
    q, k, v = torch.split(mixed_qkv, [k_dim, k_dim, v_dim], dim=-1)
    q = q.view(bsz, seqlen, nk, kd).repeat_interleave(r, dim=2)
    k = k.view(bsz, seqlen, nk, kd).repeat_interleave(r, dim=2)
    v = v.view(bsz, seqlen, nv, vd)

    def l2norm(x, eps=1e-6):
        return x * torch.rsqrt((x * x).sum(dim=-1, keepdim=True) + eps)

    q, k = l2norm(q.float()), l2norm(k.float())
    v = v.float()
    beta_t = beta.float().unsqueeze(-1)          # [b, s, nv, 1]
    gexp = g.float().exp().unsqueeze(-1)         # [b, s, nv, 1] natural-log decay
    scale = kd ** -0.5
    slot0 = slots.to(torch.long)[0]
    state = pool[slot0, 0].float().clone().unsqueeze(0)  # [b, nv, kd, vd]
    out = torch.empty(bsz, seqlen, nv, vd, dtype=torch.float32, device=v.device)
    for i in range(seqlen):
        kv_mem = (state * k[:, i].unsqueeze(-1)).sum(dim=-2)      # [b, nv, vd]
        gexp_i = gexp[:, i]                                        # [b, nv, 1]
        v_t = v[:, i] - kv_mem * gexp_i                             # [b, nv, vd]
        delta = (k[:, i].unsqueeze(-1) * v_t.unsqueeze(-2)) * beta_t[:, i].unsqueeze(-1)  # [b, nv, kd, vd]
        state = state * gexp_i.unsqueeze(-1) + delta
        out[:, i] = (state * q[:, i].unsqueeze(-1)).sum(dim=-2) * scale
    if save_state:
        pool_final = pool.clone()
        pool_final[slot0, 0] = state[0].to(pool.dtype)
        return out, pool_final
    return out, None


def max_diff(a, b):
    if a is None or b is None:
        return None
    return float((a.float() - b.float()).abs().max())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="/tmp/gdn_aiter_probe")
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dev = torch.device("cuda:0")
    vk_fn = adapter.load_aiter_vk()  # fail clearly before any state mutation
    print(f"AITER VK entry: {vk_fn.__module__}.{vk_fn.__name__}")

    results = []
    for nk, nv, label in HEAD_CONFIGS:
        kd = vd = 128
        n_slots = 8
        for seqlen in LENGTHS:
            mixed_qkv, beta, g = make_inputs(1, seqlen, nk, nv, kd, vd, dev, seed=seqlen)
            # sparse slot: use slot 5 of a pool, so slot indexing is exercised
            slot = torch.tensor([7], dtype=torch.int32, device=dev)
            pool_ref = make_pool(n_slots, nv, kd, vd, dev, seed=100 + seqlen)

            # torch reference
            o_ref, s_ref = torch_ref_grouped(mixed_qkv, beta, g, pool_ref, slot, True, nk, nv, kd, vd)

            # HIP current route (fresh pool copy)
            pool_hip = pool_ref.clone()
            o_hip, s_hip = run_hip(mixed_qkv, beta, g, pool_hip, slot, True, nk, nv, kd, vd)

            # AITER adapter (fresh pool copy)
            pool_ai = pool_ref.clone()
            o_ai, s_ai = run_aiter(mixed_qkv, beta, g, pool_ai, slot, True, nk, nv, kd, vd)

            rec = {
                "heads": label, "seqlen": seqlen,
                "out_hip_vs_ref": max_diff(o_hip, o_ref),
                "out_aiter_vs_ref": max_diff(o_ai, o_ref),
                "out_aiter_vs_hip": max_diff(o_ai, o_hip),
                "state_aiter_vs_ref": max_diff(s_ai, s_ref) if s_ref is not None else None,
                "state_aiter_vs_hip": max_diff(s_ai, s_hip) if s_hip is not None else None,
            }
            results.append(rec)
            print(json.dumps(rec))

            # save_state=False: pool must remain untouched
            pool_keep = pool_ref.clone()
            _o, _s = run_aiter(mixed_qkv, beta, g, pool_keep, slot, False, nk, nv, kd, vd)
            assert torch.equal(pool_keep, pool_ref), "save_state=False must not mutate the pool"

    # split vs whole prompt equivalence (4096 = 8 x 512), unsplit heads:
    # run the adapter chunk-by-chunk, reinjecting AITER's VK final state, and
    # compare against the whole-prompt AITER call and the HIP final state
    from aiter.ops.triton.gated_delta_net import chunk_gated_delta_rule_opt_vk
    nk, nv, _ = HEAD_CONFIGS[0]
    whole, beta, g = make_inputs(1, 4096, nk, nv, 128, 128, dev, seed=4096)
    pool_whole = make_pool(8, nv, 128, 128, dev, seed=999)
    slot_w = torch.tensor([3], dtype=torch.int32, device=dev)

    o_whole, s_whole = run_aiter(whole, beta, g, pool_whole.clone(), slot_w, True, nk, nv, 128, 128)
    _, s_whole_hip = run_hip(whole, beta, g, pool_whole.clone(), slot_w, True, nk, nv, 128, 128)

    outs = []
    vk_state = adapter.gather_states_vk(pool_whole, slot_w)  # first chunk starts from the pool state
    for i in range(0, 4096, 512):
        qs, ks, vs = torch.split(whole[:, i:i + 512], [nk * 128, nk * 128, nv * 128], dim=-1)
        q = qs.reshape(1, 512, nk, 128).contiguous()
        k = ks.reshape(1, 512, nk, 128).contiguous()
        v = vs.reshape(1, 512, nv, 128).contiguous()
        o, final = chunk_gated_delta_rule_opt_vk(
            q=q, k=k, v=v, g=g[:, i:i + 512].contiguous(), beta=beta[:, i:i + 512].contiguous(),
            initial_state=vk_state, output_final_state=True,
            use_qk_l2norm_in_kernel=True, use_exp2=False, state_dtype=torch.float32,
        )
        vk_state = final
        outs.append(o)
    o_split = torch.cat(outs, dim=1)
    split_res = {
        "split_vs_whole_aiter_out": max_diff(o_split, o_whole),
        "split_vs_whole_aiter_state_vs_whole": max_diff(
            vk_state.transpose(-1, -2).contiguous(), s_whole[3, 0]
        ),
        "split_vs_whole_aiter_state_vs_hip": max_diff(
            vk_state.transpose(-1, -2).contiguous(), s_whole_hip[3, 0]
        ),
    }
    results.append({"heads": "unsplit", "seqlen": 4096, **split_res})
    print(json.dumps(split_res, indent=1))

    # timing: HIP vs AITER at prefill lengths, 3 reps after warmup, conversions included
    timing = {}
    for seqlen in (512, 1024, 2048, 4096):
        mixed_qkv, beta, g = make_inputs(1, seqlen, 8, 24, 128, 128, dev, seed=seqlen)
        pool = make_pool(8, 24, 128, 128, dev)
        slot = torch.tensor([5], dtype=torch.int32, device=dev)
        for name, fn in (
            ("hip", lambda: run_hip(mixed_qkv, beta, g, pool, slot, True, 8, 24, 128, 128)),
            ("aiter", lambda: run_aiter(mixed_qkv, beta, g, pool, slot, True, 8, 24, 128, 128)),
        ):
            for _ in range(3):
                fn()  # JIT/warmup excluded
            torch.cuda.synchronize()
            ts = []
            for _ in range(5):
                t0 = time.perf_counter()
                fn()
                torch.cuda.synchronize()
                ts.append(time.perf_counter() - t0)
            timing[f"{name}_s{seqlen}"] = round(statistics.median(ts) * 1e3, 3)
    print(json.dumps(timing, indent=1))
    results.append({"timing_ms": timing})

    (out_dir / "probe-results.json").write_text(json.dumps(results, indent=2, default=str))
    print(f"wrote {out_dir / 'probe-results.json'}")


if __name__ == "__main__":
    main()