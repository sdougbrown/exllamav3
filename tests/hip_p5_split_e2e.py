# P5 T3: end-to-end split-vs-unsplit oracle on the real model (Qwen3.8-Flash-Next).
# Phase A: unsplit baseline (all experts GPU-resident). Phase B: static CPU split
# (EXL3_MOE_CPU_SPLIT=k, text component only, MTP head excluded by the T-gate).
# Both run the same fixed prompt through the production Generator (MTP3, greedy, 16 steps),
# capturing full logical-vocab logits per forward call. Gate: per-step max logit delta within
# the framework envelope (rows2048 precedent: 6.0) and token identity, with the CPU kernel's
# int8-activation approximation folded in.
# Run: ~/vllm-test-env/bin/python tests/hip_p5_split_e2e.py phaseA|phaseB|compare [OUT_DIR]
from __future__ import annotations
import json
import os
import sys
from pathlib import Path

MODE = sys.argv[1] if len(sys.argv) > 1 else "compare"
sys.path.insert(0, str(Path(__file__).parent.parent))
OUT = Path(sys.argv[2] if len(sys.argv) > 2 else "/tmp/p5-split-e2e")
OUT.mkdir(parents=True, exist_ok=True)

LOGICAL_VOCAB = 248077
STEPS = 16
PROMPT_TOKENS = 512


def run_phase(name: str):
    """Load the model with the current env (phase B sets the split env before import)."""
    import torch
    torch.cuda.init()
    for d in (0, 1):
        torch.cuda.set_device(d)
        torch.zeros(1, device=f"cuda:{d}")
    from exllamav3 import Cache, Config, Generator, GreedySampler, Job, Model, Tokenizer
    from exllamav3.cache import CacheLayer_quant
    config = Config.from_directory(str(Path.home() / "Models/Qwen3.8-Flash-Next-exl3-bpw3"))
    config.infer_params.ngram_stream_from_disk = True
    os.environ.setdefault("EXL3_MOE_SYNC_FREE_COUNT", "1")
    os.environ.setdefault("EXL3_QC_STAGING", "1")
    target = Model.from_config(config, component="text")
    draft = Model.from_config(config, component="mtp")
    cache = Cache(target, max_num_tokens=393216, layer_type=CacheLayer_quant,
                  k_bits=8, v_bits=8, max_batch_size=1, max_history=3)
    draft_cache = Cache(draft, max_num_tokens=393216, max_batch_size=1)
    draft.load(use_per_device=[0.0, 2.0], max_chunk_size=256, max_output_size=4,
               max_batch_size=1, verbose=False)
    target.load(use_per_device=[30.0, 30.0], max_chunk_size=512, max_output_size=32,
                max_batch_size=1, verbose=False)
    tokenizer = Tokenizer.from_config(config)
    gen = Generator(model=target, cache=cache, tokenizer=tokenizer, draft_model=draft,
                    draft_cache=draft_cache, num_draft_tokens=3,
                    max_batch_size=1, max_chunk_size=512,
                    recurrent_cache_size=256 * 1024 ** 2, cpu_cache_size=0)

    para = ("The history of the number zero begins in ancient Mesopotamia, where scribes "
        "used a placeholder in positional systems. Brahmagupta formalized zero as a number "
        "in 628 CE, defining rules for arithmetic with zero and negative quantities. Centuries "
        "later, Al-Khwarizmi's algebra spread these ideas through the Islamic world, and "
        "Fibonacci carried them to Europe, where merchants adopted Hindu-Arabic numerals for "
        "their efficiency in commerce. The printing press standardized notation; Descartes "
        "linked algebra with geometry; and by the seventeenth century zero anchored infinitesimal "
        "calculus, logarithms, and the binary systems that much later powered digital computing. "
        "Each era reinterpreted what looked like nothing at all: a placeholder, a quantity, a "
        "limit, a truth value, a voltage. Mathematics advanced whenever notation kept pace with "
        "concept, and the story of zero shows that abstract symbols can carry enormous practical "
        "weight when the surrounding culture is ready to use them. ") * 8
    enc = tokenizer.encode(para)
    if hasattr(enc, "shape"):
        ids = enc[:, :PROMPT_TOKENS] if enc.dim() == 2 else enc[:PROMPT_TOKENS].unsqueeze(0)
    else:
        if isinstance(enc[0], list):
            enc = enc[0]
        ids = torch.tensor([list(enc)[:PROMPT_TOKENS]], dtype=torch.long)
    job = Job(input_ids=ids, max_new_tokens=STEPS, stop_conditions=[], sampler=GreedySampler())
    gen.enqueue(job)
    steps, tokens = [], []
    step_count = 0

    orig_forward = target.forward
    def wrapped(input_ids, params=None):
        nonlocal step_count
        y = orig_forward(input_ids, params)
        if torch.is_tensor(y) and y.dim() >= 2 and step_count < STEPS:
            steps.append(y.detach()[..., :LOGICAL_VOCAB].float().cpu())
            step_count += 1
        return y
    target.forward = wrapped

    while gen.num_remaining_jobs():
        for _ in gen.iterate():
            pass
    seq = job.sequences[0]
    ids_out = seq.sequence_ids.torch_slice(0, None).tolist()
    tokens = ids_out[-STEPS:]
    torch.save({"logits": steps, "tokens": tokens,
                "prompt_len": PROMPT_TOKENS}, OUT / f"phase-{name}.pt")
    summary = {"phase": name, "n_steps": len(steps), "tokens": tokens,
               "logits_norm": [float(s.norm()) for s in steps[:4]]}
    (OUT / f"phase-{name}.json").write_text(json.dumps(summary, indent=1))
    print(f"[{name}] steps={len(steps)} tokens={tokens}")
    print(f"[{name}] logits norms: {[round(v,2) for v in summary['logits_norm']]}")


def compare():
    import torch
    a = torch.load(OUT / "phase-phaseA.pt", weights_only=False)
    b = torch.load(OUT / "phase-phaseB.pt", weights_only=False)
    la, lb = a["logits"], b["logits"]
    n = min(len(la), len(lb))
    deltas = []
    for i in range(n):
        d = float((la[i] - lb[i]).abs().max())
        deltas.append(d)
        am_a = int(la[i].argmax()); am_b = int(lb[i].argmax())
        print(f"step {i}: max|Δlogit| = {d:.4f}  argmax {am_a} vs {am_b} "
              f"{'SAME' if am_a == am_b else 'DIFF'}")
    tokens_match = a["tokens"] == b["tokens"]
    max_delta = max(deltas) if deltas else None
    envelope = 6.0
    gate = tokens_match and max_delta is not None and max_delta <= envelope
    out = {"n_steps": n, "max_logit_delta": max_delta, "tokens_match": tokens_match,
           "envelope": envelope, "gate_pass": gate,
           "note": "envelope = rows2048 oracle precedent (framework); CPU int8-activation "
                   "approximation of the CPU-resident tail folds in here"}
    (OUT / "t3-compare.json").write_text(json.dumps(out, indent=1))
    print(f"\nT3 split-vs-unsplit: max|Δ|={max_delta:.4f} tokens_match={tokens_match} "
          f"{'PASS' if gate else 'FAIL'}")
    return 0 if gate else 1


if __name__ == "__main__":
    if MODE == "compare":
        sys.exit(compare())
    run_phase(MODE)