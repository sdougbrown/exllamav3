"""Real-route QSA cache quality harness: terminal next-token logits and held-out-token
perplexity through the paged Generator/cache path.

Unlike eval/ppl.py (which calls model.forward directly with cache=False and a hardcoded
flash_attn_nc mode) and eval/model_diff.py (which simulates quantization), this harness
drives every row through Generator + Job with return_logits=True so the measured logits
come from the actual prefill/chunked-paged-cache route, including the QSA sparse online
attention regime for quantized caches.

For quantized cache modes, every attached CacheLayer_qsa_quant.get_kv is monkeypatched to
raise, so any full-cache dequantization fallback during attention aborts the run and each
context proves the sparse online (packed) route.

Not a pytest module. Run manually once the GPUs are free, once per mode/length:

    python tests/qsa_cache_kld.py --cache-mode fp16 --length 2048 --rows 8 --out fp16_2048.pt
    python tests/qsa_cache_kld.py --cache-mode 8,8 --length 2048 --rows 8 --out q8_2048.pt \
        --reference fp16_2048.pt
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pathlib
import sys
import tempfile
import urllib.request
import zipfile

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exllamav3 import Cache, Config, Generator, GreedySampler, Job, Model, Tokenizer
from exllamav3.cache.fp16 import CacheLayer_fp16
from exllamav3.cache.quant import CacheLayer_quant
from exllamav3.cache.qsa import CacheLayer_qsa_quant
from exllamav3.util.measures import compute_kl_div, compute_target_log_probs

MODEL = os.environ.get(
    "EXL3_FLASH_TEST_MODEL", "~/Models/Qwen3.8-Flash-Next-exl3-bpw3",
)
MAX_NUM_TOKENS = 139264
MAX_CHUNK_SIZE = 512
GPU_BUDGET = 27.0
MAX_OUTPUT_SIZE = 16

_WIKITEXT2_URL = "https://huggingface.co/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip"


def parse_args():
    parser = argparse.ArgumentParser(
        description = "Real-path terminal-logit KLD and held-out-token PPL harness",
        allow_abbrev = False,
    )
    parser.add_argument("--model", type = str, default = MODEL, help = "Model directory")
    parser.add_argument(
        "--cache-mode", type = str, default = "8,8",
        help = "'fp16' or quantized K/V bit widths as 'k,v' (e.g. 8,8 / 6,6 / 4,4)",
    )
    parser.add_argument("--length", type = int, default = 2048,
                        help = "Context tokens per row; the following token is the held-out target")
    parser.add_argument("--rows", type = int, default = 8, help = "Number of rows")
    parser.add_argument("--out", type = str, default = None,
                        help = "Save logits/targets/metadata to this .pt file")
    parser.add_argument("--reference", type = str, action = "append", default = [],
                        help = "Saved result file(s) to compare against (repeatable)")
    return parser.parse_args()


def parse_cache_mode(text):
    if text == "fp16":
        return "fp16", None, None
    parts = text.split(",")
    assert len(parts) == 2, "--cache-mode must be 'fp16' or 'k,v' with exactly two widths"
    k_bits, v_bits = (int(part) for part in parts)
    assert 2 <= k_bits <= 8 and 2 <= v_bits <= 8, "bit widths must be within 2..8"
    return "quant", k_bits, v_bits


def validate_args(length, rows, cache_kind, k_bits, v_bits):
    assert rows >= 1, "--rows must be at least 1"
    assert length >= 1, "--length must be at least 1"
    # The generated token also claims a cache slot after prefill.
    assert length + 1 <= MAX_NUM_TOKENS, \
        f"length {length} (+1 generated token) exceeds max cache tokens {MAX_NUM_TOKENS}"


def get_wikitext2_text() -> str:
    """WikiText-2 raw test split, exactly as consumed by eval/ppl.py's wiki2 spec.

    Prefers the `datasets` path used by eval/ppl.py's default eval; falls back to the
    llama.cpp-exact raw archive that eval/ppl.py uses for its GGUF-equivalent mode.
    """
    try:
        from datasets import load_dataset
        return "\n\n".join(
            load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split = "test")["text"]
        )
    except ImportError:
        pass

    cache_dir = pathlib.Path(tempfile.gettempdir()) / "llama_cpp_ppl_wikitext2"
    cache_dir.mkdir(parents = True, exist_ok = True)
    raw_path = cache_dir / "wikitext-2-raw" / "wiki.test.raw"
    if not raw_path.exists():
        zip_path = cache_dir / "wikitext-2-raw-v1.zip"
        if not zip_path.exists():
            print(f"Downloading WikiText-2 raw to {zip_path} ...", file = sys.stderr)
            urllib.request.urlretrieve(_WIKITEXT2_URL, str(zip_path))
        with zipfile.ZipFile(str(zip_path), "r") as zf:
            zf.extractall(str(cache_dir))
        zip_path.unlink(missing_ok = True)
        if not raw_path.exists():
            raise FileNotFoundError(f"Failed to extract to {raw_path}.")
    with open(raw_path, "r", encoding = "utf-8") as f:
        return f.read()


def build_rows(tokenizer, length: int, rows: int):
    """Contiguous non-overlapping rows over the tokenized test split, matching eval/ppl.py's
    row selection (stride == length). Row i: context = tokens[i*length, (i+1)*length),
    held-out target = token (i+1)*length."""
    text = get_wikitext2_text()
    tokens = tokenizer.encode(text)
    num_tokens = tokens.shape[-1]
    needed = rows * length + 1
    assert num_tokens >= needed, (
        f"wikitext-2 test split tokenizes to {num_tokens} tokens, "
        f"but {rows} rows of length {length} need {needed}; "
        f"reduce --rows or --length"
    )
    out = []
    for i in range(rows):
        start = i * length
        context = tokens[:, start:start + length]
        target = int(tokens[0, start + length].item())
        out.append((context, target))
    return out


def _gfx12_devices():
    if not (torch.version.hip and torch.cuda.is_available()):
        return []
    return [
        index for index in range(torch.cuda.device_count())
        if getattr(torch.cuda.get_device_properties(index), "gcnArchName", "").split(":", 1)[0]
        in ("gfx1200", "gfx1201")
    ]


def _budgets():
    devices = _gfx12_devices()
    if len(devices) < 2:
        raise SystemExit(f"qsa cache quality harness requires two gfx12 devices, found {devices}")
    return [GPU_BUDGET if index in devices[:2] else 0.0
            for index in range(torch.cuda.device_count())]


def install_qsa_guard(cache):
    """Raise on any full-cache dequantization (get_kv) fallback on attached QSA quant
    layers, proving every context stays on the sparse online packed route. Returns the
    patched layers and their originals for restoration."""
    guarded = [
        layer for layer in cache.layers.values()
        if isinstance(layer, CacheLayer_qsa_quant)
    ]

    def guarded_get_kv(*_args, **_kwargs):
        raise RuntimeError(
            "QSA quantized cache fell back to full-cache dequantization (get_kv) "
            "during the cache quality run"
        )

    originals = [layer.get_kv for layer in guarded]
    for layer in guarded:
        layer.get_kv = guarded_get_kv
    return guarded, originals


def restore_qsa_guard(guarded, originals):
    for layer, original in zip(guarded, originals):
        layer.get_kv = original


def collect_terminal_logits(generator, context_ids):
    """One-token greedy Job through the Generator; returns CPU fp32 terminal next-token
    logits, the sampled token id, and prefill elapsed time if reported."""
    job = Job(
        input_ids = context_ids,
        max_new_tokens = 1,
        sampler = GreedySampler(),
        return_logits = True,
    )
    generator.enqueue(job)
    logits = None
    sampled_id = None
    time_prefill = None
    while generator.num_remaining_jobs():
        for result in generator.iterate():
            if "logits" in result:
                # Move the one-token vector off the GPU immediately; do not retain it.
                logits = result["logits"].to(device = "cpu", dtype = torch.float32).clone()
            if sampled_id is None and result.get("token_ids") is not None:
                sampled_id = int(result["token_ids"].reshape(-1)[-1].item())
            if "time_prefill" in result:
                time_prefill = result["time_prefill"]
    if logits is None:
        raise RuntimeError("generator produced no logits")
    if sampled_id is None:
        sampled_id = int(torch.argmax(logits[0], dim = -1).item())
    # One forward pass without a draft model yields exactly one pre-sampling logit row.
    assert logits.shape[0] == 1, f"unexpected logit batch shape {tuple(logits.shape)}"
    assert logits.shape[1] == 1, (
        f"expected exactly one pre-sampling logit row for max_new_tokens=1, "
        f"got {logits.shape[1]}; a draft/verification path seems active"
    )
    return logits[0, 0], sampled_id, time_prefill, job


def main():
    args = parse_args()
    cache_kind, k_bits, v_bits = parse_cache_mode(args.cache_mode)
    validate_args(args.length, args.rows, cache_kind, k_bits, v_bits)

    model_path = pathlib.Path(args.model).expanduser()
    assert model_path.is_dir(), f"model not found: {model_path}"

    config = Config.from_directory(str(model_path))
    # Disk-backed n-gram streaming, same as the embedded Flash oracle / passkey harness.
    config.infer_params.ngram_stream_from_disk = True

    model = Model.from_config(config, component = "text")
    if cache_kind == "fp16":
        cache = Cache(
            model,
            max_num_tokens = MAX_NUM_TOKENS,
            layer_type = CacheLayer_fp16,
            max_batch_size = 1,
        )
        cache_mode_label = "fp16"
    else:
        cache = Cache(
            model,
            max_num_tokens = MAX_NUM_TOKENS,
            layer_type = CacheLayer_quant,
            k_bits = k_bits,
            v_bits = v_bits,
            max_batch_size = 1,
        )
        cache_mode_label = f"{k_bits},{v_bits}"

    budgets = _budgets()
    model.load(
        use_per_device = budgets,
        max_chunk_size = MAX_CHUNK_SIZE,
        max_output_size = MAX_OUTPUT_SIZE,
        max_batch_size = 1,
    )

    tokenizer = Tokenizer.from_config(config)
    generator = Generator(
        model = model,
        cache = cache,
        tokenizer = tokenizer,
        max_chunk_size = MAX_CHUNK_SIZE,
        max_batch_size = 1,
    )

    guarded, originals = ([], []) if cache_kind == "fp16" else install_qsa_guard(cache)
    if cache_kind != "fp16":
        assert guarded, "no QSA quantized cache layers attached; nothing to qualify"
    print(json.dumps({
        "event": "guard",
        "cache_mode": cache_mode_label,
        "guarded_qsa_quant_layers": len(guarded),
        "total_cache_layers": len(cache.layers),
        "sparse_regime_expected": args.length > 2051,
    }))

    rows = build_rows(tokenizer, args.length, args.rows)
    actual_vocab_size = tokenizer.actual_vocab_size
    padded_vocab_size = generator.padded_vocab_size

    saved_rows = []
    nll_sum = 0.0
    try:
        for index, (context, target) in enumerate(rows):
            logits, sampled_id, time_prefill, job = collect_terminal_logits(generator, context)
            target_log_prob = float(compute_target_log_probs(
                logits.view(1, -1), torch.tensor([target]), actual_vocab_size,
            ).item())
            target_nll = -target_log_prob
            nll_sum += target_nll

            input_sha = hashlib.sha256(
                context.to(torch.int64).numpy().tobytes()
            ).hexdigest()

            print(json.dumps({
                "event": "row",
                "row": index,
                "cache_mode": cache_mode_label,
                "length": int(context.shape[-1]),
                "target_id": target,
                "greedy_sampled_id": sampled_id,
                "target_log_prob": target_log_prob,
                "target_nll": target_nll,
                "time_prefill": time_prefill,
                "input_sha256": input_sha,
            }))

            saved_rows.append({
                "row": index,
                "length": int(context.shape[-1]),
                "target_id": target,
                "input_sha256": input_sha,
                "target_log_prob": target_log_prob,
                "time_prefill": time_prefill,
                "logits": logits,
            })
            del logits, job

        mean_nll = nll_sum / len(rows)
        summary = {
            "event": "summary",
            "cache_mode": cache_mode_label,
            "length": args.length,
            "rows": len(rows),
            "mean_target_nll": mean_nll,
            "heldout_ppl": math.exp(mean_nll),
        }
        summary.update(compare_references(args.reference, saved_rows, actual_vocab_size))
        print(json.dumps(summary))
    finally:
        restore_qsa_guard(guarded, originals)
        model.unload()
        torch.cuda.empty_cache()

    if args.out:
        payload = {
            "format": "qsa_cache_kld_v1",
            "meta": {
                "model": str(model_path),
                "cache_mode": cache_mode_label,
                "cache_kind": cache_kind,
                "k_bits": k_bits,
                "v_bits": v_bits,
                "length": args.length,
                "rows": len(saved_rows),
                "actual_vocab_size": actual_vocab_size,
                "padded_vocab_size": padded_vocab_size,
                "dataset": "wikitext-2-raw-v1:test",
            },
            "rows": saved_rows,
        }
        torch.save(payload, args.out)
        print(json.dumps({"event": "saved", "path": args.out}))


def compare_references(reference_paths, saved_rows, actual_vocab_size):
    """Verify each reference saw identical inputs/targets, then report mean KL in both
    directions between the saved terminal logits and this run's actual vocabulary."""
    if not reference_paths:
        return {}

    current = torch.stack([entry["logits"] for entry in saved_rows], dim = 0)
    results = {}
    for path in reference_paths:
        payload = torch.load(path, weights_only = False)
        ref_rows = payload["rows"]
        ref_meta = payload.get("meta", {})
        assert len(ref_rows) == len(saved_rows), (
            f"reference {path} has {len(ref_rows)} rows, current run has {len(saved_rows)}"
        )
        for ref_entry, cur_entry in zip(ref_rows, saved_rows):
            assert ref_entry["target_id"] == cur_entry["target_id"], (
                f"reference {path} row {cur_entry['row']}: target id mismatch "
                f"({ref_entry['target_id']} vs {cur_entry['target_id']})"
            )
            assert ref_entry["input_sha256"] == cur_entry["input_sha256"], (
                f"reference {path} row {cur_entry['row']}: input token hash mismatch"
            )
            assert ref_entry["length"] == cur_entry["length"], (
                f"reference {path} row {cur_entry['row']}: length mismatch"
            )
        reference = torch.stack([entry["logits"] for entry in ref_rows], dim = 0)
        assert reference.shape == current.shape, (
            f"reference {path} logits shape {tuple(reference.shape)} != "
            f"current {tuple(current.shape)}"
        )
        reference_vocab_size = ref_meta.get("actual_vocab_size")
        assert reference_vocab_size == actual_vocab_size, (
            f"reference {path} actual vocab {reference_vocab_size} != current {actual_vocab_size}"
        )
        vocab_size = actual_vocab_size
        # compute_kl_div(input, target) = KL(softmax(target) || softmax(input)) per row.
        kl_ref_to_cur = compute_kl_div(current, reference, vocab_size)
        kl_cur_to_ref = compute_kl_div(reference, current, vocab_size)
        results[f"mean_kl_ref_to_cur[{path}]"] = float(kl_ref_to_cur.mean().item())
        results[f"mean_kl_cur_to_ref[{path}]"] = float(kl_cur_to_ref.mean().item())
        results.setdefault("reference_meta", []).append({
            "path": path,
            "cache_mode": ref_meta.get("cache_mode"),
            "length": ref_meta.get("length"),
        })
    return results


if __name__ == "__main__":
    main()