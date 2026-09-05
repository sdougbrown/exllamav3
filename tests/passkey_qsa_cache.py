"""Real-route QSA quantized-cache passkey qualification.

Runs the deterministic needle-in-a-haystack protocol end to end through the Generator
against every attached QSA quantized cache layer, and fails the run if any attention
step falls back to full-cache dequantization.

Not a pytest module. Run manually once the GPUs are free:

    python tests/passkey_qsa_cache.py
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exllamav3 import Cache, Config, Generator, GreedySampler, Job, Model, Tokenizer
from exllamav3.cache.quant import CacheLayer_quant
from exllamav3.cache.qsa import CacheLayer_qsa_quant

MODEL = Path(os.environ.get(
    "EXL3_FLASH_TEST_MODEL", "~/Models/Qwen3.8-Flash-Next-exl3-bpw3",
)).expanduser()

SEEDS = (17, 29, 43, 57)
LENGTHS = (65536, 131072)
DEPTHS = (5, 25, 50, 75, 95)
MAX_NUM_TOKENS = 139264
MAX_CHUNK_SIZE = 512
GPU_BUDGET = 27.0
MAX_OUTPUT_SIZE = 16

SECRET_WORDS = [
    "the", "of", "and", "to", "in", "is", "that", "it", "for", "as", "with", "was", "his", "he",
    "be", "not", "by", "but", "have", "you", "which", "are", "on", "this", "or", "from", "at",
    "they", "an", "one", "had", "all", "we", "can", "her", "has", "there", "were", "she", "him",
    "when", "time", "more", "if", "no", "out", "so", "said", "what", "up", "its", "about", "into",
    "than", "them", "may", "over", "who", "will", "would", "years", "also", "these", "people",
    "then", "new", "some", "way", "only", "world", "year", "after", "work", "make", "three",
    "life", "down", "before", "back", "get", "day", "use", "man", "great", "very", "through",
    "just", "know", "take", "even", "like", "well", "because", "such", "different", "little",
    "still", "own", "place", "right", "small", "large", "next", "early", "young", "important",
    "few", "public", "bad", "same", "able", "water", "house", "light", "night", "tree", "river",
    "mountain", "city", "story", "letter", "morning", "under", "between", "never", "always",
    "again", "both", "those", "while", "without", "along", "being", "other", "should", "found",
    "where", "each", "many", "part", "come", "long", "here", "must", "does", "made", "could",
    "every", "sound", "number", "follow", "help", "turn", "play", "show", "animal", "sea",
    "plant", "school", "country", "sentence", "cause", "change", "answer", "study", "learn",
    "read", "write", "speak", "listen", "think", "north", "south", "east", "west", "winter",
    "summer", "spring", "autumn", "cold", "warm", "green", "blue", "white", "black", "stone",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description = "QSA quantized-cache passkey qualification (40 probes per mode)",
    )
    parser.add_argument("--model", type = str, default = str(MODEL))
    parser.add_argument("--bits", type = str, default = "8,8", help = "k_bits,v_bits")
    parser.add_argument("--lengths", type = str, default = ",".join(str(l) for l in LENGTHS))
    parser.add_argument("--seeds", type = str, default = ",".join(str(s) for s in SEEDS))
    parser.add_argument("--depths", type = str, default = ",".join(str(d) for d in DEPTHS))
    parser.add_argument("--max-new-tokens", type = int, default = 16)
    return parser.parse_args()


def _int_list(text):
    values = [int(part) for part in text.split(",") if part.strip()]
    assert values, "empty numeric list"
    return values


def _validate_args(lengths):
    for length in lengths:
        assert length <= MAX_NUM_TOKENS, \
            f"length {length} exceeds max cache tokens {MAX_NUM_TOKENS}"


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
        raise SystemExit(f"passkey qualification requires two gfx12 devices, found {devices}")
    return [GPU_BUDGET if index in devices[:2] else 0.0
            for index in range(torch.cuda.device_count())]


def build_prompt_ids(tokenizer, seed: int, target_length: int, depth_pct: int, code: str):
    """Chat-formatted haystack with the secret sentence at the given depth.

    The final prompt goes through the model's chat template with thinking disabled
    and is trimmed/extended so it lands within 32 tokens of the target without
    exceeding it.
    """
    secret = f"The secret code is {code}."
    question = ("What is the six-digit secret code mentioned in the text? "
                "Answer with only the six digits.")

    rng = random.Random(f"{seed}:{target_length}:{depth_pct}")
    filler_words = []
    for _ in range(64):
        filler_words.append(rng.choice(SECRET_WORDS))

    def assemble(num_words):
        words = list(filler_words[:num_words])
        # Extend with more random words when trimming alone cannot reach the floor.
        while len(words) < num_words:
            words.append(rng.choice(SECRET_WORDS))
        filler = " ".join(words) + "."
        insert_at = max(1, min(len(words) - 1, int(len(words) * depth_pct / 100)))
        text = " ".join(words[:insert_at]) + " " + secret + " " + " ".join(words[insert_at:]) + " " + question
        ids = tokenizer.hf_chat_template(
            [{"role": "user", "content": text}],
            add_generation_prompt = True,
            enable_thinking = False,
        )
        return ids

    def chat_len(num_words):
        return len(assemble(num_words)[0].tolist())

    # Grow until we exceed target, then binary-search the largest fitting count.
    num_words = len(filler_words)
    while chat_len(num_words) < target_length:
        next_num_words = num_words * 2
        filler_words.extend(
            rng.choice(SECRET_WORDS) for _ in range(next_num_words - len(filler_words))
        )
        num_words = next_num_words
        if num_words > 1000000:
            raise AssertionError(f"target length {target_length} too large")
    lo, hi = 1, num_words
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if chat_len(mid) <= target_length:
            lo = mid
        else:
            hi = mid - 1
    num_words = lo
    ids = assemble(num_words)
    n = len(ids[0].tolist())
    assert n <= target_length, f"prompt {n} exceeds target {target_length}"
    assert target_length - n <= 32, f"prompt {n} more than 32 tokens below target {target_length}"
    return ids, secret, n


def run_probe(generator, tokenizer, input_ids, max_new_tokens):
    job = Job(
        input_ids = input_ids,
        max_new_tokens = max_new_tokens,
        sampler = GreedySampler(),
    )
    generator.enqueue(job)
    chunks = []
    while generator.num_remaining_jobs():
        for result in generator.iterate():
            if result.get("token_ids") is not None:
                chunks.append(result["token_ids"].cpu())
    assert chunks, "generator produced no tokens"
    tokens = torch.cat(chunks, dim = -1)
    text = tokenizer.decode(tokens)
    if isinstance(text, list):
        text = text[0]
    return text


def main():
    args = parse_args()
    k_bits, v_bits = _int_list(args.bits)
    assert len(_int_list(args.bits)) == 2, "--bits must be exactly two widths"
    for width in (k_bits, v_bits):
        assert 2 <= width <= 8, f"bit width {width} outside 2..8"
    lengths = _int_list(args.lengths)
    seeds = _int_list(args.seeds)
    depths = _int_list(args.depths)
    _validate_args(lengths)

    model_path = Path(args.model).expanduser()
    assert model_path.is_dir(), f"model not found: {model_path}"

    config = Config.from_directory(str(model_path))
    # Disk-backed n-gram streaming, same as the embedded Flash oracle.
    config.infer_params.ngram_stream_from_disk = True

    model = Model.from_config(config, component = "text")
    cache = Cache(
        model,
        max_num_tokens = MAX_NUM_TOKENS,
        layer_type = CacheLayer_quant,
        k_bits = k_bits,
        v_bits = v_bits,
        max_batch_size = 1,
    )

    budgets = _budgets()
    model.load(
        use_per_device = budgets,
        max_chunk_size = MAX_CHUNK_SIZE,
        max_output_size = max(MAX_OUTPUT_SIZE, args.max_new_tokens),
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

    # Guard every attached QSA quantized cache layer: any full-cache dequant fallback
    # during attention now raises and fails the run. Non-QSA layers are left alone.
    guarded = [
        layer for layer in cache.layers.values()
        if isinstance(layer, CacheLayer_qsa_quant)
    ]
    assert len(guarded) > 0, "no QSA quantized cache layers attached; nothing to qualify"
    print(json.dumps({
        "event": "guard",
        "guarded_qsa_quant_layers": len(guarded),
        "total_cache_layers": len(cache.layers),
        "k_bits": k_bits,
        "v_bits": v_bits,
    }))
    originals = [layer.get_kv for layer in guarded]

    def guarded_get_kv(*_args, **_kwargs):
        raise RuntimeError(
            "QSA quantized cache fell back to full-cache dequantization (get_kv) "
            "during the passkey run"
        )

    for layer in guarded:
        layer.get_kv = guarded_get_kv

    try:
        hits_by_length = {length: 0 for length in lengths}
        totals_by_length = {length: 0 for length in lengths}
        total_hits = 0
        total_probes = 0

        for length in lengths:
            for seed in seeds:
                for depth in depths:
                    code = f"{(seed * 7919 + depth * 104729 + length) % 1000000:06d}"
                    input_ids, secret, actual = build_prompt_ids(
                        tokenizer, seed, length, depth, code,
                    )
                    assert actual <= length
                    completion = run_probe(generator, tokenizer, input_ids, args.max_new_tokens)
                    hit = code in completion
                    hits_by_length[length] += int(hit)
                    totals_by_length[length] += 1
                    total_hits += int(hit)
                    total_probes += 1
                    print(json.dumps({
                        "event": "probe",
                        "length_requested": length,
                        "length_actual": actual,
                        "seed": seed,
                        "depth_pct": depth,
                        "code": code,
                        "hit": hit,
                        "completion": completion,
                    }))

        summary = {
            "event": "summary",
            "k_bits": k_bits,
            "v_bits": v_bits,
            "by_length": {
                str(length): {
                    "hits": hits_by_length[length],
                    "total": totals_by_length[length],
                    "recall": hits_by_length[length] / totals_by_length[length],
                }
                for length in lengths
            },
            "overall": {
                "hits": total_hits,
                "total": total_probes,
                "recall": total_hits / total_probes,
            },
        }
        print(json.dumps(summary))
    finally:
        for layer, original in zip(guarded, originals):
            layer.get_kv = original
        model.unload()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()