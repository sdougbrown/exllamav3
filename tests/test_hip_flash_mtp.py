"""Opt-in embedded Qwen3.8-Flash-Next MTP and multirow verification oracle."""
from __future__ import annotations

from collections import Counter
import os
from pathlib import Path

import pytest
import torch

from exllamav3 import Cache, Config, Generator, GreedySampler, Job, Model, Tokenizer
from exllamav3.ext import exllamav3_ext as ext
from exllamav3.modules.block_sparse_mlp import BlockSparseMLP

MODEL = Path(os.environ.get(
    "EXL3_FLASH_TEST_MODEL", "~/Models/Qwen3.8-Flash-Next-exl3"
)).expanduser()
RUN_ORACLE = os.environ.get("EXL3_RUN_FLASH_MTP_ORACLE") == "1"
PROMPT = "The capital of France is"
STEPS = 24


def _gfx12_devices():
    if not (torch.version.hip and torch.cuda.is_available()):
        return []
    return [
        index for index in range(torch.cuda.device_count())
        if getattr(torch.cuda.get_device_properties(index), "gcnArchName", "").split(":", 1)[0]
        in ("gfx1200", "gfx1201")
    ]


def _run(generator, tokenizer, forced_tokens=None):
    job = Job(
        input_ids=tokenizer.encode(PROMPT, add_bos=True),
        max_new_tokens=STEPS + 1,
        stop_conditions=[],
        sampler=GreedySampler(),
    )
    generator.enqueue(job)
    if forced_tokens is not None:
        job.constrain_output_now(forced_tokens.contiguous())
    chunks = []
    accepted = rejected = 0
    while generator.num_remaining_jobs():
        for result in generator.iterate():
            if "token_ids" in result:
                chunks.append(result["token_ids"].cpu())
            accepted = max(accepted, result.get("accepted_draft_tokens", 0))
            rejected = max(rejected, result.get("rejected_draft_tokens", 0))
    tokens = torch.cat(chunks, dim=-1)
    assert tokens.shape == (1, STEPS)
    if job.draft_stats:
        accepted = sum(round_accepted for _position, _window, round_accepted in job.draft_stats)
        rejected = sum(window - round_accepted
                       for _position, window, round_accepted in job.draft_stats)
    return tokens, job, accepted, rejected


@pytest.mark.skipif(not RUN_ORACLE, reason="set EXL3_RUN_FLASH_MTP_ORACLE=1")
@torch.inference_mode()
def test_embedded_mtp_uses_grouped_multirow_verification_on_both_gfx12(monkeypatch):
    devices = _gfx12_devices()
    if len(devices) < 2:
        pytest.skip(f"MTP oracle requires two gfx12 devices, found {devices}")
    if not MODEL.is_dir():
        pytest.skip(f"Flash model not found: {MODEL}")

    config = Config.from_directory(str(MODEL))
    config.infer_params.ngram_stream_from_disk = True
    target = Model.from_config(config, component="text")
    draft = Model.from_config(config, component="mtp")
    target_cache = Cache(
        target, max_num_tokens=512, max_batch_size=1, max_history=3,
    )
    draft_cache = Cache(draft, max_num_tokens=512, max_batch_size=1)
    budgets = [27.0 if index in devices[:2] else 0.0
               for index in range(torch.cuda.device_count())]

    real_grouped = ext.exl3_moe_gfx12_k3
    calls = Counter()

    def grouped_spy(*args, **kwargs):
        calls[(args[0].device.index, args[0].shape[0])] += 1
        return real_grouped(*args, **kwargs)

    try:
        draft.load(
            use_per_device=budgets, max_chunk_size=256,
            max_output_size=4, max_batch_size=1,
        )
        target.load(
            use_per_device=budgets, max_chunk_size=256,
            max_output_size=4, max_batch_size=1,
        )
        tokenizer = Tokenizer.from_config(config)
        target_grouped_layers = sum(
            isinstance(module, BlockSparseMLP) and module.support_hip_grouped
            for module in target
        )
        assert target_grouped_layers > 0

        target_generator = Generator(
            model=target, cache=target_cache, tokenizer=tokenizer,
        )
        reference_tokens, _, _, _ = _run(target_generator, tokenizer)

        ext.exl3_moe_gfx12_k3 = grouped_spy
        free_generator = Generator(
            model=target,
            cache=target_cache,
            tokenizer=tokenizer,
            draft_model=draft,
            draft_cache=draft_cache,
            num_draft_tokens=3,
            record_draft_stats=True,
        )
        free_tokens, free_job, free_accepted, free_rejected = _run(
            free_generator, tokenizer,
        )
        completion = tokenizer.decode(free_tokens)
        if isinstance(completion, list):
            completion = completion[0]
        free_row4_calls = sum(
            count for (_device, rows), count in calls.items() if rows == 4
        )
        assert free_row4_calls % target_grouped_layers == 0
        free_verify_rounds = free_row4_calls // target_grouped_layers
        # One ordinary target step initializes the MTP carry. Every later verification round
        # emits one target token plus its accepted draft prefix.
        inferred_free_accepted = (STEPS - 1) - free_verify_rounds
        assert inferred_free_accepted > 0

        mtp_generator = Generator(
            model=target,
            cache=target_cache,
            tokenizer=tokenizer,
            draft_model=draft,
            draft_cache=draft_cache,
            num_draft_tokens=3,
            record_draft_stats=True,
        )
        mtp_tokens, job, forced_accepted, forced_rejected = _run(
            mtp_generator, tokenizer, reference_tokens,
        )

        assert torch.equal(mtp_tokens, reference_tokens)
        assert forced_rejected > 0
        assert sum(count for (device, rows), count in calls.items() if rows == 4) > 0
        assert sum(count for (device, rows), count in calls.items() if rows == 1) > 0
        assert set(devices[:2]) <= {device for device, _rows in calls}
        print(
            f"MTP forced counters={forced_accepted}/{forced_rejected}; "
            f"free inferred accepted={inferred_free_accepted} "
            f"rounds={free_verify_rounds} routes={calls} "
            f"completion={completion!r}"
        )
        assert completion and "Paris" in completion
    finally:
        ext.exl3_moe_gfx12_k3 = real_grouped
        target.unload()
        draft.unload()
        torch.cuda.empty_cache()
