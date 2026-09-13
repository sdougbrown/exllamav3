"""Coverage for the opt-in per-block HIP graph capture layer."""
from __future__ import annotations

import os
import types
from pathlib import Path

import pytest
import torch

from exllamav3 import Cache, Config, Generator, Model, Tokenizer
from exllamav3.modules import block_graph
from exllamav3.modules.block_graph import BlockGraphRunner, maybe_graph_forward
from exllamav3.modules.block_sparse_mlp import BlockSparseMLP
from exllamav3.modules.gated_delta_net import GatedDeltaNet

MODEL = Path(os.environ.get(
    "EXL3_FLASH_TEST_MODEL", "~/Models/Qwen3.8-Flash-Next-exl3"
)).expanduser()
PROMPT = "The capital of France is"


def _gfx12_available() -> bool:
    if not (torch.version.hip and torch.cuda.is_available()):
        return False
    for index in range(torch.cuda.device_count()):
        arch = getattr(torch.cuda.get_device_properties(index), "gcnArchName", "")
        if arch.split(":", 1)[0] in ("gfx1200", "gfx1201"):
            return True
    return False


def _fake_attn(**attrs):
    attn = GatedDeltaNet.__new__(GatedDeltaNet)
    attn.bc = None
    attn.num_v_heads = 48
    for name, value in attrs.items():
        setattr(attn, name, value)
    return attn


def _fake_mlp(**attrs):
    mlp = BlockSparseMLP.__new__(BlockSparseMLP)
    mlp.support_hip_grouped = True
    mlp.support_hip_prefill = False
    mlp.tp_mode = None
    mlp.tp_reduce = False
    for name, value in attrs.items():
        setattr(mlp, name, value)
    return mlp


def _runner(device: torch.device) -> BlockGraphRunner:
    block = types.SimpleNamespace(
        attn=_fake_attn(), mlp=_fake_mlp(), device=device, layer_idx=0, key="fake",
    )
    return BlockGraphRunner(block)


def _decode_input(device: torch.device, rows: int = 1, hidden: int = 8):
    return torch.zeros(1, rows, 1, hidden, dtype=torch.float32, device=device)


def _eligible_params(bsz: int = 1):
    state = types.SimpleNamespace(exported=False, cache=object())
    return {
        "recurrent_states": [state],
        "recurrent_slots": torch.zeros(bsz, dtype=torch.int32),
        "recurrent_history": False,
        "layer_instance": 0,
    }


def test_disabled_returns_none(monkeypatch):
    monkeypatch.setattr(block_graph, "BLOCK_GRAPH_ENABLED", False)
    block = types.SimpleNamespace(attn=object(), mlp=object())
    assert maybe_graph_forward(block, object(), {}) is None
    assert not hasattr(block, "block_graph_runner")


def test_non_gdn_mlp_block_is_checked_once(monkeypatch):
    monkeypatch.setattr(block_graph, "BLOCK_GRAPH_ENABLED", True)
    block = types.SimpleNamespace(attn=object(), mlp=object())
    assert maybe_graph_forward(block, object(), {}) is None
    assert getattr(block, "_block_graph_checked", False)
    assert not hasattr(block, "block_graph_runner")
    assert maybe_graph_forward(block, object(), {}) is None


@pytest.mark.skipif(not _gfx12_available(), reason="requires a gfx12 device")
class TestDeclines:
    @staticmethod
    def test_call_modes_decline():
        runner = _runner(torch.device("cuda", 0))
        x = _decode_input(torch.device("cuda", 0))
        for params in (
            {"prefill": True},
            {"reconstruct": True},
            {"activate_all_experts": True},
            {"autosplit_measure": True},
            {"quant_preserve": {}},
            {"capture": True},
        ):
            assert runner.slot_key(x, params) is None
        assert set(runner.stats["declines"]) == {"call_mode"}

    @staticmethod
    def test_attn_and_mlp_routes_decline():
        device = torch.device("cuda", 0)
        x = _decode_input(device)
        runner = _runner(device)
        runner.block.attn = object()
        assert runner.slot_key(x, {}) is None
        assert runner.stats["declines"]["attn_type"] == 1
        runner.block.attn = _fake_attn(qsa_indexer=object())
        assert runner.slot_key(x, {}) is None
        assert runner.stats["declines"]["qsa"] == 1
        runner.block.attn = _fake_attn(bc=object())
        assert runner.slot_key(x, {}) is None
        assert runner.stats["declines"]["attn_type"] == 2
        runner.block.attn = _fake_attn()
        runner.block.mlp = _fake_mlp(support_hip_grouped=False, support_hip_prefill=False)
        assert runner.slot_key(x, {}) is None
        assert runner.stats["declines"]["mlp_route"] == 1
        runner.block.mlp = _fake_mlp(tp_mode="col")
        assert runner.slot_key(x, {}) is None
        assert runner.stats["declines"]["tp"] == 1

    @staticmethod
    def test_io_shape_and_filter_declines():
        device = torch.device("cuda", 0)
        runner = _runner(device)
        assert runner.slot_key(torch.zeros(1, 1, 8, device=device), {}) is None
        assert runner.stats["declines"]["io"] == 1
        noncontig = _decode_input(device, rows=4, hidden=16)[..., :8]
        assert runner.slot_key(noncontig, {}) is None
        assert runner.stats["declines"]["io"] == 2
        assert runner.slot_key(_decode_input(device, rows=17), {}) is None
        assert runner.stats["declines"]["rows"] == 1
        runner.block.attn = _fake_attn(num_v_heads=8)
        assert runner.slot_key(_decode_input(device, rows=8), {}) is None
        assert runner.stats["declines"]["chunk_path"] == 1
        assert runner.slot_key(_decode_input(device, rows=7), _eligible_params()) is not None
        runner.block.device = torch.device("cuda", 1)
        assert runner.slot_key(_decode_input(device), {}) is None
        assert runner.stats["declines"]["device"] == 1

    @staticmethod
    def test_device_filter_declines(monkeypatch):
        monkeypatch.setattr(block_graph, "BLOCK_GRAPH_DEVICES", {7})
        runner = _runner(torch.device("cuda", 0))
        assert runner.slot_key(_decode_input(torch.device("cuda", 0)), {}) is None
        assert runner.stats["declines"]["device_filter"] == 1

    @staticmethod
    def test_sync_free_count_off_declines(monkeypatch):
        monkeypatch.setenv("EXL3_MOE_SYNC_FREE_COUNT", "0")
        runner = _runner(torch.device("cuda", 0))
        assert runner.slot_key(_decode_input(torch.device("cuda", 0)), {}) is None
        assert runner.stats["declines"]["sync_free_count_off"] == 1

    @staticmethod
    def test_recurrent_state_declines():
        device = torch.device("cuda", 0)
        runner = _runner(device)
        x = _decode_input(device)
        assert runner.slot_key(x, {}) is None
        assert runner.stats["declines"]["recurrent_state"] == 1
        state = types.SimpleNamespace(exported=True, cache=object())
        assert runner.slot_key(x, {"recurrent_states": [state]}) is None
        assert runner.stats["declines"]["recurrent_state"] == 2
        state.exported = False
        assert runner.slot_key(x, {"recurrent_states": [state]}) is None
        assert runner.stats["declines"]["recurrent_slots"] == 1


@pytest.mark.skipif(not _gfx12_available(), reason="requires a gfx12 device")
def test_eligible_call_produces_slot_key():
    device = torch.device("cuda", 0)
    runner = _runner(device)
    params = _eligible_params()
    key = runner.slot_key(_decode_input(device), params)
    assert key is not None
    assert key[0] == 0 and key[1] == 1 and key[3] is False
    # A second call with the same specialization maps to the same slot.
    assert runner.slot_key(_decode_input(device), params) == key


@pytest.mark.skipif(not _gfx12_available(), reason="requires a gfx12 device")
class TestCaptureReplayMachine:
    """Warmup accounting, capture, replay, and failure paths over a fake block whose
    forward is a capturable in-place double."""

    @staticmethod
    def _fake_runner(device: torch.device) -> BlockGraphRunner:
        runner = _runner(device)

        def forward(x, params, out_dtype=None):
            x.mul_(2.0)
            return x

        runner.block.forward = forward
        return runner

    @staticmethod
    def test_warmup_capture_replay_and_disable():
        device = torch.device("cuda", 0)
        runner = TestCaptureReplayMachine._fake_runner(device)
        params = _eligible_params()
        key = runner.slot_key(_decode_input(device), params)
        # Warmup calls complete logically and return None (caller stays eager).
        for k in range(1, block_graph.BLOCK_GRAPH_WARMUPS + 1):
            assert runner.maybe_forward(_decode_input(device), params) is None
            assert runner.warmups_left[key] == block_graph.BLOCK_GRAPH_WARMUPS - k + 1
        # The call after warmups captures and returns the graphed result.
        x = torch.full((1, 1, 1, 8), 3.0, dtype=torch.float32, device=device)
        y = runner.maybe_forward(x, params)
        assert torch.equal(y, torch.full_like(x, 6.0))
        assert runner.stats["captures"] == 1
        # Replay recomputes from the static buffer, not from a mutated capture input.
        x2 = torch.full((1, 1, 1, 8), 5.0, dtype=torch.float32, device=device)
        y2 = runner.maybe_forward(x2, params)
        assert torch.equal(y2, torch.full_like(x2, 10.0))
        assert runner.stats["replays"] == 2  # capture call commits one replay itself
        # A launch-time replay failure disables the slot and raises; later calls decline.
        runner.slots[key].graph = types.SimpleNamespace(
            replay=lambda: (_ for _ in ()).throw(RuntimeError("boom")))
        with pytest.raises(RuntimeError, match="boom"):
            runner.maybe_forward(x2, params)
        assert key in runner.disabled
        assert runner.maybe_forward(x2, params) is None
        assert runner.stats["declines"]["disabled_slot"] == 1

    @staticmethod
    def test_slot_source_identity_refresh():
        device = torch.device("cuda", 0)
        runner = TestCaptureReplayMachine._fake_runner(device)
        params = _eligible_params()
        for _ in range(block_graph.BLOCK_GRAPH_WARMUPS):
            runner.maybe_forward(_decode_input(device), params)
        x = torch.zeros(1, 1, 1, 8, dtype=torch.float32, device=device)
        runner.maybe_forward(x, params)  # capture + first replay
        slot = next(iter(runner.slots.values()))
        assert slot.slots_src is params["recurrent_slots"]
        # A different source object (job slot reassignment) refreshes the buffer contents.
        params2 = dict(params)
        params2["recurrent_slots"] = torch.ones(1, dtype=torch.int32)
        runner.maybe_forward(x, params2)
        assert slot.slots_src is params2["recurrent_slots"]
        assert slot.slots_dev.cpu().tolist() == [1]
        # Same object again: contents stay as uploaded (no re-copy of the old buffer).
        runner.maybe_forward(x, params2)
        assert slot.slots_dev.cpu().tolist() == [1]

    @staticmethod
    def test_total_capture_cap_declines(monkeypatch):
        device = torch.device("cuda", 0)
        monkeypatch.setattr(block_graph, "BLOCK_GRAPH_MAX_TOTAL", 1)
        monkeypatch.setattr(block_graph, "BLOCK_GRAPH_WARMUPS", 1)
        saved = block_graph._total_captures[0]
        block_graph._total_captures[0] = 1
        try:
            runner = TestCaptureReplayMachine._fake_runner(device)
            params = _eligible_params()
            assert runner.maybe_forward(_decode_input(device), params) is None  # warmup
            assert runner.maybe_forward(_decode_input(device), params) is None  # capped
            assert runner.stats["declines"]["total_cap"] == 1
            assert runner.stats["captures"] == 0
        finally:
            block_graph._total_captures[0] = saved


@pytest.mark.skipif(not (MODEL.is_dir() and _gfx12_available()),
                    reason=f"requires the flash model and a gfx12 device: {MODEL}")
class TestRealModelCapture:
    @staticmethod
    @torch.inference_mode()
    def test_graph_replay_matches_eager_generation(monkeypatch):
        monkeypatch.setattr(block_graph, "BLOCK_GRAPH_ENABLED", False)
        model = Model.from_config(Config.from_directory(str(MODEL)))
        try:
            model.load()
            tokenizer = Tokenizer.from_config(model.config)

            def generate(cache):
                return Generator(model=model, cache=cache, tokenizer=tokenizer).generate(
                    prompt=PROMPT,
                    stop_conditions=[],
                    max_new_tokens=12,
                    completion_only=True,
                    add_bos=True,
                )

            # Eager reference on its own cache.
            reference = generate(Cache(model, max_num_tokens=256, max_batch_size=1))
            assert reference

            monkeypatch.setattr(block_graph, "BLOCK_GRAPH_ENABLED", True)
            graph_cache = Cache(model, max_num_tokens=256, max_batch_size=1)
            first = generate(graph_cache)
            captures_after_first = block_graph.global_stats()["captures"]
            replays_after_first = block_graph.global_stats()["replays"]
            assert captures_after_first > 0, "no decode block was captured"
            assert replays_after_first > 0, "no graph replay ran"
            assert first == reference, "graphed pass diverged from the eager reference"
            # A second pass on the same cache replays the existing slots: same output,
            # no new captures.
            second = generate(graph_cache)
            assert second == first
            assert block_graph.global_stats()["captures"] == captures_after_first
        finally:
            block_graph.purge()
            model.unload()
            torch.cuda.empty_cache()
