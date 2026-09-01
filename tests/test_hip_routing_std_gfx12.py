"""gfx12 wave32 batch-one standard softmax MoE router."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import os

import pytest
import torch

if not (torch.version.hip and torch.cuda.is_available()):
    pytest.skip("ROCm router tests", allow_module_level = True)

from exllamav3 import Config, Model
from exllamav3.ext import exllamav3_ext as ext
from exllamav3.modules import block_sparse_mlp as bsm
from exllamav3.modules.block_sparse_mlp import BlockSparseMLP

HIDDEN = 2560
EXPERTS = 512
TOP_K = 10
GFX12_DEVICES = [
    index for index in range(torch.cuda.device_count())
    if getattr(torch.cuda.get_device_properties(index), "gcnArchName", "").split(":", 1)[0]
    in ("gfx1200", "gfx1201")
]
MODEL = Path(os.environ.get(
    "EXL3_FLASH_TEST_MODEL", "~/Models/Qwen3.8-Flash-Next-exl3"
)).expanduser()


def _require_gfx12(device_index):
    if device_index not in GFX12_DEVICES:
        pytest.skip(f"device {device_index} is not gfx12")
    assert getattr(torch.cuda.get_device_properties(device_index), "warp_size", 0) == 32
    assert hasattr(ext, "routing_std_gfx12_bsz1")


def _case(device_index=0, seed=1201):
    _require_gfx12(device_index)
    device = torch.device("cuda", device_index)
    generator = torch.Generator(device=device).manual_seed(seed)
    hidden = torch.randn((1, HIDDEN), generator=generator, device=device, dtype=torch.half) * 0.02
    gate = torch.randn((HIDDEN, EXPERTS), generator=generator, device=device, dtype=torch.half) * 0.02
    return hidden.contiguous(), gate.contiguous()


def _buffers(device):
    return (
        torch.empty((1, EXPERTS), device=device, dtype=torch.half),
        torch.empty((1, TOP_K), device=device, dtype=torch.long),
        torch.empty((1, TOP_K), device=device, dtype=torch.half),
    )


def _native(hidden, gate_t, buffers=None):
    scores, selected, weights = buffers or _buffers(hidden.device)
    ext.routing_std_gfx12_bsz1(hidden, gate_t, scores, selected, weights)
    return scores, selected, weights


def _oracle(hidden, gate):
    logits = torch.matmul(hidden, gate).float()
    values, selected = torch.topk(logits, TOP_K, dim=-1)
    return logits.half(), selected, torch.softmax(values, dim=-1).half()


@pytest.mark.parametrize("device_index", GFX12_DEVICES[:2] or [0])
@torch.inference_mode()
def test_native_router_matches_independent_torch_route(device_index):
    hidden, gate = _case(device_index)
    actual_scores, actual_selected, actual_weights = _native(hidden, gate.T.contiguous())
    expected_scores, expected_selected, expected_weights = _oracle(hidden, gate)
    torch.cuda.synchronize(device_index)

    torch.testing.assert_close(actual_scores, expected_scores, rtol=2e-3, atol=2e-3)
    assert torch.equal(actual_selected, expected_selected)
    torch.testing.assert_close(actual_weights, expected_weights, rtol=2e-3, atol=2e-3)


@torch.inference_mode()
def test_boundary_values_ties_and_repeated_launches_are_deterministic():
    _require_gfx12(0)
    device = torch.device("cuda", 0)
    hidden = torch.zeros((1, HIDDEN), device=device, dtype=torch.half)
    hidden[0, 0] = 1
    logits = torch.linspace(-1, 1, EXPERTS, device=device, dtype=torch.half)
    # Exact ties inside and immediately outside the selected boundary, plus one-half-ULP
    # neighbors. Tied IDs are unspecified by torch.topk, but ordering by value is required.
    logits[-8:] = 2
    logits[-12:-8] = torch.tensor(
        [1.5, 1.5009765625, 1.501953125, 1.5029296875], device=device, dtype=torch.half)
    gate_t = torch.zeros((EXPERTS, HIDDEN), device=device, dtype=torch.half)
    gate_t[:, 0] = logits

    outputs = []
    for _ in range(8):
        scores, selected, weights = _native(hidden, gate_t)
        outputs.append((selected.clone(), weights.clone()))
    torch.cuda.synchronize()

    for selected, weights in outputs[1:]:
        assert torch.equal(selected, outputs[0][0])
        assert torch.equal(weights, outputs[0][1])
    selected, weights = outputs[0]
    selected_logits = logits[selected[0]].float()
    assert torch.all(selected_logits[:-1] >= selected_logits[1:])
    assert selected.unique().numel() == TOP_K
    expected_weights = torch.softmax(selected_logits, dim=0).half()
    torch.testing.assert_close(weights[0], expected_weights, rtol=0, atol=0)


@torch.inference_mode()
def test_native_router_uses_current_custom_stream():
    hidden, gate = _case(0, seed=77)
    gate_t = gate.T.contiguous()
    stream = torch.cuda.Stream(device=0)
    with torch.cuda.stream(stream):
        hidden.fill_(0.03125)
        scores, selected, weights = _native(hidden, gate_t)
        done = torch.cuda.Event()
        done.record(stream)
    done.synchronize()
    expected_scores, expected_selected, expected_weights = _oracle(hidden, gate)
    torch.testing.assert_close(scores, expected_scores, rtol=2e-3, atol=2e-3)
    assert torch.equal(selected, expected_selected)
    torch.testing.assert_close(weights, expected_weights, rtol=2e-3, atol=2e-3)


def _offset_view(shape, dtype, device):
    count = torch.Size(shape).numel()
    storage = torch.empty(count + 16, dtype=dtype, device=device)
    offset = 1 if dtype != torch.long else 1
    return storage[offset:offset + count].view(shape)


@pytest.mark.parametrize("bad", [
    "hidden-shape", "hidden-dtype", "hidden-noncontiguous", "hidden-alignment",
    "gate-shape", "gate-dtype", "gate-noncontiguous", "gate-alignment",
    "scores-shape", "selected-dtype", "weights-shape",
])
@torch.inference_mode()
def test_native_router_rejects_invalid_inputs(bad):
    hidden, gate = _case(0)
    gate_t = gate.T.contiguous()
    scores, selected, weights = _buffers(hidden.device)
    if bad == "hidden-shape": hidden = hidden[:, :-1]
    elif bad == "hidden-dtype": hidden = hidden.float()
    elif bad == "hidden-noncontiguous": hidden = torch.empty((HIDDEN, 2), device=hidden.device, dtype=torch.half)[:, 0].view(1, -1)
    elif bad == "hidden-alignment": hidden = _offset_view((1, HIDDEN), torch.half, hidden.device)
    elif bad == "gate-shape": gate_t = gate_t[:, :-1]
    elif bad == "gate-dtype": gate_t = gate_t.float()
    elif bad == "gate-noncontiguous": gate_t = gate
    elif bad == "gate-alignment": gate_t = _offset_view((EXPERTS, HIDDEN), torch.half, hidden.device)
    elif bad == "scores-shape": scores = scores[:, :-1]
    elif bad == "selected-dtype": selected = selected.int()
    elif bad == "weights-shape": weights = weights[:, :-1]
    with pytest.raises(RuntimeError, match="routing_std_gfx12_bsz1"):
        ext.routing_std_gfx12_bsz1(hidden, gate_t, scores, selected, weights)


@torch.inference_mode()
def test_python_route_executes_native_and_profiles_without_aten_mm_or_topk(monkeypatch):
    hidden, gate = _case(0)
    scores, selected, weights = _buffers(hidden.device)
    cfg = SimpleNamespace(
        gate_tensor=gate, gate_tensor_t=gate.T.contiguous(), router_bias=None,
        num_experts=EXPERTS, num_experts_per_tok=TOP_K,
        router_logits_bsz1=scores, selected_experts_bsz1=selected,
        routing_weights_bsz1=weights, per_expert_scale=None,
    )
    calls = {"native": 0}
    real_native = ext.routing_std_gfx12_bsz1

    def spy(*args, **kwargs):
        calls["native"] += 1
        return real_native(*args, **kwargs)

    monkeypatch.setattr(ext, "routing_std_gfx12_bsz1", spy)
    with torch.profiler.profile(activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ]) as profile:
        actual_selected, actual_weights = bsm.routing_std(1, cfg, hidden, {})
    torch.cuda.synchronize()
    _, expected_selected, expected_weights = _oracle(hidden, gate)
    assert calls["native"] == 1
    assert torch.equal(actual_selected, expected_selected)
    torch.testing.assert_close(actual_weights, expected_weights, rtol=2e-3, atol=2e-3)
    names = [event.name for event in profile.events()]
    assert "aten::mm" not in names
    assert "aten::topk" not in names
    assert any("routing_gemv_gfx12_bsz1_kernel" in name for name in names)
    assert any("routing_std_topk_gfx12_bsz1_kernel" in name for name in names)


@pytest.mark.parametrize("mode", ["disabled", "batch", "activate-all", "scale", "bias"])
@torch.inference_mode()
def test_python_route_falls_back_for_unsupported_modes(mode, monkeypatch):
    hidden, gate = _case(0)
    bsz, params = 1, {}
    if mode == "batch":
        bsz = 2
        hidden = hidden.expand(2, -1).contiguous()
    elif mode == "activate-all": params["activate_all_experts"] = True
    elif mode == "disabled": monkeypatch.setenv("EXL3_HIP_ROUTER", "0")
    scale = torch.ones(EXPERTS, device=hidden.device, dtype=torch.bfloat16) if mode == "scale" else None
    bias = torch.zeros(EXPERTS, device=hidden.device, dtype=torch.half) if mode == "bias" else None
    cfg = SimpleNamespace(
        gate_tensor=gate, gate_tensor_t=None, router_bias=bias,
        num_experts=EXPERTS, num_experts_per_tok=TOP_K,
        router_logits_bsz1=torch.empty((1, EXPERTS), device=hidden.device, dtype=torch.half),
        selected_experts_bsz1=torch.empty((1, TOP_K), device=hidden.device, dtype=torch.long),
        routing_weights_bsz1=torch.empty((1, TOP_K), device=hidden.device, dtype=torch.half),
        per_expert_scale=scale,
    )
    calls = {"native": 0}
    real_native = ext.routing_std_gfx12_bsz1

    def spy(*args, **kwargs):
        calls["native"] += 1
        return real_native(*args, **kwargs)

    monkeypatch.setattr(ext, "routing_std_gfx12_bsz1", spy)
    selected, weights = bsm.routing_std(bsz, cfg, hidden, params)
    assert calls["native"] == 0
    if mode == "disabled":
        assert cfg.gate_tensor_t is None
    expected_logits = (hidden @ gate).float()
    if params.get("activate_all_experts"):
        expected_selected = torch.arange(EXPERTS, device=hidden.device).expand(bsz, -1)
        expected_weights = torch.softmax(expected_logits, dim=-1)
    else:
        values, expected_selected = torch.topk(expected_logits, TOP_K, dim=-1)
        expected_weights = torch.softmax(values, dim=-1)
    if scale is not None:
        expected_weights *= scale.float()[expected_selected]
    assert torch.equal(selected, expected_selected)
    torch.testing.assert_close(weights, expected_weights.half(), rtol=0, atol=0)


@torch.inference_mode()
def test_bias_route_and_missing_binding_do_not_enter_native(monkeypatch):
    hidden, gate = _case(0)
    bias = torch.linspace(-0.1, 0.1, EXPERTS, device=hidden.device, dtype=torch.half)
    cfg = SimpleNamespace(
        gate_tensor=gate, gate_tensor_t=None, router_bias=bias,
        num_experts=EXPERTS, num_experts_per_tok=TOP_K,
        router_logits_bsz1=torch.empty((1, EXPERTS), device=hidden.device, dtype=torch.half),
        selected_experts_bsz1=torch.empty((1, TOP_K), device=hidden.device, dtype=torch.long),
        routing_weights_bsz1=torch.empty((1, TOP_K), device=hidden.device, dtype=torch.half),
        per_expert_scale=None,
    )
    monkeypatch.setattr(bsm, "ext", SimpleNamespace())
    selected, weights = bsm.routing_std_bias(1, cfg, hidden, {})
    values, expected_selected = torch.topk((hidden @ gate).float() + bias.float(), TOP_K, dim=-1)
    assert torch.equal(selected, expected_selected)
    torch.testing.assert_close(weights, torch.softmax(values, dim=-1).half(), rtol=0, atol=0)


@pytest.mark.parametrize("device_index", GFX12_DEVICES[:2] or [0])
@torch.inference_mode()
def test_every_flash_router_real_weights_match_torch_on_random_states(device_index):
    _require_gfx12(device_index)
    if not MODEL.is_dir():
        pytest.skip(f"Flash model not found: {MODEL}")
    model = Model.from_config(Config.from_directory(str(MODEL)))
    routers = [module for module in model if isinstance(module, BlockSparseMLP)]
    assert routers
    device = torch.device("cuda", device_index)
    checked = 0
    try:
        for layer_index, mlp in enumerate(routers):
            mlp.load(device=device)
            try:
                cfg = mlp.routing_cfg
                if not bsm._hip_router_config_supported(cfg):
                    continue
                assert cfg.gate_tensor_t is not None
                assert cfg.gate_tensor_t.shape == (EXPERTS, HIDDEN)
                assert cfg.gate_tensor_t.untyped_storage().nbytes() == EXPERTS * HIDDEN * 2
                transpose_ptr = cfg.gate_tensor_t.data_ptr()
                bsm._prepare_hip_router_gate_t(cfg)
                assert cfg.gate_tensor_t.data_ptr() == transpose_ptr
                generator = torch.Generator(device=device).manual_seed(1201 + layer_index)
                for scale in (0.02, 0.25, 1.0):
                    hidden = torch.randn(
                        (1, HIDDEN), generator=generator, device=device, dtype=torch.half
                    ) * scale
                    logits, expected_selected, _ = _oracle(hidden, cfg.gate_tensor)
                    actual_selected, actual_weights = bsm.routing_std(1, cfg, hidden, {})
                    expected_values = logits.float().gather(1, expected_selected)
                    actual_values = logits.float().gather(1, actual_selected)
                    # fp16 router logits can tie at a rank boundary. torch.topk does not
                    # specify an ID order for equal keys, so compare the ordered score
                    # multiset and bind each weight to the ID the native route returned.
                    assert torch.equal(
                        torch.sort(actual_values, descending=True).values,
                        torch.sort(expected_values, descending=True).values,
                    ), mlp.key
                    expected_weights = torch.softmax(actual_values, dim=-1).half()
                    torch.testing.assert_close(
                        actual_weights, expected_weights, rtol=2e-3, atol=2e-3,
                        msg=lambda message: f"{mlp.key}: {message}",
                    )
                checked += 1
            finally:
                mlp.unload()
        assert checked == len(routers)
    finally:
        model.unload()
        torch.cuda.empty_cache()


@torch.inference_mode()
def test_non_gfx12_architecture_uses_python_fallback(monkeypatch):
    hidden, gate = _case(0)
    buffers = _buffers(hidden.device)
    cfg = SimpleNamespace(
        gate_tensor=gate, gate_tensor_t=None, router_bias=None,
        num_experts=EXPERTS, num_experts_per_tok=TOP_K,
        router_logits_bsz1=buffers[0], selected_experts_bsz1=buffers[1],
        routing_weights_bsz1=buffers[2], per_expert_scale=None,
    )
    monkeypatch.setattr(
        torch.cuda, "get_device_properties",
        lambda _index: SimpleNamespace(gcnArchName="gfx1100", warp_size=32),
    )
    calls = {"native": 0}
    monkeypatch.setattr(
        ext, "routing_std_gfx12_bsz1",
        lambda *_args, **_kwargs: calls.__setitem__("native", calls["native"] + 1),
    )
    selected, weights = bsm.routing_std(1, cfg, hidden, {})
    _, expected_selected, expected_weights = _oracle(hidden, gate)
    assert calls["native"] == 0
    assert torch.equal(selected, expected_selected)
    torch.testing.assert_close(weights, expected_weights, rtol=0, atol=0)
