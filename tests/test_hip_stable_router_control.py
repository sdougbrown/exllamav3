"""CPU/GPU contract tests for the opt-in diagnostic (no production imports)."""
import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from hip_stable_router_control import stable_topk, routing_std_stable


def original_router():
    path = Path(__file__).parents[1] / 'exllamav3/modules/block_sparse_mlp.py'
    node = next(n for n in ast.parse(path.read_text()).body
                if isinstance(n, ast.FunctionDef) and n.name == '_routing_std_torch')
    scope = {'torch': torch}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), scope)
    return scope['_routing_std_torch']


@pytest.fixture(params=['cpu', 'cuda'])
def device(request):
    if request.param == 'cuda' and not torch.cuda.is_available():
        pytest.skip('GPU unavailable')
    return request.param


@pytest.mark.parametrize('kind', ['cutoff', 'all', 'none', 'nextafter'])
def test_score_order_and_exact_ties(device, kind):
    scores = torch.tensor([[5., 3., 3., 3., 1.], [1., 3., 3., 3., 5.]], device=device)
    if kind == 'all':
        scores.fill_(1.)
    elif kind == 'none':
        scores = torch.tensor([[5., 1., 3., 2., 4.], [1., 2., 3., 4., 5.]], device=device)
    elif kind == 'nextafter':
        one = torch.tensor(1., device=device)
        up = torch.nextafter(one, torch.tensor(float('inf'), device=device))
        down = torch.nextafter(one, torch.tensor(0., device=device))
        scores = torch.stack([torch.stack([one, up, down, one, up])] * 2)
    values, ids = stable_topk(scores, 3)
    expected = torch.tensor([sorted(range(5), key=lambda i: (-row[i], i))[:3]
                             for row in scores.cpu().tolist()], device=device)
    assert torch.equal(ids, expected)
    assert values.is_contiguous()
    assert torch.equal(values, scores.gather(-1, expected))
    old_values, old_ids = torch.topk(scores, 3, dim=-1)
    assert torch.equal(values, old_values)
    assert torch.equal(torch.softmax(values, -1), torch.softmax(old_values, -1))
    for row in range(2):
        for slot in range(3):
            if (scores[row] == values[row, slot]).sum() == 1:
                assert ids[row, slot] == old_ids[row, slot]
    if kind == 'none':
        assert torch.equal(ids, old_ids)


@pytest.mark.parametrize('include_bias', [False, True])
@pytest.mark.parametrize('scale', [False, True])
@pytest.mark.parametrize('activate_all', [False, True])
@pytest.mark.parametrize('tied', [False, True])
def test_original_router_semantics(device, include_bias, scale, activate_all, tied):
    y = torch.eye(5, dtype=torch.float16, device=device)
    gate = torch.tensor([[5., 2., 4., 1., 3.]] * 5, dtype=y.dtype, device=device)
    if tied:
        gate.fill_(1.)
    cfg = SimpleNamespace(gate_tensor=gate, num_experts=5, num_experts_per_tok=3,
                          router_bias=torch.tensor([0., .25, 0., .25, 0.], device=device),
                          per_expert_scale=torch.arange(1, 6, device=device) / 3 if scale else None)
    params = {'activate_all_experts': activate_all}
    ids, weights = routing_std_stable(cfg, y, params, include_bias)
    old_ids, old_weights = original_router()(cfg, y, params, include_bias)
    if not tied or activate_all:
        assert torch.equal(ids, old_ids)
        assert torch.equal(weights, old_weights)
    scores = (y @ gate).float()
    if include_bias:
        scores += cfg.router_bias.float()
    expected_ids = torch.arange(5, device=device).expand(5, -1) if activate_all else torch.tensor(
        [sorted(range(5), key=lambda i: (-row[i], i))[:3] for row in scores.cpu().tolist()], device=device)
    expected_weights = torch.softmax(scores.gather(-1, expected_ids).contiguous(), -1)
    if scale:
        expected_weights *= cfg.per_expert_scale.float()[expected_ids]
    assert torch.equal(ids, expected_ids)
    assert torch.equal(weights, expected_weights.half())


def test_absent_bias_and_no_input_mutation(device):
    y = torch.eye(3, device=device)
    cfg = SimpleNamespace(gate_tensor=torch.tensor([[2., 1., 3.]] * 3, device=device),
                          num_experts=3, num_experts_per_tok=2, router_bias=None, per_expert_scale=None)
    before = y.clone(), cfg.gate_tensor.clone()
    actual = routing_std_stable(cfg, y, {}, include_bias=True)
    expected = original_router()(cfg, y, {}, include_bias=True)
    assert all(torch.equal(a, b) for a,b in zip(actual, expected))
    assert torch.equal(y, before[0]) and torch.equal(cfg.gate_tensor, before[1])


def test_installer_changes_only_torch_fallback(monkeypatch):
    import sys
    from hip_stable_router_control import install
    native = object()
    original = object()
    module = SimpleNamespace(_routing_std_torch=original, routing_std=native)
    monkeypatch.setitem(sys.modules, 'exllamav3.modules.block_sparse_mlp', module)
    restore = install()
    assert module._routing_std_torch is routing_std_stable
    assert module.routing_std is native
    restore()
    assert module._routing_std_torch is original
