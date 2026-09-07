"""Host-only Stage 4 expert-shard contract tests."""

import torch
import pytest

from exllamav3.modules import block_sparse_mlp as bsm
from exllamav3.modules.multilinear import MultiLinear


def test_global_to_local_expert_ids_and_sentinel():
    for first, last, values, expected in (
        (0, 256, [-1, 0, 255, 256, 511], [256, 0, 255, 256, 256]),
        (100, 412, [0, 99, 100, 411, 412, 511], [312, 312, 0, 311, 312, 312]),
        (0, 512, [0, 1, 511, 512], [0, 1, 511, 512]),
    ):
        assert bsm._map_expert_ids_to_local(torch.tensor(values), first, last).tolist() == expected


def test_scatter_expert_count_has_sentinel_bin():
    ids = torch.tensor([0, 2, 3, 2, 3, 3])
    counts = bsm._scatter_expert_count(ids, 5)
    assert counts.shape == (5,)
    assert counts.tolist() == [1, 0, 2, 3, 0]
    empty = bsm._scatter_expert_count(torch.empty(0, dtype=torch.long), 4)
    assert empty.tolist() == [0, 0, 0, 0]


@pytest.mark.parametrize("changes", [
    {"tp_mode": "channels"}, {"cpu_split_first": 1}, {"num_local_experts": 0},
    {"lora": True}, {"activate_all_experts": True}, {"autosplit_measure": True},
    {"reconstruct": True}, {"no_reconstruct": True}, {"bias": True},
    {"padded_down": True}, {"shape_ok": False}, {"activation_ok": False},
])
def test_expert_shard_fast_path_guard_matrix(changes):
    base = dict(tp_mode="experts", cpu_split_first=None, num_local_experts=4,
                lora=False, activate_all_experts=False, autosplit_measure=False,
                reconstruct=False, no_reconstruct=False, bias=False, padded_down=False,
                shape_ok=True, activation_ok=True)
    base.update(changes)
    assert not bsm._hip_expert_shard_eligible(**base)


@pytest.mark.parametrize("tp_mode", [None, "experts"])
def test_whole_expert_shard_fast_path_is_eligible(tp_mode):
    assert bsm._hip_expert_shard_eligible(
        tp_mode=tp_mode, cpu_split_first=None, num_local_experts=4, lora=False,
        activate_all_experts=False, autosplit_measure=False, reconstruct=False,
        no_reconstruct=False, bias=False, padded_down=False, shape_ok=True, activation_ok=True)


class _Backend:
    def __init__(self): self.calls = []
    def all_reduce(self, value, contribute):
        self.calls.append(contribute)
        if contribute: value.mul_(2)


def _mlp(shared, local=1, reduce=True):
    m = bsm.BlockSparseMLP.__new__(bsm.BlockSparseMLP)
    m.alt_residual_channel = False; m.hidden_size = 1; m.router_pre_norm = None; m.routed_pre_norm = None
    m.routing_gate = object(); m.routing_cfg = None
    m.routing_fn_calls = 0
    def routing_fn(*args):
        m.routing_fn_calls += 1
        return torch.tensor([[0]]), torch.ones(1, 1)
    m.routing_fn = routing_fn
    m.routing_device = None; m.cpu_split_first = None; m.cpu_offload = True
    m.intermediate_size = 1; m.num_local_experts = local; m.num_experts_per_tok = 1
    m.shared_experts = shared; m.shared_gate = None; m.tp_reduce = reduce
    m.tp_mode = "experts" if reduce else None
    m.routed_post_norm = None; m.shared_experts_post_norm = None; m.bc = None
    m.cpu_split_combine = lambda y, *_: y
    m.cpu_offload_forward_calls = 0
    def cpu_offload_forward(*args):
        m.cpu_offload_forward_calls += 1
        return torch.full_like(args[1], 2.0)
    m.cpu_offload_forward = cpu_offload_forward
    return m


@pytest.mark.parametrize("local,reduce,expected,calls", [(1, True, 7.0, [True]),
                                                           (0, True, 5.0, [False]),
                                                           (1, False, 5.0, [])])
def test_forward_routes_then_adds_replicated_shared(local, reduce, expected, calls):
    class Shared:
        def forward(self, x, params): return torch.full_like(x, 3.0)
    backend = _Backend()
    mlp = _mlp(Shared(), local, reduce)
    out = mlp.forward(torch.ones(1, 1), {"backend": backend})
    assert out.item() == expected
    assert backend.calls == calls
    assert mlp.routing_fn_calls == 1
    assert mlp.cpu_offload_forward_calls == 1


@pytest.mark.parametrize("bsz", [1, 32, 33])
@pytest.mark.parametrize("reduce,expected,calls", [(False, 6.0, []), (True, 8.0, [True])])
def test_forward_gated_shared_accumulates_directly(monkeypatch, bsz, reduce, expected, calls):
    class Shared:
        def forward(self, x, params): return torch.full_like(x, 3.0)

    class Gate:
        inner = type("Inner", (), {"weight": torch.ones(1, 1)})()
        def forward(self, x, params): return torch.ones_like(x)

    def add_gate(y, z, out): out.add_(y + z)
    def add_gate_proj(y, x, out, weight): out.add_(y + x @ weight)
    monkeypatch.setattr(bsm.ext, "add_sigmoid_gate", add_gate)
    monkeypatch.setattr(bsm.ext, "add_sigmoid_gate_proj", add_gate_proj)

    backend = _Backend()
    mlp = _mlp(Shared(), reduce=reduce)
    mlp.shared_gate = Gate()
    out = mlp.forward(torch.ones(bsz, 1), {"backend": backend})
    assert torch.all(out == expected)
    assert backend.calls == calls


def test_multilinear_pointer_table_is_local_list():
    class Inner:
        K = 3; mcg = 1; mul1 = 2; bias = None; softcap = False
        def __init__(self):
            self.suh = torch.empty(2); self.svh = torch.empty(2); self.trellis = torch.empty(2)
    class L:
        quant_type = "exl3"; softcap = False; post_scale = 1.0
        in_features = 2; out_features = 2
        def __init__(self): self.inner = Inner()
    ls = [L(), L()]
    table = MultiLinear(torch.device("cpu"), ls)
    for ptrs, values in (
        (table.ptrs_trellis, [x.inner.trellis.data_ptr() for x in ls]),
        (table.ptrs_suh, [x.inner.suh.data_ptr() for x in ls]),
        (table.ptrs_svh, [x.inner.svh.data_ptr() for x in ls]),
    ):
        assert ptrs.numel() == len(ls)
        assert ptrs.tolist() == values
    ids = bsm._map_expert_ids_to_local(torch.tensor([4, 5, 6]), 4, 6)
    assert ids.tolist() == [0, 1, 2]
    assert bsm._scatter_expert_count(ids, 3).tolist() == [1, 1, 1]
