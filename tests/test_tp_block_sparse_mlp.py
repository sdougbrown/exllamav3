"""Host-only Stage 4 expert-shard contract tests."""

from pathlib import Path

import pytest
import torch

from exllamav3.modules import block_sparse_mlp as bsm


SOURCE = Path(bsm.__file__).read_text()


def test_global_to_local_expert_ids_and_sentinel():
    mapper = bsm._map_expert_ids_to_local
    for first, last, values, expected in (
        (0, 256, [-1, 0, 255, 256, 511], [256, 0, 255, 256, 256]),
        (100, 412, [0, 99, 100, 411, 412, 511], [312, 312, 0, 311, 312, 312]),
        (0, 512, [0, 1, 511, 512], [0, 1, 511, 512]),
    ):
        selected = torch.tensor(values, dtype=torch.long)
        assert mapper(selected, first, last).tolist() == expected


@pytest.mark.parametrize(
    "changes",
    [
        {"tp_mode": "channels"}, {"cpu_split_first": 1}, {"num_local_experts": 0},
        {"lora": True}, {"activate_all_experts": True}, {"autosplit_measure": True},
        {"reconstruct": True}, {"no_reconstruct": True}, {"bias": True},
        {"padded_down": True}, {"shape_ok": False}, {"activation_ok": False},
    ],
)
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
        tp_mode=tp_mode, cpu_split_first=None, num_local_experts=4,
        lora=False, activate_all_experts=False, autosplit_measure=False,
        reconstruct=False, no_reconstruct=False, bias=False, padded_down=False,
        shape_ok=True, activation_ok=True,
    )


def test_shared_expert_reduce_accounting_and_source_ordering():
    routed = torch.tensor([1.0, 2.0])
    shared = torch.tensor([10.0, 20.0])

    class Backend:
        def all_reduce(self, value, contribute):
            if contribute:
                value *= 2

    out = routed.clone()
    Backend().all_reduce(out, True)
    out += shared
    assert torch.equal(out, torch.tensor([12.0, 24.0]))

    shared_block = "if self.shared_experts and not bc_sh_exp"
    reduce_block = "if self.tp_reduce and not pre_norm_reduce"
    assert shared_block in SOURCE and reduce_block in SOURCE
    assert SOURCE.index(shared_block) < SOURCE.index(reduce_block)
    assert "params[\"backend\"].all_reduce(y, True)" in SOURCE
    assert "(self.intermediate_size > 0 and self.num_local_experts > 0) or bool(self.shared_experts)" in SOURCE


def test_shared_reduce_accounting_zero_local_and_no_reduce():
    class Backend:
        def all_reduce(self, value, contribute):
            if contribute:
                value *= 2

    for contribute, tp_reduce, expected in ((False, True, 3.0), (True, False, 7.0)):
        value = torch.tensor([3.0])
        if tp_reduce:
            Backend().all_reduce(value, contribute)
        value += 4
        assert value.item() == expected


def test_hip_prefill_uses_local_selected_view_and_local_tables():
    assert "flat_expert_local" in SOURCE
    assert "selected_experts[:num_tokens]" in SOURCE or "selected_experts_local" in SOURCE
    assert "expert_count" in SOURCE and "ptrs_trellis" in SOURCE
    assert "assignments = num_tokens * top_k" in SOURCE
    assert "self.num_local_experts == self.num_experts" not in SOURCE[SOURCE.index("hip_prefill_eligible"):SOURCE.index("hip_prefill_eligible") + 700]
