# Host-only (no GPU) tests for the TP contract of GatedRMSNorm: the tp_export payload must carry
# "gate_activation" and tp_import / tp_import_split must pass the BC activation flag matching load().
import pytest
import torch
import torch.nn as nn
from unittest import mock

from exllamav3.modules import gated_rmsnorm as gated_rmsnorm_module
from exllamav3.modules.gated_rmsnorm import GatedRMSNorm
from exllamav3.model.model_tp_shared import SMProducer, SMConsumer


@pytest.fixture
def arena():
    producer = SMProducer(buffer_size = 1 << 20)
    with mock.patch("exllamav3.model.model_tp_shared.torch.cuda.set_device"):
        consumer = SMConsumer(producer_imp = producer, device = torch.device("cpu"), pin_memory = False)
    yield producer, consumer
    consumer.close()
    producer.close()


@pytest.fixture
def bc_spy():
    calls = []

    def spy(*args):
        calls.append(args)
        return None

    with mock.patch.object(gated_rmsnorm_module.ext, "BC_GatedRMSNorm", spy), \
         mock.patch("torch.cuda.synchronize"), \
         mock.patch("torch.cuda.set_device"):
        yield calls


def make_module(activation):
    module = GatedRMSNorm(
        config = None,
        key = "test.gnorm",
        rms_norm_eps = 1e-5,
        gate_activation = activation,
    )
    module.device = torch.device("cpu")
    module.weight = nn.Parameter(torch.randn(8, 8, dtype = torch.float32))
    return module


@pytest.mark.parametrize("activation", ["sigmoid", "silu"])
def test_tp_export_import_roundtrip_gate_activation(arena, bc_spy, activation):
    producer, consumer = arena
    module = make_module(activation)

    exported = module.tp_export(plan = {}, producer = producer)
    assert exported["kwargs"]["gate_activation"] == activation

    local_context = {"consumer": consumer, "device": torch.device("cpu")}
    imported = GatedRMSNorm.tp_import(local_context, exported, plan = {})

    assert imported.gate_activation == activation
    assert torch.equal(imported.weight, module.weight)

    assert len(bc_spy) == 1
    call = bc_spy[0]
    assert len(call) == 6
    assert call[-1] == (1 if activation == "sigmoid" else 0)


@pytest.mark.parametrize("activation", ["sigmoid", "silu"])
@pytest.mark.parametrize("importer", ["tp_import", "tp_import_split"])
def test_bc_gate_act_flag_per_importer(arena, bc_spy, activation, importer):
    producer, consumer = arena
    module = make_module(activation)

    exported = module.tp_export(plan = {}, producer = producer)
    local_context = {"consumer": consumer, "device": torch.device("cpu")}
    if importer == "tp_import":
        GatedRMSNorm.tp_import(local_context, exported, plan = {})
    else:
        GatedRMSNorm.tp_import_split(local_context, exported, plan = {}, split = (0, 8))

    assert len(bc_spy) == 1
    assert len(bc_spy[0]) == 6
    assert bc_spy[0][-1] == (1 if activation == "sigmoid" else 0)


def test_sigmoid_forward_equivalence_after_tp_import(arena, bc_spy):
    producer, consumer = arena
    torch.manual_seed(0)
    module = make_module("sigmoid")
    # Per-channel weight so the fp32 sigmoid fallback broadcasts against x of last-dim 8
    module.weight = nn.Parameter(torch.randn(8, dtype = torch.float32))

    exported = module.tp_export(plan = {}, producer = producer)
    local_context = {"consumer": consumer, "device": torch.device("cpu")}
    imported = GatedRMSNorm.tp_import(local_context, exported, plan = {})

    x = torch.randn(2, 5, 8, dtype = torch.float32)
    gate = torch.randn(2, 5, 8, dtype = torch.float32)
    params = {}

    y_orig = module.forward(x, params, gate = gate)
    y_imported = imported.forward(x, params, gate = gate)

    assert torch.equal(y_orig, y_imported)