# Host-only tests for PLELayer tensor-parallel bring-up (T4-T7, T11, T12).
# These run with HIP_VISIBLE_DEVICES= (no GPU): any torch.cuda call must be patched out.
# They are failing-first: they error against the current code until PLELayer gains
# make_tp_allocation / tp_export / tp_import / load_local and the exported look-up branch
# in forward.

from __future__ import annotations

import pytest
import torch
from torch import nn
from types import SimpleNamespace
from unittest import mock

from exllamav3.modules.ple import PLELayer, PLELayerState
from exllamav3.modules.linear import Linear
from exllamav3.modules.rmsnorm import RMSNorm
from exllamav3.modules.ngram_embedding import NGramEmbedding
from exllamav3.loader.safetensors import DiskTensorHandle
from exllamav3.model.model_tp_shared import SMProducer, SMConsumer
from exllamav3.model.model_tp_alloc import TPAllocation, TPAllocator
from exllamav3.model.model_tp_fn import mp_model_append


# --------------------------------------------------------------------------------------
# Shared facilities
# --------------------------------------------------------------------------------------

@pytest.fixture()
def arena():
    producer = SMProducer(buffer_size = 1 << 20)
    with mock.patch("exllamav3.model.model_tp_shared.torch.cuda.set_device"):
        consumer = SMConsumer(
            producer_imp = producer,
            device = torch.device("cpu"),
            pin_memory = False,
        )
        yield producer, consumer
    consumer.close()
    producer.close()


def make_ple() -> PLELayer:
    m = PLELayer(
        config = None,
        key = "test.ple",
        layer_idx = -1,
        hidden_size = 8,
        hc_mult = 2,
        ple_embed_dim = 640,
        ngram_size = 3,
        heads_per_ngram = 2,
        eos_token_id = 0,
        conv_kernel_size = 3,
        rms_norm_eps = 1e-5,
        mm_token_id = None,
    )
    m.device = torch.device("cpu")
    m.conv_w = torch.randn(m.hc_mult * m.hidden_size, 1, m.conv_kernel_size, dtype = torch.half)

    # Fake-load the n-gram child: fp16 disk-streaming mode, one shard handle, aux hash params
    emb = m.ple_embedding
    emb.device = torch.device("cpu")
    emb.mode = "fp16_disk"
    emb.K = None
    emb.handles = [DiskTensorHandle(
        key = "test.ple.ple_embedding.ngram_embedding.shard_0.weight",
        filename = "/virtual/ngram_table.bin",
        abs_offset = 0,
        shape = [4, 160],
        dtype = torch.float16,
    )]
    emb.rows_per_shard = 4
    emb.num_rows = 4
    emb._row_dtype = torch.float16
    emb.head_offsets = torch.tensor([0], dtype = torch.long)
    emb.head_vocab_sizes = torch.tensor([4], dtype = torch.long)
    emb.layer_multipliers = torch.tensor([0, 2, 4], dtype = torch.long)
    emb.head_bias = None

    w = torch.randn(16, dtype = torch.half)
    m.norm_key.weight = nn.Parameter(w, requires_grad = False)
    m.norm_query.weight = nn.Parameter(w.clone(), requires_grad = False)
    m.norm_conv.weight = nn.Parameter(w.clone(), requires_grad = False)
    for norm in (m.norm_key, m.norm_query, m.norm_conv):
        norm.device = torch.device("cpu")
    return m


def local_context(consumer):
    return {"consumer": consumer, "device": torch.device("cpu")}


# --------------------------------------------------------------------------------------
# T4 — tp_export structure
# --------------------------------------------------------------------------------------

def test_t4_tp_export_structure(arena):
    producer, _ = arena
    m = make_ple()
    st = PLELayerState(m, max_batch_size = 2, max_history = 4, cache_id = 123)
    m.recurrent_layers.append(st)

    with mock.patch.object(Linear, "tp_export", return_value = {"cls": Linear, "stub": True}):
        exported = m.tp_export(plan = {}, producer = producer)

    assert exported["cls"] is PLELayer
    assert exported["kwargs"] == {
        "key": "test.ple",
        "layer_idx": -1,
        "hidden_size": 8,
        "hc_mult": 2,
        "ple_embed_dim": 640,
        "ngram_size": 3,
        "heads_per_ngram": 2,
        "eos_token_id": 0,
        "conv_kernel_size": 3,
        "rms_norm_eps": 1e-5,
        "stream_from_disk": True,
        "out_dtype": None,
        "mm_token_id": None,
        "qmap": None,
    }
    for name in ("ple_embedding", "key_proj", "value_proj", "norm_key", "norm_query", "norm_conv"):
        assert name in exported, f"missing child export: {name}"
    assert exported["ple_embedding"]["cls"] is NGramEmbedding
    assert exported["norm_key"]["cls"] is RMSNorm
    assert "method" in exported["conv_w"], "conv_w must be a real producer descriptor"
    assert exported["recurrent_layers"] == [st.tp_export(None)]
    assert exported["recurrent_layers"][0]["args"] == {
        "cache_id": 123,
        "max_history": 4,
        "max_batch_size": 2,
    }


# --------------------------------------------------------------------------------------
# T5 — tp_import round trip
# --------------------------------------------------------------------------------------

def test_t5_tp_import_round_trip(arena):
    producer, consumer = arena
    m = make_ple()
    st = PLELayerState(m, max_batch_size = 2, max_history = 4, cache_id = 123)
    m.recurrent_layers.append(st)

    with mock.patch.object(Linear, "tp_export", return_value = {"cls": Linear, "stub": True}):
        exported = m.tp_export(plan = {}, producer = producer)

    with mock.patch("torch.cuda.synchronize"), \
         mock.patch.object(Linear, "tp_import", staticmethod(
             lambda local_context, exported, plan, **kw: SimpleNamespace(key = "stub"))):
        imported = PLELayer.tp_import(local_context(consumer), exported, plan = {})

    assert isinstance(imported, PLELayer)
    assert torch.equal(imported.conv_w, m.conv_w)

    # n-gram child: real module, disk-streaming mode preserved, handle metadata intact,
    # aux hash tensors bitwise equal
    emb = imported.ple_embedding
    assert isinstance(emb, NGramEmbedding)
    assert emb.mode == "fp16_disk"
    h0 = emb.handles[0]
    h0_ref = m.ple_embedding.handles[0]
    assert h0.filename == h0_ref.filename
    assert h0.abs_offset == h0_ref.abs_offset
    assert h0.shape == h0_ref.shape
    assert h0.dtype == h0_ref.dtype
    assert torch.equal(emb.head_offsets, m.ple_embedding.head_offsets)
    assert torch.equal(emb.head_vocab_sizes, m.ple_embedding.head_vocab_sizes)
    assert torch.equal(emb.layer_multipliers, m.ple_embedding.layer_multipliers)
    assert emb.head_bias is None and m.ple_embedding.head_bias is None
    assert emb.rows_per_shard == m.ple_embedding.rows_per_shard
    assert emb.num_rows == m.ple_embedding.num_rows
    assert emb._row_dtype == m.ple_embedding._row_dtype

    # norms MUST be imported (not stubs), weights bitwise equal
    assert isinstance(imported.norm_key, RMSNorm)
    assert torch.equal(imported.norm_key.weight.data, m.norm_key.weight.data)

    # registration rebuilt: index 0 ple_embedding, then key_proj, value_proj, norm_key,
    # norm_query, norm_conv
    mods = imported.modules
    assert isinstance(mods, list) and len(mods) == 6
    assert mods[0] is imported.ple_embedding
    assert mods[1] is imported.key_proj
    assert mods[2] is imported.value_proj
    assert mods[3] is imported.norm_key
    assert mods[4] is imported.norm_query
    assert mods[5] is imported.norm_conv

    # recurrent state rebuilt into the TP look-up and allocated via load_local
    assert set(imported.tp_recurrent_lookup.keys()) == {123}
    rli = imported.tp_recurrent_lookup[123]
    assert isinstance(rli, PLELayerState)
    assert rli.cache_id == 123
    assert rli.device == torch.device("cpu")
    assert rli.conv_state.dtype == torch.half
    assert rli.id_state.dtype == torch.long
    assert (rli.id_state == 0).all()
    assert imported.device == torch.device("cpu")


# --------------------------------------------------------------------------------------
# T6 — forward exported-branch selection
# --------------------------------------------------------------------------------------

def test_t6_forward_exported_branch(arena):
    m = make_ple()
    win = m.conv_state_len
    ctx = m.ple_embedding.context_len
    st = PLELayerState(m, max_batch_size = 1, max_history = 4, cache_id = 99)
    st.alloc(torch.device("cpu"))
    m.forward_streams = lambda streams, history, params, conv_state = None: (
        torch.zeros_like(streams),
        torch.zeros(1, m.hc_mult * m.hidden_size, win + 2, dtype = torch.half),
    )
    m.tp_recurrent_lookup = {99: st}

    x = torch.zeros(1, 2, m.hc_mult, m.hidden_size, dtype = torch.float32)
    params = {
        "input_ids": torch.zeros(1, 3, dtype = torch.long),
        "recurrent_slots": torch.zeros(1, dtype = torch.long),
        "recurrent_states": [SimpleNamespace(exported = True, cache = 99)],
    }
    out = m.forward(x, params)
    assert torch.equal(out, x)

    # non-exported path still routes through cache.get_recurrent_layer
    m2 = make_ple()
    m2.forward_streams = lambda streams, history, params, conv_state = None: (
        torch.zeros_like(streams),
        torch.zeros(1, m2.hc_mult * m2.hidden_size, win + 2, dtype = torch.half),
    )
    params2 = {
        "input_ids": torch.zeros(1, 3, dtype = torch.long),
        "recurrent_slots": torch.zeros(1, dtype = torch.long),
        "recurrent_states": [SimpleNamespace(
            exported = False,
            cache = SimpleNamespace(get_recurrent_layer = lambda li: st),
        )],
    }
    out2 = m2.forward(x, params2)
    assert torch.equal(out2, x)


# --------------------------------------------------------------------------------------
# T7 — PLELayerState export args + load_local alloc-only semantics
# --------------------------------------------------------------------------------------

def test_t7_state_export_and_load_local():
    m = make_ple()
    st = PLELayerState(m, max_batch_size = 2, max_history = 4, cache_id = 777)
    exp = st.tp_export(plan = None)
    assert exp["cls"] is PLELayerState
    assert exp["args"] == {"cache_id": 777, "max_history": 4, "max_batch_size": 2}

    rebuilt = exp["cls"](m, **exp["args"])
    assert rebuilt.device is None

    conv_w_before = m.conv_w
    m.recurrent_layers.append(rebuilt)
    m.load_local(torch.device("cpu"))
    assert rebuilt.device == torch.device("cpu")
    assert rebuilt.conv_state.dtype == torch.half
    assert rebuilt.id_state.dtype == torch.long
    assert (rebuilt.id_state == 0).all()
    # load_local touched nothing else: conv_w untouched by identity, no stc access
    assert m.conv_w is conv_w_before
    assert m.config is not None


# --------------------------------------------------------------------------------------
# T11 — make_tp_allocation replicated accounting
# --------------------------------------------------------------------------------------

def test_t11_make_tp_allocation_replicated():
    m = make_ple()
    comps = m.make_tp_allocation({})
    assert len(comps) == 1
    c = comps[0]
    assert isinstance(c, TPAllocation)
    assert c.key == m.key
    assert c.storage_per_device > 0
    assert c.storage_to_split == 0
    assert c.channels_to_split == 1
    assert not any(
        "key_proj" in k or "ple_embedding" in k
        for k in [cc.key for cc in comps]
    )

    alloc = TPAllocator(comps, num_tokens = 1024, output_num_tokens = 256, dev_limits = {})
    _, storage, _ = alloc.initial_split([8 * 1024**3, 8 * 1024**3])
    assert storage == [c.storage_per_device, c.storage_per_device]


# --------------------------------------------------------------------------------------
# T12 — mp_model_append populates recurrent_modules
# --------------------------------------------------------------------------------------

def test_t12_mp_model_append_recurrent_modules():
    m = make_ple()
    with mock.patch.object(PLELayer, "tp_import", return_value = m):
        ctx = {
            "modules": [],
            "kv_modules": [],
            "recurrent_modules": [],
            "device": 0,
            "plan": {0: {}, 1: {}},
        }
        assert mp_model_append(ctx, {"cls": PLELayer}) is None
        mp_model_append(ctx, {"cls": PLELayer})
    assert ctx["modules"] == [m, m]
    assert ctx["recurrent_modules"] == [m, m]
    assert ctx["kv_modules"] == []