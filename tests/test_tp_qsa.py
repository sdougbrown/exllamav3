# Host-only tests for QSA attention + cache side-plane TP bring-up (Stage 3, T1-T7).
# These run with HIP_VISIBLE_DEVICES= (no GPU): any torch.cuda call must be patched out.
# They are failing-first: they error against the current code until QSAIndexer gains the
# ctor injection kwargs + tp_export/tp_import, CacheLayer_qsa(_quant) gains
# plane_storage_size(), and Attention gains the indexer accounting branch in
# make_tp_allocation plus the QSA tp_export/import path.

from __future__ import annotations

import pytest
import torch
from torch import nn
from types import SimpleNamespace
from unittest import mock

from exllamav3.constants import PAGE_SIZE
from exllamav3.cache.fp16 import CacheLayer_fp16
from exllamav3.cache.quant import CacheLayer_quant
from exllamav3.cache.qsa import CacheLayer_qsa, CacheLayer_qsa_quant
from exllamav3.modules.attn import Attention
from exllamav3.modules.linear import Linear
from exllamav3.modules.qsa_indexer import QSAIndexer
from exllamav3.modules.rmsnorm import RMSNorm
from exllamav3.model.model_tp_alloc import TPAllocation
from exllamav3.model.model_tp_shared import SMProducer, SMConsumer


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


def make_indexer(head_dim: int = 32, hidden_size: int = 16, **kw) -> QSAIndexer:
    defaults = dict(
        config = None,
        key = "test.qsa",
        hidden_size = hidden_size,
        n_heads = 2,
        kv_heads = 1,
        head_dim = head_dim,
        token_budget = 8,
        compress_ratio = 4,
        rms_norm_eps = 1e-6,
    )
    defaults.update(kw)
    m = QSAIndexer(**defaults)
    m.device = torch.device("cpu")
    # fake-load the norms so weights_numel()/forward paths see real weights
    for norm in (m.q_layernorm, m.k_layernorm):
        norm.device = torch.device("cpu")
        norm.weight = nn.Parameter(torch.randn(head_dim, dtype = torch.half))
        norm._numel = norm.weight.numel()
    return m


def make_attention(head_dim: int = 256, kv_heads: int = 2, q_heads: int = 4) -> Attention:
    idx = make_indexer(head_dim = head_dim, hidden_size = 64)
    attn = Attention(
        config = None,
        key = "test.attn",
        layer_idx = 0,
        hidden_size = 64,
        head_dim = head_dim,
        num_q_heads = q_heads,
        num_kv_heads = kv_heads,
        rope_settings = None,
        key_q = "q_proj",
        key_k = "k_proj",
        key_v = "v_proj",
        key_o = "o_proj",
        qsa_indexer = idx,
    )
    attn.device = torch.device("cpu")
    return attn


# --------------------------------------------------------------------------------------
# T1 — ctor injection kwargs ([C1])
# --------------------------------------------------------------------------------------

def test_t1_ctor_injection_kwargs():
    m = make_indexer()
    assert isinstance(m.index_qk_proj, Linear)
    assert isinstance(m.q_layernorm, RMSNorm)
    assert isinstance(m.k_layernorm, RMSNorm)
    assert m.q_layernorm.constant_bias == 1.0
    assert m.k_layernorm.constant_bias == 1.0
    assert m.index_qk_proj.out_features_unpadded == (2 + 1) * 32
    assert m.modules == [m.index_qk_proj, m.q_layernorm, m.k_layernorm]

    # injected children are used by identity (the TP import path)
    fake_proj = SimpleNamespace(key = "inj.proj")
    fake_q = SimpleNamespace(key = "inj.q")
    fake_k = SimpleNamespace(key = "inj.k")
    m2 = QSAIndexer(
        config = None, key = "test.qsa2", hidden_size = 16, n_heads = 2, kv_heads = 1,
        head_dim = 32, token_budget = 8, compress_ratio = 4, rms_norm_eps = 1e-6,
        index_qk_proj = fake_proj, q_layernorm = fake_q, k_layernorm = fake_k,
    )
    assert m2.index_qk_proj is fake_proj
    assert m2.q_layernorm is fake_q
    assert m2.k_layernorm is fake_k
    assert m2.modules == [fake_proj, fake_q, fake_k]


# --------------------------------------------------------------------------------------
# T2 — tp_export structure ([C2])
# --------------------------------------------------------------------------------------

def test_t2_tp_export_structure(arena):
    producer, _ = arena
    m = make_indexer()
    with mock.patch.object(Linear, "tp_export", return_value = {"cls": Linear, "stub": True}):
        exported = m.tp_export(plan = {}, producer = producer)

    assert exported["cls"] is QSAIndexer
    assert exported["kwargs"] == {
        "key": "test.qsa",
        "hidden_size": 16,
        "n_heads": 2,
        "kv_heads": 1,
        "head_dim": 32,
        "token_budget": 8,
        "compress_ratio": 4,
        "rms_norm_eps": 1e-6,
    }
    assert exported["index_qk_proj"] == {"cls": Linear, "stub": True}
    assert exported["q_layernorm"]["cls"] is RMSNorm
    assert exported["k_layernorm"]["cls"] is RMSNorm
    assert exported["device"] == torch.device("cpu")


# --------------------------------------------------------------------------------------
# T3 — tp_import round trip ([C2])
# --------------------------------------------------------------------------------------

def test_t3_tp_import_round_trip(arena):
    producer, consumer = arena
    m = make_indexer()
    with mock.patch.object(Linear, "tp_export", return_value = {"cls": Linear, "stub": True}):
        exported = m.tp_export(plan = {}, producer = producer)

    with mock.patch.object(Linear, "tp_import", staticmethod(
            lambda local_context, exported, plan, **kw: SimpleNamespace(key = "stub"))), \
         mock.patch("torch.cuda.synchronize"):
        imported = QSAIndexer.tp_import(
            {"consumer": consumer, "device": torch.device("cpu")}, exported, plan = {})

    assert isinstance(imported, QSAIndexer)
    assert imported.index_qk_proj.key == "stub"
    assert isinstance(imported.q_layernorm, RMSNorm)
    assert isinstance(imported.k_layernorm, RMSNorm)
    assert torch.equal(imported.q_layernorm.weight.data, m.q_layernorm.weight.data)
    assert torch.equal(imported.k_layernorm.weight.data, m.k_layernorm.weight.data)
    mods = imported.modules
    assert len(mods) == 3
    assert mods[0] is imported.index_qk_proj
    assert mods[1] is imported.q_layernorm
    assert mods[2] is imported.k_layernorm
    assert imported.device == torch.device("cpu")


# --------------------------------------------------------------------------------------
# T4 — CacheLayer_qsa plane_storage_size ([C3])
# --------------------------------------------------------------------------------------

def test_t4_qsa_plane_storage_size():
    idx = SimpleNamespace(head_dim = 32, compress_ratio = 4)
    attn = SimpleNamespace(qsa_indexer = idx, num_kv_heads = 1, head_dim = 32)
    cl = CacheLayer_qsa(None, attn, cache_id = 0, max_num_tokens = 4 * PAGE_SIZE)
    num_pages = 4
    plane = (num_pages * PAGE_SIZE * 32 + num_pages * (PAGE_SIZE // 4) * 32) * 2
    assert cl.plane_storage_size() == plane
    assert cl.storage_size() == CacheLayer_fp16.storage_size(cl) + plane

    cl.alloc(torch.device("cpu"))
    assert cl.raw_k.shape == cl.raw_k_shape
    assert cl.pooled.shape == cl.pooled_shape
    assert cl.raw_k.dtype == torch.half
    assert cl.pooled.dtype == torch.half
    cl.free()
    assert cl.raw_k is None and cl.pooled is None


# --------------------------------------------------------------------------------------
# T5 — CacheLayer_qsa_quant plane_storage_size ([C3])
# --------------------------------------------------------------------------------------

def test_t5_qsa_quant_plane_storage_size():
    idx = SimpleNamespace(head_dim = 32, compress_ratio = 4)
    attn = SimpleNamespace(qsa_indexer = idx, num_kv_heads = 1, head_dim = 32)
    cl = CacheLayer_qsa_quant(None, attn, cache_id = 0, max_num_tokens = 4 * PAGE_SIZE,
                              k_bits = 8, v_bits = 8)
    num_pages = 4
    plane = (num_pages * PAGE_SIZE * 32 + num_pages * (PAGE_SIZE // 4) * 32) * 2
    assert cl.plane_storage_size() == plane
    assert cl.storage_size() == CacheLayer_quant.storage_size(cl) + plane

    cl.alloc(torch.device("cpu"))
    assert cl.raw_k.shape == cl.raw_k_shape
    assert cl.pooled.shape == cl.pooled_shape
    assert cl.raw_k.dtype == torch.half
    assert cl.pooled.dtype == torch.half
    cl.free()
    assert cl.raw_k is None and cl.pooled is None


# --------------------------------------------------------------------------------------
# T6 — Attention.make_tp_allocation indexer accounting ([C4])
# --------------------------------------------------------------------------------------

def test_t6_make_tp_allocation_indexer_accounting():
    attn = make_attention()
    for cid in (0, 1):
        attn.cache_layers.append(CacheLayer_qsa(None, attn, cache_id = cid, max_num_tokens = 4 * PAGE_SIZE))
    assert len(attn.cache_layers) == 2

    # Linear.storage_size/recons_size need stc (config is None here); pin them so the
    # accounting is fully deterministic. Also capture the indexer's mocked qk-proj
    # storage + the norms' real (pre-load) weights_numel inside the mock context.
    with mock.patch.object(Linear, "storage_size", return_value = 1000), \
         mock.patch.object(Linear, "recons_size", return_value = 500):
        indexer_storage = attn.qsa_indexer.index_qk_proj.storage_size() + \
            2 * attn.qsa_indexer.q_layernorm.weights_numel() + \
            2 * attn.qsa_indexer.k_layernorm.weights_numel()
        comps = attn.make_tp_allocation({})

    assert len(comps) == 1
    c = comps[0]
    assert isinstance(c, TPAllocation)
    assert c.key == attn.key
    # channel math unchanged: head_dim 256 -> channel_width 1, channels_to_split 2
    assert c.channel_width == 1
    assert c.channels_to_split == 2

    # planes: 2 layers x (raw num_pages*PAGE_SIZE*hd + pooled num_pages*(PAGE_SIZE//cr)*hd) fp16
    hd = attn.head_dim
    cr = attn.qsa_indexer.compress_ratio
    num_pages = 4 * PAGE_SIZE // PAGE_SIZE
    plane = (num_pages * PAGE_SIZE * hd + num_pages * (PAGE_SIZE // cr) * hd) * 2
    assert c.storage_per_device == indexer_storage + 2 * plane
    # split storage: q/k/v/o (mocked 1000 each) + main K/V of both layers
    main_kv = CacheLayer_fp16.storage_size(attn.cache_layers[0])
    assert c.storage_to_split == 4 * 1000 + 2 * main_kv
    # overhead_d: hidden fp16 row + indexer recons (mocked)
    assert c.overhead_per_device == 64 * 2 + 500
    assert c.recons_temp == 500


# --------------------------------------------------------------------------------------
# T7 — Attention.tp_export/tp_import with indexer ([C5]/[C6])
# --------------------------------------------------------------------------------------

def test_t7_attention_tp_export_import_with_indexer(arena):
    producer, consumer = arena
    attn = make_attention()
    for cid in (0, 1):
        attn.cache_layers.append(CacheLayer_qsa(None, attn, cache_id = cid, max_num_tokens = 4 * PAGE_SIZE))

    with mock.patch.object(Linear, "tp_export", return_value = {"cls": Linear, "stub": True}):
        exported = attn.tp_export(plan = {}, producer = producer)

    assert "qsa_indexer" in exported
    assert exported["qsa_indexer"]["cls"] is QSAIndexer
    assert exported["cache_layers"][0]["cls"] is CacheLayer_qsa

    with mock.patch.object(Linear, "tp_import", staticmethod(
            lambda local_context, exported, plan, **kw: SimpleNamespace(key = "stub"))), \
         mock.patch.object(Linear, "tp_import_split", staticmethod(
            lambda local_context, exported, plan, split, **kw: SimpleNamespace(key = "stub"))), \
         mock.patch("torch.cuda.synchronize"):
        imported = Attention.tp_import(
            {"consumer": consumer, "device": torch.device("cpu")},
            exported, plan = {"test.attn": (0, 2, "heads")})

    assert isinstance(imported, Attention)
    assert isinstance(imported.qsa_indexer, QSAIndexer)
    assert imported.qsa_indexer.index_qk_proj.key == "stub"
    assert isinstance(imported.qsa_indexer.q_layernorm, RMSNorm)
    assert torch.equal(imported.qsa_indexer.q_layernorm.weight.data,
                       attn.qsa_indexer.q_layernorm.weight.data)
    assert torch.equal(imported.qsa_indexer.k_layernorm.weight.data,
                       attn.qsa_indexer.k_layernorm.weight.data)
    assert len(imported.cache_layers) == 2
    assert all(isinstance(cl, CacheLayer_qsa) for cl in imported.cache_layers)
    # load_local ran on import: the side planes are allocated, not just constructed
    assert imported.cache_layers[0].raw_k is not None
    assert imported.cache_layers[0].pooled is not None
    assert imported.has_split_cache
    assert imported.tp_reduce
    assert imported.qsa_indexer.device == torch.device("cpu")
