"""
Stage 2 GPU oracles: numeric end-to-end checks for module TP serialization, gated behind
EXL3_TP_GPU_TEST=1 and an exclusive GPU window. Skipped on every host-only run; import must
stay clean without CUDA.
"""

import os
from types import SimpleNamespace
from unittest import mock

import pytest
import torch
from torch import nn

from exllamav3.modules.gated_rmsnorm import GatedRMSNorm
from exllamav3.modules.ple import PLELayer, PLELayerState
from exllamav3.modules.linear import Linear
from exllamav3.modules.ngram_embedding import NGramEmbedding
from exllamav3.model.model_tp_shared import SMProducer, SMConsumer
from exllamav3.loader.safetensors import DiskTensorHandle
from exllamav3.modules.quant.exl3_lib.ngram_codec import (
    ROW_DIM,
    mul1_codebook,
    words_per_row,
    pack_rows,
)

GPU_TESTS = os.environ.get("EXL3_TP_GPU_TEST") == "1"


def _gpu_gate():
    if not GPU_TESTS:
        pytest.skip("GPU oracles require EXL3_TP_GPU_TEST=1 and an exclusive GPU window")
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        pytest.skip("requires a visible HIP device")


def test_gated_rmsnorm_sigmoid_gpu_import_and_forward():
    _gpu_gate()
    module = GatedRMSNorm(config=None, key="test.gnorm", rms_norm_eps=1e-5, gate_activation="sigmoid")
    module.device = torch.device("cuda:0")
    module.weight = nn.Parameter(torch.randn(8, 8, dtype=torch.bfloat16, device="cuda:0"))

    producer = SMProducer(buffer_size=1 << 20)
    consumer = SMConsumer(producer_imp=producer, device=0, pin_memory=False)
    try:
        exported = module.tp_export(plan={}, producer=producer)
        imported = GatedRMSNorm.tp_import({"consumer": consumer, "device": 0}, exported, plan={})
    finally:
        consumer.close()
        producer.close()

    assert imported.bc is not None
    assert imported.gate_activation == "sigmoid"

    torch.manual_seed(0)
    x = torch.randn(2, 5, 8, dtype=torch.bfloat16, device="cuda:0").contiguous()
    gate = torch.randn(2, 5, 8, dtype=torch.bfloat16, device="cuda:0").contiguous()
    y = imported.forward(x, {}, gate=gate)
    h = x.float()
    h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + 1e-5)
    h = imported.weight.float() * h
    h = h * torch.sigmoid(gate.float())
    ref = h.to(y.dtype)
    assert torch.allclose(y.cpu(), ref.cpu(), rtol=1e-2, atol=1e-3)
    del x, gate, y, h, ref


def test_ple_gpu_import_roundtrip():
    _gpu_gate()
    m = PLELayer(
        config=None,
        key="test.ple",
        layer_idx=-1,
        hidden_size=8,
        hc_mult=2,
        ple_embed_dim=640,
        ngram_size=3,
        heads_per_ngram=2,
        eos_token_id=0,
        conv_kernel_size=3,
        rms_norm_eps=1e-5,
        mm_token_id=None,
    )
    hc_hidden = m.hc_mult * m.hidden_size
    m.device = torch.device("cuda:0")
    m.conv_w = torch.randn(hc_hidden, 1, 3, dtype=torch.half, device="cuda:0").contiguous()

    # N-gram child in fp16_disk mode: metadata-only handle, no backing file needed for export
    m.ple_embedding.device = torch.device("cuda:0")
    m.ple_embedding.mode = "fp16_disk"
    m.ple_embedding.K = None
    m.ple_embedding.handles = [
        DiskTensorHandle(key="t.weight", filename="placeholder.safetensors", abs_offset=0,
                         shape=[16, ROW_DIM], dtype=torch.half)
    ]
    m.ple_embedding.rows_per_shard = 16
    m.ple_embedding.num_rows = 16
    m.ple_embedding._row_dtype = torch.half
    m.ple_embedding.tables = None

    for norm in (m.norm_key, m.norm_query, m.norm_conv):
        norm.device = torch.device("cuda:0")
        norm.weight = nn.Parameter(torch.randn(hc_hidden, dtype=torch.half, device="cuda:0"))

    m.recurrent_layers.append(PLELayerState(m, max_batch_size=1, max_history=2, cache_id=55))

    def _fake_export(self, plan, producer):
        return {"cls": Linear, "stub": True}

    @staticmethod
    def _fake_import(local_context, exported, plan):
        return SimpleNamespace(key="stub")

    with mock.patch.object(Linear, "tp_export", _fake_export), \
            mock.patch.object(Linear, "tp_import", _fake_import):
        producer = SMProducer(buffer_size=1 << 20)
        consumer = SMConsumer(producer_imp=producer, device=0, pin_memory=False)
        try:
            exported = m.tp_export(plan={}, producer=producer)
            imported = PLELayer.tp_import({"consumer": consumer, "device": 0}, exported, plan={})
        finally:
            consumer.close()
            producer.close()

    assert imported.conv_w.is_cuda
    assert torch.equal(imported.conv_w, m.conv_w)
    state = imported.tp_recurrent_lookup[55]
    assert isinstance(state, PLELayerState)
    assert state.conv_state.device.type == "cuda"
    assert imported.device == 0

    # Conv-path self-consistency: identical weights round-tripped bitwise through the real
    # consumer must give bitwise-identical conv output (torch ops only, no Linear)
    bsz, seq = 1, 4
    col = torch.randn(bsz, m.hc_mult * m.hidden_size, m.conv_state_len + seq,
                      dtype=torch.half, device="cuda:0")
    o_orig, _ = m._short_conv(col.transpose(1, 2), conv_state=None)
    o_imp, _ = imported._short_conv(col.transpose(1, 2), conv_state=None)
    assert torch.equal(o_orig.cpu(), o_imp.cpu())


def test_ngram_gpu_forward_matches_reference(tmp_path):
    _gpu_gate()
    R, K = 16, 4
    words = words_per_row(K)
    states = torch.randint(0, 1 << K, (R, 160), dtype=torch.int64)
    scales = torch.rand(R, dtype=torch.float16) * 0.1
    packed = pack_rows(states, scales, K)
    f = tmp_path / "table.trellis"
    f.write_bytes(packed.to(torch.int16).reshape(-1).view(torch.uint8).numpy().tobytes())

    m = NGramEmbedding(
        config=None,
        key="test.ngram",
        ngram_size=3,
        heads_per_ngram=2,
        ple_embed_dim=640,
        eos_token_id=0,
        stream_from_disk=True,
        out_dtype=torch.half,
    )
    m.device = torch.device("cuda:0")
    m.mode = "trellis_disk"
    m.K = 4
    m.handles = [
        DiskTensorHandle(key="t.trellis", filename=str(f), abs_offset=0, shape=[R, words],
                         dtype=torch.int16)
    ]
    m.rows_per_shard = R
    m.num_rows = R
    m._row_dtype = None
    m.head_offsets = torch.tensor([0, 0, 0, 0], dtype=torch.long)
    m.head_vocab_sizes = torch.tensor([R] * 4, dtype=torch.long)
    m.layer_multipliers = torch.tensor([0, 2, 4], dtype=torch.long)
    m.head_bias = torch.zeros(4, dtype=torch.half, device="cuda:0")
    m.codebook = mul1_codebook("cuda:0")
    m.tables = None

    ids = torch.tensor([[1, 2, 3, 4, 5, 6]], dtype=torch.long)
    fast = m.forward(ids, {})
    ref = m.forward_reference(ids, {})
    assert fast.shape == ref.shape
    assert torch.allclose(fast.cpu(), ref.cpu(), rtol=1e-3, atol=1e-3)