# Host-only tests for NGramEmbedding tensor-parallel bring-up (T8-T10).
# These run with HIP_VISIBLE_DEVICES= (no GPU): any torch.cuda call must be patched out.
# They are failing-first: they error against the current code until NGramEmbedding gains
# tp_export / tp_import.

from __future__ import annotations

import pytest
import torch
from unittest import mock

from exllamav3.modules.ngram_embedding import NGramEmbedding
from exllamav3.loader.safetensors import DiskTensorHandle
from exllamav3.model.model_tp_shared import SMProducer, SMConsumer
from exllamav3.modules.quant.exl3_lib.ngram_codec import mul1_codebook, words_per_row


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


def make_ngram(mode: str = "fp16_disk") -> NGramEmbedding:
    m = NGramEmbedding(
        config = None,
        key = "test.ngram",
        ngram_size = 3,
        heads_per_ngram = 2,
        ple_embed_dim = 640,
        eos_token_id = 0,
        stream_from_disk = True,
        out_dtype = torch.half,
        qmap = None,
    )
    m.device = torch.device("cpu")
    m.mode = mode
    if mode == "fp16_disk":
        m.K = None
        m.handles = [DiskTensorHandle(
            key = "test.ngram.shard_0.weight",
            filename = "/virtual/ngram.bin",
            abs_offset = 0,
            shape = [4, 160],
            dtype = torch.float16,
        )]
        m.rows_per_shard = 4
        m.num_rows = 4
        m._row_dtype = torch.float16
    elif mode == "trellis_disk":
        m.K = 4
        m.handles = [DiskTensorHandle(
            key = "test.ngram.trellis",
            filename = "/virtual/t.bin",
            abs_offset = 0,
            shape = [4, words_per_row(4)],
            dtype = torch.int16,
        )]
        m.rows_per_shard = 4
        m.num_rows = 4
        m._row_dtype = None
    elif mode == "fp16_ram":
        m.K = None
        m.tables = [torch.randn(4, 160, dtype = torch.float16)]
        m.rows_per_shard = 4
        m.num_rows = 4
        m._row_dtype = torch.float16
    else:
        raise ValueError(mode)
    m.head_offsets = torch.tensor([0, 0, 0, 0], dtype = torch.long)
    m.head_vocab_sizes = torch.tensor([16, 16, 16, 16], dtype = torch.long)
    m.layer_multipliers = torch.tensor([0, 2, 4], dtype = torch.long)
    m.head_bias = None
    m.codebook = None
    return m


# --------------------------------------------------------------------------------------
# T8 — export guard + export structure
# --------------------------------------------------------------------------------------

def test_t8a_ram_mode_hard_fail(arena):
    producer, _consumer = arena
    m = make_ngram(mode = "fp16_ram")
    with pytest.raises(AssertionError):
        m.tp_export(plan = {}, producer = producer)


def test_t8b_disk_mode_export_structure(arena):
    producer, _consumer = arena
    m = make_ngram(mode = "fp16_disk")

    exported = m.tp_export(plan = {}, producer = producer)

    assert exported["cls"] is NGramEmbedding
    assert exported["kwargs"] == {
        "key": "test.ngram",
        "ngram_size": 3,
        "heads_per_ngram": 2,
        "ple_embed_dim": 640,
        "eos_token_id": 0,
        "stream_from_disk": True,
        "out_dtype": torch.half,
        "qmap": None,
    }
    assert exported["mode"] == "fp16_disk"
    assert exported["K"] is None
    assert exported["rows_per_shard"] == 4
    assert exported["num_rows"] == 4
    assert exported["_row_dtype"] == "torch.float16"
    assert exported["handles"] == [{
        "key": "test.ngram.shard_0.weight",
        "filename": "/virtual/ngram.bin",
        "abs_offset": 0,
        "shape": [4, 160],
        "dtype": "torch.float16",
    }]

    # aux tensors must be real producer sends
    for name in ("head_offsets", "head_vocab_sizes", "layer_multipliers"):
        d = exported[name]
        assert isinstance(d, dict) and "method" in d, f"{name} is not a producer send descriptor"
    assert exported["head_bias"]["method"] == "none_tensor"


# --------------------------------------------------------------------------------------
# T9 — import round trip
# --------------------------------------------------------------------------------------

def test_t9a_import_round_trip_fp16(arena):
    producer, consumer = arena
    m = make_ngram(mode = "fp16_disk")
    exported = m.tp_export(plan = {}, producer = producer)

    local_context = {"consumer": consumer, "device": torch.device("cpu")}
    imported = NGramEmbedding.tp_import(local_context, exported, plan = {})

    assert isinstance(imported, NGramEmbedding)
    assert imported.mode == "fp16_disk"
    assert imported.K is None
    assert imported.rows_per_shard == 4
    assert imported.num_rows == 4
    assert imported._row_dtype is torch.float16

    h = imported.handles[0]
    assert isinstance(h, DiskTensorHandle)
    assert h.key == "test.ngram.shard_0.weight"
    assert h.filename == "/virtual/ngram.bin"
    assert h.abs_offset == 0
    assert h.shape == [4, 160]
    assert h.dtype == torch.float16
    assert h.num_rows == 4
    assert h.row_bytes == 160 * 2

    assert torch.equal(imported.head_offsets, m.head_offsets)
    assert torch.equal(imported.head_vocab_sizes, m.head_vocab_sizes)
    assert torch.equal(imported.layer_multipliers, m.layer_multipliers)

    assert imported.head_bias is None
    assert imported.codebook is None
    assert imported._pin is None
    assert imported._pin_events == {}
    assert imported.device == torch.device("cpu")


def test_t9b_import_rebuilds_trellis_codebook(arena):
    producer, consumer = arena
    m = make_ngram(mode = "trellis_disk")
    exported = m.tp_export(plan = {}, producer = producer)

    local_context = {"consumer": consumer, "device": torch.device("cpu")}
    imported = NGramEmbedding.tp_import(local_context, exported, plan = {})

    assert imported.mode == "trellis_disk"
    assert imported.codebook is not None
    assert torch.equal(imported.codebook, mul1_codebook(torch.device("cpu")))


# --------------------------------------------------------------------------------------
# T10 — multi-shard descriptors + functional handle on a real temp file
# --------------------------------------------------------------------------------------

def test_t10a_multi_shard_descriptors(arena):
    producer, consumer = arena
    m = make_ngram(mode = "fp16_disk")
    m.handles = [
        DiskTensorHandle(
            key = "test.ngram.shard_0.weight",
            filename = "/virtual/n.bin",
            abs_offset = 0,
            shape = [4, 160],
            dtype = torch.float16,
        ),
        DiskTensorHandle(
            key = "test.ngram.shard_1.weight",
            filename = "/virtual/n.bin",
            abs_offset = 4 * 160 * 2,
            shape = [4, 160],
            dtype = torch.float16,
        ),
    ]
    m.num_rows = 8
    exported = m.tp_export(plan = {}, producer = producer)

    assert len(exported["handles"]) == 2
    assert exported["handles"][0]["abs_offset"] == 0
    assert exported["handles"][1]["abs_offset"] == 4 * 160 * 2
    for d in exported["handles"]:
        assert d["filename"] == "/virtual/n.bin"
        assert d["shape"] == [4, 160]
        assert d["dtype"] == "torch.float16"

    local_context = {"consumer": consumer, "device": torch.device("cpu")}
    imported = NGramEmbedding.tp_import(local_context, exported, plan = {})

    assert [h.abs_offset for h in imported.handles] == [0, 4 * 160 * 2]
    assert imported.rows_per_shard == 4
    assert imported.num_rows == 8


def test_t10b_real_file_handle_smoke(arena, tmp_path):
    producer, consumer = arena
    table = torch.arange(4 * 160, dtype = torch.float16).view(4, 160)
    f = tmp_path / "table.bin"
    try:
        raw = table.numpy().tobytes()
    except RuntimeError:
        raw = table.reshape(-1).view(torch.uint8).numpy().tobytes()
    f.write_bytes(raw)

    m = make_ngram(mode = "fp16_disk")
    m.handles = [DiskTensorHandle(
        key = "t.weight",
        filename = str(f),
        abs_offset = 0,
        shape = [4, 160],
        dtype = torch.float16,
    )]
    exported = m.tp_export(plan = {}, producer = producer)

    local_context = {"consumer": consumer, "device": torch.device("cpu")}
    imported = NGramEmbedding.tp_import(local_context, exported, plan = {})

    idx = torch.tensor([0, 2, 3])
    assert torch.equal(imported.handles[0].read_rows(idx), table[[0, 2, 3]])
    assert torch.equal(imported.handles[0].read_range(1, 3), table[1:3])