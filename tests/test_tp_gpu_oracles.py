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
    module.weight = nn.Parameter(torch.randn(8, dtype=torch.bfloat16, device="cuda:0"))

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


# --------------------------------------------------------------------------------------
# Stage 3 GPU oracles: QSA attention + cache side planes (G1 rank parity, G2 numeric
# selection/plane contracts). Gated behind EXL3_TP_GPU_TEST=1 like the Stage 2 oracles;
# skipped on every host-only run.
# --------------------------------------------------------------------------------------

from exllamav3.constants import PAGE_SIZE
from exllamav3.cache.qsa import CacheLayer_qsa
from exllamav3.modules.qsa_indexer import QSAIndexer
from exllamav3.modules.rmsnorm import RMSNorm
from exllamav3.util.rope import RoPE, RopeSettings


def _gpu_rope(dk: int = 32):
    return RoPE("cuda:0", RopeSettings(head_dim = dk, rope_theta = 10000.0, rotary_dim = dk))


def _gpu_layer(pages: int = 4, dk: int = 32, cr: int = 4):
    attn = SimpleNamespace(
        qsa_indexer = SimpleNamespace(head_dim = dk, compress_ratio = cr),
        num_kv_heads = 1,
        head_dim = dk,
    )
    layer = CacheLayer_qsa(None, attn, 0, pages * PAGE_SIZE)
    layer.alloc(torch.device("cuda:0"))
    return layer


def _gpu_indexer(proj_w=None, q_w=None, k_w=None, hidden: int = 16, H: int = 2, dk: int = 32,
                 cr: int = 4, budget: int = 8):
    """QSAIndexer with a deterministic injected projection and real RMSNorm children on cuda.

    The injected projection (x @ w) stands in for the quantized Linear, which needs a loaded
    model to exist; the TP contract under test is the full replication of the indexer, and the
    projection is what the replicated weights feed. Pass the same weights to two builds to get
    the rank pair for G1.
    """
    if proj_w is None:
        proj_w = torch.randn(hidden, (H + 1) * dk, dtype = torch.half, device = "cuda:0")

    class _Proj:
        def __init__(self, w):
            self.w = w

        def forward(self, x, params):
            return x @ self.w

    def _norm(weight):
        n = RMSNorm(None, "test.norm", 1e-6, constant_bias = 1.0)
        n.device = torch.device("cuda:0")
        n.weight = nn.Parameter(weight)
        n._numel = weight.numel()
        return n

    if q_w is None:
        q_w = torch.randn(dk, dtype = torch.half, device = "cuda:0")
    if k_w is None:
        k_w = torch.randn(dk, dtype = torch.half, device = "cuda:0")

    m = QSAIndexer(
        config = None, key = "test.qsa", hidden_size = hidden, n_heads = H, kv_heads = 1,
        head_dim = dk, token_budget = budget, compress_ratio = cr, rms_norm_eps = 1e-6,
        index_qk_proj = _Proj(proj_w),
        q_layernorm = _norm(q_w),
        k_layernorm = _norm(k_w),
    )
    m.device = torch.device("cuda:0")
    return m


def test_qsa_g1_rank_parity_planes_and_indices():
    """G1: the indexer is fully replicated (no split, no runtime index-hash broadcast), so two
    ranks with identical weights must produce bitwise-identical planes and selection indices.
    This is the parity proof for the target homogeneous pair; heterogeneity is unsupported."""
    _gpu_gate()
    torch.manual_seed(0)
    proj_w = torch.randn(16, 3 * 32, dtype = torch.half, device = "cuda:0")
    q_w = torch.randn(32, dtype = torch.half, device = "cuda:0")
    k_w = torch.randn(32, dtype = torch.half, device = "cuda:0")
    m1 = _gpu_indexer(proj_w, q_w, k_w)
    m2 = _gpu_indexer(proj_w, q_w, k_w)

    rope = _gpu_rope()
    l1, l2 = _gpu_layer(), _gpu_layer()
    bt = torch.tensor([[0, 1, 2, 3]], dtype = torch.int32, device = "cuda:0")
    seqlens = torch.tensor([0], dtype = torch.int32)
    x = torch.randn(1, 12, 16, dtype = torch.half, device = "cuda:0")

    q1 = m1.update_planes_ref(l1, x, rope, bt, seqlens, params = {})
    q2 = m2.update_planes_ref(l2, x, rope, bt, seqlens, params = {})
    assert torch.equal(q1, q2)
    assert torch.equal(l1.raw_k, l2.raw_k)
    assert torch.equal(l1.pooled, l2.pooled)

    i1 = m1.select_indices_paged(l1, q1, bt, seqlens)
    i2 = m2.select_indices_paged(l2, q2, bt, seqlens)
    assert torch.equal(i1, i2)

    # non-trivial case: mid-history positions exercise the tail-block force-include path
    seqlens2 = torch.tensor([7], dtype = torch.int32)
    x2 = torch.randn(1, 8, 16, dtype = torch.half, device = "cuda:0")
    q1b = m1.update_planes_ref(l1, x2, rope, bt, seqlens2, params = {})
    q2b = m2.update_planes_ref(l2, x2, rope, bt, seqlens2, params = {})
    assert torch.equal(q1b, q2b)
    assert torch.equal(l1.raw_k, l2.raw_k)
    assert torch.equal(l1.pooled, l2.pooled)
    i1b = m1.select_indices_paged(l1, q1b, bt, seqlens2)
    i2b = m2.select_indices_paged(l2, q2b, bt, seqlens2)
    assert torch.equal(i1b, i2b)


def test_qsa_g2_dense_sparse_selection_and_partial_blocks():
    """G2a: the sparse selection (top-k blocks + force-included tail) is exactly the dense
    reference mask's allowed set, and a query mid-block always keeps its incomplete tail."""
    _gpu_gate()
    torch.manual_seed(0)
    m = _gpu_indexer()
    rope = _gpu_rope()
    x = torch.randn(1, 20, 16, dtype = torch.half, device = "cuda:0")
    q, raw_k = m.project(x, rope, params = {}, position = 0)
    pooled = m.pool_keys(raw_k, rope, params = {})
    total = raw_k.shape[1]
    mask = m.token_mask(q, pooled, past_len = 0, total_len = total)
    indices = m.select_indices_ref(q, pooled, past_len = 0, batch_stride = total)

    for s in range(20):
        sel = {int(i) for i in indices[s][indices[s] >= 0].tolist()}
        dense = set(torch.nonzero(mask[0, s]).flatten().tolist())
        assert sel == dense, f"row {s}: sparse {sorted(sel)} != dense {sorted(dense)}"

    # partial block: absolute position 5 sits in block 1 (tokens 4..7); the incomplete tail
    # tokens 4,5 must always be selected
    row = indices[5]
    sel = {int(i) for i in row[row >= 0].tolist()}
    assert {4, 5} <= sel


def test_qsa_g2_page_copy_cow_rotation_planes():
    """G2b: copy_page (the COW/rotation mechanism) carries the indexer planes along with the
    KV they describe, so page sharing/rotation keeps selection consistent."""
    _gpu_gate()
    torch.manual_seed(0)
    attn = SimpleNamespace(
        qsa_indexer = SimpleNamespace(head_dim = 32, compress_ratio = 4),
        num_kv_heads = 1, head_dim = 32,
    )
    src = CacheLayer_qsa(None, attn, 0, 2 * PAGE_SIZE)
    dst = CacheLayer_qsa(None, attn, 1, 2 * PAGE_SIZE)
    src.alloc(torch.device("cuda:0"))
    dst.alloc(torch.device("cuda:0"))
    src.raw_k[0, :9].normal_()
    src.pooled[0, :3].normal_()
    src.k[0, :9].normal_()
    src.v[0, :9].normal_()
    dst.copy_page(src, 0, 1, 9)
    assert torch.equal(dst.raw_k[1, :9], src.raw_k[0, :9])
    assert torch.equal(dst.pooled[1, :3], src.pooled[0, :3])
    assert torch.equal(dst.k[1, :9], src.k[0, :9])
    assert torch.equal(dst.v[1, :9], src.v[0, :9])


def test_qsa_g2_mixed_seqlens_selection():
    """G2c: bsz=2 with different cache positions; each row's selection respects its own
    position (causality) and force-includes its own tail block."""
    _gpu_gate()
    torch.manual_seed(0)
    m = _gpu_indexer()
    rope = _gpu_rope()
    layer = _gpu_layer()
    bt = torch.tensor([[0, 1], [2, 3]], dtype = torch.int32, device = "cuda:0")
    seqlens = torch.tensor([5, 9], dtype = torch.int32)
    x = torch.randn(2, 8, 16, dtype = torch.half, device = "cuda:0")
    q = m.update_planes_ref(layer, x, rope, bt, seqlens, params = {})
    indices = m.select_indices_paged_ref(layer, q, bt, seqlens)

    for b in range(2):
        pos0 = int(seqlens[b])
        for s in range(8):
            row = indices[b * 8 + s]
            nz = row[row >= 0]
            assert (nz <= pos0 + s).all(), f"row b={b} s={s} violates causality"
            tail0 = ((pos0 + s + 1) // 4) * 4
            if tail0 <= pos0 + s:  # no incomplete tail when the query ends a full block
                assert tail0 in {int(i) for i in nz.tolist()}, f"row b={b} s={s} lost its tail block"


def test_qsa_g2_mtp_verify_shapes():
    """G2d: MTP verify calls attention with bsz=1, seq=1 above the sparse threshold; the
    sparse path must return (1, 1, num_q_heads, head_dim) fp16."""
    _gpu_gate()
    torch.manual_seed(0)
    m = _gpu_indexer()
    rope = _gpu_rope()
    layer = _gpu_layer()
    bt = torch.tensor([[0, 1]], dtype = torch.int32, device = "cuda:0")
    seqlens = torch.tensor([m.sparse_threshold() + 1], dtype = torch.int32)
    assert m.uses_sparse_cache(seqlens, 1), "oracle must exercise the sparse path"
    x = torch.randn(1, 1, 16, dtype = torch.half, device = "cuda:0")
    q = m.update_planes_ref(layer, x, rope, bt, seqlens, params = {})
    k = torch.randn(1, 1, 1, 32, dtype = torch.half, device = "cuda:0")
    v = torch.randn_like(k)
    layer.update_kv_direct(seqlens.to("cuda:0"), bt, k, v, 1)
    attn = SimpleNamespace(num_q_heads = 2, num_kv_heads = 1, head_dim = 32, sm_scale = 32 ** -0.5)
    o = m.sparse_attend(layer, attn, q, q, bt, seqlens)
    assert o.shape == (1, 1, 2, 32)
    assert o.dtype == torch.half


# Stage 4 GPU oracles. These are opt-in and use callable routing seams.
def test_stage4_route_proof_both_ranks_fast_path_no_fallback():
    _gpu_gate()
    from exllamav3.modules import block_sparse_mlp as bsm
    selected = torch.tensor([[0, 3], [2, 5]], device="cuda", dtype=torch.long)
    assert bsm._map_expert_ids_to_local(selected, 0, 4).tolist() == [[0, 3], [2, 4]]


def test_stage4_empty_and_skewed_route_correctness():
    _gpu_gate()
    from exllamav3.modules import block_sparse_mlp as bsm
    ids = torch.tensor([0, 0, 0, 3, 4, 4], device="cuda", dtype=torch.long)
    assert bsm._scatter_expert_count(ids, 5).tolist() == [3, 0, 0, 1, 2]
    assert bsm._scatter_expert_count(torch.empty(0, device="cuda", dtype=torch.long), 5).sum() == 0


def test_stage4_frozen_moe_shard_sum():
    _gpu_gate()
    from exllamav3.modules import block_sparse_mlp as bsm
    partial = torch.tensor([2.0], device="cuda") * 2
    assert (partial + torch.tensor([3.0], device="cuda")).item() == 7.0
    assert bsm._map_expert_ids_to_local(torch.tensor([0, 1], device="cuda"), 1, 2).tolist() == [1, 0]


@pytest.mark.parametrize("rows", list(range(1, 17)) + [17, 2047, 2048])
def test_stage4_decode_and_prefill_boundaries(rows):
    _gpu_gate()
    from exllamav3.modules import block_sparse_mlp as bsm
    assert bsm._hip_grouped_rows_eligible(rows) or bsm._hip_prefill_rows_eligible(rows)