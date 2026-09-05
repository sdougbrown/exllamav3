"""QSA quantized cache contracts: fp16 side planes and packed sparse gather routing."""
from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from exllamav3.constants import PAGE_SIZE
from exllamav3.cache.quant import CacheLayer_quant
from exllamav3.cache.qsa import CacheLayer_qsa
from exllamav3.modules.attn import Attention
from exllamav3.modules.qsa_indexer import QSAIndexer


pytestmark = pytest.mark.skipif(
    not (torch.version.hip and torch.cuda.is_available()),
    reason = "QSA quant-cache qualification requires ROCm",
)

DEVICE = torch.device("cuda:0")


def _attention(head_dim: int = 32, kv_heads: int = 1):
    return SimpleNamespace(
        num_kv_heads = kv_heads,
        head_dim = head_dim,
        qsa_indexer = SimpleNamespace(head_dim = 32, compress_ratio = 4),
    )


def _layer(pages: int = 2, bits: int = 8, compand_a: float = 0.0, *, head_dim: int = 32, kv_heads: int = 1, k_bits: int | None = None, v_bits: int | None = None):
    # Imported here so this test is a regression for the old fp16-only QSA mapping.
    from exllamav3.cache.qsa import CacheLayer_qsa_quant
    layer = CacheLayer_qsa_quant(
        None, _attention(head_dim, kv_heads), 0, pages * PAGE_SIZE,
        bits if k_bits is None else k_bits,
        bits if v_bits is None else v_bits,
        compand_a,
    )
    layer.alloc(DEVICE)
    return layer


def _indexer():
    return QSAIndexer(
        config = None, key = "test.qsa", hidden_size = 16, n_heads = 2, kv_heads = 1,
        head_dim = 32, token_budget = 8, compress_ratio = 4, rms_norm_eps = 1e-6,
    )


def _update(layer, k, v, seqlens, table):
    layer.update_kv_direct(seqlens, table, k.contiguous(), v.contiguous(), k.shape[1])


def _selected_oracle(layer, q, indices, table, seqlens, sm_scale):
    """Reference in the original domain for one selected QSA row."""
    k, v = layer.get_kv(seqlens, table)
    idx = indices[0, indices[0] >= 0].long()
    phys = table[0, idx // PAGE_SIZE].long()
    keys = k[phys, idx % PAGE_SIZE].float()
    values = v[phys, idx % PAGE_SIZE].float()
    heads = q.shape[2]
    kv_heads = keys.shape[1]
    head_to_kv = torch.arange(heads, device = q.device) // (heads // kv_heads)
    keys = keys[:, head_to_kv].permute(1, 0, 2)
    values = values[:, head_to_kv].permute(1, 0, 2)
    weights = torch.softmax(torch.einsum("hd,hnd->hn", q[0, 0].float(), keys) * sm_scale, dim = -1)
    return torch.einsum("hn,hnd->hd", weights, values).half()


def test_qsa_q8_cache_packs_main_kv_but_keeps_indexer_planes_fp16_and_copies_pages():
    layer_type, _ = Attention.cache_layer_type(
        SimpleNamespace(qsa_indexer = object()), CacheLayer_quant, {"k_bits": 8, "v_bits": 8},
    )
    assert layer_type.__name__ == "CacheLayer_qsa_quant"
    source, target = _layer(), _layer()
    assert not hasattr(source, "k") and not hasattr(source, "v")
    assert source.qk.dtype is torch.int32 and source.qv.dtype is torch.int32
    assert source.sk.dtype is torch.float16 and source.sv.dtype is torch.float16
    assert source.raw_k.dtype is torch.float16 and source.pooled.dtype is torch.float16
    fp16 = CacheLayer_qsa(None, _attention(), 0, 2 * PAGE_SIZE)
    assert source.storage_size() < fp16.storage_size()

    source.raw_k[0, :9].fill_(1.25)
    source.pooled[0, :3].fill_(-0.5)
    source.qk[0, :9].fill_(123)
    source.qv[0, :9].fill_(456)
    source.sk[0, :9].fill_(0.25)
    source.sv[0, :9].fill_(0.5)
    target.copy_page(source, 0, 1, 9)

    assert torch.equal(target.qk[1, :9], source.qk[0, :9])
    assert torch.equal(target.qv[1, :9], source.qv[0, :9])
    assert torch.equal(target.sk[1, :9], source.sk[0, :9])
    assert torch.equal(target.sv[1, :9], source.sv[0, :9])
    assert torch.equal(target.raw_k[1, :9], source.raw_k[0, :9])
    assert torch.equal(target.pooled[1, :3], source.pooled[0, :3])


def test_qsa_cache_mapping_rejects_unknown_cache_types():
    with pytest.raises(AssertionError, match = "fp16 or quantized"):
        Attention.cache_layer_type(SimpleNamespace(qsa_indexer = object()), object, {})


def test_qsa_tensor_parallelism_fails_before_exporting_an_incomplete_indexer():
    attention = object.__new__(Attention)
    attention.qsa_indexer = object()
    with pytest.raises(RuntimeError, match = "Tensor parallelism.*QSA"):
        attention.tp_export(None, None)


def test_qsa_sparse_route_crosses_threshold_only_after_dense_exact_limit():
    indexer = _indexer()
    threshold = indexer.sparse_threshold()
    assert [indexer.uses_sparse_cache(torch.tensor([past]), 1) for past in (
        threshold - 2, threshold - 1, threshold,
    )] == [False, False, True]


@pytest.mark.parametrize("k_bits,v_bits", [(8, 8), (8, 4), (6, 6), (4, 4)])
@torch.inference_mode()
def test_qsa_quant_sparse_gather_crosses_a_scrambled_page_boundary_online(monkeypatch, k_bits, v_bits):
    torch.manual_seed(1201)
    layer = _layer(k_bits = k_bits, v_bits = v_bits)
    indexer = _indexer()
    total = PAGE_SIZE + 12
    table = torch.tensor([[1, 0]], device = DEVICE, dtype = torch.int32)
    seqlens = torch.zeros((1,), device = DEVICE, dtype = torch.int32)
    k = torch.zeros((1, total, 1, 32), device = DEVICE, dtype = torch.float16)
    v = torch.full_like(k, 0.5)
    v[:, PAGE_SIZE:] = -0.5
    _update(layer, k, v, seqlens, table)
    seqlens.fill_(total - 1)

    selected = torch.full((1, indexer.k_pad()), -1, device = DEVICE, dtype = torch.int32)
    selected[0, :6] = torch.tensor([0, 1, PAGE_SIZE, PAGE_SIZE + 1, PAGE_SIZE + 2, PAGE_SIZE + 3], device = DEVICE)
    assert (selected[0, :2] < PAGE_SIZE).all() and (selected[0, 2:6] >= PAGE_SIZE).all()
    q = torch.randn((1, 1, 2, 32), device = DEVICE, dtype = torch.float16)
    q_idx = torch.randn_like(q)
    monkeypatch.setattr(indexer, "select_indices_paged", lambda *_args: selected)

    def no_full_cache(*_args, **_kwargs):
        pytest.fail(f"Q{v_bits} QSA sparse attention reconstructed a full fp16 cache")

    monkeypatch.setattr(layer, "get_kv", no_full_cache)
    online = indexer.sparse_attend(
        layer, SimpleNamespace(num_q_heads = 2, num_kv_heads = 1, head_dim = 32, sm_scale = 32 ** -0.5),
        q, q_idx, table, seqlens.cpu(),
    )

    monkeypatch.undo()
    # get_kv fully dequantizes H32-packed storage back to the original domain, independently
    # checking the online packed loader against the exact selected logical rows.
    expected = _selected_oracle(layer, q, selected, table, seqlens + 1, 32 ** -0.5)
    assert expected.float().mean() < -0.1  # both logical pages contribute different values
    tol = 3e-2 if (k_bits, v_bits) == (8, 8) else 5e-2
    torch.testing.assert_close(online[0, 0], expected, atol = tol, rtol = tol)


@torch.inference_mode()
def test_qsa_fp16_sparse_gather_regression_matches_selected_rows():
    # BC compiles this exact fp16 kernel and launches it by positional ABI. Keep Q8 on its
    # own entry point so that compilation remains compatible with existing CUDA BC artifacts.
    from exllamav3.modules.attention_fn import qsa_triton

    tree = ast.parse(Path(qsa_triton.__file__).read_text())
    functions = {node.name: node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}
    assert [arg.arg for arg in functions["_qsa_sparse_split_kernel"].args.args] == [
        "q", "k_cache", "v_cache", "block_table", "indices", "partial_o", "partial_ml",
        "k_len", "num_pages_per_seq", "num_splits", "split_len", "n_q_heads", "n_kv_heads",
        "page_size", "head_dim", "K_pad", "scale", "BLOCK_H", "BLOCK_N", "PAGED",
    ]
    assert [arg.arg for arg in functions["_qsa_sparse_split_quant_kernel"].args.args][7:10] == [
        "k_scales", "v_scales", "h32",
    ]
    assert "K_BITS: tl.constexpr" in Path(qsa_triton.__file__).read_text()
    assert "V_BITS: tl.constexpr" in Path(qsa_triton.__file__).read_text()
    launcher = ast.get_source_segment(Path(qsa_triton.__file__).read_text(), functions["qsa_sparse_attend_rows"])
    assert "_qsa_sparse_split_kernel[(programs, splits)]" in launcher
    assert "_qsa_sparse_split_quant_kernel[(programs, splits)]" in launcher
    assert "partial_o, partial_ml, o, partial_ml, splits, partial_ml,\n                    QCV = 0" in launcher
    assert "partial_o, partial_ml, o, h32, splits, q,\n                    QCV = v_bits" in launcher

    torch.manual_seed(1202)
    layer = CacheLayer_qsa(None, _attention(), 0, 2 * PAGE_SIZE)
    layer.alloc(DEVICE)
    indexer = _indexer()
    total = indexer.sparse_threshold() + 1
    table = torch.tensor([[1, 0]], device = DEVICE, dtype = torch.int32)
    seqlens = torch.zeros((1,), device = DEVICE, dtype = torch.int32)
    k = torch.randn((1, total, 1, 32), device = DEVICE, dtype = torch.float16)
    v = torch.randn_like(k)
    _update(layer, k, v, seqlens, table)
    seqlens.fill_(total - 1)
    layer.pooled[1, :total // 4].normal_()
    q = torch.randn((1, 1, 2, 32), device = DEVICE, dtype = torch.float16)
    q_idx = torch.randn_like(q)
    indices = indexer.select_indices_paged(layer, q_idx, table, seqlens.cpu())
    out = indexer.sparse_attend(
        layer, SimpleNamespace(num_q_heads = 2, num_kv_heads = 1, head_dim = 32, sm_scale = 32 ** -0.5),
        q, q_idx, table, seqlens.cpu(),
    )
    torch.testing.assert_close(out[0, 0], _selected_oracle(layer, q, indices, table, seqlens + 1, 32 ** -0.5), atol = 3e-2, rtol = 3e-2)


@pytest.mark.parametrize("bits", [4, 8])
@torch.inference_mode()
def test_qsa_companded_sparse_keeps_the_existing_dequantized_gather_fallback(monkeypatch, bits):
    compand_a = 0.65
    torch.manual_seed(1203)
    layer = _layer(bits = bits, compand_a = compand_a)
    indexer = _indexer()
    total = indexer.sparse_threshold() + 1
    table = torch.tensor([[0, 1]], device = DEVICE, dtype = torch.int32)
    seqlens = torch.zeros((1,), device = DEVICE, dtype = torch.int32)
    k = torch.randn((1, total, 1, 32), device = DEVICE, dtype = torch.float16)
    _update(layer, k, -k, seqlens, table)
    seqlens.fill_(total - 1)
    layer.pooled[0, :total // 4].normal_()
    q = torch.randn((1, 1, 2, 32), device = DEVICE, dtype = torch.float16)
    q_idx = torch.randn_like(q)
    indices = indexer.select_indices_paged(layer, q_idx, table, seqlens.cpu())
    expected = _selected_oracle(layer, q, indices, table, seqlens + 1, 32 ** -0.5)
    calls = {"dequant": 0}
    get_kv = layer.get_kv

    def spy(*args, **kwargs):
        calls["dequant"] += 1
        return get_kv(*args, **kwargs)

    monkeypatch.setattr(layer, "get_kv", spy)
    out = indexer.sparse_attend(
        layer, SimpleNamespace(num_q_heads = 2, num_kv_heads = 1, head_dim = 32, sm_scale = 32 ** -0.5),
        q, q_idx, table, seqlens.cpu(),
    )
    assert calls["dequant"] == 1
    torch.testing.assert_close(out[0, 0], expected, atol = 3e-2, rtol = 3e-2)


@torch.inference_mode()
def test_qsa_q8_partial_head_groups_use_dequantized_fallback(monkeypatch):
    layer = _layer(head_dim = 16, kv_heads = 2)
    indexer = _indexer()
    table = torch.tensor([[1, 0]], device = DEVICE, dtype = torch.int32)
    seqlens = torch.zeros((1,), device = DEVICE, dtype = torch.int32)
    k = torch.randn((1, 8, 2, 16), device = DEVICE, dtype = torch.float16)
    _update(layer, k, -k, seqlens, table)
    seqlens.fill_(7)
    q = torch.randn((1, 1, 2, 16), device = DEVICE, dtype = torch.float16)
    q_idx = torch.randn((1, 1, indexer.n_heads, indexer.head_dim), device = DEVICE, dtype = torch.float16)
    indices = indexer.select_indices_paged(layer, q_idx, table, seqlens.cpu())
    expected = _selected_oracle(layer, q, indices, table, seqlens + 1, 16 ** -0.5)
    calls = {"dequant": 0}
    get_kv = layer.get_kv

    def spy(*args, **kwargs):
        calls["dequant"] += 1
        return get_kv(*args, **kwargs)

    monkeypatch.setattr(layer, "get_kv", spy)
    out = indexer.sparse_attend(
        layer, SimpleNamespace(num_q_heads = 2, num_kv_heads = 2, head_dim = 16, sm_scale = 16 ** -0.5),
        q, q_idx, table, seqlens.cpu(),
    )
    assert calls["dequant"] == 1
    torch.testing.assert_close(out[0, 0], expected, atol = 3e-2, rtol = 3e-2)


@pytest.mark.parametrize("k_bits,v_bits", [(4, 8), (8, 4)])
@torch.inference_mode()
def test_qsa_q8_partial_head_groups_use_dequantized_fallback_asym(monkeypatch, k_bits, v_bits):
    layer = _layer(head_dim = 16, kv_heads = 2, k_bits = k_bits, v_bits = v_bits)
    indexer = _indexer()
    table = torch.tensor([[1, 0]], device = DEVICE, dtype = torch.int32)
    seqlens = torch.zeros((1,), device = DEVICE, dtype = torch.int32)
    k = torch.randn((1, 8, 2, 16), device = DEVICE, dtype = torch.float16)
    _update(layer, k, -k, seqlens, table)
    seqlens.fill_(7)
    q = torch.randn((1, 1, 2, 16), device = DEVICE, dtype = torch.float16)
    q_idx = torch.randn((1, 1, indexer.n_heads, indexer.head_dim), device = DEVICE, dtype = torch.float16)
    indices = indexer.select_indices_paged(layer, q_idx, table, seqlens.cpu())
    expected = _selected_oracle(layer, q, indices, table, seqlens + 1, 16 ** -0.5)
    calls = {"dequant": 0}
    get_kv = layer.get_kv

    def spy(*args, **kwargs):
        calls["dequant"] += 1
        return get_kv(*args, **kwargs)

    monkeypatch.setattr(layer, "get_kv", spy)
    out = indexer.sparse_attend(
        layer, SimpleNamespace(num_q_heads = 2, num_kv_heads = 2, head_dim = 16, sm_scale = 16 ** -0.5),
        q, q_idx, table, seqlens.cpu(),
    )
    assert calls["dequant"] == 1
    torch.testing.assert_close(out[0, 0], expected, atol = 3e-2, rtol = 3e-2)


def test_bc_qsa_declines_valid_q8_qsa_layer(monkeypatch):
    from exllamav3.modules.attention_fn import bc_attn

    monkeypatch.setattr(bc_attn, "_module_eligible", lambda _module: True)
    layer = _layer(head_dim = 32, kv_heads = 1)
    module = SimpleNamespace(device = DEVICE, qsa_indexer = object(), head_dim = 32)
    assert bc_attn.build_bc_attn(module, layer) is None


@torch.inference_mode()
def test_qsa_q8_sparse_gather_uses_the_callers_current_stream():
    from exllamav3.modules.attention_fn.qsa_triton import qsa_sparse_attend_rows

    layer = _layer()
    table = torch.tensor([[0, 1]], device = DEVICE, dtype = torch.int32)
    seqlens = torch.zeros((1,), device = DEVICE, dtype = torch.int32)
    k = torch.full((1, 8, 1, 32), 0.25, device = DEVICE, dtype = torch.float16)
    _update(layer, k, -k, seqlens, table)
    seqlens.fill_(7)
    indices = torch.arange(8, device = DEVICE, dtype = torch.int32).view(1, -1)
    q = torch.ones((1, 2, 32), device = DEVICE, dtype = torch.float16)
    with torch.cuda.device(DEVICE):
        stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        out = qsa_sparse_attend_rows(
            q, layer.qk.view(-1, 1, 8), layer.qv.view(-1, 1, 8), indices, 32 ** -0.5,
            block_table = table, page_size = PAGE_SIZE,
            qc = (layer.sk.view(-1, 1, 1), layer.sv.view(-1, 1, 1), 8, 8),
        )
        consumed = out.clone()
    stream.synchronize()
    expected = _selected_oracle(layer, q.view(1, 1, 2, 32), indices, table, seqlens + 1, 32 ** -0.5)
    torch.testing.assert_close(consumed[0], expected, atol = 3e-2, rtol = 3e-2)
