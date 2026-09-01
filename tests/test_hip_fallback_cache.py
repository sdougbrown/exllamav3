"""Focused ROCm contracts for the PyTorch q-cache and DSA fallbacks."""
from __future__ import annotations

import pytest
import torch

IS_ROCM = torch.version.hip is not None and torch.cuda.is_available()
if IS_ROCM:
    from exllamav3.ext import exllamav3_ext as ext
else:
    ext = None

pytestmark = pytest.mark.skipif(not IS_ROCM, reason = "ROCm fallback contract tests")

DEVICE = "cuda"
PAGE = 256


def _cache(pool: int, bits: int, dim: int = 32):
    groups = dim // 32
    return (
        torch.zeros((pool, PAGE, groups * bits), device = DEVICE, dtype = torch.int32),
        torch.zeros((pool, PAGE, groups), device = DEVICE, dtype = torch.float16),
    )


@pytest.mark.parametrize("bits", range(2, 9))
@pytest.mark.parametrize("compand_a", [0.0, 0.65])
def test_quant_cache_cont_all_bitrates_are_finite(bits, compand_a):
    torch.manual_seed(0)
    # The mix of endpoints and tiny values exercises midpoint boundaries after H32.
    endpoints = torch.tensor(
        [-1.0, -0.8751, -0.875, -0.8749, -0.5001, -0.5, -0.4999, -0.1251,
         -0.125, -0.1249, 0.1249, 0.125, 0.1251, 0.4999, 0.5, 0.5001,
         0.8749, 0.875, 0.8751, 1.0, -0.75, -0.25, 0.25, 0.75,
         -0.0625, 0.0625, -0.9375, 0.9375, -0.375, 0.375, -0.625, 0.625],
        device = DEVICE,
    )
    x = torch.cat((
        endpoints,
        torch.tensor([0.0, -0.0, 2.0 ** -24, -(2.0 ** -24)] * 8, device = DEVICE),
        torch.randn(64, device = DEVICE) * 0.2,
    )).reshape(4, 32).half()
    packed = torch.empty((4, bits), device = DEVICE, dtype = torch.int32)
    scales = torch.empty((4, 1), device = DEVICE, dtype = torch.float16)
    restored = torch.empty_like(x)
    ext.quant_cache_cont(x, packed, scales, compand_a)
    ext.dequant_cache_cont(packed, scales, restored, compand_a)
    assert torch.isfinite(scales).all()
    assert torch.isfinite(restored).all()
    err = restored.float() - x.float()
    rmse = torch.sqrt((err * err).mean()).item()
    cosine = torch.nn.functional.cosine_similarity(restored.float().flatten(), x.float().flatten(), dim = 0).item()
    assert rmse < {2: .16, 3: .09, 4: .05, 5: .03, 6: .015, 7: .008, 8: .005}[bits]
    assert cosine > 0.90


def test_quant_cache_cont_validates_shapes_and_dtypes():
    x = torch.zeros((1, 32), device = DEVICE, dtype = torch.float16)
    packed = torch.empty((1, 4), device = DEVICE, dtype = torch.int32)
    scales = torch.empty((1, 1), device = DEVICE, dtype = torch.float16)
    with pytest.raises(TypeError):
        ext.quant_cache_cont(x.float(), packed, scales)
    with pytest.raises(ValueError):
        ext.quant_cache_cont(torch.empty((1, 31), device = DEVICE, dtype = torch.float16), packed, scales)
    with pytest.raises(ValueError):
        ext.quant_cache_cont(x, torch.empty((1, 9), device = DEVICE, dtype = torch.int32), scales)
    with pytest.raises(TypeError):
        ext.dequant_cache_cont(packed, scales.float(), x)


@pytest.mark.parametrize("k_bits,v_bits", [(2, 8), (8, 2), (3, 7)])
@pytest.mark.parametrize("dim", [64, 128])
def test_paged_cache_append_dequant_window_and_compact_scratch(k_bits, v_bits, dim):
    pool, bsz, pages = 6, 2, 2
    table = torch.tensor([[3, 1], [5, 0]], device = DEVICE, dtype = torch.int32)
    lengths = torch.tensor([254, 254], device = DEVICE, dtype = torch.int32)
    kq, ks = _cache(pool, k_bits, dim)
    vq, vs = _cache(pool, v_bits, dim)
    k_new = torch.stack((torch.full((2, 1, dim), 0.25), torch.full((2, 1, dim), -0.5))).half().to(DEVICE)
    v_new = -k_new
    ext.quant_cache_paged(k_new, kq, ks, v_new, vq, vs, lengths, table, PAGE, 2, 0.65, True)
    lengths.add_(2)

    kd = torch.full((pool, PAGE, 1, dim), float("nan"), device = DEVICE, dtype = torch.float16)
    vd = torch.full_like(kd, float("nan"))
    ext.dequant_cache_paged(kq, ks, kd, vq, vs, vd, lengths, table, PAGE, -1, 0.65)
    for batch in range(bsz):
        physical = table[batch, 0].item()
        assert torch.isfinite(kd[physical, 254:256]).all()
        assert torch.isfinite(vd[physical, 254:256]).all()
    assert torch.isnan(kd[2]).all()  # an unmapped pool page remains untouched

    window = torch.full_like(kd, float("nan"))
    ext.dequant_cache_paged(kq, ks, window, vq, vs, torch.full_like(kd, float("nan")), lengths, table, PAGE, 1, 0.65)
    for batch in range(bsz):
        physical = table[batch, 0].item()
        # Native dequant skips whole launch blocks, so the block overlapping the one-token
        # window is decoded in full.
        assert torch.isfinite(window[physical, 254:256]).all()

    scratch_k = torch.full((bsz * pages, PAGE, 1, dim), float("nan"), device = DEVICE, dtype = torch.float16)
    scratch_v = torch.full_like(scratch_k, float("nan"))
    lengths.sub_(2)
    ext.dequant_cache_paged_window(kq, ks, scratch_k, vq, vs, scratch_v, lengths, table, PAGE, 2, 0.65)
    for batch in range(bsz):
        assert torch.isfinite(scratch_k[batch * pages, 254:256]).all()
        assert torch.isfinite(scratch_v[batch * pages, 254:256]).all()
        assert torch.isnan(scratch_k[batch * pages + 1]).all()


@pytest.mark.parametrize("dim", [64])
def test_dequant_cache_paged_window_overlaps_chunk_boundary(dim):
    kq, ks = _cache(2, 8, dim)
    vq, vs = _cache(2, 8, dim)
    table = torch.tensor([[0, 1]], device = DEVICE, dtype = torch.int32)
    groups = dim // 32
    tokens_per_block = 256 // ((groups + 3) // 4)
    total = tokens_per_block + 5
    lengths = torch.zeros((1,), device = DEVICE, dtype = torch.int32)
    src = torch.arange(total, device = DEVICE, dtype = torch.float16).view(1, -1, 1).expand(1, -1, dim).contiguous()
    ext.quant_cache_paged(src, kq, ks, src, vq, vs, lengths, table, PAGE, total, 0.0, True)
    lengths.fill_(total)
    outk = torch.full((2, PAGE, 1, dim), float("nan"), device = DEVICE, dtype = torch.float16)
    outv = torch.full_like(outk, float("nan"))
    ext.dequant_cache_paged(kq, ks, outk, vq, vs, outv, lengths, table, PAGE, 4, 0.0)
    assert torch.isnan(outk[0, tokens_per_block - 1]).all()
    assert torch.isfinite(outk[0, tokens_per_block:tokens_per_block + 5]).all()
    assert torch.isfinite(outk[1, :5]).all()


def _selected(out: torch.Tensor) -> torch.Tensor:
    return out[out >= 0].to(torch.long)


@pytest.mark.parametrize("bound", [0, 1, 8, 99])
def test_dsa_topk_scalar_t_ptr_never_emits_out_of_bound_indices(bound):
    scores = torch.arange(32, device = DEVICE, dtype = torch.float16).reshape(4, 8)
    out = torch.empty((4, 4), device = DEVICE, dtype = torch.int32)
    ext.dsa_topk(scores, out, 4, torch.tensor([bound], device = DEVICE, dtype = torch.int32), 0)
    for row in out:
        selected = _selected(row)
        limit = min(max(bound, 0), 8)
        expected = torch.arange(max(0, limit - 4), limit, device = DEVICE, dtype = torch.long)
        assert torch.equal(selected, expected)
        assert (row[selected.numel():] == -1).all()


def test_dsa_topk_per_sequence_bounds_special_scores_and_strides():
    # CUDA ranks the fp16 bit key, then emits selected entries in ascending input-index order.
    raw = torch.tensor([-1024, 0x7c00, 0x7e00, -512, -32768, 0, 1, -32767, 0x7bff, -1025], device = DEVICE, dtype = torch.int16)
    special = raw.view(torch.float16)
    storage = torch.empty((4, 16), device = DEVICE, dtype = torch.float16)
    storage[:, :10] = special
    scores = storage[:, :10]  # row-strided, dense innermost dimension
    out = torch.empty((4, 6), device = DEVICE, dtype = torch.int32)
    bounds = torch.tensor([0, 1, 10, 99], device = DEVICE, dtype = torch.int32)
    ext.dsa_topk(scores, out, 5, bounds, 1)
    assert _selected(out[0]).numel() == 0
    assert _selected(out[1]).numel() == 0  # the only in-bound value is -inf
    assert torch.equal(_selected(out[2]), torch.tensor([1, 2, 5, 6, 8], device = DEVICE))
    assert torch.equal(_selected(out[3]), torch.tensor([1, 2, 5, 6, 8], device = DEVICE))

    all_valid = torch.empty((1, 9), device = DEVICE, dtype = torch.int32)
    ext.dsa_topk(scores[:1], all_valid, 9, torch.tensor([10], device = DEVICE, dtype = torch.int32), 0)
    assert torch.equal(_selected(all_valid[0]), torch.tensor([1, 2, 4, 5, 6, 7, 8, 9], device = DEVICE))

    with pytest.raises(ValueError):
        ext.dsa_topk(scores.t(), torch.empty((10, 6), device = DEVICE, dtype = torch.int32), 1)
    with pytest.raises(ValueError):
        ext.dsa_topk(scores, torch.empty((6, 4), device = DEVICE, dtype = torch.int32).t(), 1)
