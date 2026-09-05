# GPU lifetime tests for the NGramEmbedding reusable pinned staging buffers
# (uids/inverse/heads/packed). The fast path refills them on the CPU every forward
# while the previous fill's H2D copies may still be queued on the consuming device:
# a delayed DMA read must observe the data that was staged when its copy was issued,
# so the module fences each refill on per-device events recorded after the copies.
#
# Delay mechanism: torch.cuda._sleep holds the current stream on-device, guaranteeing
# that copies enqueued behind it have not executed when the host proceeds. An event
# recorded after fill A must still be pending when fill B's host code runs — that is
# the measured proof that the reuse window is open (not inferred from timing).
#
# The module under test is the real NGramEmbedding fast path (real ngram_hash_cpu /
# _gather_rows / non_blocking H2D / ngram_dequant); only the table contents are
# synthetic (trellis_ram mode) and the reference oracle is the module's own
# forward_reference, which bypasses the staging buffers entirely.

import pytest
import torch

from exllamav3.modules.ngram_embedding import NGramEmbedding
from exllamav3.modules.quant.exl3_lib.ngram_codec import mul1_codebook, words_per_row

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU only")

DEV = torch.device("cuda", 0)
NGRAM_SIZE = 3
HPN = 2
H = (NGRAM_SIZE - 1) * HPN
K = 4
WORDS = words_per_row(K)
ROWS = 5000
EOS = 10_000_000  # never appears in generated ids
SLEEP_CYCLES = int(4e8)  # device-side hold long enough to outlast all host code below


def _make_module():
    torch.manual_seed(7)
    sizes = torch.tensor([997, 1009, 1013, 1019], dtype=torch.int64)
    offsets = torch.tensor([0, 997, 2006, 3019], dtype=torch.int64)
    mults = torch.tensor([31, 37, 41], dtype=torch.int64)
    mod = NGramEmbedding(None, "test.ngram", NGRAM_SIZE, HPN, H * 160, EOS)
    mod.device = DEV
    mod.mode = "trellis_ram"
    mod.K = K
    raw = torch.randint(-2**15, 2**15 - 1, (ROWS, WORDS), dtype=torch.int16)
    raw[:, 0] = 0x3C00  # fp16 scale word = 1.0; random bits can encode NaN and would
                        # degrade the fast-vs-reference oracle (unrelated to the race)
    mod.tables = [raw]
    mod.rows_per_shard = ROWS
    mod.num_rows = ROWS
    mod.head_vocab_sizes = sizes
    mod.head_offsets = offsets
    mod.layer_multipliers = mults
    mod.head_bias = torch.zeros((H, 160), dtype=torch.half, device=DEV)
    mod.codebook = mul1_codebook(DEV)
    return mod


def _ids(seed, hi=32000):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, hi, (1, NGRAM_SIZE - 1 + 64), generator=g)


def _race(module, ids_a, ids_b):
    """fill A behind a device hold, refill B with no host sync in between; returns
    (out_a, out_b, window_open)."""
    torch.cuda._sleep(SLEEP_CYCLES)
    out_a = module.forward(ids_a, {})
    ev = torch.cuda.Event()
    ev.record()  # fires only after A's queued copies have executed
    window_open = not ev.query()
    out_b = module.forward(ids_b, {})  # CPU refill of the same staging buffers
    torch.cuda.synchronize()
    return out_a, out_b, window_open


def test_fast_path_matches_reference_synced():
    # Oracle sanity: with a sync between fills, the fast path and the staging-free
    # reference agree bitwise.
    with torch.cuda.device(DEV):
        mod = _make_module()
        for seed in (11, 23, 47):
            ids = _ids(seed)
            out = mod.forward(ids, {})
            torch.cuda.synchronize()
            assert torch.equal(out, mod.forward_reference(ids, {}))


def test_refill_fenced_against_delayed_copies():
    # The race: B's CPU refill happens while A's H2D reads are provably still queued
    # (window_open). The event fence must make B wait, so A's output stays intact.
    # Before the fence existed this sequence corrupted A (and could fault the device
    # out-of-bounds when B's overwritten inverse exceeded A's unique-row count).
    with torch.cuda.device(DEV):
        mod = _make_module()
        mod.forward(_ids(11), {})
        mod.forward(_ids(23), {})
        torch.cuda.synchronize()
        out_a, out_b, window_open = _race(mod, _ids(11), _ids(23))
        assert window_open, "delayed-copy window did not open; harness invalid"
        assert torch.equal(out_a, mod.forward_reference(_ids(11), {}))
        assert torch.equal(out_b, mod.forward_reference(_ids(23), {}))


def test_refill_fenced_inbounds_corruption_signature():
    # Same race with B drawn from a tiny token range so B's unique-row count (and thus
    # its inverse values) stays below A's: pre-fix this corrupted A's gather in-bounds
    # (A's rows silently became B's) without faulting the device. The fence must keep
    # A's output equal to its own reference.
    with torch.cuda.device(DEV):
        mod = _make_module()
        mod.forward(_ids(11, 32000), {})
        mod.forward(_ids(23, 8), {})
        torch.cuda.synchronize()
        out_a, out_b, window_open = _race(mod, _ids(11, 32000), _ids(23, 8))
        assert window_open
        assert torch.equal(out_a, mod.forward_reference(_ids(11, 32000), {}))
        assert torch.equal(out_b, mod.forward_reference(_ids(23, 8), {}))


def test_shared_module_interleaved_fills():
    # Cross-job pattern: one shared module instance, three fills alternating with no
    # host sync and the device held busy. Every fill except the first overlaps a
    # previous fill's queued copies.
    with torch.cuda.device(DEV):
        mod = _make_module()
        seeds = [11, 23, 47, 83, 97]
        for s in seeds:
            mod.forward(_ids(s), {})
        torch.cuda.synchronize()
        torch.cuda._sleep(SLEEP_CYCLES)
        outs = [mod.forward(_ids(s), {}) for s in seeds]
        torch.cuda.synchronize()
        for s, out in zip(seeds, outs):
            assert torch.equal(out, mod.forward_reference(_ids(s), {}))


def test_growth_refill_while_pending():
    # Buffer growth replaces the staging tensors while the previous fill's copies are
    # still queued. The fence runs before _grow_pin, so the old buffers are retired
    # (DMA-complete) before they are freed and the new fill is safe.
    with torch.cuda.device(DEV):
        mod = _make_module()
        short = _ids(11)
        long_ids = torch.randint(0, 32000, (1, NGRAM_SIZE - 1 + 256),
                                 generator=torch.Generator().manual_seed(23))
        mod.forward(short, {})
        torch.cuda.synchronize()
        torch.cuda._sleep(SLEEP_CYCLES)
        out_a = mod.forward(short, {})
        out_b = mod.forward(long_ids, {})  # grows uids/inverse/heads/packed
        torch.cuda.synchronize()
        assert torch.equal(out_a, mod.forward_reference(short, {}))
        assert torch.equal(out_b, mod.forward_reference(long_ids, {}))


def test_unload_clears_pin_events():
    mod = _make_module()
    with torch.cuda.device(DEV):
        mod.forward(_ids(11), {})
        torch.cuda.synchronize()
    assert mod._pin or mod._pin_events
    mod.unload()
    assert mod._pin is None and mod._pin_events == {}