# GPU lifetime tests for the generator's pinned prefill uploads:
#
# 1. Job._prefill_staged_cache_seqlens: a per-job two-slot pinned ring. Each slot is
#    refilled only after the per-device events recorded at the end of the chunk that
#    consumed it have fired, so a DMA read of the slot can never observe a newer value.
#    This exercises the real Job methods (generator=None falls back to the current
#    device, mirroring the single-device test rig).
# 2. Sequence.build_block_index_tensor: each call allocates a FRESH pinned tensor, so
#    replacement/cancellation frees the old one while its non_blocking H2D may still be
#    queued. Safety there relies on PyTorch's caching host allocator deferring reuse of
#    blocks with outstanding recorded events — measured here directly, not assumed.
#
# Delay mechanism and window proof as in test_ngram_staging_lifetime.py.

import pytest
import torch

from exllamav3.generator.job import Job

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU only")

DEV = torch.device("cuda", torch.cuda.current_device())
SLEEP_CYCLES = int(4e8)


def _job(monkeypatch):
    monkeypatch.setenv("EXL3_PREFILL_ASYNC_UPLOADS", "1")
    return Job(torch.tensor([[1, 2, 3]]), max_new_tokens = 4)


def test_unfenced_slot_corrupts_under_delay(monkeypatch):
    # Harness validity: the SAME refill pattern with the ring's event wait bypassed
    # corrupts — the delayed DMA read observes the newer value. This proves the test
    # rig can detect the failure mode the ring exists to prevent.
    monkeypatch.setenv("EXL3_PREFILL_ASYNC_UPLOADS", "0")
    job = Job(torch.tensor([[1, 2, 3]]), max_new_tokens = 4)
    slot = torch.zeros(1, dtype = torch.int32, pin_memory = True)
    with torch.cuda.device(DEV):
        torch.cuda._sleep(SLEEP_CYCLES)
        d1 = slot.to(DEV, non_blocking = True)   # copy of the first value queued behind the hold
        slot[0] = 99                              # unfenced immediate refill
        torch.cuda.synchronize()
        assert d1.item() == 99, "unfenced refill did not leak the newer value to the device"


def test_cache_seqlens_ring_waits_for_dma(monkeypatch):
    job = _job(monkeypatch)
    with torch.cuda.device(DEV):
        torch.cuda._sleep(SLEEP_CYCLES)
        s0 = job._prefill_staged_cache_seqlens(5)      # parity 0
        assert job._prefill_current_parity == 0
        d0 = s0.to(DEV, non_blocking = True)           # consumer copy, queued behind hold
        job._prefill_record_cache_seqlens_dma()        # end-of-chunk event, generator None -> dev set {0}
        s1 = job._prefill_staged_cache_seqlens(7)      # parity 1: no wait by design
        assert job._prefill_current_parity == 1
        d1 = s1.to(DEV, non_blocking = True)
        s2 = job._prefill_staged_cache_seqlens(9)      # parity 0 refill: must fence on the event
        assert s2 is s0, "ring did not alternate back to slot 0"
        torch.cuda.synchronize()
        assert d0.item() == 5, "slot refilled before its DMA read completed"
        assert d1.item() == 7


def test_cache_seqlens_ring_double_wrap(monkeypatch):
    # Two full ring cycles under sustained device backlog: slot values must remain
    # exactly what each queued copy was issued with.
    job = _job(monkeypatch)
    with torch.cuda.device(DEV):
        torch.cuda._sleep(SLEEP_CYCLES)
        dev_vals = []
        for v in (5, 7, 9, 11, 13, 15):
            s = job._prefill_staged_cache_seqlens(v)
            dev_vals.append(s.to(DEV, non_blocking = True))
            job._prefill_record_cache_seqlens_dma()
        torch.cuda.synchronize()
        assert [d.item() for d in dev_vals] == [5, 7, 9, 11, 13, 15]


def test_cache_seqlens_ring_covers_second_device(monkeypatch):
    # Real serving config spans two devices ([30, 30]); each layer's get_for_device
    # upload runs on that layer device's current stream, so the end-of-chunk fence
    # must record on EVERY consuming device. With events recorded for devices 0 and 1
    # but the consuming copy queued on device 1 only, the parity-0 refill must block
    # on the device-1 event: a missing per-device event would let the host overwrite
    # the slot before the DMA read.
    if not torch.cuda.device_count() >= 2:
        pytest.skip("requires at least two CUDA devices")
    job = _job(monkeypatch)
    dev1 = torch.device("cuda", 1)

    class _FakeModel:
        active_devices = [0, 1]

    class _FakeGen:
        model = _FakeModel()
        draft_model = None

    job.generator = _FakeGen()
    s0 = job._prefill_staged_cache_seqlens(5)
    with torch.cuda.device(dev1):
        torch.cuda._sleep(SLEEP_CYCLES)              # hold device 1's stream
        d0 = s0.to(dev1, non_blocking = True)        # queued behind the hold on dev 1
    job._prefill_record_cache_seqlens_dma()          # events on dev 0 AND dev 1
    ev1 = job._prefill_cs_events[job._prefill_current_parity][1]
    assert not ev1.query(), "device-1 event fired before its queued copy"
    job._prefill_staged_cache_seqlens(7)
    job._prefill_staged_cache_seqlens(9)         # parity-0 refill: must fence on dev-1 event
    torch.cuda.synchronize()
    assert d0.item() == 5, "slot refilled before the device-1 DMA read completed"


def test_block_index_replacement_survives_delayed_copy(monkeypatch):
    # build_block_index_tensor replacement: the old pinned tensor is freed while its
    # H2D copy is still queued; the caching host allocator must defer reuse of the
    # block until the recorded copy event fires. Mirrors Sequence page (re)allocation
    # and requeue, where block_index_tensor is rebuilt from scratch.
    monkeypatch.setenv("EXL3_PREFILL_ASYNC_UPLOADS", "1")
    with torch.cuda.device(DEV):
        torch.cuda._sleep(SLEEP_CYCLES)
        pairs = []
        for i in range(64):
            old = torch.full((1, 16), 1000 + i, dtype = torch.int32).pin_memory()
            d_old = old.to(DEV, non_blocking = True)
            del old                                   # freed while copy is queued
            new = torch.full((1, 16), 2000 + i, dtype = torch.int32).pin_memory()
            d_new = new.to(DEV, non_blocking = True)
            del new
            pairs.append((d_old, d_new))
        torch.cuda.synchronize()
        for i, (d_old, d_new) in enumerate(pairs):
            assert d_old[0, 0].item() == 1000 + i, \
                f"freed pinned block was reused before its queued copy read it (iter {i})"
            assert d_new[0, 0].item() == 2000 + i


def test_block_index_growth_under_delayed_copy(monkeypatch):
    # Buffer growth: successive rebuilds with growing page counts free progressively
    # larger pinned blocks while copies of earlier (smaller) ones are still queued.
    # Each allocation carries a unique sentinel pattern (iteration-tagged values) so
    # any block reuse before its queued copy executed is detectable regardless of
    # size-class overlap.
    monkeypatch.setenv("EXL3_PREFILL_ASYNC_UPLOADS", "1")
    with torch.cuda.device(DEV):
        torch.cuda._sleep(SLEEP_CYCLES)
        outs = []
        for it, pages in enumerate((1, 4, 16, 64, 256)):
            t = (torch.arange(pages, dtype = torch.int32) + (it + 1) * 1000) \
                .pin_memory().view(1, pages)
            d = t.to(DEV, non_blocking = True)
            del t
            outs.append(d)
        torch.cuda.synchronize()
        for it, (pages, d) in enumerate(zip((1, 4, 16, 64, 256), outs)):
            exp = torch.arange(pages, dtype = torch.int32, device = DEV) + (it + 1) * 1000
            assert torch.equal(d.view(pages), exp), \
                f"freed pinned block reused before its queued copy read it (iter {it})"