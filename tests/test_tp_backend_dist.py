"""
Stage 1 transport backend tests: TPBackendNCCL as a pure torch.distributed backend.

Tier A (mocked): no processes, no GPU. Asserts the backend is a pure distributed
implementation (zero native-extension references, no fallback object), rank mapping,
in-place contribution handling, zero-width gather skipping, and no dtype casts.

Tier B (gloo, 2 spawned host-only processes): broadcast <=2KB and >2KB, exact
all_reduce in fp32/fp16/bf16, contribution=False, uneven / subset / zero-width gathers
with sorted concat order, 50 repeated rounds, clean close, and a single-joiner init
timeout.

Tier C (RCCL, 2 ranks on gfx1201): gated behind EXL3_TP_RCCL_TEST=1 and requires an
exclusive GPU window. Never run on shared GPUs.
"""

import inspect
import multiprocessing
import os
import socket
from datetime import timedelta
from unittest import mock

import pytest
import torch
import torch.distributed as dist

from exllamav3.model.model_tp_backend import TPBackendNCCL


# ---------------------------------------------------------------------------
# Tier A: mocked torch.distributed
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_dist():
    calls = []

    def _record(name):
        def fn(*args, **kwargs):
            calls.append((name, args, kwargs))
            return None
        return fn

    def _recv(tensor, src):
        calls.append(("recv", (tensor,), {"src": src}))
        tensor.fill_(float(src))
        return None

    patchers = [
        mock.patch.object(dist, "init_process_group", _record("init_process_group")),
        mock.patch.object(dist, "all_reduce", _record("all_reduce")),
        mock.patch.object(dist, "broadcast", _record("broadcast")),
        mock.patch.object(dist, "barrier", _record("barrier")),
        mock.patch.object(dist, "destroy_process_group", _record("destroy_process_group")),
        mock.patch.object(dist, "send", _record("send")),
        mock.patch.object(dist, "recv", _recv),
    ]
    for p in patchers:
        p.start()
    try:
        yield calls
    finally:
        for p in patchers:
            p.stop()


def _make_backend(mock_dist, device=0, active_devices=None, output_device=0, backend="gloo", **kw):
    active_devices = [0, 1] if active_devices is None else active_devices
    return TPBackendNCCL(
        device=device,
        active_devices=active_devices,
        output_device=output_device,
        init_method="tcp://127.0.0.1:1",
        master=(device == 0),
        uuid="tier-a",
        backend=backend,
        timeout_s=30.0,
        **kw,
    )


def _calls_of(calls, name):
    return [c for c in calls if c[0] == name]


def test_tier_a_no_native_references():
    """The backend must be pure torch.distributed: no ext.* calls, no native fallback."""
    from exllamav3.model import model_tp_backend as mtb
    src = inspect.getsource(mtb.TPBackendNCCL)
    assert "ext." not in src, "TPBackendNCCL must not reference the native extension"
    assert "TPBackendNative" not in src, "TPBackendNCCL must not construct the native fallback"


def test_tier_a_no_fallback_object(mock_dist):
    b = _make_backend(mock_dist)
    assert not hasattr(b, "fallback")


def test_tier_a_rank_mapping(mock_dist):
    b = _make_backend(mock_dist, device=1, active_devices=[0, 1, 2])
    assert b.rank == 1
    assert b.world_size == 3
    name, args, kwargs = mock_dist[0]
    assert name == "init_process_group"
    assert kwargs["rank"] == 1
    assert kwargs["world_size"] == 3
    assert kwargs["timeout"] == timedelta(seconds=30.0)
    assert kwargs["device_id"] is None  # gloo: no device id


def test_tier_a_nccl_device_id(mock_dist):
    """The nccl path sets device_id; the warmup must not touch a GPU in tests."""
    with mock.patch("torch.ones", return_value=torch.ones((6,))):
        _make_backend(mock_dist, device=0, backend="nccl")
    name, args, kwargs = mock_dist[0]
    assert name == "init_process_group"
    assert kwargs["device_id"] == torch.device(0)


def test_tier_a_all_reduce_contribution_false(mock_dist):
    b = _make_backend(mock_dist)
    t = torch.full((8,), 3.0)
    b.all_reduce(t, contribution=False)
    # zeroed in place, then reduced in place: exactly one all_reduce on the same tensor
    assert (t == 0).all(), "non-contributor tensor must be zeroed before the reduce"
    ars = _calls_of(mock_dist, "all_reduce")
    assert len(ars) == 2  # warmup + this call
    assert ars[-1][1][0] is t, "all_reduce must run on the caller's tensor, no temp copy"


def test_tier_a_all_reduce_contribution_true(mock_dist):
    b = _make_backend(mock_dist)
    t = torch.full((8,), 3.0)
    b.all_reduce(t, contribution=True)
    assert (t == 3.0).all(), "contributor tensor must not be zeroed"
    ars = _calls_of(mock_dist, "all_reduce")
    assert ars[-1][1][0] is t


def test_tier_a_all_reduce_no_dtype_cast(mock_dist):
    b = _make_backend(mock_dist)
    for dtype in (torch.float32, torch.float16, torch.bfloat16):
        t = torch.ones((8,), dtype=dtype)
        b.all_reduce(t)
        assert t.dtype == dtype, f"all_reduce must not cast {dtype}"
        ars = _calls_of(mock_dist, "all_reduce")
        assert ars[-1][1][0] is t
        assert ars[-1][1][0].dtype == dtype


def test_tier_a_broadcast_rank_mapping(mock_dist):
    b = _make_backend(mock_dist, device=1, active_devices=[0, 1, 2])
    t = torch.ones((4,))
    b.broadcast(t, src_device=2)
    bcasts = _calls_of(mock_dist, "broadcast")
    assert len(bcasts) == 1
    assert bcasts[0][1][0] is t
    assert bcasts[0][2]["src"] == 2


def test_tier_a_gather_ldim_zero_skip_dst(mock_dist):
    """Zero-width participants are skipped on the destination: no recv, no concat slot."""
    b = _make_backend(mock_dist, device=0, active_devices=[0, 1, 2], output_device=0)
    t = torch.full((3,), 1.0)
    out = torch.empty((8,))
    b.gather(t, out, [0, 1, 2], 0, [3, 0, 5])
    recvs = _calls_of(mock_dist, "recv")
    assert len(recvs) == 1
    assert recvs[0][2]["src"] == 2
    assert out[:3].tolist() == [1.0] * 3
    assert out[3:].tolist() == [2.0] * 5


def test_tier_a_gather_ldim_zero_skip_sender(mock_dist):
    """A zero-width participant must not send."""
    b = _make_backend(mock_dist, device=1, active_devices=[0, 1, 2], output_device=0)
    t = torch.empty((0,))
    b.gather(t, None, [0, 1, 2], 0, [3, 0, 5])
    assert _calls_of(mock_dist, "send") == []


def test_tier_a_gather_sender_sends(mock_dist):
    b = _make_backend(mock_dist, device=2, active_devices=[0, 1, 2], output_device=0)
    t = torch.full((5,), 4.0)
    b.gather(t, None, [0, 1, 2], 0, [3, 0, 5])
    sends = _calls_of(mock_dist, "send")
    assert len(sends) == 1
    assert sends[0][1][0] is t
    assert sends[0][2]["dst"] == 0


def test_tier_a_gather_output_last(mock_dist):
    """Production ordering: output device last in active_devices, gather still works."""
    b = _make_backend(mock_dist, device=1, active_devices=[0, 1], output_device=1)
    t = torch.full((5,), 2.0)
    out = torch.empty((8,))
    b.gather(t, out, [0, 1], 1, [3, 5])
    recvs = _calls_of(mock_dist, "recv")
    assert len(recvs) == 1
    assert recvs[0][2]["src"] == 0
    assert out[:3].tolist() == [0.0] * 3
    assert out[3:].tolist() == [2.0] * 5


def test_tier_a_gather_small_same_path(mock_dist):
    b = _make_backend(mock_dist, device=0, active_devices=[0, 1], output_device=0)
    t = torch.full((2,), 1.0)
    out = torch.empty((6,))
    b.gather_small(t, out, [0, 1], 0, [2, 4])
    recvs = _calls_of(mock_dist, "recv")
    assert len(recvs) == 1
    assert out[:2].tolist() == [1.0] * 2
    assert out[2:].tolist() == [1.0] * 4  # mock recv fills with src rank 1


def test_tier_a_fwd_barrier(mock_dist):
    b = _make_backend(mock_dist)
    b.fwd_barrier()
    assert _calls_of(mock_dist, "barrier")


def test_tier_a_close(mock_dist):
    b = _make_backend(mock_dist)
    b.close()
    assert len(_calls_of(mock_dist, "barrier")) == 1
    assert len(_calls_of(mock_dist, "destroy_process_group")) == 1
    # double close is safe
    b.close()
    assert len(_calls_of(mock_dist, "barrier")) == 1
    assert len(_calls_of(mock_dist, "destroy_process_group")) == 1


def test_tier_a_cpu_slot_noop(mock_dist):
    """The -1 CPU helper slot skips process-group init entirely and close() is a no-op."""
    b = _make_backend(mock_dist, device=-1)
    assert b.device == -1
    b.close()
    assert _calls_of(mock_dist, "init_process_group") == []
    assert _calls_of(mock_dist, "barrier") == []
    assert _calls_of(mock_dist, "destroy_process_group") == []


def test_init_pg_native_guard():
    """init_pg must reject the native backend when the pg_* extension is absent (ROCm)."""
    from exllamav3.ext import exllamav3_ext as ext
    from exllamav3.model.model_tp_fn import init_pg
    if hasattr(ext, "pg_init_context"):
        pytest.skip("native backend available in this build; guard not applicable")
    with mock.patch("torch.cuda.set_device"):
        with pytest.raises(NotImplementedError):
            init_pg(0, [0, 1], 0, {"type": "native", "init_method": "tcp://127.0.0.1:1", "uuid": "x"})


# ---------------------------------------------------------------------------
# Tier B: gloo, 2 spawned host-only processes
# ---------------------------------------------------------------------------

def _free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _tier_b_collectives(backend, rank, world_size, device="cpu"):
    out_device = world_size - 1

    backend.fwd_barrier()

    # broadcast small (<= 2KB) and large (> 2KB)
    for n in (64, 1024):
        expected = torch.arange(n, dtype=torch.float32, device=device) + 1.0
        t = expected.clone() if rank == 0 else torch.empty(n, dtype=torch.float32, device=device)
        backend.broadcast(t, src_device=0)
        assert torch.equal(t, expected), f"broadcast {n} mismatch"

    # all_reduce exact in fp32 / fp16 / bf16
    for dtype in (torch.float32, torch.float16, torch.bfloat16):
        t = torch.full((16,), float(rank + 1), dtype=dtype, device=device)
        backend.all_reduce(t)
        expected = float(world_size * (world_size + 1) // 2)
        assert torch.all(t == expected), f"all_reduce {dtype} mismatch: {t.tolist()}"

    # contribution=False: only rank 0 contributes, both ranks receive the full sum
    t = torch.full((16,), float(rank + 1), dtype=torch.float32, device=device)
    backend.all_reduce(t, contribution=(rank == 0))
    assert torch.all(t == 1.0), f"contribution=False mismatch: {t.tolist()}"

    # gather: uneven widths, sorted concat order
    gd, ldims = [0, 1], [3, 5]
    out = torch.empty((8,), device=device) if rank == out_device else None
    t = torch.full((ldims[rank],), float(rank + 1), device=device)
    backend.gather(t, out, gd, out_device, ldims)
    if rank == out_device:
        assert out[:3].tolist() == [1.0] * 3, f"uneven gather head: {out.tolist()}"
        assert out[3:].tolist() == [2.0] * 5, f"uneven gather tail: {out.tolist()}"

    # gather: zero-width participant (rank 0 contributes nothing)
    gd, ldims = [0, 1], [0, 5]
    out = torch.empty((5,), device=device) if rank == out_device else None
    t = torch.full((ldims[rank],), float(rank + 1), device=device)
    backend.gather(t, out, gd, out_device, ldims)
    if rank == out_device:
        assert out.tolist() == [2.0] * 5, f"zero-width gather: {out.tolist()}"

    # gather: subset (only the output device participates)
    gd, ldims = [1], [7]
    out = torch.empty((7,), device=device) if rank == out_device else None
    t = torch.full((7,), 9.0, device=device)
    backend.gather(t, out, gd, out_device, ldims)
    if rank == out_device:
        assert out.tolist() == [9.0] * 7, f"subset gather: {out.tolist()}"

    # gather: zero-width output device (only rank 0 contributes)
    gd, ldims = [0, 1], [5, 0]
    out = torch.empty((5,), device=device) if rank == out_device else None
    t = torch.full((ldims[rank],), float(rank + 1), device=device)
    backend.gather(t, out, gd, out_device, ldims)
    if rank == out_device:
        assert out.tolist() == [1.0] * 5, f"zero-width dst gather: {out.tolist()}"

    # gather_small: same send/recv path
    gd, ldims = [0, 1], [2, 4]
    out = torch.empty((6,), device=device) if rank == out_device else None
    t = torch.full((ldims[rank],), float(rank + 1), device=device)
    backend.gather_small(t, out, gd, out_device, ldims)
    if rank == out_device:
        assert out[:2].tolist() == [1.0] * 2, f"gather_small head: {out.tolist()}"
        assert out[2:].tolist() == [2.0] * 4, f"gather_small tail: {out.tolist()}"

    # 50 rounds of broadcast + all_reduce
    for i in range(50):
        t = (torch.full((8,), float(i), dtype=torch.float32, device=device) if rank == 0
             else torch.empty((8,), dtype=torch.float32, device=device))
        backend.broadcast(t, src_device=0)
        assert torch.all(t == float(i)), f"round {i} broadcast mismatch"
        backend.all_reduce(t)
        assert torch.all(t == float(i) * world_size), f"round {i} all_reduce mismatch"


def _tier_b_worker(rank, world_size, port, results):
    try:
        backend = TPBackendNCCL(
            device=rank,
            active_devices=list(range(world_size)),
            output_device=world_size - 1,
            init_method=f"tcp://127.0.0.1:{port}",
            master=(rank == 0),
            uuid="tier-b",
            backend="gloo",
            timeout_s=60.0,
        )
        try:
            _tier_b_collectives(backend, rank, world_size)
        finally:
            backend.close()
        assert not dist.is_initialized(), "process group must be destroyed after close"
        results.put(("ok", None))
    except Exception as e:
        results.put(("error", (type(e).__name__, str(e))))


def test_tier_b_gloo_collectives():
    port = _free_port()
    ctx = multiprocessing.get_context("spawn")
    results = ctx.Queue()
    procs = []
    for rank in range(2):
        p = ctx.Process(target=_tier_b_worker, args=(rank, 2, port, results))
        p.start()
        procs.append(p)
    for p in procs:
        p.join(120)
    for p in procs:
        assert p.exitcode == 0, f"Tier B worker exited with code {p.exitcode}"
    outcomes = [results.get(timeout=10) for _ in procs]
    for kind, payload in outcomes:
        assert kind == "ok", f"Tier B worker failed: {payload}"


def _tier_b_timeout_joiner(rank, port, results):
    try:
        TPBackendNCCL(
            device=rank,
            active_devices=[0, 1],
            output_device=0,
            init_method=f"tcp://127.0.0.1:{port}",
            master=True,
            uuid="tier-b-timeout",
            backend="gloo",
            timeout_s=5.0,
        )
        results.put(("no-error", None))
    except Exception as e:
        results.put(("error", (type(e).__name__, str(e))))


def _tier_b_timeout_bystander(results):
    # Never joins the process group; the joiner must time out on its own
    results.put(("ok", None))


def test_tier_b_single_joiner_timeout():
    port = _free_port()
    ctx = multiprocessing.get_context("spawn")
    results = ctx.Queue()
    joiner = ctx.Process(target=_tier_b_timeout_joiner, args=(0, port, results))
    bystander = ctx.Process(target=_tier_b_timeout_bystander, args=(results,))
    joiner.start()
    bystander.start()
    joiner.join(60)
    bystander.join(60)
    assert joiner.exitcode == 0, f"joiner exited with code {joiner.exitcode}"
    assert bystander.exitcode == 0, f"bystander exited with code {bystander.exitcode}"
    outcomes = [results.get(timeout=10) for _ in range(2)]
    kinds = {k for k, _ in outcomes}
    assert "error" in kinds, f"expected the single joiner to time out, got {outcomes}"
    err = next(p for k, p in outcomes if k == "error")
    assert err[0] in ("DistStoreError", "TimeoutError"), f"unexpected exception: {err}"


# ---------------------------------------------------------------------------
# Tier C: RCCL 2-rank on gfx1201 (gated; needs an exclusive GPU window)
# ---------------------------------------------------------------------------

def _tier_c_worker(rank, world_size, port, results):
    try:
        backend = TPBackendNCCL(
            device=rank,
            active_devices=list(range(world_size)),
            output_device=world_size - 1,
            init_method=f"tcp://127.0.0.1:{port}",
            master=(rank == 0),
            uuid="tier-c",
            backend="nccl",
            timeout_s=60.0,
        )
        try:
            _tier_b_collectives(backend, rank, world_size, device=f"cuda:{rank}")
        finally:
            backend.close()
        assert not dist.is_initialized(), "process group must be destroyed after close"
        # Proxy for zero residual KFD handles: no GPU allocations left after teardown
        assert torch.cuda.memory_allocated(rank) == 0, "residual GPU allocations after close"
        results.put(("ok", None))
    except Exception as e:
        results.put(("error", (type(e).__name__, str(e))))


@pytest.mark.skipif(
    os.environ.get("EXL3_TP_RCCL_TEST") != "1",
    reason="Tier C RCCL 2-rank test requires EXL3_TP_RCCL_TEST=1, HIP, and 2 GPUs in an exclusive window",
)
def test_tier_c_rccl_two_rank():
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        pytest.skip("Tier C requires 2 visible HIP devices")
    port = _free_port()
    ctx = multiprocessing.get_context("spawn")
    results = ctx.Queue()
    procs = []
    for rank in range(2):
        p = ctx.Process(target=_tier_c_worker, args=(rank, 2, port, results))
        p.start()
        procs.append(p)
    for p in procs:
        p.join(180)
    for p in procs:
        assert p.exitcode == 0, f"Tier C worker exited with code {p.exitcode}"
    outcomes = [results.get(timeout=10) for _ in procs]
    for kind, payload in outcomes:
        assert kind == "ok", f"Tier C worker failed: {payload}"
