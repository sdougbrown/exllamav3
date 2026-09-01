import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pytest
import torch
from exllamav3 import Config, Model
from exllamav3.ext import exllamav3_ext as ext

torch.set_printoptions(precision = 5, sci_mode = False, linewidth = 200)

# -----------------------------------------------------------------------------------
# ROCm HIP EXL3 decode (GEMV) correctness oracle.
#
# Rationale: on a ROCm-only box (no CUDA GPU available) the HIP decode GEMV cannot be
# diffed against a CUDA reference. Instead we use the engine's own known-correct path as
# the oracle: the `reconstruct=True` path (PyTorch reconstruct of EXL3 weights + rocBLAS
# fp16 GEMM) is the reference, and the `reconstruct=False` path (which the HIP GEMV serves)
# must reproduce it to fp16 tolerance. This is the same comparison pattern test_qgemm.py
# applies to its two internal paths, applied at decode batch sizes (m <= 8, the MMODE-0/1
# regimes) where the GEMV path is the entire story.
#
# ROUTE PROOF (hard requirement): comparing reconstruct=False against reconstruct=True is
# only meaningful if reconstruct=False actually ran the HIP GEMV. On a build where the GEMV
# is not wired, both sides silently take the same reconstruct path and the comparison
# passes vacuously. Therefore every comparison case spies on the extension bindings and
# asserts:
#   * ext.exl3_gemv exists (FAIL, never skip, if the binding is absent), and
#   * it was called for reconstruct=False, and
#   * ext.reconstruct / ext.reconstruct_slice were NOT called (no silent fallback).
#
# Skips cleanly when there is no HIP build / ROCm device / test model present.
#
# Model: set EXL3_TEST_MODEL to a compatible Qwen3.8-27B EXL3 model directory.
# -----------------------------------------------------------------------------------

# Decode regime: batch sizes that hit the GEMV path (rows <= 144) and specifically the
# m==1 fast path plus small-m (2..8). m==1 is autoregressive decode.
BATCH_SIZES = [1, 2, 8]

# Per-arch proj keys exercising decode-relevant shapes. Keys must match the loaded
# model's module names (Qwen3.8 hybrid: "model.language_model" prefix; full-attention
# layers are 3, 7, ...; layer 0 is linear_attention).
TEST_KEYS = [
    ("model.language_model.layers.3.self_attn.q_proj", "model.language_model.layers.3.input_layernorm"),
    ("model.language_model.layers.3.self_attn.k_proj", "model.language_model.layers.3.input_layernorm"),
    ("model.language_model.layers.3.self_attn.v_proj", "model.language_model.layers.3.input_layernorm"),
    ("model.language_model.layers.3.self_attn.o_proj", None),   # in = heads*head_dim, not the hidden norm output
    ("model.language_model.layers.0.mlp.up_proj", "model.language_model.layers.0.post_attention_layernorm"),
    ("model.language_model.layers.0.mlp.gate_proj", "model.language_model.layers.0.post_attention_layernorm"),
    ("model.language_model.layers.0.mlp.down_proj", None),
    ("model.language_model.layers.0.linear_attn.in_proj_qkv", "model.language_model.layers.0.input_layernorm"),
]

# The default Qwen3.8 fixture uses the mul1 codebook. Set EXL3_TEST_MODEL to the
# Qwen3.5-9B fixture to exercise the MCG codebook with the same oracle matrix.
DEFAULT_MODEL = os.path.expanduser("~/Models/Qwen3.8-27B-exl3")
MODEL = os.environ.get("EXL3_TEST_MODEL", DEFAULT_MODEL)


def _roc_available():
    # ROCm build active and at least one visible device
    if not (torch.version.hip and torch.cuda.is_available()):
        return False
    return True


def _require_gfx12():
    if not _roc_available():
        pytest.skip("ROCm build / device not available")
    if not hasattr(ext, "exl3_gemv_supported") or not ext.exl3_gemv_supported(0):
        pytest.skip("synthetic GEMV oracle is limited to gfx1200/gfx1201")
    arch = getattr(torch.cuda.get_device_properties(0), "gcnArchName", "")
    if not arch.startswith(("gfx1200", "gfx1201")):
        pytest.skip(f"HIP GEMV oracle requires gfx1200/gfx1201, got {arch or 'unknown'}")


@pytest.fixture(scope="module")
def model():
    _require_gfx12()
    if not os.path.isdir(MODEL):
        pytest.skip(f"Test model not found: {MODEL} (set EXL3_TEST_MODEL)")
    config = Config.from_directory(MODEL)
    return Model.from_config(config)


def _spy_ext_route():
    """Temporarily wrap ext.exl3_gemv / ext.reconstruct / ext.reconstruct_slice /
    ext.reconstruct_had_slice / ext.hgemm with call-counting spies so a silent
    fallback into ANY reconstruction path cannot slip through. Returns (calls, restore)."""
    calls = {"gemv": 0, "reconstruct": 0, "reconstruct_slice": 0, "reconstruct_had_slice": 0, "hgemm": 0}
    real_gemv = ext.exl3_gemv
    real_reconstruct = ext.reconstruct
    real_reconstruct_slice = getattr(ext, "reconstruct_slice", None)
    real_reconstruct_had_slice = getattr(ext, "reconstruct_had_slice", None)
    real_hgemm = ext.hgemm

    def gemv_spy(*args, **kwargs):
        calls["gemv"] += 1
        return real_gemv(*args, **kwargs)

    def reconstruct_spy(*args, **kwargs):
        calls["reconstruct"] += 1
        return real_reconstruct(*args, **kwargs)

    def reconstruct_slice_spy(*args, **kwargs):
        calls["reconstruct_slice"] += 1
        return real_reconstruct_slice(*args, **kwargs)

    def reconstruct_had_slice_spy(*args, **kwargs):
        calls["reconstruct_had_slice"] += 1
        return real_reconstruct_had_slice(*args, **kwargs)

    def hgemm_spy(*args, **kwargs):
        calls["hgemm"] += 1
        return real_hgemm(*args, **kwargs)

    ext.exl3_gemv = gemv_spy
    ext.reconstruct = reconstruct_spy
    if real_reconstruct_slice is not None:
        ext.reconstruct_slice = reconstruct_slice_spy
    if real_reconstruct_had_slice is not None:
        ext.reconstruct_had_slice = reconstruct_had_slice_spy
    ext.hgemm = hgemm_spy

    def restore():
        ext.exl3_gemv = real_gemv
        ext.reconstruct = real_reconstruct
        if real_reconstruct_slice is not None:
            ext.reconstruct_slice = real_reconstruct_slice
        if real_reconstruct_had_slice is not None:
            ext.reconstruct_had_slice = real_reconstruct_had_slice
        ext.hgemm = real_hgemm

    return calls, restore


# The livescope model fixture used by the comparison test loads individual modules and MUST
# unload them even when an assertion fails, or aggregate modules stay resident for later tests.
def _load_then_compare(model, linear_key, norm_key, batch_size):
    # Route proof: a missing binding is a FAILURE, not a skip -- the whole point of this
    # suite is to prove the HIP GEMV route executes.
    assert hasattr(ext, "exl3_gemv"), \
        "ext.exl3_gemv is missing: the HIP decode GEMV is not wired into the extension"

    norm = model.find_module(norm_key) if norm_key else None
    linear = model.find_module(linear_key)
    try:
        if norm:
            norm.load(device="cuda")
        linear.load(device="cuda")

        torch.manual_seed(0)
        x = torch.randn((1, batch_size, linear.in_features), dtype=torch.float16, device="cuda")
        if norm:
            x = norm.forward(x, {})

        # Spied GEMV forward: proves the route and forbids a silent reconstruct fallback
        calls, restore = _spy_ext_route()
        try:
            x_gemv = linear.forward(x, {"reconstruct": False})
        finally:
            restore()
        assert calls["gemv"] > 0, \
            f"{linear_key} (m={batch_size}): reconstruct=False did not call ext.exl3_gemv"
        assert calls["reconstruct"] == 0, \
            f"{linear_key} (m={batch_size}): reconstruct=False silently fell back to ext.reconstruct"
        assert calls["reconstruct_slice"] == 0, \
            f"{linear_key} (m={batch_size}): reconstruct=False silently fell back to ext.reconstruct_slice"
        assert calls["reconstruct_had_slice"] == 0, \
            f"{linear_key} (m={batch_size}): reconstruct=False silently fell back to ext.reconstruct_had_slice"
        assert calls["hgemm"] == 0, \
            f"{linear_key} (m={batch_size}): reconstruct=False silently reached ext.hgemm (reconstruct fallback)"

        # Reference forward, unspied
        x_ref = linear.forward(x, {"reconstruct": True})

        # The complete gfx1201 matrix below observed a 0.00390625 maximum absolute
        # deviation. 0.01 is over 2.5x that fp16-quantization step without inheriting
        # the upstream suite's much looser 0.05 tolerance.
        tol = 0.01
        torch.testing.assert_close(x_gemv, x_ref, rtol=tol, atol=tol)

        # Report-grade deviation metrics for reproducing the measured tolerance bound.
        d = (x_gemv.float() - x_ref.float()).abs()
        rel = d / x_ref.float().abs().clamp_min(1e-3)
        return d.max().item(), rel.max().item()
    finally:
        linear.unload()
        if norm:
            norm.unload()


@torch.inference_mode()
def test_hip_gemv_direct_binding_accepts_rank3_inputs():
    """The direct HIP binding must accept arbitrary-rank contiguous leading dims."""
    _require_gfx12()
    size_k = size_n = 128
    A = torch.randn((1, 1, size_k), dtype=torch.float16, device="cuda") * 1e-3
    B = torch.full((size_k // 16, size_n // 16, 4 * 16), 0x1111, dtype=torch.int16, device="cuda")
    suh = (torch.randint(0, 2, (size_k,), device="cuda") * 2 - 1).to(torch.float16)
    svh = (torch.randint(0, 2, (size_n,), device="cuda") * 2 - 1).to(torch.float16)

    actual = torch.empty((1, 1, size_n), dtype=torch.float16, device="cuda")
    A_had = torch.empty_like(A)
    ext.exl3_gemv(A, B, actual, suh, A_had, svh, False, False)

    A_ref = torch.empty_like(A)
    weight = torch.empty((size_k, size_n), dtype=torch.float16, device="cuda")
    expected = torch.empty_like(actual)
    ext.had_r_128(A.view(1, size_k), A_ref.view(1, size_k), suh, None, 1.0)
    ext.reconstruct(weight, B, 4, False, False)
    ext.hgemm(A_ref.view(1, size_k), weight, expected.view(1, size_n))
    ext.had_r_128(expected.view(1, size_n), expected.view(1, size_n), None, svh, 1.0)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0.03)


@pytest.mark.parametrize(
    "mcg, multiplier, addend",
    [
        pytest.param(False, 89226354, 64248484, id="plain"),
        pytest.param(True, 0xCBAC1FED, 0, id="mcg"),
    ],
)
@torch.inference_mode()
def test_procedural_codebook_matches_lop3_zero_state(mcg, multiplier, addend):
    """The portable codebook expression must match the original PTX lop3 LUT 0x6a.

    A zero packed trellis decodes every weight from state zero, making the expected
    fp16 codebook value independent of the trellis layout and reconstruction path.
    """
    _require_gfx12()
    K = 4
    packed = torch.zeros((8, 8, K * 16), dtype=torch.int16, device="cuda")
    weight = torch.empty((128, 128), dtype=torch.float16, device="cuda")
    ext.reconstruct(weight, packed, K, mcg, False)

    product = (0 * multiplier + addend) & 0xFFFFFFFF
    lop3 = 0x3B603B60 ^ (product & 0x8FFF8FFF)
    halves = torch.frombuffer(bytearray(lop3.to_bytes(4, "little")), dtype=torch.float16)
    expected = halves.sum(dtype=torch.float16)

    torch.testing.assert_close(weight, torch.full_like(weight, expected), rtol=0, atol=0)


@pytest.mark.parametrize(
    "K, mcg, mul1",
    [
        pytest.param(4, False, False, id="k4-plain"),
        pytest.param(4, True, False, id="k4-mcg"),
        pytest.param(4, False, True, id="k4-mul1"),
        pytest.param(2, True, False, id="k2-mcg"),
        pytest.param(2, False, True, id="k2-mul1"),
        pytest.param(3, True, False, id="k3-mcg"),
        pytest.param(3, False, True, id="k3-mul1"),
    ],
)
@torch.inference_mode()
def test_hip_gemv_synthetic_oracle(K, mcg, mul1):
    """Every HIP-instantiated bitrate/codebook family matches the reconstruct oracle."""
    _require_gfx12()
    torch.manual_seed(1234 + K + 10 * mcg + 100 * mul1)
    size_k = size_n = 128
    # These nonzero repeated cycles are valid finite trellises for their codebooks. Random
    # packed words are not valid test data because several intentionally decode to fp16 NaNs.
    packed_cycle = {
        (4, False, False): 0x1111,
        (4, True, False): 0x2222,
        (4, False, True): 0x1111,
        (2, True, False): 0x5555,
        (2, False, True): 0x5555,
        (3, True, False): 0x2492,
        (3, False, True): 0x2492,
    }[K, mcg, mul1]
    A = torch.randn((1, size_k), dtype=torch.float16, device="cuda") * 1e-3
    B = torch.full((size_k // 16, size_n // 16, K * 16), packed_cycle,
                   dtype=torch.int16, device="cuda")
    suh = (torch.randint(0, 2, (size_k,), device="cuda") * 2 - 1).to(torch.float16)
    svh = (torch.randint(0, 2, (size_n,), device="cuda") * 2 - 1).to(torch.float16)

    actual = torch.empty((1, size_n), dtype=torch.float16, device="cuda")
    A_had = torch.empty_like(A)
    ext.exl3_gemv(A, B, actual, suh, A_had, svh, mcg, mul1)

    A_ref = torch.empty_like(A)
    weight = torch.empty((size_k, size_n), dtype=torch.float16, device="cuda")
    expected = torch.empty_like(actual)
    ext.had_r_128(A, A_ref, suh, None, 1.0)
    ext.reconstruct(weight, B, K, mcg, mul1)
    ext.hgemm(A_ref, weight, expected)
    ext.had_r_128(expected, expected, None, svh, 1.0)

    # All seven cases were bit-identical on gfx1201. A 0.03 absolute allowance retains
    # modest fp16 rounding headroom without hiding an extraction/codebook mismatch.
    torch.testing.assert_close(actual, expected, rtol=0, atol=0.03)


@pytest.mark.parametrize("test_key", TEST_KEYS)
@pytest.mark.parametrize("batch_size", BATCH_SIZES)
@torch.inference_mode()
def test_hip_gemv_matches_reference(model, test_key, batch_size):
    """HIP decode GEMV (reconstruct=False) must match the reconstruct=True reference,
    and must actually route through the GEMV kernel to do it."""
    _load_then_compare(model, test_key[0], test_key[1], batch_size)


@torch.inference_mode()
def test_hip_gemv_route_only(model):
    """Focused m==1 route proof on one representative projection, independent of the
    numeric matrix: the GEMV binding fires and the reconstruct fallback does not."""
    assert hasattr(ext, "exl3_gemv"), "ext.exl3_gemv is missing"
    norm = model.find_module("model.language_model.layers.3.input_layernorm")
    linear = model.find_module("model.language_model.layers.3.self_attn.q_proj")
    try:
        norm.load(device="cuda")
        linear.load(device="cuda")
        torch.manual_seed(0)
        x = torch.randn((1, 1, linear.in_features), dtype=torch.float16, device="cuda")
        x = norm.forward(x, {})
        calls, restore = _spy_ext_route()
        try:
            y = linear.forward(x, {"reconstruct": False})
        finally:
            restore()
        assert calls["gemv"] == 1, f"expected exactly one exl3_gemv call, got {calls['gemv']}"
        assert calls["reconstruct"] == 0
        assert calls["reconstruct_slice"] == 0
        assert calls["reconstruct_had_slice"] == 0
        assert calls["hgemm"] == 0, "reconstruct=False must not reach ext.hgemm"
        assert y.shape[-1] == linear.out_features
        assert torch.isfinite(y.float()).all()
    finally:
        linear.unload()
        norm.unload()


@torch.inference_mode()
def test_hip_gemv_rows_over_limit_falls_back(model):
    """A K=4 decode batch with nine rows must not enter the m <= 8 GEMV kernel."""
    linear = model.find_module("model.language_model.layers.3.self_attn.q_proj")
    try:
        linear.load(device="cuda")
        assert linear.inner.K == 4
        assert linear.inner.mcg != linear.inner.mul1
        x = torch.randn((1, 9, linear.in_features), dtype=torch.float16, device="cuda")
        calls, restore = _spy_ext_route()
        try:
            y = linear.forward(x, {"reconstruct": False})
        finally:
            restore()
        assert calls["gemv"] == 0, "m=9 must bypass the m <= 8 GEMV kernel"
        assert calls["reconstruct"] > 0 and calls["hgemm"] > 0
        assert torch.isfinite(y.float()).all()
    finally:
        linear.unload()


@torch.inference_mode()
def test_hip_gemv_env_opt_out_falls_back(model, monkeypatch):
    """EXL3_GEMV=0 disables only the Python HIP route; direct C++ semantics are unchanged."""
    linear = model.find_module("model.language_model.layers.3.self_attn.q_proj")
    try:
        linear.load(device="cuda")
        assert linear.inner.K == 4
        x = torch.randn((1, 1, linear.in_features), dtype=torch.float16, device="cuda")
        monkeypatch.setenv("EXL3_GEMV", "0")
        calls, restore = _spy_ext_route()
        try:
            y = linear.forward(x, {"reconstruct": False})
        finally:
            restore()
        assert calls["gemv"] == 0
        assert calls["reconstruct"] > 0 and calls["hgemm"] > 0
        assert torch.isfinite(y.float()).all()
    finally:
        linear.unload()


@torch.inference_mode()
def test_hip_gemv_ineligible_falls_back(model):
    """The fixture's lm_head is EXL3 at 6 bpw (K=6 > 4): the GEMV kernel is not eligible and
    reconstruct=False MUST take the reconstruct fallback (exl3_gemv never called)."""
    assert hasattr(ext, "exl3_gemv")
    linear = model.find_module("lm_head")
    try:
        linear.load(device="cuda")
        assert linear.inner.K == 6
        torch.manual_seed(0)
        x = torch.randn((1, 1, linear.in_features), dtype=torch.float16, device="cuda")
        calls, restore = _spy_ext_route()
        try:
            y = linear.forward(x, {"reconstruct": False})
        finally:
            restore()
        assert calls["gemv"] == 0, "K=6 module must not route to the GEMV kernel"
        reconstruct_calls = calls["reconstruct"] + calls["reconstruct_slice"] + calls["reconstruct_had_slice"]
        assert reconstruct_calls > 0, "K=6 module must fall back to a reconstruct path"
        assert torch.isfinite(y.float()).all()
    finally:
        linear.unload()


@torch.inference_mode()
def test_hip_decode_smoke(model):
    """One-token autoregressive decode sanity: greedy output runs the m==1 (MMODE0) decode path."""
    from exllamav3 import Cache, Tokenizer, Generator

    # Use a FRESH model instance so a full model.load() here cannot interleave with the
    # partially-loaded state left by the comparison test's shared fixture.
    if not os.path.isfile(os.path.join(MODEL, "tokenizer.json")):
        pytest.skip(f"No tokenizer.json in {MODEL} (set EXL3_TEST_MODEL)")
    fresh = Model.from_config(model.config)
    cache = Cache(fresh, max_num_tokens=2048, max_batch_size=1)  # must exist before model.load()
    fresh.load(device="cuda")
    tokenizer = Tokenizer.from_config(fresh.config)
    gen = Generator(model=fresh, cache=cache, tokenizer=tokenizer)
    calls, restore = _spy_ext_route()
    try:
        response = gen.generate(
            prompt="Q: The capital of France is\nA:",
            stop_conditions=[],
            max_new_tokens=8,
            completion_only=True,
            add_bos=True,
        )
    finally:
        restore()
        fresh.unload()
    # The point is that generation RAN through the HIP decode path without erroring.
    # A model may emit <eos> at the first token, so require only that a completion was
    # produced, not that it be non-trivial text. Ineligible layers such as the K=6
    # lm_head may reconstruct during sampling, but at least one decode projection must GEMV.
    assert response is not None
    assert calls["gemv"] > 0, "decode smoke test never called ext.exl3_gemv"
