import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pytest
import torch
from exllamav3 import Config, Model

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
# Skips cleanly when there is no HIP build / ROCm device / test model present.
# Re-enables the decode-relevant kernels that PR #283 stubbed (GEMV / reconstruct / qgemm).
#
# Model: set EXL3_TEST_MODEL to a small pre-quantized EXL3 model directory.
#        Defaults to a Qwen under ~/Models; update once a model is downloaded.
# -----------------------------------------------------------------------------------

# Decode regime: batch sizes that hit the GEMV path (rows <= 144) and specifically the
# m==1 fast path plus small-m (2..8). m==1 is autoregressive decode.
BATCH_SIZES = [1, 2, 8]

# Per-arch proj keys exercising decode-relevant shapes. Mirrors test_qgemm's set.
TEST_KEYS = [
    ("model.layers.0.self_attn.q_proj", "model.layers.0.input_layernorm"),
    ("model.layers.0.self_attn.k_proj", "model.layers.0.input_layernorm"),
    ("model.layers.0.self_attn.v_proj", "model.layers.0.input_layernorm"),
    ("model.layers.0.self_attn.o_proj", "model.layers.0.input_layernorm"),
    ("model.layers.0.mlp.up_proj", "model.layers.0.post_attention_layernorm"),
    ("model.layers.0.mlp.gate_proj", "model.layers.0.post_attention_layernorm"),
    ("model.layers.0.mlp.down_proj", None),
    ("lm_head", "model.norm"),
]

DEFAULT_MODEL = os.path.expanduser("~/Models/Qwen3.5-9B-exl3")
MODEL = os.environ.get("EXL3_TEST_MODEL", DEFAULT_MODEL)


def _roc_available():
    # ROCm build active and at least one visible device
    if not (torch.version.hip and torch.cuda.is_available()):
        return False
    return True


@pytest.fixture(scope="module")
def model():
    if not _roc_available():
        pytest.skip("ROCm build / device not available")
    if not os.path.isdir(MODEL):
        pytest.skip(f"Test model not found: {MODEL} (set EXL3_TEST_MODEL)")
    config = Config.from_directory(MODEL)
    return Model.from_config(config)


# The livescope model fixture used by the comparison test loads individual modules and MUST
# unload them even when an assertion fails, or aggregate modules stay resident for later tests.
def _load_then_compare(model, linear_key, norm_key, batch_size):
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

        x_gemv = linear.forward(x, {"reconstruct": False})
        x_ref = linear.forward(x, {"reconstruct": True})
        # fp16-tolerance; the two internal paths are not bit-identical, only close.
        # 0.05 matches upstream test_qgemm.py; for decode/m==1 this is the same comparison
        # the upstream suite applies, just at decode batch sizes.
        tol = 0.05
        torch.testing.assert_close(x_gemv, x_ref, rtol=tol, atol=tol)
    finally:
        linear.unload()
        if norm:
            norm.unload()


@pytest.mark.parametrize("test_key", TEST_KEYS)
@pytest.mark.parametrize("batch_size", BATCH_SIZES)
@torch.inference_mode()
def test_hip_gemv_matches_reference(model, test_key, batch_size):
    """HIP decode GEMV (reconstruct=False) must match the reconstruct=True reference."""
    _load_then_compare(model, test_key[0], test_key[1], batch_size)


@torch.inference_mode()
def test_hip_decode_smoke(model):
    """One-token autoregressive decode sanity: greedy output runs the m==1 (MMODE0) decode path."""
    from exllamav3 import Cache, Tokenizer, Generator

    # Use a FRESH model instance so a full model.load() here cannot interleave with the
    # partially-loaded state left by the comparison test's shared fixture.
    if not os.path.isfile(os.path.join(MODEL, "tokenizer.json")):
        pytest.skip(f"No tokenizer.json in {MODEL} (set EXL3_TEST_MODEL)")
    fresh = Model.from_config(model.config)
    fresh.load(device="cuda")
    cache = Cache(fresh, max_num_tokens=2048, max_batch_size=1)
    tokenizer = Tokenizer.from_config(fresh.config)
    gen = Generator(model=fresh, cache=cache, tokenizer=tokenizer)
    response = gen.generate(
        prompt="Q: The capital of France is\nA:",
        stop_conditions=[],
        max_new_tokens=8,
        completion_only=True,
        add_bos=True,
    )
    # The point is that generation RAN through the HIP decode path without erroring.
    # A model may emit <eos> at the first token, so require only that a completion was
    # produced, not that it be non-trivial text.
    assert response is not None
