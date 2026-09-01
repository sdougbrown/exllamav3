# Experimental gfx12 ROCm decode backend

This page documents the implemented Heterogeneous-compute Interface for Portability (HIP) decode path. It accelerates EXL3 matrix-vector multiplication (GEMV) on RDNA4 and provides correctness fallbacks outside that envelope.

## What runs on gfx12

The runtime gate in `exllamav3.modules.quant.exl3.LinearEXL3` sends a layer to HIP GEMV only when:

- ROCm is active and `ext.exl3_gemv_supported(device)` is true;
- the GPU is `gfx1200` or `gfx1201`;
- the decode batch is `rows <= 8`;
- `in_features` and `out_features` are both multiples of 128;
- `K` is 2, 3, or 4;
- the existing codebook flags match the instantiated family (`mcg` / `mul1` when required).

On those gfx12 devices, the real EXL3 decode kernel uses `v_wmma_f32_16x16x16_f16` with fp32 accumulation. That is the supported ROCm acceleration path here; it is not a general matrix fused multiply-add (MFMA) backend.

## What falls back

Anything outside the envelope above uses the existing reconstruct path:

- rows `> 8`
- unsupported bitrate / codebook combinations
- `EXL3_GEMV=0`
- non-gfx12 ROCm devices

The fallback is `reconstruct + hgemm`. That path is correctness-first. It is not a fused performance kernel, and it does not try to match CUDA decode throughput.

The rest of the ROCm shims in `ext_fallbacks.py` follow the same rule: they preserve behavior with PyTorch implementations for correctness, not speed.

## Why the HIP launches are split

HIP does not use the CUDA cooperative-grid decode path.

Instead it runs three stream-ordered launches:

1. input Hadamard
2. ordinary GEMV
3. output Hadamard

The earlier cooperative-launch experiment was removed after the Heterogeneous System Architecture (HSA) runtime crashed during shutdown. CUDA keeps its separate cooperative path with `grid.sync()`.

## Why LDS staging is mandatory on gfx12

The gfx12 implementation stages decoded trellis words and wave matrix multiply-accumulate (WMMA) operands through warp-private local data share (LDS).

That is not an optimization choice. On gfx12, feeding WMMA from the shuffle path caused hardware exceptions, so the HIP kernel disables that path.

## What is still deferred

Out of scope for this backend slice:

- large-throughput matrix-matrix multiplication (GEMM, `rows > 144`)
- int8 GEMV
- CDNA MFMA
- the quantizer
- CUDA graphs / `bc_*`

## Validation snapshot

Current local verification on gfx1201 includes:

- 39 GEMV matrix, routing, and codebook tests
- 88 cache / reconstruct tests
- 61 multi-head latent attention (MLA) / DeepSeek sparse attention (DSA) tests
- forced-identical-context 16-step logits oracles for mul1 and MCG models

The Qwen3.8-27B mul1 oracle reports:

- top-1 agreement: 16/16
- top-5 overlap: at least 4/5 tokens at every step
- representative max absolute logit delta: 0.4043
- worst-step mean delta: 0.0661
- overall mean delta: 0.0399
- fallback-only `-inf` positions carry <1e-6 softmax mass per step

The Qwen3.5-9B MCG oracle reports:

- top-1 agreement: 16/16
- exact top-5 agreement: 16/16 steps
- max absolute logit delta: 0.21875
- worst-step mean delta: 0.01867
- overall mean delta: 0.00903

Warmed decode measurements on gfx1201:

| Model | Trial size | HIP GEMV median | Reconstruct median | Speedup |
| --- | ---: | ---: | ---: | ---: |
| Qwen3.8-27B | 64 tokens | 14.534 tok/s | 4.603 tok/s | 3.16x |
| Qwen3.5-9B | 128 tokens | 36.114 tok/s | 14.847 tok/s | 2.43x |

These are single-model, single-prompt measurements on one GPU. They are useful proof points, not general performance claims.

## MCG compatibility

The Qwen3.5-9B fixture exposed an incorrect portable translation of the original PTX `lop3` expression. The fix restores the exact LUT `0x6a` semantics for plain and MCG procedural codebooks. A zero-state reconstruction test now checks the codebook value independently of both GEMV and the reconstruct implementation.

## Build and test

```sh
cd exllamav3
export HIPCXX=hipcc PYTORCH_ROCM_ARCH=gfx1201 MAX_JOBS=4
pip install . --no-build-isolation
python -c "from exllamav3 import ext; assert ext.exllamav3_ext.exl3_gemv_supported(0)"

EXL3_TEST_MODEL=/path/to/Qwen3.8-27B-exl3 \
  python -m pytest tests/test_hip_gemv_decode.py -q

EXL3_TEST_MODEL=/path/to/Qwen3.8-27B-exl3 \
  python tests/hip_gemv_logits_oracle.py

# Repeat with an MCG fixture.
EXL3_TEST_MODEL=/path/to/Qwen3.5-9B-exl3 \
  python tests/hip_gemv_logits_oracle.py
```
