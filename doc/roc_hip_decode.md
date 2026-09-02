# Experimental gfx12 ROCm decode backend

This page documents the implemented Heterogeneous-compute Interface for Portability (HIP) decode path. It accelerates EXL3 matrix-vector multiplication (GEMV) on RDNA4 and provides correctness fallbacks outside that envelope.

## What runs on gfx12

The runtime gate in `exllamav3.modules.quant.exl3.LinearEXL3` sends a layer to HIP GEMV only when:

- ROCm is active and `ext.exl3_gemv_supported(device)` is true;
- the GPU is `gfx1200` or `gfx1201`;
- the decode batch is `rows <= 8`;
- `in_features` and `out_features` are both multiples of 128;
- `K` is from 2 through 6;
- the existing codebook flags match the instantiated family (`mcg` / `mul1` when required); K5 and K6 require one of those two codebooks.

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

- 98 GEMV matrix, routing, codebook, K5, and K6 tests per model fixture
- 88 cache / reconstruct tests
- 61 multi-head latent attention (MLA) / DeepSeek sparse attention (DSA) tests
- 34 native ROCm hyperconnection route, numerical, stream, validation, and fallback tests
- 63 grouped K3 MoE route, numerical, multirow, safety, LoRA, and fallback tests, plus opt-in full Flash and MTP oracles
- 44 native ROCm shared-gate fusion, stream, validation, and fallback tests
- 25 gfx12 batch-one router numerical, tie, stream, validation, fallback, and real-layer tests
- forced-identical-context 16-step logits oracles for mul1 and MCG models

The Qwen3.8-27B mul1 oracle reports:

- top-1 agreement: 16/16
- top-5 overlap: at least 4/5 tokens at every step
- observed max absolute logit delta: 0.488281
- worst-step mean delta: 0.046886
- overall mean delta: 0.027793
- fallback-only `-inf` positions carry <1e-6 softmax mass per step

The Qwen3.5-9B MCG oracle reports:

- top-1 agreement: 16/16
- exact top-5 agreement: 16/16 steps
- max absolute logit delta: 0.234375
- worst-step mean delta: 0.017942
- overall mean delta: 0.009152

Warmed decode measurements on gfx1201:

| Model | Trial size | K6 reconstruct median | Direct K6 median | Uplift |
| --- | ---: | ---: | ---: | ---: |
| Qwen3.8-27B | 64 tokens | 14.569 tok/s | 16.250 tok/s | 11.5% |
| Qwen3.5-9B | 128 tokens | 36.522 tok/s | 47.311 tok/s | 29.5% |

Each median is three trials after a 32-token warmup, using one model, prompt, and GPU. The K6-reconstruct arm preserves direct GEMV for K2-K4 and falls back only for K6.

Reusing each assembled WMMA A fragment across adjacent N tiles provides another measured gain:

| Model | Before A reuse | With A reuse | Uplift |
| --- | ---: | ---: | ---: |
| Qwen3.8-27B | 16.315 tok/s | 17.325 tok/s | 6.2% |
| Qwen3.5-9B | 47.315 tok/s | 51.917 tok/s | 9.7% |

Specializing the m=1 reduction for its single valid output row raises Qwen3.8-27B from 17.285 to 18.115 tok/s (4.8%). Qwen3.5-9B rises from 51.880 to 54.411 tok/s (4.9%). This reduces kernel LDS use from 28,672 to 14,336 bytes and raises occupancy from two to four blocks per compute unit.

Compared with the earlier all-reconstruct baseline, the current backend measures 3.94x faster on Qwen3.8-27B (4.603 to 18.115 tok/s). Qwen3.5-9B measures 3.66x faster (14.847 to 54.411 tok/s). These cross-run ratios show the cumulative backend improvement; the tables and percentages above are controlled incremental comparisons.

Steady-state profiles after prefill show the K6 vocabulary head changing from reconstruct plus rocBLAS hgemm to one direct kernel:

| Model | Steps | K6 reconstruct + hgemm | Direct K6 | Total self GPU time change |
| --- | ---: | ---: | ---: | ---: |
| Qwen3.8-27B | 8 | 80.543 ms | 18.791 ms | 483.397 ms → 423.119 ms |
| Qwen3.5-9B | 12 | 97.071 ms | 21.174 ms | 292.792 ms → 217.324 ms |

A-fragment reuse reduces total self GPU time from 423.119 to 400.882 ms for the eight-step Qwen3.8 profile. The dominant K4 mul1/fp16 wide kernel falls from 163.596 to 154.410 ms. For the twelve-step Qwen3.5 profile, time falls from 217.324 to 197.155 ms. Its K4 MCG/fp32 narrow kernel falls from 67.561 to 57.072 ms.

The one-row reduction lowers the Qwen3.8 wide K4 kernel again, from 154.413 to 140.053 ms in a controlled profile. The corresponding Qwen3.5 wide K4 kernel falls from 51.207 to 42.012 ms; its narrow kernel is effectively unchanged.

## Qwen3.8-Flash-Next target

The `turboderp/Qwen3.8-Flash-Next-exl3` 3.05-bpw checkpoint runs target-only with a layer split across two gfx1201 GPUs. Its 32.64-GB PLE n-gram table stays file-backed and streams only selected rows. A balanced 27/27-GB load budget allocated approximately 28.6 and 24.4 GB on the two GPUs with a 32K cache.

A short greedy smoke produced `Paris. Paris is the capital` and executed the direct EXL3 route. Adding K5 support moves the model's high-quality shared-expert and other K5 projections off reconstruct+hgemm:

- warmed target-only decode: 6.588 to 7.687 tok/s, a 16.7% gain;
- four-token self-device profile: 875.654 to 752.568 ms, a 14.1% reduction;
- K5 reconstruction: 1,232 launches to zero;
- real shared-expert gate/up/down K5 projections match reconstruct+hgemm on both gfx1201 devices.

The existing fused GatedResidual and hyperconnection kernels are also available on wave32 ROCm devices. They replace the decode reference path's small matrix multiplications and elementwise chains with two `gr_mix` launches and one in-place `hc_apply` launch per residual site. Unsupported wave sizes retain the PyTorch path. In a controlled same-process comparison, fusion improved the 32-token median from 7.787 to 14.085 tok/s. Over four tokens, it reduced `aten::mm` from 1,552 to 392 calls and removed 7,736 HIP launches. A separate fresh-process run measured a 13.553 tok/s median.

Keeping the PLE table in RAM did not improve that fresh-process result: disk streaming measured 13.553 tok/s versus 13.410 tok/s from RAM. RAM residency also increased model load time from 17.0 to 30.9 seconds. The streamed path remains the recommended default and avoids reserving 32.64 GB of host memory.

A dedicated gfx12 batch-one MoE route groups the ten selected K3/mul1 experts into device-resident gate/up/down launches. It preserves duplicate expert slots and uses a deterministic fp32 weighted reduction. Strict eligibility keeps batch, conversion, TP, LoRA, activation-limit, unsupported-shape, and non-gfx12 calls on the established path.

Two warmed free-running comparisons measured grouped medians of 24.59 and 24.83 tok/s, versus 14.21 and 14.63 tok/s without grouping. Individual grouped trials ranged from 20.03 to 29.95 tok/s because generated token paths select different PLE rows and experts. Under one fixed forced-token path, the median improved from 12.315 to 23.697 tok/s, a reproducible 92.4% gain. The four-token profile reduced K3 GEMV calls from 5,808 to 48 and HIP activities from 43,240 to 13,288.

The scalar shared-expert gate also uses the native fused dot–sigmoid–accumulate kernel on ROCm. This removes 192 small matrix multiplications and 960 device activities over four tokens. Controlled fixed-token decode improved from 26.211 to 29.133 tok/s, while the free-running median improved from 25.851 to 28.670 tok/s.

A gfx12 batch-one standard router replaces each remaining 2560-by-512 rocBLAS projection and PyTorch top-k with two wave32 kernels. It keeps a 2.5-MiB transposed gate per layer, 120 MiB across the 48-layer model. The four-token profile removed 192 `aten::mm` and 192 `aten::topk` calls; the native router used 3.067 ms total. Controlled free-running time fell from 34.508 to 32.916 ms/token, equivalent to 28.98 and 30.38 tok/s.

Three retained gfx12 optimizations from `d189d3a`, `0bbf261`, and `69acc17` carry that target further. The fixed-K QSA selector is native only for gfx1200/gfx1201 wave32 inputs with fp16 scores, int32 indices, `k == 512`, a 2D score matrix with 128-aligned row stride, and no `t_ptr` / `t_seq` override; every other shape or device uses the Python fallback. It preserves the exact fp16 bit-key order used in tests: `-inf` and negative NaNs stay out, positive NaNs rank above `+inf`, and 69acc17 adds stable block-parallel compaction while preserving ascending-index threshold ties. On gfx1201, the same-harness R512/T32768 native selector improved from 6.298 to 0.396 ms, a 15.9x speedup over the prior serial native path; the real Flash 128K selector improved from 7.131 to 0.852 ms, an 8.37x speedup. Full 128K prefill improved from 136.183 to 123.601 s (9.2%), the late 512-row chunk from 572.7 to 466.1 ms (18.6%), and 128K target decode from 22.59 to 38.27 tok/s (69.4%). A 12K full prefill run was noise-level/neutral.

The standard router is native for rows 1 through 8 on gfx1200/gfx1201 wave32 when the router config is the supported 512-expert/10-top-k half-precision shape with no router bias, per-expert scale, or activate-all path. `RoutingCFG` owns one persistent multirow workspace reused in stream order, and like existing per-module inference workspaces it assumes the loaded model is not invoked concurrently on independent streams. Unsupported rows, disabled routing, missing bindings, and other mismatched configs still fall back to the torch path. In isolated router runs, that native path was 1.5x-2.6x faster, and controlled concurrent decode improved by 1.5%-5.5% across 2-8 jobs.

The grouped path's first real-model layer difference is 1.49e-8 maximum in fp32 output. Qwen4Exp recurrent state amplifies numerical noise. A 16-step fallback repeat measured a 3.63 maximum logit delta and 0.270 mean delta. The grouped pass measured 4.44 and 0.274. Both retained 15/16 top-1 agreement with the reference pass and at least 4/5 top-five overlap. The opt-in full-model oracle includes that fallback control and verifies grouped execution on both gfx1201 devices.

The gfx12 throughput route extends the same direct K3 arithmetic to 6–512 prefill rows. It sorts assignments by expert, processes up to 16 rows per wide WMMA tile, and reduces through the inverse permutation in deterministic expert order. The shared workspaces add approximately 119 MiB per device. Five controlled pp511 trials improved from a 357.93 tok/s fallback median to 855.11 tok/s, a 138.9% gain. Under the profiler, HIP launches fell from 136,043 to 13,305; aggregate GPU time fell from 986 to 484 ms. A prefill-only 128-thread schedule later reduced the warm layer-harness median from 2.207 to 2.083 ms and grouped-body GPU time from 1.857 to 1.697 ms. Fixed-prompt pp511 medians improved from 902.32 to 926.67 tok/s; decode and MTP retain their existing schedules.

The gfx12 GDN prefill kernel uses two V partitions instead of the decode-oriented four. This retains the four-way layout for sequences up to five rows and leaves NVIDIA unchanged. A controlled pp511 sweep was noise-level, 894.48 versus 896.61 tok/s. Profiled recurrent-kernel time fell from 97.45 to 78.34 ms, while aggregate GPU time fell from 484 to 468 ms. An alternating 12K sweep improved from a 660.69 to 675.48 tok/s median, a 2.24% gain.

AITER can complement the runtime one operation at a time; Triton is not an all-or-nothing dependency. The installed AITER exposes chunked GDN, paged-attention, and GEMM APIs. Its GDN API requires expanding this checkpoint's 16 shared q/k heads to 48 value heads. Both tested chunk variants accumulated non-finite full-model state. Its A16W16 assembly path also has no gfx1201 kernel or Triton GEMM configuration. No AITER route is enabled.

The 30.4 tok/s target-only short-context median is above the historical 19.43 tok/s llama.cpp target-only result for this host.

The checkpoint also includes its complete 6,200-tensor MTP head; no separate EXL3 draft download is required. The draft adds approximately 1.25 GB on GPU0. Grouped K3 execution now supports verification windows of up to five rows. This moves MTP3 from 17.50 to 49.79 tok/s in a controlled on/off comparison. It also preserves duplicate routing slots and deterministic per-token reductions.

Across three warmed 64-token trials, target-only measured 38.34 tok/s. MTP1/2/3/4 measured 47.91, 50.19, 53.56, and 43.30 tok/s respectively. MTP3 is the recommended short-context setting. Its three trials accepted 40–43 draft tokens and rejected 20–32 while producing coherent output. Speculative jobs now honor the same maximum output length as target-only jobs instead of reserving an unused full draft window.

The MTP3 result is above the historical 45–50 tok/s llama.cpp short-context MTP band, but the prompt and active-context lengths differ. An initial 4,095-token random-word prompt produced only one accepted draft token out of 96 proposals. That result was prompt-confounded rather than a context-length boundary.

A controlled sweep kept the same `The capital of France is` suffix while varying the prefix from 511 to 4,095 tokens. MTP3 accepted 14–20 draft tokens and rejected 19–43, with no discontinuity at QSA's 2,051-token sparse threshold. At 12K it accepted 18 and rejected 27 while decoding at 37.60 tok/s. MTP acceptance is workload-sensitive, but it does not inherently collapse when sparse QSA activates.

Sparse-QSA target decode measured 32.38 tok/s at 11,999 tokens. The original 256-token prefill chunks reached 547.5 tok/s. Loading with `max_chunk_size=512` raised 12K prefill to 643.1 tok/s; a value of 1,024 regressed to 627.4 tok/s. `Generator` retains its independent 2,048-token default unless its `max_chunk_size` is also set.

With 512-token chunks, synthetic prefill remained stable as context grew:

| Active context | Prefill | Target decode |
| ---: | ---: | ---: |
| 11,999 | 643.1 tok/s | 32.45 tok/s |
| 32,767 | 598.0 tok/s | 28.33 tok/s |
| 65,535 | 592.8 tok/s | 28.65 tok/s |
| 131,071 | 568.6 tok/s | 29.07 tok/s |
| 179,999 | 563.6 tok/s | 31.69 tok/s |

The 180K result retained 87.6% of the 12K prefill rate instead of falling toward 300 tok/s. Retrieval quality is not yet qualified. A synthetic 12K passkey prompt failed under both sparse QSA and a forced-dense control, so it did not isolate the selector.

This is a foundation result, not the final serving profile. Qwen4Exp currently uses layer split rather than tensor parallelism. MTP3 works beyond the sparse threshold, but its workload-dependent acceptance and sparse-QSA retrieval quality still need broader qualification. The remaining short-context bottlenecks are now the grouped K3 WMMA body, recurrent GDN, and the remaining reconstructed K5 projections. Warmed 511-token prefill reached 896.61 tok/s with both throughput changes. Long-context retrieval quality still needs a representative control before comparison with sparse-QSA serving results.

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