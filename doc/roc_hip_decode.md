# ROCm HIP EXL3 decode port — design notes

Branch: `hip-decode-backend` (worktree `~/Code/_wt/exllamav3-hip`, based on `origin/dev` @ v1.4.5)
Scope: **decode only** (autoregressive GEMV). Not the quantizer, not rows>144 GEMM, not bc_* CUDA-graph attention.
Canonical plan: `~/Code/_notes/plans/2026-08-31-exllamav3-hip-decode-backend.md`

## Why the decode GEMV is the whole game for decode

From `exllamav3/modules/quant/exl3.py`:
```
rows <= AUTO_RECONSTRUCT_THRESHOLD (144)  -> GEMV path (exl3_gemv)   <- ALL autoregressive decode
rows >  144                               -> reconstruct + hgemm       <- large prefill (deferred)
```
The GEMV kernel (`exllamav3_ext/quant/exl3_gemv_kernel.cuh`) is therefore the decode
hot path. It is a QTIP-style small-m matvec:
- warps split k, stream `B` (the packed trellis) to registers via `ld.global.cs` (DCS) behind a prefetch ring,
- resolve the two-word bit windows of the trellis in-warp via `__shfl_sync`,
- one fp16 `mma.m16n8k16` per 16x16 weight tile, fp16-accumulate, then fold to fp32 on a cadence,
- cross-warp reduction over k-splits in shared memory,
- **cooperative launch** with two `grid.sync()` delimiters for the input hadamard (A) and output hadamard (C) stages.

CFG0 "narrow" (512 threads, 2 n-tiles/warp, 16 k-splits) for attention projection sizes;
CFG1 "wide" (256 threads, 4 n-tiles/warp, 8 k-splits) for large-n FFN. MMODE0 = m==1 fast path.

## NVIDIA mm a that must become AMD MFMA

The single PTX site to port:
```cpp
"mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {c0,c1}, {a0..a3}, {b0,b1}, {c0,c1}"
```
- `A` = m16 x k16 fp16 (fed as two `FragB` = 4 fp16/lane: `a01`, `a23`),
- `B` = k16 x n8 fp16 (`FragB` = 2 fp16/lane),
- `C` = m16 x n8 fp16-accumulate (`FragC_h` = Vec<half2,2>, 4 fp16/lane).
- The kernel computes one 16x16 tile as **two** of these along n (top 8 cols `ch[t][0]`, bottom 8 cols `ch[t][1]`).

### AMD shape mapping (the design decision)
There is no fp16-accumulate MFMA and no n8 variant on AMD. The natural target is:

```
v_mfma_f32_16x16x16_f16        (M=16, N=16, K=16, fp16 in, fp32 accumulate)
```
Key differences versus the CUDA code:
1. **One MFMA covers the whole 16x16 tile (N=16)** — replaces the *two* m16n8k16 calls (`ch[t][0]`, `ch[t][1]` collapse into one `HipFragC`). Simpler.
2. **Accumulation is fp32 natively** — the CUDA kernel's fp16-accumulate-with-cadence-fold (`ch` + `acc0` + `FOLD`) is replaced by a single fp32 accumulator. Numerically **at least as accurate**; removes the fold logic.
3. **Fragment layout differs** — A/B/C are distributed across the 32 lanes differently than NVIDIA. The `__shfl_sync` trellis-word streaming still resolves the same decoded half values; only the register arrangement into the MFMA differs. This is the bit-level work that **must be finalized on a device** (see Oracle).
4. `mma_ab_h`'s special m==1 handling (only the active row's 4 fp16 are nonzero) maps to feeding a mostly-zero A fragment — same trick, driven by the oracle-verified layout.

Target builtin (compiler/toolkit-version dependent — finalize at build):
`__builtin_amdgcn_mfma_f32_16x16x16_f16` (or hipWMMA/rocWMMA gate tile API; prefer raw builtin for parity with the CUDA kernels).

## Port map

| Piece | Action | Confidence |
|---|---|---|
| `exl3_dq.cuh` | near-verbatim (funnelshift) | portable |
| `codebook.cuh` | replace `lop3.b32` with portable bitwise; keep `__dp4a` | portable |
| `hadamard_inner.cuh` | near-verbatim | portable |
| `__shfl_sync` / `__ldcs` streaming | HIP-native | portable |
| `mma_ab_h` → `mma_ab_h_hip` | MFMA rewrite + fragment mapping | device-time TODO |
| cooperative `grid.sync()` | prefer splitting hadamard-I / hadamard-O into separate launches to avoid cooperative-sync dependence on ROCm (robustness win) | decision |
| `reconstruct.cu`, `hgemm.cu` | deferred; hipBLASLt/rocBLAS for hgemm | deferred |
| bc_* / CUDA-graph attention | route to Triton (amdgpu) | deferred |

## Oracle strategy (why we need no CUDA GPU)

PR #283 guarantees a pure-PyTorch fallback for every missing kernel. The existing engine's
`reconstruct=true` path (PyTorch reconstruct + rocBLAS fp16 gemm) is a known-correct
reference. Therefore **HIP decode GEMV must reproduce `reconstruct=true` output** to fp16
tolerance — the exact same pattern `test_qgemm.py` uses to compare its two internal paths.
Tests in `tests/test_hip_gemv_decode.py` implement this and also re-enable the PR-#283-excluded
kernel tests as they come online.

## Open questions / device-time TODOs
- [ ] Exact MFMA A-fragment lane↔element layout to keep the m==1 fast path cheap; verify with oracle.
- [ ] Cooperative launch availability & robustness on gfx1201; fallback to split hadamard launches.
- [ ] `__builtin_amdgcn_mfma_f32_16x16x16_f16` calling convention on the installed ROCm/HIP toolkit.
- [ ] Build environment: PyTorch ROCm wheel, `hipcc`, no flash_attn for HIP (Triton path carries attention).
- [ ] Perf gate decode tok/s vs ~14 tok/s #283 baseline vs CUDA reference.
