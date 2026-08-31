#pragma once

// HIP (ROCm) tensor-core fragment vocabulary for the EXL3 decode GEMV.
//
// WIP SCAFFOLD - not yet compiled or validated. This mirrors the shapes used by
// exl3_gemv_kernel.cuh (which consumes ../ptx.cuh on the CUDA path) so a HIP build can
// swap the tensor-core call site. See doc/roc_hip_decode.md for the full mapping and
// the device-time TODOs. Nothing here is wired into the kernel yet.
//
// Currently active worktree branch: hip-decode-backend (based on origin/dev).
// Correctness oracle: compare this GEMV against the reconstruct=true reference path
// (see tests/test_hip_gemv_decode.py). No CUDA GPU is required for validation.

#include <cstdint>

#if defined(__HIPCC__)

// ---- fp16 helpers / little glue (HIP provides __funnelshift_r, __dp4a, __shfl_*) ----

using half = __half;

template <typename T, int n>
struct HipVec
{
    T elems[n];
    __device__ T& operator[](int i) { return elems[i]; }
    __device__ const T& operator[](int i) const { return elems[i]; }
};

// Fragment roles, mirrored from ../ptx.cuh names so the kernel body ports 1:1.
using FragA   = HipVec<half2, 4>;         // m16 A operand (M x K)  -- NVIDIA: FragA
using FragB   = HipVec<half2, 2>;         // n8 B operand / A pieces -- NVIDIA: FragB
using FragC   = HipVec<float, 4>;         // 16x16 fp32 accumulator (one per lane)
using FragC_h = HipVec<half2, 2>;         // legacy fp16-accum shape, kept for shape parity

// -----------------------------------------------------------------------------------
// mma m16n8k16  ->  v_mfma_f32_16x16x16_f16
//
// The CUDA site:
//   mma_ab_h(const FragB& a01, const FragB& a23, const FragB& b, FragC_h& c)
//     "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 ..."
//   per 16x16 tile: two m16n8k16 along n  (ch[t][0] = cols 0..7, ch[t][1] = cols 8..15).
//
// On AMD, ONE MFMA computes the full 16x16 output with fp32 accumulation, so the two
// per-tile calls collapse into one and the fp16->fp32 cadence-fold is dropped.
//
// A operand: m16 x k16 fp16. NOTE the CUDA kernel feeds only the active row's 4 fp16
// (a01, a23) and zeros the rest; that m==1 fast-path trick must be re-derived for the
// AMD A-fragment lane layout (TODO below).
//
// B operand: k16 x n16 fp16 (full tile; was n8 per call).
// C operand: 16x16 fp32 accumulator, 4 float per lane.
// -----------------------------------------------------------------------------------
__device__ __forceinline__ void mma_ab_h_hip(
    const FragB& a01,
    const FragB& a23,
    const FragB& b_hi,   // tentative: B split across two FragB to keep the n16 B in 4 fp16/lane
    const FragB& b_lo,
    FragC& c)
{
    // TODO(device): finalize the MD - lane<->element mapping for A,B,C against the
    // oracle before trusting this shape. The builtin calling convention differs across
    // ROCm/HIP toolkit versions; confirm on the installed toolchain.
    //
    // Representative target (parameters enum values to verify):
    //   __builtin_amdgcn_mfma_f32_16x16x16_f16(A, B, C, cbsz, abid)
    // where A,B are the lane's fp16 halves and C the lane's 4x fp32 accumulators.
}

// m==1 fast path hook. The CUDA kernel zero-fills most of the A fragment because only
// row 0 is nonzero at MMODE 0. On AMD the equivalent is "feed a mostly-zero A fragment".
__device__ __forceinline__ void mma_ab_h_hip_m1(
    const FragB& a01,
    const FragB& a23,
    const FragB& b_hi,
    const FragB& b_lo,
    FragC& c)
{
    // TODO(device): MMODE 0 specialization once the base MFMA layout is oracle-verified.
    mma_ab_h_hip(a01, a23, b_hi, b_lo, c);
}

#endif  // __HIPCC__
