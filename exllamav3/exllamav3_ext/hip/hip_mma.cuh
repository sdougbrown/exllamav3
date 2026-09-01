#pragma once

// HIP (ROCm) tensor-core fragment vocabulary + mma_ab_h_hip for the EXL3 decode GEMV.
//
// STATUS (be honest, read this):
//   * gfx12 (RDNA4: gfx1200/gfx1201) path:  ON-DEVICE ORACLE-VERIFIED.  The fragment
//     layout below was derived empirically on a gfx1201 (ROCm 7.2.4) and validated
//     bit-exact (D == A*B + C, fp32, all 256 C elements, m==1 and m==8 feeds).
//     See doc/roc_hip_decode.md and the probes behind it.
//   * CDNA (gfx90a/gfx94x/gfx950, v_mfma_*) path: RESEARCH + SCAFFOLD ONLY.  Shapes and
//     the builtin signature are confirmed by the ROCm 7.2.4 compiler and the shipped
//     ComposableKernel headers, but the per-lane <-> element placement is NOT verified
//     (no CDNA hardware/ISA doc reachable in this session) and MUST be oracle-verified
//     before use.  Its device branch intentionally does nothing so broad ROCm builds
//     succeed; exl3_gemv_try_launch rejects it before any GEMV kernel can launch.
//
// There is no fp16-accumulate tensor core on AMD: both families accumulate in fp32
// natively, so the CUDA kernel's fp16-accumulate + cadence-fold (ch / acc0 / FOLD) is
// replaced by a single fp32 accumulator (FragC / FragC8).  Numerically >= CUDA.

#include <cstdint>

#if defined(__HIPCC__)

#include <hip/hip_fp16.h>

using half = __half;

// ---- fragment shapes (mirrored from ../ptx.cuh names so the kernel body ports 1:1) ----

template <typename T, int n>
struct HipVec
{
    T elems[n];
    __device__ T& operator[](int i) { return elems[i]; }
    __device__ const T& operator[](int i) const { return elems[i]; }
};

using FragA   = HipVec<half2, 4>;   // m16 A operand / gfx12 WMMA operand: 8 fp16 per lane
// FragB: when compat_rocm.cuh is already in scope (any ROCm extension TU that includes
// util.cuh), defer to its layout-identical FragB instead of redefining the name; include
// compat_rocm.cuh (directly or via hadamard_inner.cuh) before this header
#ifndef EXL3_ROCM_FRAGB
using FragB   = HipVec<half2, 2>;   // n8 B operand / A pieces: 4 fp16 per lane (CUDA shape parity)
#endif
using FragC   = HipVec<float, 4>;   // CDNA MFMA accumulator: 4 fp32 per lane  (fp32 accumulate)
using FragC8  = HipVec<float, 8>;   // gfx12 WMMA accumulator: 8 fp32 per lane (fp32 accumulate)
using FragC_h = HipVec<half2, 2>;   // legacy fp16-accumulate shape, kept for shape parity only

// WMMA operand registers on gfx12: 8 fp16 per lane (A and B) -> one 128-bit vector reg.
// Use the HIP vector types so the compiler sees the required v8f16 register form.
#if defined(__gfx1200__) || defined(__gfx1201__)
using HipFp16x8 = __attribute__((__vector_size__(8 * sizeof(__fp16)))) __fp16;
using HipFp32x8 = __attribute__((__vector_size__(8 * sizeof(float)))) float;
#endif

// =====================================================================================
// gfx12 (RDNA4):  v_wmma_f32_16x16x16_f16   (wave32; the only tensor core on this HW)
// =====================================================================================
// D[16x16 f32] = A[16x16 f16] * B[16x16 f16] + C[16x16 f32], one instruction per tile.
//
// FRAGMENT LAYOUT (ORACLE-VERIFIED on gfx1201, ROCm 7.2.4; derived empirically, then
// validated bit-exact end-to-end against the fp32 reference):
//     A: lane l, reg r 0..7  ->  A[row = l&15][k = (l>>4)*8 + r]        (lane = one row)
//     B: lane l, reg r 0..7  ->  B[k = (l>>4)*8 + r][col = l&15]        (lane = one column)
//     C: lane l, reg r 0..7  ->  C[row = (l>>4)*8 + r][col = l&15]      (lane = one column)
// (k-split between the half-warps: lanes 0..15 hold k 0..7, lanes 16..31 hold k 8..15.)
//
// NOTE on the A/B operand sizes: both take 8 fp16 per lane, exactly the shape of the
// EXL3 kernel's decoded 16x16 tile (b_hi + b_lo = FragB x2 = one full B operand) and of
// FragA (a01 + a23 padded to 8 halves).
//
// Fragments are assembled through a warp-private LDS staging buffer instead of
// __shfl_sync: ROCm 7.2.4 clang (gfx1201) lowers __shfl_sync to a ds_bpermute +
// exec-mask software emulation that deterministically raises a hardware exception when
// the shuffled half2 feeds a WMMA operand (pattern: half2 built from two unaligned
// __half loads -> shfl -> wmma).  Keep this in mind for the kernel's own trellis-word
// streaming shfls.

// warp-private staging: [warp][cu_lane][0..3] = a01[0],a01[1],a23[0],a23[1]
// (reused for b_hi/b_lo with slots [0..1] -> b_hi, [2..3] -> b_lo)
#define HIP_MMA_STG_WARPS 16   // EXL3 CFG0 max warps (512 threads); bump if CFG grows
__shared__ __half2 hip_mma_stg[HIP_MMA_STG_WARPS][32][4];

__device__ __forceinline__ int hip_mma_warp_id()
{
    return (int)(threadIdx.x >> 5);
}

// Build the gfx12 WMMA A operand (8 halves) from the kernel's a01/a23 fragments
// (CUDA m16n8k16 A-fragment semantics, PTX ISA 9.7.15.5.8):
//   lane l: g=l>>2, t=l&3  a01[0] = A[g][2t],A[g][2t+1]   a23[0] = A[g][2t+8],A[g][2t+9]
//   (a01[1]/a23[1] = rows g+8, always zero in EXL3 GEMV: m <= 8).
#if defined(__gfx1200__) || defined(__gfx1201__)   // ---- gfx12 only: assembly helpers ----
__device__ __forceinline__ HipFp16x8 assemble_a_frag_gfx12(const FragB& a01, const FragB& a23)
{
    const int lane = (int)(threadIdx.x & 31);
    const int wid  = hip_mma_warp_id();
    __half2 (*stg)[4] = hip_mma_stg[wid];

    stg[lane][0] = a01[0]; stg[lane][1] = a01[1];
    stg[lane][2] = a23[0]; stg[lane][3] = a23[1];
    __syncwarp();

    HipFp16x8 a = {};
    uint32_t* aw = reinterpret_cast<uint32_t*>(&a);   // 8 halves = 4 dwords, little-endian
    const int row = lane & 15;                 // AMD A row this lane contributes
    if (row < 8)                               // rows 8..15 of the m16 tile are never fed
        for (int j = 0; j < 4; ++j)
        {
            // AMD reg pair (2j,2j+1) = A[row][k = 8*(lane>>4)+2j (+1)]:
            // CUDA src lane 4*row+j, half2 a01[0] (k<8) / a23[0] (k>=8)
            const __half2 v = stg[4 * row + j][2 * (lane >> 4)];
            aw[j] = *reinterpret_cast<const uint32_t*>(&v);
        }
    return a;
}

// Build the gfx12 WMMA B operand (8 halves) from the decoded 16x16 tile b_hi/b_lo
// (CUDA m16n8k16 B-fragment semantics, PTX ISA 9.7.15.5.8, two n8 halves of the tile):
//   lane l: g=l>>2, t=l&3  b_hi[0] = B[2t][g],B[2t+1][g]   b_hi[1] = B[2t+8][g],B[2t+9][g]
//   b_lo = same k's at cols g+8.
__device__ __forceinline__ HipFp16x8 assemble_b_frag_gfx12(const FragB& b_hi, const FragB& b_lo)
{
    const int lane = (int)(threadIdx.x & 31);
    const int wid  = hip_mma_warp_id();
    __half2 (*stg)[4] = hip_mma_stg[wid];

    stg[lane][0] = b_hi[0]; stg[lane][1] = b_hi[1];
    stg[lane][2] = b_lo[0]; stg[lane][3] = b_lo[1];
    __syncwarp();

    HipFp16x8 b = {};
    uint32_t* bw = reinterpret_cast<uint32_t*>(&b);
    const int col = lane & 15;                 // AMD B column this lane contributes
    for (int j = 0; j < 4; ++j)
    {
        // AMD reg pair (2j,2j+1) = B[k = 8*(lane>>4)+2j (+1)][col]:
        // CUDA src lane 4*(col&7)+j, half2 (b_hi|b_lo)[(k>>3)&1] with reg = lane>>4
        const __half2 v = stg[4 * (col & 7) + j][(col >= 8 ? 2 : 0) + (lane >> 4)];
        bw[j] = *reinterpret_cast<const uint32_t*>(&v);
    }
    return b;
}

// C fragment <-> tile map for the kernel epilogue:
//   C(lane, reg) = C[row = (lane>>4)*8 + reg][col = lane & 15]
// (documented here so the kernel's reduction stage can consume FragC8 directly.)
#endif  // __gfx1200__ || __gfx1201__

// mma m16n8k16 x2 (n=16)  ->  one v_wmma_f32_16x16x16_f16
// A from a01/a23, B from b_hi(=f0, cols 0..7) + b_lo(=f1, cols 8..15), fp32 accumulate.
template <typename FragC_t>
__device__ __forceinline__ void mma_ab_h_hip(
    const FragB& a01,
    const FragB& a23,
    const FragB& b_hi,
    const FragB& b_lo,
    FragC_t& c)
{
#if defined(__gfx1200__) || defined(__gfx1201__)
    static_assert(sizeof(FragC_t) == sizeof(HipFp32x8),
                  "gfx12 path accumulates in 8 fp32 per lane (FragC8)");
    HipFp16x8 a = assemble_a_frag_gfx12(a01, a23);
    HipFp16x8 b = assemble_b_frag_gfx12(b_hi, b_lo);
    HipFp32x8 d = *reinterpret_cast<HipFp32x8*>(&c);
    d = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, d);
    *reinterpret_cast<HipFp32x8*>(&c) = d;
#elif defined(__gfx90a__) || defined(__gfx94__) || defined(__gfx950__)
    // CDNA MFMA layout is not oracle-verified. This inert body exists only so a fat ROCm
    // build can compile; exl3_gemv_try_launch rejects CDNA at runtime.
    (void)a01; (void)a23; (void)b_hi; (void)b_lo; (void)c;
#else
    (void)a01; (void)a23; (void)b_hi; (void)b_lo; (void)c;   // host pass / unknown arch: parse-only
#endif
}

// m==1 fast-path hook. The gfx12 assembly already produces the mostly-zero A fragment
// (rows 8..15 zero, rows m..7 come back zero from the kernel's zeroed sources), so the
// generic path is correct for m==1; a register-cost cut (skip staging when row == 0 only)
// is a later optimization.
template <typename FragC_t>
__device__ __forceinline__ void mma_ab_h_hip_m1(
    const FragB& a01,
    const FragB& a23,
    const FragB& b_hi,
    const FragB& b_lo,
    FragC_t& c)
{
    mma_ab_h_hip(a01, a23, b_hi, b_lo, c);
}

// =====================================================================================
// CDNA (gfx90a/gfx94x/gfx950): v_mfma_f32_16x16x16_f16 -- RESEARCH NOTES (SCAFFOLD)
// =====================================================================================
// Builtin (ROCm 7.2.4 clang; confirmed by compile + disassembly for -mcpu=gfx90a, and
// by the shipped ComposableKernel headers at /opt/rocm 7.2.4, include/ck/.../amd_xdlops.hpp):
//   __builtin_amdgcn_mfma_f32_16x16x16f16(HipFp16x4 a, HipFp16x4 b, HipFp32x4 c, 0, 0, 0)
//     -> v_mfma_f32_16x16x16_f16      (M=16, N=16, K=16; f16 in, fp32 accumulate)
//   A, B: 4 fp16 per lane (2 dwords); C: 4 fp32 per lane; cbsz=abid=blgp=0 (CK convention).
// ARCHITECTURALLY CERTAIN: one 16x16x16 instruction covers the full N=16 tile (vs two
// m16n8k16 on CUDA); fp32 accumulate; no fp16-accumulate variant.
// NOT VERIFIED (MUST-ORACLE): the per-lane (row,k)/(k,col)/(row,col) placement, AND the
// wave64 wrinkle -- CDNA MFMA data spans a 64-lane wavefront, so a 32-lane HIP warp
// alone does not hold a full 16x16 fragment; block/warp pairing (cbsz/abid) must be
// settled on device.  The AMD ISA manual page was not reachable from this session and
// the shipped CK headers encode the layout only implicitly (xdlops_gemm.hpp /
// amd_wmma.hpp), so nothing here is claimed correct until the gfx90a/94x oracle runs.
// TODO(device, oracle-verify): fill the CDNA lane<->element tables and enable this path.

#endif  // __HIPCC__
