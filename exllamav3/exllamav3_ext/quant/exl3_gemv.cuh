#pragma once

#include <ATen/Tensor.h>
#include <cuda_runtime.h>

// QTIP-style small-m GEMV path (see exl3_gemv_kernel.cuh). Launched from exl3_gemm when the
// heuristic applies; also exposed directly for testing. Same kernel arguments as
// exl3_gemm_kernel, so graph recording/patching is identical.

// True when `device` can execute the GEMV tensor-core kernel. On ROCm this is limited to
// the oracle-verified gfx1200/gfx1201 WMMA implementations.
bool exl3_gemv_supported(int device);

#if defined(USE_ROCM)
// gfx12 decode/verification grouped MoE path for the Qwen3.8 Flash K3/mul1 expert shape.
// Supports 1..5 token rows; routing IDs and weights remain device-resident and duplicate
// assignment slots are preserved independently per token.
void exl3_moe_gfx12_k3
(
    const at::Tensor& A,
    at::Tensor& output,
    const at::Tensor& selected,
    const at::Tensor& weights,
    const at::Tensor& gate_trellis,
    const at::Tensor& gate_suh,
    const at::Tensor& gate_svh,
    const at::Tensor& up_trellis,
    const at::Tensor& up_suh,
    const at::Tensor& up_svh,
    const at::Tensor& down_trellis,
    const at::Tensor& down_suh,
    const at::Tensor& down_svh,
    at::Tensor& gu_had,
    at::Tensor& gu_out,
    at::Tensor& down_had,
    at::Tensor& down_out
);
#endif

// Try to dispatch a GEMM call to the GEMV kernel. Returns false (launching nothing) if the
// call is not eligible. On success *launched_kernel receives the kernel pointer for graph
// recording. `force` bypasses the shape heuristic but not the hard constraints.
bool exl3_gemv_try_launch
(
    void** kernel_args,
    int size_m,
    int size_k,
    int size_n,
    int K,
    int cb,
    bool c_fp32,
    bool has_su_sv,
    int device,
    cudaStream_t stream,
    void** launched_kernel,
    bool force
);

// Direct entry point (testing): errors if the call is not hard-eligible
void exl3_gemv
(
    const at::Tensor& A,
    const at::Tensor& B,
    at::Tensor& C,
    const c10::optional<at::Tensor>& suh,
    const c10::optional<at::Tensor>& A_had,
    const c10::optional<at::Tensor>& svh,
    bool mcg,
    bool mul1
);
