#if defined(USE_ROCM)
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#else
#include <cuda_fp16.h>
#endif
#include "exl3_gemv.cuh"
#include "hadamard.cuh"

#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#if !defined(USE_ROCM)
#include <cooperative_groups.h>
namespace cg = cooperative_groups;
#endif
#include "../util.h"
#include "../util.cuh"
#include "exl3_gemv_kernel.cuh"
#include "exl3_devctx.cuh"
#include <cstdio>
#include <cstring>
#include <map>
#include <mutex>

/*
QTIP-style small-m GEMV path, kernel in exl3_gemv_kernel.cuh. Dispatched from exl3_gemm via
exl3_gemv_try_launch when the shape heuristic applies, or forced through the exl3_gemv entry
point. Kernel arguments and graph parameter offsets are identical to exl3_gemm_kernel.

Env: EXL3_GEMV = 0 disables the path, 1/unset = heuristic (default), 2 = use wherever the hard
constraints allow (testing). EXL3_GEMV_DEBUG=1 prints each newly queried kernel occupancy.

The CUDA heuristic envelope (measured on RTX 3090 at 4 bpw and m <= 8) favors the narrow
config at attention-projection sizes (n <= 4096) and the wide config at large-n/small-k FFN
sizes. HIP gfx12 additionally supports m <= 16 through its 16-row WMMA mode.
*/

#if defined(USE_ROCM)
constexpr int EXL3_GEMV_ROCM_MAX_M = 16;


namespace {

constexpr int MOE_HIDDEN = 2560;
constexpr int MOE_TOP_K = 10;
constexpr int MOE_MAX_ROWS = 16;
constexpr int MOE_PREFILL_MAX_ROWS = 2048;
constexpr int MOE_PREFILL_ROWS_PER_CHUNK = 16;
constexpr int MOE_PREFILL_MAX_EXPERT_ROWS = MOE_PREFILL_MAX_ROWS * MOE_TOP_K;
constexpr int MOE_PREFILL_CHUNKS_PER_EXPERT =
    MOE_PREFILL_MAX_EXPERT_ROWS / MOE_PREFILL_ROWS_PER_CHUNK;
constexpr int MOE_THREADS = 512;
constexpr float HAD_SCALE = 0.088388347648f;

template <bool PRE_SCALE, bool FP32>
__global__ __launch_bounds__(32)
void moe_had_rows_kernel
(
    const void* input,
    void* output,
    const int64_t* selected,
    const int64_t* scale_tables_0,
    const int64_t* scale_tables_1,
    int assignments,
    int width,
    int experts,
    bool token_input
)
{
    const int matrix = blockIdx.x;
    const int slot = matrix % assignments;
    const int projection = matrix / assignments;
    const int64_t expert = selected[slot];
    const int64_t* scale_table = projection == 0 ? scale_tables_0 : scale_tables_1;
    const size_t output_offset = (size_t) matrix * width + blockIdx.y * 128;
    const bool valid_expert = expert >= 0 && expert < experts;
    const int64_t scale_ptr = valid_expert ? scale_table[expert] : 0;
    if (!scale_ptr)
    {
        if constexpr (FP32)
        {
            float* output_ptr = reinterpret_cast<float*>(output) + output_offset;
            for (int idx = threadIdx.x; idx < 128; idx += blockDim.x) output_ptr[idx] = 0.0f;
        }
        else
        {
            half* output_ptr = reinterpret_cast<half*>(output) + output_offset;
            for (int idx = threadIdx.x; idx < 128; idx += blockDim.x) output_ptr[idx] = __float2half_rn(0.0f);
        }
        return;
    }

    const half* scale = reinterpret_cast<const half*>(scale_ptr);
    const size_t input_row = token_input ? slot / MOE_TOP_K : matrix;
    const size_t input_offset = input_row * width + blockIdx.y * 128;

    if constexpr (FP32)
        had_ff_r_128_inner<PRE_SCALE, !PRE_SCALE>
        (
            reinterpret_cast<const float*>(input) + input_offset,
            reinterpret_cast<float*>(output) + output_offset,
            scale,
            HAD_SCALE
        );
    else
        had_hf_r_128_inner<PRE_SCALE, !PRE_SCALE>
        (
            reinterpret_cast<const half*>(input) + input_offset,
            reinterpret_cast<half*>(output) + output_offset,
            scale,
            HAD_SCALE
        );
}

__global__ void moe_silu_mul_kernel(const half* gate, const half* up, half* output, int count)
{
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < count; idx += blockDim.x * gridDim.x)
    {
        const float g = __half2float(gate[idx]);
        const half activated = __float2half_rn(g / (1.0f + expf(-g)));
        output[idx] = __hmul(activated, up[idx]);
    }
}

template <bool FP32, bool TWO_PROJECTIONS>
__global__ __launch_bounds__(MOE_THREADS)
void moe_grouped_gemv_k3_kernel
(
    const half* A,
    const int64_t* selected,
    const int64_t* B_table_0,
    const int64_t* B_table_1,
    void* C,
    int size_k,
    int size_n,
    int experts,
    int assignments
)
{
    const int slot = blockIdx.y;
    const int projection = TWO_PROJECTIONS ? blockIdx.z : 0;
    const int64_t expert = selected[slot];
    const int64_t* B_table = projection == 0 ? B_table_0 : B_table_1;
    const size_t matrix = (size_t) projection * assignments + slot;
    void* C_row = FP32
        ? static_cast<void*>(reinterpret_cast<float*>(C) + matrix * size_n)
        : static_cast<void*>(reinterpret_cast<half*>(C) + matrix * size_n);
    const bool valid_expert = expert >= 0 && expert < experts;
    const int64_t B_ptr = valid_expert ? B_table[expert] : 0;
    if (!B_ptr)
    {
        const int col = blockIdx.x * 32 + threadIdx.x;
        if (threadIdx.x < 32 && col < size_n)
        {
            if constexpr (FP32) reinterpret_cast<float*>(C_row)[col] = 0.0f;
            else reinterpret_cast<half*>(C_row)[col] = __float2half_rn(0.0f);
        }
        return;
    }

    const uint16_t* B = reinterpret_cast<const uint16_t*>(B_ptr);
    const half* A_row = A + matrix * size_k;
    exl3_gemv_kernel_body<3, FP32, 2, 0, 0, true>
    (A_row, B, C_row, 1, size_k, size_n, nullptr, nullptr, nullptr, nullptr);
}

__global__ void moe_weighted_reduce_kernel
(
    const float* rows,
    const int64_t* selected,
    const half* weights,
    float* output,
    int width
)
{
    const int row = blockIdx.y;
    rows += (size_t) row * MOE_TOP_K * width;
    selected += row * MOE_TOP_K;
    weights += row * MOE_TOP_K;
    output += (size_t) row * width;
    for (int col = blockIdx.x * blockDim.x + threadIdx.x;
         col < width; col += blockDim.x * gridDim.x)
    {
        // Match the established fallback's expert-sorted accumulation order independently
        // for each token. Ties retain routing-slot order, so duplicates remain distinct.
        unsigned used = 0;
        float sum = 0.0f;
        #pragma unroll
        for (int rank = 0; rank < MOE_TOP_K; ++rank)
        {
            int best_slot = -1;
            int64_t best_expert = INT64_MAX;
            #pragma unroll
            for (int slot = 0; slot < MOE_TOP_K; ++slot)
            {
                const bool available = !(used & (1u << slot));
                const int64_t expert = selected[slot];
                if (available && (expert < best_expert ||
                    (expert == best_expert && (best_slot < 0 || slot < best_slot))))
                {
                    best_expert = expert;
                    best_slot = slot;
                }
            }
            used |= 1u << best_slot;
            const float weighted = __fmul_rn(
                rows[(size_t) best_slot * width + col], __half2float(weights[best_slot]));
            sum = __fadd_rn(sum, weighted);
        }
        output[col] = sum;
    }
}

__global__ void moe_prefill_metadata_kernel
(
    const int64_t* expert_count,
    const int64_t* order,
    int64_t* expert_offsets,
    int64_t* inverse_order,
    int* expert_chunks,
    int* num_chunks_out,
    int experts,
    int assignments
)
{
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < assignments; idx += blockDim.x * gridDim.x)
    {
        const int64_t original_slot = order[idx];
        if (original_slot >= 0 && original_slot < assignments)
            inverse_order[original_slot] = idx;
    }

    if (blockIdx.x == 0 && threadIdx.x == 0)
    {
        int64_t offset = 0;
        int num_chunks = 0;
        expert_offsets[0] = 0;
        for (int expert = 0; expert < experts; ++expert)
        {
            const int64_t count = expert_count[expert];
            offset += count;
            expert_offsets[expert + 1] = offset;
            if (count > 0 && count <= MOE_PREFILL_MAX_EXPERT_ROWS)
                for (int chunk = 0;
                     chunk * MOE_PREFILL_ROWS_PER_CHUNK < count; ++chunk)
                    expert_chunks[num_chunks++] =
                        expert * MOE_PREFILL_CHUNKS_PER_EXPERT + chunk;
        }
        *num_chunks_out = num_chunks;
    }
}

template <bool PRE_SCALE, bool FP32>
__global__ __launch_bounds__(32)
void moe_prefill_had_rows_kernel
(
    const void* input,
    void* output,
    const int64_t* selected,
    const int64_t* order,
    const int64_t* expert_count,
    const int64_t* scale_tables_0,
    const int64_t* scale_tables_1,
    int assignments,
    int width,
    int experts,
    bool token_input
)
{
    const int matrix = blockIdx.x;
    const int sorted_slot = matrix % assignments;
    const int projection = matrix / assignments;
    const int64_t original_slot = order[sorted_slot];
    const bool valid_order = original_slot >= 0 && original_slot < assignments;
    const int64_t expert = valid_order ? selected[original_slot] : -1;
    const int64_t* scale_table = projection == 0 ? scale_tables_0 : scale_tables_1;
    const size_t output_offset = (size_t) matrix * width + blockIdx.y * 128;
    const bool valid_expert = expert >= 0 && expert < experts &&
        expert_count[expert] <= MOE_PREFILL_MAX_EXPERT_ROWS;
    const int64_t scale_ptr = valid_expert ? scale_table[expert] : 0;
    if (!scale_ptr)
    {
        if constexpr (FP32)
        {
            float* output_ptr = reinterpret_cast<float*>(output) + output_offset;
            for (int idx = threadIdx.x; idx < 128; idx += blockDim.x) output_ptr[idx] = 0.0f;
        }
        else
        {
            half* output_ptr = reinterpret_cast<half*>(output) + output_offset;
            for (int idx = threadIdx.x; idx < 128; idx += blockDim.x)
                output_ptr[idx] = __float2half_rn(0.0f);
        }
        return;
    }

    const half* scale = reinterpret_cast<const half*>(scale_ptr);
    const size_t input_row = token_input ? original_slot / MOE_TOP_K : matrix;
    const size_t input_offset = input_row * width + blockIdx.y * 128;
    if constexpr (FP32)
        had_ff_r_128_inner<PRE_SCALE, !PRE_SCALE>
        (
            reinterpret_cast<const float*>(input) + input_offset,
            reinterpret_cast<float*>(output) + output_offset,
            scale,
            HAD_SCALE
        );
    else
        had_hf_r_128_inner<PRE_SCALE, !PRE_SCALE>
        (
            reinterpret_cast<const half*>(input) + input_offset,
            reinterpret_cast<half*>(output) + output_offset,
            scale,
            HAD_SCALE
        );
}

template <bool FP32, bool TWO_PROJECTIONS, int CFG>
__global__ __launch_bounds__(CFG == 0 ? 512 : CFG == 1 ? 256 : 128)
void moe_prefill_grouped_gemv_k3_kernel
(
    const half* A,
    const int64_t* expert_offsets,
    const int* expert_chunks,
    const int* num_chunks,
    const int64_t* B_table_0,
    const int64_t* B_table_1,
    void* C,
    int size_k,
    int size_n,
    int experts,
    int assignments
)
{
    if (blockIdx.y >= *num_chunks) return;
    const int expert_chunk = expert_chunks[blockIdx.y];
    const int expert = expert_chunk / MOE_PREFILL_CHUNKS_PER_EXPERT;
    const int chunk = expert_chunk % MOE_PREFILL_CHUNKS_PER_EXPERT;

    const int64_t start = expert_offsets[expert];
    const int64_t end = expert_offsets[expert + 1];
    const int64_t row0 = start + chunk * MOE_PREFILL_ROWS_PER_CHUNK;
    const int rows = min((int64_t) MOE_PREFILL_ROWS_PER_CHUNK, end - row0);
    if (rows <= 0) return;

    const int projection = TWO_PROJECTIONS ? blockIdx.z : 0;
    const int64_t* B_table = projection == 0 ? B_table_0 : B_table_1;
    const int64_t B_ptr = B_table[expert];
    const size_t matrix = (size_t) projection * assignments + row0;
    void* C_rows = FP32
        ? static_cast<void*>(reinterpret_cast<float*>(C) + matrix * size_n)
        : static_cast<void*>(reinterpret_cast<half*>(C) + matrix * size_n);
    if (!B_ptr)
    {
        constexpr int cols = CFG == 0 ? 32 : 64;
        for (int idx = threadIdx.x; idx < rows * cols; idx += blockDim.x)
        {
            const int row = idx / cols;
            const int col = blockIdx.x * cols + idx % cols;
            if (col < size_n)
            {
                if constexpr (FP32)
                    reinterpret_cast<float*>(C_rows)[(size_t) row * size_n + col] = 0.0f;
                else
                    reinterpret_cast<half*>(C_rows)[(size_t) row * size_n + col] = __float2half_rn(0.0f);
            }
        }
        return;
    }

    exl3_gemv_kernel_body<3, FP32, 2, 2, CFG, true>
    (
        A + matrix * size_k,
        reinterpret_cast<const uint16_t*>(B_ptr),
        C_rows,
        rows,
        size_k,
        size_n,
        nullptr,
        nullptr,
        nullptr,
        nullptr
    );
}

__global__ void moe_prefill_weighted_reduce_kernel
(
    const float* sorted_rows,
    const int64_t* selected,
    const half* weights,
    const int64_t* inverse_order,
    const int64_t* expert_count,
    float* output,
    int rows,
    int experts,
    int width
)
{
    const int row = blockIdx.y;
    if (row >= rows) return;
    selected += row * MOE_TOP_K;
    weights += row * MOE_TOP_K;
    output += (size_t) row * width;
    for (int col = blockIdx.x * blockDim.x + threadIdx.x;
         col < width; col += blockDim.x * gridDim.x)
    {
        unsigned used = 0;
        float sum = 0.0f;
        #pragma unroll
        for (int rank = 0; rank < MOE_TOP_K; ++rank)
        {
            int best_slot = -1;
            int64_t best_expert = INT64_MAX;
            #pragma unroll
            for (int slot = 0; slot < MOE_TOP_K; ++slot)
            {
                const bool available = !(used & (1u << slot));
                const int64_t expert = selected[slot];
                if (available && (expert < best_expert ||
                    (expert == best_expert && (best_slot < 0 || slot < best_slot))))
                {
                    best_expert = expert;
                    best_slot = slot;
                }
            }
            used |= 1u << best_slot;
            if (best_expert >= 0 && best_expert < experts &&
                expert_count[best_expert] <= MOE_PREFILL_MAX_EXPERT_ROWS)
            {
                const int64_t original_slot = (int64_t) row * MOE_TOP_K + best_slot;
                const int64_t sorted_slot = inverse_order[original_slot];
                if (sorted_slot >= 0 && sorted_slot < (int64_t) rows * MOE_TOP_K)
                {
                    const float weighted = __fmul_rn(
                        sorted_rows[(size_t) sorted_slot * width + col],
                        __half2float(weights[best_slot]));
                    sum = __fadd_rn(sum, weighted);
                }
            }
        }
        output[col] = sum;
    }
}

void check_ptr_table
(
    const at::Tensor& table,
    int64_t experts,
    const at::Device& device,
    const char* name
)
{
    TORCH_CHECK_DTYPE(table, kLong);
    TORCH_CHECK(table.device() == device && table.is_contiguous() && table.dim() == 1 &&
                table.numel() == experts, name, " must be a contiguous same-device pointer table");
}

void check_moe_alignment(const at::Tensor& tensor, const char* name)
{
    TORCH_CHECK(reinterpret_cast<uintptr_t>(tensor.data_ptr()) % 16 == 0,
                "exl3_moe_gfx12_k3: ", name, " must be 16-byte aligned");
}

} // namespace

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
)
{
    const at::cuda::OptionalCUDAGuard device_guard(A.device());
    hipStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    int device;
    cuda_check(hipGetDevice(&device));
    TORCH_CHECK(exl3_gemv_supported(device), "exl3_moe_gfx12_k3 requires gfx1200/gfx1201");

    TORCH_CHECK(A.is_cuda() && A.is_contiguous() && A.dtype() == at::kHalf &&
                A.dim() == 2 && A.size(1) == MOE_HIDDEN &&
                A.size(0) >= 1 && A.size(0) <= MOE_MAX_ROWS,
                "exl3_moe_gfx12_k3 requires contiguous fp16[R, 2560], R=1..16");
    const int rows = A.size(0);
    const int assignments = rows * MOE_TOP_K;
    TORCH_CHECK(output.device() == A.device() && output.is_contiguous() && output.dtype() == at::kFloat &&
                output.dim() == 2 && output.size(0) == rows && output.size(1) == MOE_HIDDEN,
                "exl3_moe_gfx12_k3 output must be contiguous fp32[R, 2560]");
    TORCH_CHECK(selected.device() == A.device() && selected.is_contiguous() && selected.dtype() == at::kLong &&
                selected.dim() == 2 && selected.size(0) == rows && selected.size(1) == MOE_TOP_K,
                "exl3_moe_gfx12_k3 selected must be contiguous device int64[R, 10]");
    TORCH_CHECK(weights.device() == A.device() && weights.is_contiguous() && weights.dtype() == at::kHalf &&
                weights.dim() == 2 && weights.size(0) == rows && weights.size(1) == MOE_TOP_K,
                "exl3_moe_gfx12_k3 weights must be contiguous device fp16[R, 10]");

    const int64_t experts = gate_trellis.numel();
    TORCH_CHECK(experts > 0, "exl3_moe_gfx12_k3 requires nonempty expert tables");
    check_ptr_table(gate_trellis, experts, A.device(), "gate_trellis");
    check_ptr_table(gate_suh, experts, A.device(), "gate_suh");
    check_ptr_table(gate_svh, experts, A.device(), "gate_svh");
    check_ptr_table(up_trellis, experts, A.device(), "up_trellis");
    check_ptr_table(up_suh, experts, A.device(), "up_suh");
    check_ptr_table(up_svh, experts, A.device(), "up_svh");
    check_ptr_table(down_trellis, experts, A.device(), "down_trellis");
    check_ptr_table(down_suh, experts, A.device(), "down_suh");
    check_ptr_table(down_svh, experts, A.device(), "down_svh");

    TORCH_CHECK(down_had.numel() % assignments == 0,
                "down_had has the wrong shape");
    const int intermediate = down_had.numel() / assignments;
    TORCH_CHECK(intermediate == 640 || intermediate == 768,
                "exl3_moe_gfx12_k3 intermediate width must be 640 or 768");
    TORCH_CHECK(gu_had.device() == A.device() && gu_had.is_contiguous() && gu_had.dtype() == at::kHalf &&
                gu_had.numel() == 2 * assignments * MOE_HIDDEN,
                "gu_had must be contiguous fp16[2 * R * 10, 2560]");
    TORCH_CHECK(gu_out.device() == A.device() && gu_out.is_contiguous() && gu_out.dtype() == at::kHalf &&
                gu_out.numel() == 2 * assignments * intermediate,
                "gu_out has the wrong shape");
    TORCH_CHECK(down_had.device() == A.device() && down_had.is_contiguous() && down_had.dtype() == at::kHalf &&
                down_had.numel() == assignments * intermediate,
                "down_had has the wrong shape");
    TORCH_CHECK(down_out.device() == A.device() && down_out.is_contiguous() && down_out.dtype() == at::kFloat &&
                down_out.numel() == assignments * MOE_HIDDEN,
                "down_out must be contiguous fp32[R * 10, 2560]");

    check_moe_alignment(A, "A");
    check_moe_alignment(output, "output");
    check_moe_alignment(gu_had, "gu_had");
    check_moe_alignment(gu_out, "gu_out");
    check_moe_alignment(down_had, "down_had");
    check_moe_alignment(down_out, "down_out");

    // MultiLinear constructs these same-device tables once and keeps their source tensors
    // alive. Entries remain an internal raw-pointer trust boundary: checking every pointer on
    // the host here would synchronize and copy all tables on every decode call.
    const int64_t* selected_ptr = reinterpret_cast<const int64_t*>(selected.data_ptr());
    const half* weights_ptr = reinterpret_cast<const half*>(weights.data_ptr());
    const int64_t* gt = reinterpret_cast<const int64_t*>(gate_trellis.data_ptr());
    const int64_t* gsuh = reinterpret_cast<const int64_t*>(gate_suh.data_ptr());
    const int64_t* gsvh = reinterpret_cast<const int64_t*>(gate_svh.data_ptr());
    const int64_t* ut = reinterpret_cast<const int64_t*>(up_trellis.data_ptr());
    const int64_t* usuh = reinterpret_cast<const int64_t*>(up_suh.data_ptr());
    const int64_t* usvh = reinterpret_cast<const int64_t*>(up_svh.data_ptr());
    const int64_t* dt = reinterpret_cast<const int64_t*>(down_trellis.data_ptr());
    const int64_t* dsuh = reinterpret_cast<const int64_t*>(down_suh.data_ptr());
    const int64_t* dsvh = reinterpret_cast<const int64_t*>(down_svh.data_ptr());

    dim3 had_gu_grid(2 * assignments, MOE_HIDDEN / 128);
    moe_had_rows_kernel<true, false><<<had_gu_grid, 32, 0, stream>>>
    (A.data_ptr(), gu_had.data_ptr(), selected_ptr, gsuh, usuh,
     assignments, MOE_HIDDEN, experts, true);

    dim3 gu_grid(intermediate / 32, assignments, 2);
    moe_grouped_gemv_k3_kernel<false, true><<<gu_grid, MOE_THREADS, 0, stream>>>
    (reinterpret_cast<const half*>(gu_had.data_ptr()), selected_ptr, gt, ut,
     gu_out.data_ptr(), MOE_HIDDEN, intermediate, experts, assignments);

    moe_had_rows_kernel<false, false><<<dim3(2 * assignments, intermediate / 128), 32, 0, stream>>>
    (gu_out.data_ptr(), gu_out.data_ptr(), selected_ptr, gsvh, usvh,
     assignments, intermediate, experts, false);

    const int activation_count = assignments * intermediate;
    const half* gu_ptr = reinterpret_cast<const half*>(gu_out.data_ptr());
    moe_silu_mul_kernel<<<CEIL_DIVIDE(activation_count, 256), 256, 0, stream>>>
    (gu_ptr, gu_ptr + activation_count, reinterpret_cast<half*>(down_had.data_ptr()), activation_count);

    moe_had_rows_kernel<true, false><<<dim3(assignments, intermediate / 128), 32, 0, stream>>>
    (down_had.data_ptr(), down_had.data_ptr(), selected_ptr, dsuh, dsuh,
     assignments, intermediate, experts, false);

    dim3 down_grid(MOE_HIDDEN / 32, assignments, 1);
    moe_grouped_gemv_k3_kernel<true, false><<<down_grid, MOE_THREADS, 0, stream>>>
    (reinterpret_cast<const half*>(down_had.data_ptr()), selected_ptr, dt, dt,
     down_out.data_ptr(), intermediate, MOE_HIDDEN, experts, assignments);

    moe_had_rows_kernel<false, true><<<dim3(assignments, MOE_HIDDEN / 128), 32, 0, stream>>>
    (down_out.data_ptr(), down_out.data_ptr(), selected_ptr, dsvh, dsvh,
     assignments, MOE_HIDDEN, experts, false);

    moe_weighted_reduce_kernel<<<dim3(CEIL_DIVIDE(MOE_HIDDEN, 256), rows), 256, 0, stream>>>
    (reinterpret_cast<const float*>(down_out.data_ptr()), selected_ptr, weights_ptr,
     reinterpret_cast<float*>(output.data_ptr()), MOE_HIDDEN);
    cuda_check(hipPeekAtLastError());
}

void exl3_moe_gfx12_k3_prefill
(
    const at::Tensor& A,
    at::Tensor& output,
    const at::Tensor& selected,
    const at::Tensor& weights,
    const at::Tensor& order,
    const at::Tensor& expert_count,
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
    at::Tensor& down_out,
    at::Tensor& expert_offsets,
    at::Tensor& inverse_order,
    at::Tensor& expert_chunks,
    at::Tensor& chunk_count
)
{
    const at::cuda::OptionalCUDAGuard device_guard(A.device());
    hipStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    int device;
    cuda_check(hipGetDevice(&device));
    TORCH_CHECK(exl3_gemv_supported(device),
                "exl3_moe_gfx12_k3_prefill requires gfx1200/gfx1201");

    TORCH_CHECK(A.is_cuda() && A.is_contiguous() && A.dtype() == at::kHalf &&
                A.dim() == 2 && A.size(1) == MOE_HIDDEN &&
                A.size(0) >= 2 && A.size(0) <= MOE_PREFILL_MAX_ROWS,
                "exl3_moe_gfx12_k3_prefill requires contiguous fp16[R, 2560], R=2..2048");
    const int rows = A.size(0);
    const int assignments = rows * MOE_TOP_K;
    TORCH_CHECK(output.device() == A.device() && output.is_contiguous() &&
                output.dtype() == at::kFloat && output.dim() == 2 &&
                output.size(0) == rows && output.size(1) == MOE_HIDDEN,
                "prefill output must be contiguous fp32[R, 2560]");
    TORCH_CHECK(selected.device() == A.device() && selected.is_contiguous() &&
                selected.dtype() == at::kLong && selected.dim() == 2 &&
                selected.size(0) == rows && selected.size(1) == MOE_TOP_K,
                "prefill selected must be contiguous device int64[R, 10]");
    TORCH_CHECK(weights.device() == A.device() && weights.is_contiguous() &&
                weights.dtype() == at::kHalf && weights.sizes() == selected.sizes(),
                "prefill weights must be contiguous device fp16[R, 10]");
    TORCH_CHECK(order.device() == A.device() && order.is_contiguous() &&
                order.dtype() == at::kLong && order.dim() == 1 && order.numel() == assignments,
                "prefill order must be contiguous device int64[R * 10]");

    const int64_t experts = gate_trellis.numel();
    TORCH_CHECK(experts > 0, "prefill requires nonempty expert tables");
    TORCH_CHECK(expert_count.device() == A.device() && expert_count.is_contiguous() &&
                expert_count.dtype() == at::kLong && expert_count.dim() == 1 &&
                expert_count.numel() == experts + 1,
                "prefill expert_count must be contiguous device int64[E + 1]");
    check_ptr_table(gate_trellis, experts, A.device(), "gate_trellis");
    check_ptr_table(gate_suh, experts, A.device(), "gate_suh");
    check_ptr_table(gate_svh, experts, A.device(), "gate_svh");
    check_ptr_table(up_trellis, experts, A.device(), "up_trellis");
    check_ptr_table(up_suh, experts, A.device(), "up_suh");
    check_ptr_table(up_svh, experts, A.device(), "up_svh");
    check_ptr_table(down_trellis, experts, A.device(), "down_trellis");
    check_ptr_table(down_suh, experts, A.device(), "down_suh");
    check_ptr_table(down_svh, experts, A.device(), "down_svh");

    TORCH_CHECK(gu_out.numel() % (2 * assignments) == 0, "prefill gu_out has the wrong shape");
    const int intermediate = gu_out.numel() / (2 * assignments);
    TORCH_CHECK(intermediate == 640 || intermediate == 768,
                "prefill intermediate width must be 640 or 768");
    TORCH_CHECK(gu_had.device() == A.device() && gu_had.is_contiguous() &&
                gu_had.dtype() == at::kHalf &&
                gu_had.numel() == 2 * assignments * MOE_HIDDEN,
                "prefill gu_had must be contiguous fp16[2 * R * 10, 2560]");
    TORCH_CHECK(gu_out.device() == A.device() && gu_out.is_contiguous() &&
                gu_out.dtype() == at::kHalf &&
                gu_out.numel() == 2 * assignments * intermediate,
                "prefill gu_out has the wrong shape");
    TORCH_CHECK(down_out.device() == A.device() && down_out.is_contiguous() &&
                down_out.dtype() == at::kFloat &&
                down_out.numel() == assignments * MOE_HIDDEN,
                "prefill down_out must be contiguous fp32[R * 10, 2560]");
    TORCH_CHECK(expert_offsets.device() == A.device() && expert_offsets.is_contiguous() &&
                expert_offsets.dtype() == at::kLong && expert_offsets.numel() == experts + 1,
                "prefill expert_offsets must be contiguous device int64[E + 1]");
    TORCH_CHECK(inverse_order.device() == A.device() && inverse_order.is_contiguous() &&
                inverse_order.dtype() == at::kLong && inverse_order.numel() == assignments,
                "prefill inverse_order must be contiguous device int64[R * 10]");
    TORCH_CHECK(expert_chunks.device() == A.device() && expert_chunks.is_contiguous() &&
                expert_chunks.dtype() == at::kInt &&
                expert_chunks.numel() >= experts * MOE_PREFILL_CHUNKS_PER_EXPERT,
                "prefill expert_chunks workspace is too small");
    TORCH_CHECK(chunk_count.device() == A.device() && chunk_count.is_contiguous() &&
                chunk_count.dtype() == at::kInt && chunk_count.numel() == 1,
                "prefill chunk_count must be a same-device int32 scalar workspace");

    check_moe_alignment(A, "A");
    check_moe_alignment(output, "output");
    check_moe_alignment(gu_had, "gu_had");
    check_moe_alignment(gu_out, "gu_out");
    check_moe_alignment(down_out, "down_out");

    const int64_t* selected_ptr = reinterpret_cast<const int64_t*>(selected.data_ptr());
    const half* weights_ptr = reinterpret_cast<const half*>(weights.data_ptr());
    const int64_t* order_ptr = reinterpret_cast<const int64_t*>(order.data_ptr());
    const int64_t* counts_ptr = reinterpret_cast<const int64_t*>(expert_count.data_ptr());
    int64_t* offsets_ptr = reinterpret_cast<int64_t*>(expert_offsets.data_ptr());
    int64_t* inverse_ptr = reinterpret_cast<int64_t*>(inverse_order.data_ptr());
    int* chunks_ptr = reinterpret_cast<int*>(expert_chunks.data_ptr());
    int* chunk_count_ptr = reinterpret_cast<int*>(chunk_count.data_ptr());
    const int64_t* gt = reinterpret_cast<const int64_t*>(gate_trellis.data_ptr());
    const int64_t* gsuh = reinterpret_cast<const int64_t*>(gate_suh.data_ptr());
    const int64_t* gsvh = reinterpret_cast<const int64_t*>(gate_svh.data_ptr());
    const int64_t* ut = reinterpret_cast<const int64_t*>(up_trellis.data_ptr());
    const int64_t* usuh = reinterpret_cast<const int64_t*>(up_suh.data_ptr());
    const int64_t* usvh = reinterpret_cast<const int64_t*>(up_svh.data_ptr());
    const int64_t* dt = reinterpret_cast<const int64_t*>(down_trellis.data_ptr());
    const int64_t* dsuh = reinterpret_cast<const int64_t*>(down_suh.data_ptr());
    const int64_t* dsvh = reinterpret_cast<const int64_t*>(down_svh.data_ptr());

    cuda_check(hipMemsetAsync(inverse_ptr, 0xff, assignments * sizeof(int64_t), stream));
    moe_prefill_metadata_kernel<<<CEIL_DIVIDE(assignments, 256), 256, 0, stream>>>
    (counts_ptr, order_ptr, offsets_ptr, inverse_ptr, chunks_ptr, chunk_count_ptr,
     experts, assignments);

    dim3 had_gu_grid(2 * assignments, MOE_HIDDEN / 128);
    moe_prefill_had_rows_kernel<true, false><<<had_gu_grid, 32, 0, stream>>>
    (A.data_ptr(), gu_had.data_ptr(), selected_ptr, order_ptr, counts_ptr, gsuh, usuh,
     assignments, MOE_HIDDEN, experts, true);

    const int chunk_slots = CEIL_DIVIDE(assignments, MOE_PREFILL_ROWS_PER_CHUNK) + experts;
    dim3 gu_grid(intermediate / 64, chunk_slots, 2);
    moe_prefill_grouped_gemv_k3_kernel<false, true, 2><<<gu_grid, 128, 0, stream>>>
    (reinterpret_cast<const half*>(gu_had.data_ptr()), offsets_ptr, chunks_ptr, chunk_count_ptr,
     gt, ut,
     gu_out.data_ptr(), MOE_HIDDEN, intermediate, experts, assignments);

    moe_prefill_had_rows_kernel<false, false>
        <<<dim3(2 * assignments, intermediate / 128), 32, 0, stream>>>
    (gu_out.data_ptr(), gu_out.data_ptr(), selected_ptr, order_ptr, counts_ptr, gsvh, usvh,
     assignments, intermediate, experts, false);

    const int activation_count = assignments * intermediate;
    half* gu_ptr = reinterpret_cast<half*>(gu_out.data_ptr());
    moe_silu_mul_kernel<<<CEIL_DIVIDE(activation_count, 256), 256, 0, stream>>>
    (gu_ptr, gu_ptr + activation_count, gu_ptr, activation_count);

    moe_prefill_had_rows_kernel<true, false>
        <<<dim3(assignments, intermediate / 128), 32, 0, stream>>>
    (gu_out.data_ptr(), gu_out.data_ptr(), selected_ptr, order_ptr, counts_ptr, dsuh, dsuh,
     assignments, intermediate, experts, false);

    dim3 down_grid(MOE_HIDDEN / 64, chunk_slots, 1);
    moe_prefill_grouped_gemv_k3_kernel<true, false, 2><<<down_grid, 128, 0, stream>>>
    (reinterpret_cast<const half*>(gu_out.data_ptr()), offsets_ptr, chunks_ptr, chunk_count_ptr,
     dt, dt,
     down_out.data_ptr(), intermediate, MOE_HIDDEN, experts, assignments);

    moe_prefill_had_rows_kernel<false, true>
        <<<dim3(assignments, MOE_HIDDEN / 128), 32, 0, stream>>>
    (down_out.data_ptr(), down_out.data_ptr(), selected_ptr, order_ptr, counts_ptr, dsvh, dsvh,
     assignments, MOE_HIDDEN, experts, false);

    moe_prefill_weighted_reduce_kernel
        <<<dim3(CEIL_DIVIDE(MOE_HIDDEN, 256), rows), 256, 0, stream>>>
    (reinterpret_cast<const float*>(down_out.data_ptr()), selected_ptr, weights_ptr,
     inverse_ptr, counts_ptr, reinterpret_cast<float*>(output.data_ptr()),
     rows, experts, MOE_HIDDEN);
    cuda_check(hipPeekAtLastError());
}

#endif // USE_ROCM

static int exl3_gemv_env_mode()
{
    const char* env = std::getenv("EXL3_GEMV");
    if (!env) return 1;
    return atoi(env);
}

static bool exl3_gemv_debug()
{
    static const bool enabled = []
    {
        const char* env = std::getenv("EXL3_GEMV_DEBUG");
        return env && env[0] == '1' && env[1] == '\0';
    }();
    return enabled;
}

bool exl3_gemv_supported(int device)
{
#if defined(USE_ROCM)
    static std::map<int, bool> support_cache;
    static std::mutex support_cache_mtx;
    std::lock_guard<std::mutex> lock(support_cache_mtx);
    auto it = support_cache.find(device);
    if (it != support_cache.end()) return it->second;

    hipDeviceProp_t prop;
    if (hipGetDeviceProperties(&prop, device) != hipSuccess) return false;
    const char* arch = prop.gcnArchName;
    bool supported = (!std::strncmp(arch, "gfx1200", 7) || !std::strncmp(arch, "gfx1201", 7))
        && (arch[7] == '\0' || arch[7] == ':');
    support_cache[device] = supported;
    return supported;
#else
    (void) device;
    return true;
#endif
}

// -1 = default per bits, 0 = force shuffle extraction, 1 = force smem staging (testing)
#if !defined(USE_ROCM)
static int exl3_gemv_env_smem()
{
    const char* env = std::getenv("EXL3_GEMV_SMEM");
    if (!env) return -1;
    return atoi(env);
}
#endif

// -1: not eligible, 0: narrow config, 1: wide config. narrow_coresident = number of narrow-config
// blocks that fit on the device at once (its grid is one block per 32 output columns)
static int exl3_gemv_cfg(int cc, int size_m, int size_k, int size_n, int K, int cb, int mode, int narrow_coresident)
{
    if (mode == 0) return -1;
#if defined(USE_ROCM)
    if (K < 2 || K > 6) return -1;
#else
    if (K < 2 || K > 4) return -1;
#endif
    if (K != 4 && cb == 0) return -1;
#if defined(USE_ROCM)
    // K2 has no HIP MMODE-2 instantiation; its established envelope ends at eight rows.
    if (K == 2 && size_m > EXL3_GEMV_MAX_M) return -1;
    if (size_m > EXL3_GEMV_ROCM_MAX_M) return -1;
#else
    if (size_m > EXL3_GEMV_MAX_M) return -1;
#endif
    if (size_k % 128 || size_n % 128) return -1;
    //if (cc != CC_AMPERE) return -1;  // measured win on Ampere; Ada/Blackwell are memory-bound here
    if (mode == 2) return size_n <= 8192 ? 0 : 1;
    if (mode == 3) return 0;   // testing: force narrow config
    if (mode == 4) return 1;   // testing: force wide config

    // The narrow config wins (up to ~30%) whenever its grid fits in a single co-resident wave;
    // in the 1..2-wave zone the trailing partial wave costs more than the kernel gains unless
    // per-group work is small (small k). The wide config covers a band of large-n shapes with
    // small-to-mid k. Everything else runs the regular block-pipelined kernel.
    // Per-bits envelopes: 2 bpw is decode-bound and won at every measured shape on both archs;
    // 3 bpw wins everywhere on Ada but only in the narrow envelope on Ampere
    if (K == 2) return size_n <= 8192 ? 0 : 1;
    if (K == 3 && cc == CC_ADA) return size_n <= 8192 ? 0 : 1;
    if (size_n / 32 <= narrow_coresident) return 0;
    if (size_k <= 2048 && size_n <= 8192) return 0;
    if (K == 3) return -1;
    if (size_n >= 8192 && size_k <= 4096) return 1;
    if (size_n >= 8192 && size_n <= 10240 && size_k <= 5120 && cc == CC_AMPERE) return 1;
    return -1;
}

static void* exl3_gemv_select_kernel(int bits, int cb, bool c_fp32, int mmode, int cfg, bool smem)
{
#if defined(USE_ROCM)
    // HIP: only the smem-staged extraction is instantiated. gfx12 WMMA operands must not
    // be fed through __shfl_sync (hardware exception), so SMEM_STAGE is forced true and
    // the `smem` argument is ignored.
    #define SEL(bits_, cb_, fp32_, mm_, cfg_) \
        if (bits == bits_ && cb == cb_ && c_fp32 == fp32_ && mmode == mm_ && cfg == cfg_) \
            return (void*) exl3_gemv_kernel<bits_, fp32_, cb_, mm_, cfg_, true>;
    #define SEL_GRID(bits_, cb_) \
        SEL(bits_, cb_, false, 0, 0) SEL(bits_, cb_, false, 0, 1) \
        SEL(bits_, cb_, false, 1, 0) SEL(bits_, cb_, false, 1, 1) \
        SEL(bits_, cb_, true,  0, 0) SEL(bits_, cb_, true,  0, 1) \
        SEL(bits_, cb_, true,  1, 0) SEL(bits_, cb_, true,  1, 1)
    #define SEL_MMODE2_GRID(bits_, cb_) \
        SEL(bits_, cb_, false, 2, 0) SEL(bits_, cb_, false, 2, 1) \
        SEL(bits_, cb_, true,  2, 0) SEL(bits_, cb_, true,  2, 1)
    SEL_GRID(4, 0) SEL_GRID(4, 1) SEL_GRID(4, 2)
    SEL_GRID(2, 1) SEL_GRID(2, 2)
    SEL_GRID(3, 1) SEL_GRID(3, 2)
    SEL_GRID(5, 1) SEL_GRID(5, 2)
    SEL_GRID(6, 1) SEL_GRID(6, 2)
    SEL_MMODE2_GRID(3, 2) SEL_MMODE2_GRID(4, 1) SEL_MMODE2_GRID(4, 2)
    SEL_MMODE2_GRID(5, 2) SEL_MMODE2_GRID(6, 1) SEL_MMODE2_GRID(6, 2)
    #undef SEL_MMODE2_GRID
    #undef SEL_GRID
    #undef SEL
#else
    #define SEL(bits_, cb_, fp32_, mm_, cfg_, sm_) \
        if (bits == bits_ && cb == cb_ && c_fp32 == fp32_ && mmode == mm_ && cfg == cfg_ && smem == sm_) \
            return (void*) exl3_gemv_kernel<bits_, fp32_, cb_, mm_, cfg_, sm_>;
    #define SEL_GRID(bits_, cb_, sm_) \
        SEL(bits_, cb_, false, 0, 0, sm_) SEL(bits_, cb_, false, 0, 1, sm_) \
        SEL(bits_, cb_, false, 1, 0, sm_) SEL(bits_, cb_, false, 1, 1, sm_) \
        SEL(bits_, cb_, true,  0, 0, sm_) SEL(bits_, cb_, true,  0, 1, sm_) \
        SEL(bits_, cb_, true,  1, 0, sm_) SEL(bits_, cb_, true,  1, 1, sm_)
    SEL_GRID(4, 0, false) SEL_GRID(4, 1, false) SEL_GRID(4, 2, false)
    SEL_GRID(2, 1, false) SEL_GRID(2, 2, false) SEL_GRID(2, 1, true) SEL_GRID(2, 2, true)
    SEL_GRID(3, 1, false) SEL_GRID(3, 2, false) SEL_GRID(3, 1, true) SEL_GRID(3, 2, true)
    #undef SEL_GRID
    #undef SEL
#endif
    return nullptr;
}

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
#if defined(USE_ROCM)
    hipStream_t stream,
#else
    cudaStream_t stream,
#endif
    void** launched_kernel,
    bool force
)
{
    // Free integer checks first; the env read (~64 ns) and device queries only run for calls
    // that could actually take this path
    if (!has_su_sv) return false;
#if defined(USE_ROCM)
    if (K < 2 || K > 6) return false;
#else
    if (K < 2 || K > 4) return false;
#endif
    if (K != 4 && cb == 0) return false;
#if defined(USE_ROCM)
    // K2 has no HIP MMODE-2 instantiation; its established envelope ends at eight rows.
    if (K == 2 && size_m > EXL3_GEMV_MAX_M) return false;
    if (size_m > EXL3_GEMV_ROCM_MAX_M) return false;
#else
    if (size_m > EXL3_GEMV_MAX_M) return false;
#endif
    if (size_k % 128 || size_n % 128) return false;
#if defined(USE_ROCM)
    if (!exl3_gemv_supported(device)) return false;
#endif

    int mode = force ? 2 : exl3_gemv_env_mode();
    if (mode == 0) return false;
    int cc = DevCtx::instance().get_cc(device);
    // if (cc != CC_AMPERE) return false;
#if defined(USE_ROCM)
    int mmode = size_m == 1 ? 0 : (size_m <= EXL3_GEMV_MAX_M ? 1 : 2);
#else
    int mmode = size_m == 1 ? 0 : 1;
#endif
    int num_sms = DevCtx::instance().get_num_sms(device);

    // CUDA's cooperative grid is capped at full co-residency. HIP retains the same grid sizing
    // for the shape heuristic and grid-stride main loop, but uses an ordinary launch.
    static std::map<void*, int> occ_cache[MAX_DEVICES];
    auto& cache = occ_cache[device];
    auto occupancy = [&] (void* kernel, int block_dim) -> int
    {
        auto it = cache.find(kernel);
        if (it != cache.end()) return it->second;
        int blocks_per_sm;
#if defined(USE_ROCM)
        cuda_check(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm, kernel, block_dim, 0));
#else
        cuda_check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm, kernel, block_dim, 0));
#endif
        cache[kernel] = blocks_per_sm;
        if (exl3_gemv_debug())
            std::fprintf(stderr, "EXL3_GEMV_DEBUG device=%d kernel=%p block_dim=%d blocks_per_sm=%d\n",
                         device, kernel, block_dim, blocks_per_sm);
        return blocks_per_sm;
    };

#if defined(USE_ROCM)
    // Extraction style on HIP: smem staging is the only safe mode (gfx12 shfl->WMMA hazard),
    // so the EXL3_GEMV_SMEM override is ignored and staging is always on
    bool smem = true;
#else
    // Extraction style: shuffle by default, smem staging selectable per call for evaluation
    bool smem = exl3_gemv_env_smem() == 1;
#endif

    void* narrow_kernel = exl3_gemv_select_kernel(K, cb, c_fp32, mmode, 0, smem);
    if (!narrow_kernel) return false;
    int narrow_coresident = occupancy(narrow_kernel, 512) * num_sms;

    int cfg = exl3_gemv_cfg(cc, size_m, size_k, size_n, K, cb, mode, narrow_coresident);
    if (cfg < 0) return false;

    void* kernel = cfg == 0 ? narrow_kernel : exl3_gemv_select_kernel(K, cb, c_fp32, mmode, cfg, smem);
    if (!kernel) return false;

    int block_dim = cfg == 0 ? 512 : 256;
    int cols = cfg == 0 ? 32 : 64;

    int max_blocks = occupancy(kernel, block_dim) * num_sms;
    int grid = MIN(size_n / cols, max_blocks);
    if (grid < 1) return false;

    cuda_check
    (
#if defined(USE_ROCM)
        hipLaunchKernel
#else
        cudaLaunchCooperativeKernel
#endif
        (
            kernel,
            dim3(grid),
            dim3(block_dim),
            kernel_args,
            0,
            stream
        )
    );

    if (launched_kernel) *launched_kernel = kernel;
    return true;
}

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
)
{
    const at::cuda::OptionalCUDAGuard device_guard(A.device());
#if defined(USE_ROCM)
    hipStream_t stream = at::cuda::getCurrentCUDAStream().stream();
#else
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
#endif

    TORCH_CHECK_DIM(B, 3);
    TORCH_CHECK(A.dim() >= 1 && C.dim() >= 1, "exl3_gemv requires non-scalar A and C");
    TORCH_CHECK_SHAPES(A, -1, B, 0, 16);
    TORCH_CHECK_SHAPES(C, -1, B, 1, 16);
    TORCH_CHECK_DTYPE(A, kHalf);
    TORCH_CHECK_DTYPE(B, kShort);
    bool c_fp32 = C.dtype() == at::kFloat;
    if (!c_fp32) TORCH_CHECK_DTYPE(C, kHalf);
    TORCH_CHECK(!(mcg && mul1), "Specified both mcg and mul1")
    TORCH_CHECK(A.is_contiguous() && B.is_contiguous() && C.is_contiguous(),
                "exl3_gemv requires contiguous A, B and C");
    TORCH_CHECK(B.device() == A.device() && C.device() == A.device(),
                "exl3_gemv requires A, B and C on the same device");

    int size_m = 1;
    int dim = A.dim();
    for (int d = 0; d < dim - 1; ++d) size_m *= A.size(d);
    int size_k = A.size(-1);
    int size_n = B.size(1) * 16;
    int K = B.size(2) / 16;

    TORCH_CHECK(suh.has_value() && A_had.has_value() && svh.has_value(),
                "exl3_gemv requires suh, A_had and svh");
    const at::Tensor& suh_t = suh.value();
    const at::Tensor& A_had_t = A_had.value();
    const at::Tensor& svh_t = svh.value();
    TORCH_CHECK_DTYPE(suh_t, kHalf);
    TORCH_CHECK_DTYPE(A_had_t, kHalf);
    TORCH_CHECK_DTYPE(svh_t, kHalf);
    TORCH_CHECK(suh_t.device() == A.device() && A_had_t.device() == A.device() &&
                svh_t.device() == A.device(),
                "exl3_gemv requires workspaces on the A device");
    TORCH_CHECK(suh_t.is_contiguous() && A_had_t.is_contiguous() && svh_t.is_contiguous(),
                "exl3_gemv requires contiguous workspaces");
    TORCH_CHECK(suh_t.numel() == size_k, "exl3_gemv suh size mismatch");
    TORCH_CHECK(A_had_t.sizes() == A.sizes(), "exl3_gemv A_had shape mismatch");
    TORCH_CHECK(svh_t.numel() == size_n, "exl3_gemv svh size mismatch");

    const half* suh_ptr = (const half*) suh_t.data_ptr();
    half* A_had_ptr = (half*) A_had_t.data_ptr();
    const half* svh_ptr = (const half*) svh_t.data_ptr();

    int cb = 0;
    if (mcg) cb = 1;
    if (mul1) cb = 2;

    int device;
#if defined(USE_ROCM)
    hipGetDevice(&device);
#else
    cudaGetDevice(&device);
#endif
    int* locks = DevCtx::instance().get_locks(device);

#if defined(USE_ROCM)
    at::Tensor A_view = A.view({size_m, size_k});
    at::Tensor A_had_view = A_had.value().view({size_m, size_k});
    had_r_128(A_view, A_had_view, suh, c10::nullopt, 1.0f);
    const half* A_ptr = (const half*) A_had_view.data_ptr();
#else
    const half* A_ptr = (const half*) A.data_ptr();
#endif
    const uint16_t* B_ptr = (const uint16_t*) B.data_ptr();
    void* C_ptr = (void*) C.data_ptr();

    void* kernel_args[] =
    {
        (void*)& A_ptr,
        (void*)& B_ptr,
        (void*)& C_ptr,
        (void*)& size_m,
        (void*)& size_k,
        (void*)& size_n,
        (void*)& locks,
        (void*)& suh_ptr,
        (void*)& A_had_ptr,
        (void*)& svh_ptr
    };

    bool ok = exl3_gemv_try_launch
    (
        kernel_args, size_m, size_k, size_n, K, cb, c_fp32,
        true, device, stream, nullptr, true
    );
    TORCH_CHECK(ok, "exl3_gemv: call is not eligible for the GEMV kernel");

#if defined(USE_ROCM)
    at::Tensor C_view = C.view({size_m, size_n});
    had_r_128(C_view, C_view, c10::nullopt, svh, 1.0f);
    cuda_check(hipPeekAtLastError());
#else
    cuda_check(cudaPeekAtLastError());
#endif
}
