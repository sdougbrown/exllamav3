#if defined(USE_ROCM)

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include "dsa_topk_gfx12.cuh"
#include "util.cuh"

#include <climits>
#include <cstdint>
#include <cstring>

namespace {

constexpr int WAVE_SIZE = 32;
constexpr int THREADS = 1024;
constexpr int K = 512;
constexpr int K_PAD = 512;
constexpr uint16_t KEY_NEG_INF = 0x03ff;

__device__ __forceinline__ uint16_t topk_key(uint16_t bits)
{
    return bits & 0x8000 ? static_cast<uint16_t>(~bits) : static_cast<uint16_t>(bits | 0x8000);
}

__device__ __forceinline__ void find_descending_bucket(const int* hist, int target, int* result)
{
    int total = 0;
    for (int bucket = 255; bucket >= 0; --bucket)
    {
        if (total + hist[bucket] >= target)
        {
            result[0] = bucket;
            result[1] = total;
            return;
        }
        total += hist[bucket];
    }
    result[0] = -1;
    result[1] = total;
}

__global__ __launch_bounds__(THREADS)
void dsa_topk_gfx12_kernel(const half* __restrict__ scores, int* __restrict__ out, int width, int stride)
{
    const int tid = threadIdx.x;
    const half* row_scores = scores + static_cast<size_t>(blockIdx.x) * stride;
    int* row_out = out + static_cast<size_t>(blockIdx.x) * K_PAD;

    __shared__ int hist[256];
    __shared__ int search[2];

    if (tid < 256) hist[tid] = 0;
    __syncthreads();
    for (int64_t index = tid; index < static_cast<int64_t>(width); index += THREADS)
    {
        const uint16_t key = topk_key(__half_as_ushort(row_scores[index]));
        if (key > KEY_NEG_INF) atomicAdd(&hist[key >> 8], 1);
    }
    __syncthreads();
    if (tid == 0) find_descending_bucket(hist, K, search);
    __syncthreads();

    const int high_bucket = search[0];
    const int count_above_high = search[1];
    uint16_t threshold = KEY_NEG_INF;
    int ties_needed = 0;
    if (high_bucket >= 0)
    {
        if (tid < 256) hist[tid] = 0;
        __syncthreads();
        for (int64_t index = tid; index < static_cast<int64_t>(width); index += THREADS)
        {
            const uint16_t key = topk_key(__half_as_ushort(row_scores[index]));
            if (key > KEY_NEG_INF && (key >> 8) == high_bucket)
                atomicAdd(&hist[key & 0xff], 1);
        }
        __syncthreads();
        if (tid == 0) find_descending_bucket(hist, K - count_above_high, search);
        __syncthreads();
        threshold = static_cast<uint16_t>((high_bucket << 8) | search[0]);
        ties_needed = K - count_above_high - search[1];
    }

    // The score histograms are fully parallel; this short ordered pass deliberately has one
    // owner so the public API's ascending-index tie and output order remain exact.
    if (tid == 0)
    {
        int emitted = 0;
        for (int index = 0; index < width; ++index)
        {
            const uint16_t key = topk_key(__half_as_ushort(row_scores[index]));
            if (key > threshold && key > KEY_NEG_INF)
                row_out[emitted++] = index;
            else if (key == threshold && key > KEY_NEG_INF && ties_needed)
            {
                row_out[emitted++] = index;
                --ties_needed;
            }
        }
        for (int index = emitted; index < K_PAD; ++index) row_out[index] = -1;
    }
}

bool is_gfx12_wave32(int device)
{
    hipDeviceProp_t prop;
    if (hipGetDeviceProperties(&prop, device) != hipSuccess) return false;
    const char* arch = prop.gcnArchName;
    const bool gfx12 = (!std::strncmp(arch, "gfx1200", 7) || !std::strncmp(arch, "gfx1201", 7)) &&
                       (arch[7] == '\0' || arch[7] == ':');
    return gfx12 && prop.warpSize == WAVE_SIZE;
}

} // namespace

void dsa_topk_gfx12(const at::Tensor& scores, at::Tensor& indices)
{
    TORCH_CHECK(scores.is_cuda() && indices.is_cuda(), "dsa_topk_gfx12 requires device tensors");
    const at::cuda::OptionalCUDAGuard device_guard(scores.device());
    const int device = scores.get_device();
    TORCH_CHECK(is_gfx12_wave32(device), "dsa_topk_gfx12 requires gfx1200/gfx1201 with wave32");
    TORCH_CHECK(scores.device() == indices.device(), "dsa_topk_gfx12 tensors must share a device");
    TORCH_CHECK(scores.dtype() == at::kHalf && indices.dtype() == at::kInt,
                "dsa_topk_gfx12 requires fp16 scores and int32 indices");
    TORCH_CHECK(scores.dim() == 2 && indices.dim() == 2 && scores.size(0) == indices.size(0),
                "dsa_topk_gfx12 requires matching 2D tensors");
    TORCH_CHECK(scores.size(1) <= INT_MAX && scores.stride(0) <= INT_MAX,
                "dsa_topk_gfx12 requires score width and row stride <= INT_MAX");
    TORCH_CHECK(scores.size(1) >= K && scores.stride(1) == 1 && scores.stride(0) >= scores.size(1) &&
                scores.stride(0) % 128 == 0,
                "dsa_topk_gfx12 requires QSA scores with T >= 512 and a 128-aligned row stride");
    TORCH_CHECK(indices.sizes() == at::IntArrayRef({scores.size(0), K_PAD}) && indices.is_contiguous(),
                "dsa_topk_gfx12 requires contiguous int32 [R, 512] output");
    if (scores.size(0) == 0) return;

    hipStream_t stream = at::cuda::getCurrentCUDAStream(device).stream();
    dsa_topk_gfx12_kernel<<<scores.size(0), THREADS, 0, stream>>>(
        reinterpret_cast<const half*>(scores.data_ptr()), reinterpret_cast<int*>(indices.data_ptr()),
        static_cast<int>(scores.size(1)), static_cast<int>(scores.stride(0)));
    cuda_check(hipPeekAtLastError());
}

#endif // USE_ROCM
