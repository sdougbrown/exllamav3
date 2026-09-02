#pragma once

#if defined(USE_ROCM)

#include <ATen/Tensor.h>

// Fixed QSA selection shape: fp16 (R, T), T >= 512, and contiguous int32 (R, 512) output.
void dsa_topk_gfx12(const at::Tensor& scores, at::Tensor& indices);

#endif
