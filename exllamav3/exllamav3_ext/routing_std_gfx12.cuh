#pragma once

#if defined(USE_ROCM)

#include <ATen/Tensor.h>

// Decode-only standard MoE router for gfx1200/gfx1201. The gate is supplied as a
// persistent contiguous (experts, hidden) transpose so the bsz-1 GEMV is coalesced.
void routing_std_gfx12_bsz1
(
    const at::Tensor& hidden,
    const at::Tensor& gate_t,
    at::Tensor& scores,
    at::Tensor& topk_indices,
    at::Tensor& topk_weights
);

#endif
