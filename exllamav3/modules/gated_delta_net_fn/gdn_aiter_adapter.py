"""Optional AITER chunked prefill backend for headwise 128x128 GDN (Stage 3).

EXL3's default prefill route for the headwise gated delta rule is the
sequence-serial HIP recurrence (`cuda_recurrent_gated_delta_rule_kernel_128`),
which loops over prompt tokens. When `fla` is absent this route is selected for
every prefill. This adapter offers a chunked alternative through the installed
AITER package's `chunk_gated_delta_rule_opt_vk`, opt-in via
`EXL3_GDN_AITER_PREFILL=1`; the original route remains the default.

Contract:
  * The persistent EXL3 state pool keeps its native [slots, history, H, K, V]
    FP32 layout. Only the states of the slots served by this call are
    transposed to AITER's [N, H, V, K] contract at the adapter boundary, and
    the final state is transposed back on write-back. Decode layout, the whole
    pool, history/rewind, and MTP are untouched.
  * EXL3's g is natural-log decay (gdn.hip: -softplus * exp(a_log), consumed as
    exp(g)), so the adapter pins `use_exp2=False`. In-kernel q/k L2
    normalization matches the HIP kernel (eps 1e-6): use_qk_l2norm_in_kernel=True.
  * The call signature is pinned to installed AITER 0.1.19
    (`chunk_gated_delta_rule_opt_vk`); newer-API parameters
    (`prefill_metadata`, `snapshot_dtype`) seen in local AITER sources are
    deliberately NOT passed.
  * Grouped heads (num_v_heads = num_k_heads * r, e.g. 48/16 or TP shards
    24/8) are required; anything else stays on the current route.
  * Conversion, gather/scatter and metadata costs happen inside this adapter so
    benchmark timing includes them.
"""
from __future__ import annotations

import os

import torch

AITER_MIN_HEAD_DIM = 128


def optin_enabled() -> bool:
    return os.environ.get("EXL3_GDN_AITER_PREFILL", "0") == "1"


def load_aiter_vk():
    """Lazy import of the installed AITER VK prefill entry. An explicit opt-in
    with a missing or incompatible AITER must fail clearly, before any state
    mutation."""
    try:
        from aiter.ops.triton.gated_delta_net import chunk_gated_delta_rule_opt_vk
    except (ModuleNotFoundError, ImportError) as e:
        raise RuntimeError(
            "EXL3_GDN_AITER_PREFILL=1 requested the AITER GDN prefill backend, "
            f"but the installed aiter package does not provide "
            f"chunk_gated_delta_rule_opt_vk: {e}"
        ) from e
    return chunk_gated_delta_rule_opt_vk


def aiter_eligible(
    seqlen: int,
    history: bool,
    channelwise_g: bool,
    k_head_dim: int,
    v_head_dim: int,
    num_v_heads: int,
    num_k_heads: int,
) -> bool:
    """True only for headwise (not KDA), history-free, 128x128 grouped-head
    prefill shapes. Everything else keeps the current route."""
    if history or channelwise_g:
        return False
    if k_head_dim != 128 or v_head_dim != 128:
        return False
    if num_k_heads <= 0 or num_v_heads % num_k_heads != 0:
        return False
    return True


def gather_states_vk(
    recurrent_state: torch.Tensor | None,
    recurrent_slots: torch.Tensor | None,
) -> torch.Tensor | None:
    """Select this call's slots from the native pool and transpose the per-head
    state to AITER's [N, H, V, K] layout. The pool is read, never written."""
    if recurrent_state is None:
        return None
    slots = recurrent_slots.to(torch.long)
    states = recurrent_state.index_select(0, slots)[:, 0]  # [N, H, K, V]
    return states.transpose(-1, -2).contiguous()           # [N, H, V, K]


def scatter_states_vk(
    final_state_vk: torch.Tensor | None,
    recurrent_state: torch.Tensor,
    recurrent_slots: torch.Tensor,
) -> None:
    """Transpose AITER's [N, H, V, K] final state back into the native pool at
    the served slots. Untouched slots are never modified."""
    if final_state_vk is None or recurrent_state is None:
        return
    slots = recurrent_slots.to(torch.long)
    states_kv = final_state_vk.transpose(-1, -2).to(recurrent_state.dtype)
    recurrent_state[slots, 0] = states_kv


def aiter_chunked_gdn_prefill(
    mixed_qkv: torch.Tensor,
    beta: torch.Tensor,
    g: torch.Tensor,
    recurrent_state: torch.Tensor,
    recurrent_slots: torch.Tensor,
    save_state: bool,
    num_k_heads: int,
    num_v_heads: int,
    k_head_dim: int,
    v_head_dim: int,
    chunk_gated_delta_rule_opt_vk=None,
) -> torch.Tensor:
    """Chunked AITER prefill including all state conversion costs. Returns
    core_attn_out [b, s, num_v_heads, v_head_dim] bf16, matching the current
    route's contract."""
    if chunk_gated_delta_rule_opt_vk is None:
        chunk_gated_delta_rule_opt_vk = load_aiter_vk()

    bsz, seqlen, fdim = mixed_qkv.shape
    k_dim = num_k_heads * k_head_dim
    v_dim = num_v_heads * v_head_dim
    q, k, v = torch.split(mixed_qkv, [k_dim, k_dim, v_dim], dim=-1)
    # contiguous copies: AITER's l2norm/views require them (mixed_qkv is strided)
    q = q.reshape(bsz, seqlen, num_k_heads, k_head_dim).contiguous()
    k = k.reshape(bsz, seqlen, num_k_heads, k_head_dim).contiguous()
    v = v.reshape(bsz, seqlen, num_v_heads, v_head_dim).contiguous()

    initial_state = gather_states_vk(recurrent_state, recurrent_slots)

    core_attn_out = torch.empty(
        (bsz, seqlen, num_v_heads, v_head_dim),
        dtype=torch.bfloat16,
        device=mixed_qkv.device,
    )
    o, final_state = chunk_gated_delta_rule_opt_vk(
        q=q,
        k=k,
        v=v,
        o=core_attn_out,
        g=g,
        beta=beta,
        scale=None,                      # default: k_head_dim ** -0.5
        initial_state=initial_state,
        output_final_state=save_state,
        use_qk_l2norm_in_kernel=True,    # matches the HIP kernel's in-kernel l2norm
        cu_seqlens=None,
        use_chunk_hip=False,             # installed AITER forces Triton on gfx1201 anyway
        state_dtype=torch.float32,
        use_exp2=False,                  # EXL3 g is natural-log decay
    )
    if save_state and recurrent_state is not None:
        scatter_states_vk(final_state, recurrent_state, recurrent_slots)
    return core_attn_out