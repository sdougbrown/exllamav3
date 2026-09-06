"""Opt-in diagnostic tie policy; never imported by production routing.

Expert columns are already in ascending ID order. Stable descending sort
therefore resolves only exact ties by lower ID, without score perturbations.
The native small-row router and its matmul precision are deliberately untouched.
"""
import importlib

import torch


def stable_topk(scores, k):
    ids = torch.argsort(scores, dim=-1, descending=True, stable=True)[..., :k].contiguous()
    return scores.gather(-1, ids).contiguous(), ids


def routing_std_stable(cfg, y, params, include_bias=False):
    scores = torch.matmul(y, cfg.gate_tensor).float()
    if include_bias and cfg.router_bias is not None:
        scores = scores + cfg.router_bias.float()
    if params.get('activate_all_experts'):
        selected = torch.arange(cfg.num_experts, dtype=torch.long, device=y.device).expand(y.shape[0], -1)
        weights = torch.softmax(scores, dim=-1)
    else:
        values, selected = stable_topk(scores, cfg.num_experts_per_tok)
        weights = torch.softmax(values, dim=-1)
    if cfg.per_expert_scale is not None:
        weights = weights * cfg.per_expert_scale.float()[selected]
    return selected, weights.half()


def install():
    module = importlib.import_module('exllamav3.modules.block_sparse_mlp')
    original = module._routing_std_torch
    module._routing_std_torch = routing_std_stable

    def restore():
        module._routing_std_torch = original

    return restore
