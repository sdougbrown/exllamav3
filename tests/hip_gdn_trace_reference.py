"""CPU FP64 exposed-stage checks and exact FP32 atomic-order enumeration."""
import argparse
import itertools
import json
from pathlib import Path

import torch


def gamma(n):
    return n * 2. ** -24 / (1. - n * 2. ** -24)


def bounded(actual, reference, bound):
    error = (actual.double() - reference).abs()
    assert bool((error <= bound).all()), (float(error.max()), float((error / bound.clamp_min(1e-300)).max()))
    return {'max_abs': float(error.max()), 'max_error_over_bound': float((error / bound.clamp_min(1e-300)).max())}


def check(snap, inputs):
    kh, vh = inputs['qkv'].shape[-1] // 128, snap['k'].shape[0]
    kh = (kh - vh) // 2
    initial = inputs['initial'][0, 0].double()
    k = snap['k'].double()
    q = snap['q'].double()
    updated = snap['state'][0].double()
    report = {}
    for name, vector, state in [('dot1', k, initial), ('dot2', q, updated)]:
        products = vector.unsqueeze(-1) * state
        parts = products.view(vh, 4, 32, 128)
        report[name + '_partials'] = bounded(snap[name + '_partials'], parts.sum(-2),
                                               gamma(64) * parts.abs().sum(-2) + 64 * 2. ** -149)
        partials = snap[name + '_partials'].float()
        matched = torch.zeros_like(snap[name], dtype=torch.bool)
        for order in itertools.permutations(range(4)):
            value = torch.zeros_like(snap[name])
            for i in order:
                value = value + partials[:, i]
            matched |= value == snap[name]
        assert bool(matched.all()), name
        report[name + '_matches_fp32_partial_permutation'] = True
        report[name + '_reduced'] = bounded(snap[name], products.sum(-2),
                                             gamma(132) * products.abs().sum(-2) + 132 * 2. ** -149)
    v = inputs['qkv'][0, 0, 2 * kh * 128:].double().view(vh, 128)
    decay = snap['decay'].double()
    beta = snap['beta'].double()
    dot1 = snap['dot1'].double()
    update = k.unsqueeze(-1) * ((v - dot1 * decay) * beta).unsqueeze(-2)
    reference = initial * decay.unsqueeze(-2) + update
    magnitude = (initial * decay.unsqueeze(-2)).abs() + k.abs().unsqueeze(-1) * (
        (v.abs() + (dot1 * decay).abs()) * beta.abs()).unsqueeze(-2)
    report['state_update_given_exposed_decay_and_dot1'] = bounded(updated, reference,
                                                                 gamma(8) * magnitude + 8 * 2. ** -149)
    output_ref = (q.unsqueeze(-1) * updated).sum(-2) * (128 ** -.5)
    arithmetic = gamma(134) * (q.unsqueeze(-1) * updated).abs().sum(-2) * (128 ** -.5)
    report['bf16_output'] = bounded(snap['output'][0, 0], output_ref,
                                    arithmetic + 2. ** -7 * (output_ref.abs() + arithmetic) + 2. ** -133)
    return report


def main(out, control=None):
    torch.set_num_threads(8)
    inputs = torch.load(out / 'trace-inputs.pt', weights_only=True)
    assert inputs['initial'][0].count_nonzero() > 0
    assert all(t.isfinite().all() for t in inputs.values())
    base = torch.load(out / 'trace-base.pt', weights_only=True)
    other = torch.load(out / 'trace-different.pt', weights_only=True)
    fixed_inputs = all(torch.equal(base[k], other[k]) for k in ('q', 'k', 'decay', 'beta', 'dot1_partials'))
    dot1_changed = not torch.equal(base['dot1'], other['dot1'])
    dot2_fixed_partials = torch.equal(base['dot2_partials'], other['dot2_partials'])
    dot2_changed = not torch.equal(base['dot2'], other['dot2'])
    checks = {'base': check(base, inputs), 'other': check(other, inputs)}
    report = {'fixed_q_k_decay_beta_dot1_partials': fixed_inputs, 'dot1_changed': dot1_changed,
              'dot2_fixed_partials': dot2_fixed_partials, 'dot2_changed': dot2_changed,
              'atomic_order_confirmed': fixed_inputs and (dot1_changed or (dot2_fixed_partials and dot2_changed)),
              'checks': checks,
              'limits': 'Exposed-stage bounds use captured normalized q/k and exp(g); they do not bound those intrinsics or full-model logits.'}
    if control is not None:
        controlled_inputs = torch.load(control / 'trace-inputs.pt', weights_only=True)
        assert all(torch.equal(v, controlled_inputs[k]) for k,v in inputs.items())
        controlled = torch.load(control / 'trace-base.pt', weights_only=True)
        for name in ('dot1', 'dot2'):
            canonical = torch.zeros_like(controlled[name])
            for part in range(4):
                canonical = canonical + controlled[name + '_partials'][:, part]
            assert torch.equal(canonical, controlled[name])
        report['fixed_order_is_exact_fp32_0_1_2_3'] = True
        report['fixed_order_reference'] = check(controlled, inputs)
        report['fixed_order_vs_atomic'] = {k: {'different': int((v != controlled[k]).sum()),
            'max_abs': float((v.float() - controlled[k].float()).abs().max())} for k,v in base.items()}
    ((control or out) / 'trace-reference.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    assert report['atomic_order_confirmed']


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--control', type=Path)
    args = p.parse_args()
    main(args.out, args.control)
