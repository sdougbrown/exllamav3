"""Trace a one-token rule with a frozen, real nonzero prefix state."""
import argparse
import json
import os
from pathlib import Path

from hip_moe_isolation import digest, load_harness


def run(out, frozen_path, repeats, inputs_path=None):
    load_harness()
    import torch
    from exllamav3.ext import exllamav3_ext as ext
    torch.set_num_threads(8)
    frozen = torch.load(frozen_path, weights_only=True)
    args = [a.cuda() if isinstance(a, torch.Tensor) else a for a in frozen['args']]
    slot = int(args[9][0]) if args[9] is not None else 0
    state = args[3][slot:slot+1, :1].clone()
    # The prefix is evaluated once; all traced calls clone this one owned result.
    prefix = 256
    shape = (1, prefix, args[6], args[8])
    temp = torch.empty(shape, dtype=torch.bfloat16, device='cuda')
    if inputs_path is None:
        ext.cuda_recurrent_gated_delta_rule(*[a[:, :prefix].contiguous() for a in args[:3]],
                                           state, temp, *args[5:9], None, False)
        qkv, g, beta = [a[:, prefix:prefix+1].contiguous() for a in args[:3]]
        initial = torch.cat([state, torch.zeros_like(state)], 0)
    else:
        saved = torch.load(inputs_path, weights_only=True)
        qkv, g, beta, initial = [saved[k].cuda() for k in ('qkv', 'g', 'beta', 'initial')]
    torch.save({'qkv': qkv.cpu(), 'g': g.cpu(), 'beta': beta.cpu(), 'initial': initial.cpu()}, out / 'trace-inputs.pt')
    before = [digest(t) for t in (qkv, g, beta, initial)]
    output = torch.empty((1, 1, args[6], args[8]), dtype=torch.bfloat16, device='cuda')
    os.environ['EXL3_DIAGNOSTIC_GDN_TRACE'] = '1'
    base = None
    records = []
    for rep in range(repeats):
        current = initial.clone()
        current[1].fill_(float('nan'))
        ext.cuda_recurrent_gated_delta_rule(qkv, g, beta, current, output, *args[5:9], None, False)
        trace = current[1].flatten()[:args[6] * 16 * 128].view(args[6], 16, 128).cpu().clone()
        assert torch.isfinite(trace[:, :14]).all(), 'trace planes were not written'
        snap = {'q': trace[:, 0], 'k': trace[:, 1], 'dot1_partials': trace[:, 2:6],
                'dot1': trace[:, 6], 'decay': trace[:, 7], 'dot2_partials': trace[:, 8:12],
                'dot2': trace[:, 12], 'beta': trace[:, 13],
                'state': current[0].cpu().clone(), 'output': output.cpu().clone()}
        if base is None:
            base = snap
            torch.save(base, out / 'trace-base.pt')
        record = {k: {'different': int((t != base[k]).sum()),
                      'max_abs': float((t.float() - base[k].float()).abs().max()),
                      'digest': digest(t)} for k,t in snap.items()}
        records.append(record)
        if any(s['different'] for s in record.values()) and not (out / 'trace-different.pt').exists():
            torch.save(snap, out / 'trace-different.pt')
        (out / 'trace.json').write_text(json.dumps({'input_digests': before, 'prefix_tokens': prefix,
                                                   'records': records}, indent=2))
        print(json.dumps(record), flush=True)
    assert before == [digest(t) for t in (qkv, g, beta, initial)]
    if os.environ.get('EXL3_DIAGNOSTIC_GDN_FIXED_ORDER') == '1':
        assert all(s['different'] == 0 for r in records for s in r.values())
        assert all(len({r[name]['digest'] for r in records}) == 1 for name in base)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--frozen', required=True, type=Path)
    p.add_argument('--repeats', type=int, default=50)
    p.add_argument('--inputs', type=Path)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    assert not (args.out / 'trace.json').exists()
    run(args.out, args.frozen, args.repeats, args.inputs)
