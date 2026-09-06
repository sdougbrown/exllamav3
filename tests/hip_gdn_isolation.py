"""Frozen real-call diagnostic for the recurrent GDN rule, outside model state."""
import argparse
import json
import os
from pathlib import Path

from hip_moe_isolation import digest, load_harness


def install_capture(out):
    import torch
    from exllamav3.ext import exllamav3_ext as ext
    original = ext.cuda_recurrent_gated_delta_rule
    path = out / 'gdn-frozen.pt'
    assert not path.exists(), path
    captured = False

    def wrapper(*args):
        nonlocal captured
        if captured or args[0].shape[1] != 512:
            return original(*args)
        captured = True
        frozen = [a.detach().cpu().clone() if isinstance(a, torch.Tensor) and i != 4 else
                  None if i == 4 else a for i, a in enumerate(args)]
        original(*args)
        torch.save({'args': frozen, 'output': args[4].cpu().clone(),
                    'final_state': args[3].cpu().clone()}, path)
        print('CAPTURED GDN', path, flush=True)

    ext.cuda_recurrent_gated_delta_rule = wrapper

    def restore():
        ext.cuda_recurrent_gated_delta_rule = original

    return restore


def capture(out):
    b = load_harness()
    from exllamav3.ext import exllamav3_ext as ext
    from hip_stable_router_control import install
    b.apply_gates(0, 0)
    config, model, cache, tokenizer, generator = b.load_model(b.DEFAULT_MODEL_DIR, 512)
    restore_router = install()
    restore_capture = install_capture(out)
    wrapped = ext.cuda_recurrent_gated_delta_rule

    class Captured(BaseException):
        pass

    def stop_after_capture(*args):
        wrapped(*args)
        if (out / 'gdn-frozen.pt').exists():
            raise Captured

    ext.cuda_recurrent_gated_delta_rule = stop_after_capture
    ids = b.prompt_for(b.build_corpus(), 12288)
    try:
        job = b._enqueue_job(generator, ids, 1, 'real_gdn_capture')
        b.run_prefill(generator, job, timed=False)
        raise AssertionError('GDN capture not reached')
    except Captured:
        (out / 'capture.json').write_text(json.dumps({
            'source': 'first 512-token GDN rule call after model load, real 12K corpus prompt',
            'prompt_sha256': b.canonical_sha256(ids), 'map': b.module_device_map(model),
            'stable_router_ties': True, 'memory': b.memory_snapshot()}, indent=2))
    finally:
        restore_capture()
        restore_router()


def replay(out, repeats, tokens=None):
    load_harness()
    import torch
    from exllamav3.ext import exllamav3_ext as ext
    from test_gated_delta_rule import _torch_gated_delta_rule
    torch.set_num_threads(8)
    frozen = torch.load(out / 'gdn-frozen.pt', weights_only=True)
    args = [a.cuda() if isinstance(a, torch.Tensor) else a for a in frozen['args']]
    if tokens:
        for i in (0, 1, 2):
            args[i] = args[i][:, :tokens].contiguous()
    initial = args[3].clone()
    args[4] = torch.empty((*args[0].shape[:2], args[6], args[8]), dtype=torch.bfloat16, device='cuda')
    before = {str(i): digest(a) for i, a in enumerate(args) if isinstance(a, torch.Tensor) and i != 4}
    records = []
    base = None
    for rep in range(repeats):
        args[3].copy_(initial)
        args[4].fill_([0., float('nan'), 123., -77.][rep % 4])
        assert before == {str(i): digest(a) for i, a in enumerate(args)
                          if isinstance(a, torch.Tensor) and i != 4}
        ext.cuda_recurrent_gated_delta_rule(*args)
        snap = {'output': args[4].cpu().clone(), 'state': args[3].cpu().clone()}
        if base is None:
            base = snap
            torch.save(base, out / 'gdn-base.pt')
        stats = {k: {'digest': digest(t), 'different': int((t != base[k]).sum()),
                     'max_abs': float((t.float() - base[k].float()).abs().max()),
                     'finite': bool(t.isfinite().all())} for k, t in snap.items()}
        rec = {'rep': rep, 'stats': stats}
        records.append(rec)
        print(json.dumps(rec), flush=True)
        (out / 'gdn-replay.json').write_text(json.dumps({'input_digests': before, 'records': records}, indent=2))
        if any(s['different'] for s in stats.values()) and not (out / 'gdn-different.pt').exists():
            torch.save(snap, out / 'gdn-different.pt')
    args[3].copy_(initial)
    assert before == {str(i): digest(a) for i, a in enumerate(args)
                      if isinstance(a, torch.Tensor) and i != 4}
    cpu = [a.cpu() if isinstance(a, torch.Tensor) else a for a in args]
    ref_out, ref_state = _torch_gated_delta_rule(cpu[0], cpu[1], cpu[2], cpu[3], cpu[9], cpu[10], *cpu[5:9])
    reference = {}
    for name, actual, expected in [('output', base['output'], ref_out), ('state', base['state'], ref_state)]:
        delta = (actual.float() - expected.float()).abs()
        reference[name] = {'max_abs': float(delta.max()), 'rms': float(delta.square().mean().sqrt()),
                           'reference': 'tests/test_gated_delta_rule.py::_torch_gated_delta_rule',
                           'rtol': .05, 'atol': .05}
        torch.testing.assert_close(actual, expected, rtol=.05, atol=.05)
    (out / 'gdn-reference.json').write_text(json.dumps(reference, indent=2))
    if os.environ.get('EXL3_DIAGNOSTIC_GDN_FIXED_ORDER') == '1':
        assert all(s['different'] == 0 and s['finite'] for r in records for s in r['stats'].values())
        assert all(len({r['stats'][name]['digest'] for r in records}) == 1 for name in base)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--capture', action='store_true')
    p.add_argument('--repeats', type=int, default=12)
    p.add_argument('--tokens', type=int)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    assert not (args.out / 'gdn-replay.json').exists()
    if args.capture:
        capture(args.out)
    else:
        replay(args.out, args.repeats, args.tokens)
