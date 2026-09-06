"""Manual gfx12 MoE isolation using the local validated-prefill harness.

Capture saves one real call and its quantized experts. Replay bypasses generator
state; module and router modes distinguish expert arithmetic from top-k ties.
EXL3_VALIDATED_PREFILL_HARNESS selects the launcher-matching setup helper;
EXL3_FLASH_TEST_MODEL selects the checkpoint for capture/module/router modes.
Use a fresh --out directory (hard-link frozen.pt there for additional replays).
The exposed-stage bounds do not qualify the intervening GEMVs or full-model logits.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path


def digest(t):
    import torch
    return hashlib.sha256(t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def load_harness():
    path = Path(os.environ.get('EXL3_VALIDATED_PREFILL_HARNESS',
        str(Path.home() / 'Serve/hosts/rocky/bench-prefill-validated.py'))).expanduser()
    spec = importlib.util.spec_from_file_location('validated', path)
    b = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(b)
    return b


def capture(out):
    b = load_harness()
    import torch
    from exllamav3.ext import exllamav3_ext as ext
    b.apply_gates(0, 0)
    config, model, cache, tokenizer, generator = b.load_model(os.environ['EXL3_FLASH_TEST_MODEL'], 512)
    block = next(m for m, instance, idx in model.fwd_modules if idx == 2)
    mlp = block.mlp
    original = ext.exl3_moe_gfx12_k3_prefill

    # Bypass the generator's per-job Exception handler and terminate the drive loop.
    class Captured(BaseException):
        pass

    def intercept(*args):
        assert args[0].shape == (512, 2560)
        # Capture owned values, not pointer-table addresses or mutable module scratch.
        names = ('x', 'output', 'selected', 'weights', 'order', 'counts')
        frozen = {name: t.detach().cpu().clone() for name, t in zip(names, args[:6]) if name != 'output'}
        frozen['projections'] = []
        for linears in (mlp.gates, mlp.ups, mlp.downs):
            parts = [[getattr(l.inner, attr).detach().cpu().clone() for l in linears]
                     for attr in ('trellis', 'suh', 'svh')]
            frozen['projections'].append(parts)
        original(*args)
        frozen['output'] = args[1].detach().cpu().clone()
        torch.save(frozen, out / 'frozen.pt')
        manifest = {'env': {k: os.environ.get(k) for k in b._LAUNCHER_ENV},
                    'map': b.module_device_map(model), 'torch': torch.__version__, 'hip': torch.version.hip,
                    'devices': [str(torch.cuda.get_device_properties(d)) for d in range(torch.cuda.device_count())],
                    'inputs': {k: digest(v) for k, v in frozen.items() if isinstance(v, torch.Tensor)},
                    'weights': [[[digest(t) for t in part] for part in proj] for proj in frozen['projections']],
                    'memory': b.memory_snapshot(), 'kfd_after_capture': b.kfd_evicted_ms(os.getpid())}
        (out / 'capture.json').write_text(json.dumps(manifest, indent=2))
        print('CAPTURED', out / 'frozen.pt', flush=True)
        raise Captured

    ext.exl3_moe_gfx12_k3_prefill = intercept
    try:
        job = b._enqueue_job(generator, b.prompt_for(b.build_corpus(), 12288), 1, 'moe_capture')
        b.run_prefill(generator, job, timed=False)
        raise AssertionError('grouped MoE not reached')
    except Captured:
        pass
    finally:
        ext.exl3_moe_gfx12_k3_prefill = original


def input_hadamard_reference(x, scales, precise=False):
    """CPU reference: fp16 prescale, radix-2 butterfly, rounded fp32 constant."""
    import torch
    dtype = torch.float64 if precise else torch.float32
    v = (x.float() * scales.float()).half().to(dtype).reshape(-1, 128)
    for stride in (1, 2, 4, 8, 16, 32, 64):
        pairs = v.reshape(-1, 2, stride)
        lo, hi = pairs[:, 0].clone(), pairs[:, 1].clone()
        pairs[:, 0], pairs[:, 1] = lo + hi, lo - hi
    result = v * float(torch.tensor(0.088388347648, dtype=torch.float32))
    return result.reshape(x.shape) if precise else result.half().reshape(x.shape)


def check_exposed_stages(inputs, projections, buffers):
    """Check metadata, input transform and sorted reduction without grouped kernels."""
    import torch
    selected, order, counts = (inputs[k].cpu() for k in ('selected', 'order', 'counts'))
    experts = len(projections[0][0])
    expected_offsets = torch.cat((torch.zeros(1,dtype=torch.long), counts[:experts].cumsum(0)))
    assert torch.equal(buffers['offsets'], expected_offsets)
    assert torch.equal(buffers['inverse'], order.argsort())
    chunks = [expert * 320 + chunk for expert in range(experts)
              for chunk in range((int(counts[expert]) + 15) // 16)]
    assert torch.equal(buffers['chunks'], torch.tensor(chunks, dtype=torch.int))
    assert int(buffers['chunk_count']) == len(chunks)
    sorted_experts = selected.flatten()[order]
    sorted_x = inputs['x'].cpu()[order // 10]
    expected_had = []
    had_bounds = []
    for projection in projections[:2]:
        scales = torch.stack([t.cpu() for t in projection[1]])[sorted_experts.clamp(max=experts-1)]
        had = input_hadamard_reference(sorted_x, scales, precise=True)
        # -Ofast permits reassociation. Bound any sum tree of 128 prescaled half
        # values plus the normalization multiply, then one final fp16 rounding.
        prescaled = (sorted_x.float() * scales.float()).half().double().reshape(-1,128)
        gamma = (128 * 2**-24) / (1 - 128 * 2**-24)
        fp32_bound = prescaled.abs().sum(-1,keepdim=True) * gamma * float(torch.tensor(0.088388347648))
        fp32_bound = fp32_bound.expand(-1,128).reshape(had.shape)
        half_bound = (had.abs() + fp32_bound) * 2**-11 + 2**-25
        bound = fp32_bound + half_bound
        had[sorted_experts == experts] = 0
        bound[sorted_experts == experts] = 0
        expected_had.append(had)
        had_bounds.append(bound)
    expected_had = torch.cat(expected_had)
    had_bounds = torch.cat(had_bounds)
    error = (buffers['gu_had'].double() - expected_had).abs()
    assert (error <= had_bounds).all(), f'Hadamard bound exceeded at {int((error > had_bounds).sum())} coordinates'
    had_stats = {'max_abs_vs_fp64': float(error.max()),
                 'max_error_over_bound': float((error / had_bounds.clamp_min(1e-300)).max()),
                 'different_from_fp64_rounded_half': int((buffers['gu_had'] != expected_had.half()).sum())}
    rows = selected.shape[0]
    weights = inputs['weights'].cpu().float()
    assignments = buffers['down_out'][buffers['inverse']].reshape(rows, 10, -1)
    result = torch.zeros_like(buffers['output'], dtype=torch.float64)
    magnitude = torch.zeros_like(result)
    slot_order = selected.argsort(dim=1, stable=True)
    for rank in range(10):
        slots = slot_order[:, rank]
        term = assignments[torch.arange(rows), slots].double() * weights[torch.arange(rows), slots,None].double()
        result += term
        magnitude += term.abs()
    # Ten fp32 products and ten fp32 additions, allowing reassociation/contraction
    # under -Ofast. The fp64 reference products and their sum have negligible error.
    gamma = (20 * 2**-24) / (1 - 20 * 2**-24)
    reduce_bound = gamma * magnitude + 20 * 2**-149
    reduce_error = (buffers['output'].double() - result).abs()
    assert (reduce_error <= reduce_bound).all()
    return {'metadata_exact': True, 'input_hadamard_bound_passed': True, 'input_hadamard': had_stats,
            'weighted_reduction_bound_passed': True,
            'weighted_reduction': {'max_abs_vs_fp64': float(reduce_error.max()),
                                   'max_error_over_bound': float((reduce_error/reduce_bound).max())},
            'limit': 'gate/up, activation and down arithmetic are not independently qualified by these checks'}


def replay(out, repeats, rows=None, sentinel=False):
    load_harness()
    import torch
    from exllamav3.ext import exllamav3_ext as ext
    torch.set_num_threads(8)
    frozen = torch.load(out / 'frozen.pt', weights_only=True)
    projections = [[[t.cuda() for t in part] for part in proj] for proj in frozen['projections']]
    ptrs = [torch.tensor([t.data_ptr() for t in part], dtype=torch.long, device='cuda') for proj in projections for part in proj]
    inputs = {k: frozen[k].cuda() for k in ('x', 'selected', 'weights', 'order', 'counts')}
    if rows is not None or sentinel:
        for k in ('x','selected','weights'):
            inputs[k] = inputs[k][:rows].clone()
        if sentinel:
            inputs['selected'][-1] = len(projections[0][0])
        inputs['order'] = inputs['selected'].flatten().argsort(stable=True)
        inputs['counts'] = torch.bincount(inputs['selected'].flatten(), minlength=len(projections[0][0])+1)
    modified_inputs = rows is not None or sentinel
    before = {k: digest(t) for k, t in inputs.items()}
    weights_before = [[ [digest(t) for t in part] for part in proj] for proj in projections]
    rows, hidden = inputs['x'].shape
    assignments = rows * 10
    experts = len(projections[0][0])
    intermediate = projections[0][2][0].numel()
    specs = {'output': ((rows, hidden), torch.float32),
             'gu_had': ((2 * assignments, hidden), torch.float16),
             'gu_out': ((2 * assignments, intermediate), torch.float16),
             'down_out': ((assignments, hidden), torch.float32),
             'offsets': ((experts + 1,), torch.long), 'inverse': ((assignments,), torch.long),
             'chunks': ((experts * 320,), torch.int), 'chunk_count': ((1,), torch.int)}
    buffers = {k: torch.empty(shape, dtype=dtype, device='cuda') for k, (shape, dtype) in specs.items()}
    base = None
    results = []
    for rep in range(repeats):
        poison = (0., float('nan'), 123., -77.)[rep % 4]
        for k, t in buffers.items():
            t.fill_(poison if t.is_floating_point() else -123)
        ext.exl3_moe_gfx12_k3_prefill(inputs['x'], buffers['output'], inputs['selected'], inputs['weights'],
            inputs['order'], inputs['counts'], *ptrs, *(buffers[k] for k in ('gu_had', 'gu_out', 'down_out', 'offsets', 'inverse', 'chunks', 'chunk_count')))
        torch.cuda.synchronize()
        snap = {k: t.cpu().clone() for k,t in buffers.items()}
        snap['chunks'] = snap['chunks'][:int(snap['chunk_count'])]
        if base is None:
            base = snap
            torch.save(base, out / 'replay-base.pt')
        stats = {}
        for k, t in snap.items():
            delta = (t.float() - base[k].float()).abs()
            stats[k] = {'digest': digest(t), 'different': int((t != base[k]).sum()),
                        'max_abs': float(delta.max()) if t.numel() else 0., 'finite': bool(t.isfinite().all())}
        result = {'rep': rep, 'poison': str(poison), 'stats': stats}
        results.append(result)
        print(json.dumps(result), flush=True)
        (out / 'replay.json').write_text(json.dumps(results, indent=2))
        if any(s['different'] for s in stats.values()) and not (out / 'replay-different.pt').exists():
            torch.save(snap, out / 'replay-different.pt')
    assert before == {k: digest(t) for k,t in inputs.items()}
    assert weights_before == [[[digest(t) for t in part] for part in proj] for proj in projections]
    (out / 'inputs-unchanged.json').write_text(json.dumps(before, indent=2))
    assert all(s['different'] == 0 and s['finite'] for r in results for s in r['stats'].values())
    if not modified_inputs:
        assert torch.equal(base['output'], frozen['output'])
    reference = check_exposed_stages(inputs, projections, base)
    (out/'exposed-stage-reference.json').write_text(json.dumps(reference,indent=2))


def module_replay(out, repeats, force_routing=False, stable_ties=False):
    load_harness()
    import torch
    from exllamav3 import Config, Model
    from exllamav3.ext import exllamav3_ext as ext
    torch.set_num_threads(8)
    frozen = torch.load(out / 'frozen.pt', weights_only=True)
    model = Model.from_config(Config.from_directory(os.environ['EXL3_FLASH_TEST_MODEL']))
    mlp = model.find_module('model.language_model.layers.0.mlp')
    mlp.load(device=torch.device('cuda:0'))
    assert not mlp.router_pre_norm and not mlp.routed_pre_norm and not mlp.alt_residual_channel
    x = frozen['x'].cuda().view(1, 512, 2560)
    restore_router = None
    if stable_ties:
        from hip_stable_router_control import install
        restore_router = install()
    if force_routing:
        selected = frozen['selected'].cuda()
        weights = frozen['weights'].cuda()
        mlp.routing_fn = lambda *a, **kw: (selected, weights)
    label = 'module-forced' if force_routing else 'module-replay'
    base = {}
    current = {}
    def record(name, t):
        value = t.detach().cpu().clone()
        base.setdefault(name, value)
        current[name] = {'digest': digest(value), 'different': int((value != base[name]).sum()),
                         'max_abs': float((value.float() - base[name].float()).abs().max()),
                         'finite': bool(value.isfinite().all())}
    original = ext.exl3_moe_gfx12_k3_prefill
    def grouped(*args):
        for name,t in zip(('x','output_unused','selected','weights','order','counts'), args[:6]):
            if name != 'output_unused':
                record(name,t)
        original(*args)
        record('routed', args[1])
    ext.exl3_moe_gfx12_k3_prefill = grouped
    wrappers = []
    for name, module in [('shared',mlp.shared_experts), ('shared_gate',mlp.shared_gate)]:
        original_forward = module.forward
        def wrapped(*a, _orig=original_forward, _name=name, **kw):
            y = _orig(*a, **kw)
            record(_name,y)
            return y
        module.forward = wrapped
        wrappers.append((module,original_forward))
    results = []
    try:
        for rep in range(repeats):
            current = {}
            y = mlp.forward(x, {})
            record('output', y)
            results.append({'rep':rep, 'stats':current})
            print(json.dumps(results[-1]),flush=True)
            (out/f'{label}.json').write_text(json.dumps(results,indent=2))
        torch.save(base,out/f'{label}-base.pt')
        if force_routing or stable_ties:
            assert all(s['different'] == 0 and s['finite'] for r in results for s in r['stats'].values())
            assert all(len({r['stats'][name]['digest'] for r in results}) == 1 for name in base)
    finally:
        if restore_router is not None:
            restore_router()
        ext.exl3_moe_gfx12_k3_prefill = original
        for m,f in wrappers:
            m.forward = f
        mlp.unload()


def router_replay(out, repeats, stable_ties=False):
    load_harness()
    import torch
    from exllamav3 import Config, Model
    from exllamav3.ext import exllamav3_ext as ext
    from exllamav3.modules.block_sparse_mlp import _hip_router_call_supported
    torch.set_num_threads(8)
    frozen = torch.load(out / 'frozen.pt', weights_only=True)
    model = Model.from_config(Config.from_directory(os.environ['EXL3_FLASH_TEST_MODEL']))
    mlp = model.find_module('model.language_model.layers.0.mlp')
    mlp.load(device=torch.device('cuda:0'))
    x = frozen['x'].cuda()
    gate = mlp.routing_cfg.gate_tensor
    scores = torch.matmul(x, gate).float()
    scores_cpu = scores.cpu().clone()
    torch.save({'scores': scores_cpu, 'gate': gate.cpu().clone()}, out/'router-frozen.pt')
    expected_values = scores_cpu.sort(dim=-1, descending=True).values[:, :10]
    records = []
    base = None
    all_ids = []
    for rep in range(repeats):
        repeated_scores = torch.matmul(x, gate).float()
        assert torch.equal(scores, repeated_scores)
        old_values, old_indices = torch.topk(scores, 10, dim=-1)
        if stable_ties:
            from hip_stable_router_control import stable_topk
            values, indices = stable_topk(scores, 10)
            assert values.is_contiguous()
            assert torch.equal(values, old_values)
            assert torch.equal(torch.softmax(values, -1).half(), torch.softmax(old_values, -1).half())
            unique = (scores.unsqueeze(1) == values.unsqueeze(-1)).sum(-1) == 1
            assert torch.equal(indices[unique], old_indices[unique])
            oracle = torch.tensor([sorted(range(row.numel()), key=lambda e: (-float(row[e]), e))[:10]
                                   for row in scores_cpu])
            assert torch.equal(indices.cpu(), oracle)
        else:
            values, indices = old_values, old_indices
        cpu_ids = indices.cpu().clone()
        cpu_values = values.cpu().clone()
        gathered = scores_cpu.gather(1, cpu_ids)
        if base is None:
            base = cpu_ids
        if stable_ties:
            assert torch.equal(cpu_ids, base)
        assert torch.equal(cpu_values, expected_values)
        assert torch.equal(gathered, expected_values)
        assert all(len(set(row)) == 10 for row in cpu_ids.tolist())
        # Distinguish slot permutations from a different expert at the tied cutoff.
        changed_sets = [r for r in range(512) if set(cpu_ids[r].tolist()) != set(base[r].tolist())]
        for row in changed_sets:
            changed = set(cpu_ids[row].tolist()) ^ set(base[row].tolist())
            assert all(scores_cpu[row, e] == expected_values[row, -1] for e in changed)
        rec = {'rep': rep, 'score_digest': digest(repeated_scores), 'id_digest': digest(cpu_ids),
               'different_slots': int((cpu_ids != base).sum()), 'changed_expert_set_rows': changed_sets,
               'all_selected_scores_match_cpu_order_statistics': True,
               'changed_membership_only_at_exact_cutoff_ties': True}
        all_ids.append(cpu_ids)
        records.append(rec)
        print(json.dumps(rec), flush=True)
    torch.save(all_ids, out/'router-selected-repeats.pt')
    result = {'route': {'ext_routing_std_present': hasattr(ext,'routing_std'),
                        'gfx12_small_row_eligible': _hip_router_call_supported(512, mlp.routing_cfg, x, {}),
                        'function': str(mlp.routing_fn), 'scores_dtype': str(scores.dtype),
                        'gate_dtype': str(gate.dtype)},
              'records': records,
              'cutoff_tie_rows': [r for r in range(512) if (scores_cpu[r] == expected_values[r,-1]).sum() > 1]}
    (out/'router-replay.json').write_text(json.dumps(result,indent=2))
    mlp.unload()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['capture', 'replay', 'module', 'router'])
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=12)
    controls = parser.add_mutually_exclusive_group()
    controls.add_argument('--force-routing', action='store_true')
    controls.add_argument('--stable-router-ties', action='store_true',
                          help='diagnostic only: stable exact ties in the torch router fallback')
    parser.add_argument('--rows', type=int, choices=[2,16,17,32,512])
    parser.add_argument('--sentinel', action='store_true')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    marker = {'capture':'frozen.pt', 'replay':'replay.json', 'router':'router-replay.json',
              'module':'module-forced.json' if args.force_routing else 'module-replay.json'}[args.mode]
    if (args.out/marker).exists():
        parser.error(f'{args.out/marker} already exists; use a fresh output directory')
    if args.repeats < 2:
        parser.error('--repeats must be at least 2')
    if args.mode == 'capture':
        capture(args.out)
    elif args.mode == 'module':
        module_replay(args.out, args.repeats, args.force_routing, args.stable_router_ties)
    elif args.mode == 'router':
        router_replay(args.out, args.repeats, args.stable_router_ties)
    else:
        replay(args.out, args.repeats, args.rows, args.sentinel)
