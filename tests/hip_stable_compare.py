"""Exact coordinate comparisons of the validated forced-control artifacts."""
import argparse
import json
from pathlib import Path

import torch


def delta(a, b):
    assert a.shape == b.shape and a.dtype == b.dtype
    d = a.float() - b.float()
    return {'bitwise_equal': torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)),
            'different': int((a != b).sum()), 'max_abs': float(d.abs().max()),
            'rms': float(d.square().mean().sqrt()), 'finite': bool(a.isfinite().all() and b.isfinite().all())}


def compare(out, partial_length=None):
    torch.set_num_threads(8)
    result = json.loads((out / ('correctness_partial.json' if partial_length else 'correctness.json')).read_text())
    if partial_length:
        result['lengths'] = {str(partial_length): result['lengths'][str(partial_length)]}
    assert result['protocol'] == 'forced-control'
    assert result['identity']['logical_vocab'] == 248077
    comparisons = []
    for length, entry in result['lengths'].items():
        ref_tokens = entry['reference_tokens']
        ref_prefixes = entry['reference_prefix_shas']
        assert ref_tokens and len(ref_prefixes) == len(ref_tokens)
        expected_runs = {('baseline', r) for r in range(result['reps'])} | {
            (name, r) for name in ('count', 'both') for r in range(result['cfg_reps'])}
        assert len(entry['runs']) == len(expected_runs)
        assert {(r['config'], r['rep']) for r in entry['runs']} == expected_runs
        assert entry['runs'][0]['role'] == 'reference-unforced'
        for run in entry['runs'][1:]:
            assert run['role'] == 'forced' and run['prefix_asserted']
            assert run['n_steps_actual'] == len(ref_tokens)
            assert run['prefix_shas'] == ref_prefixes
            assert all(run['forced_flags'])
            assert [s['next_token'] for s in run['steps']] == ref_tokens
            tag = f"{run['config']}_rep{run['rep']}_L{length}"
            record = {'tag': tag, 'steps': []}
            for step in range(len(ref_tokens)):
                a = torch.load(out / f'logits_baseline_rep0_L{length}_step{step}.pt', weights_only=True)
                b = torch.load(out / f'logits_{tag}_step{step}.pt', weights_only=True)
                assert a.numel() == 248077
                record['steps'].append(delta(a, b))
            a = torch.load(out / f'planes_baseline_rep0_L{length}.pt', weights_only=True)
            b = torch.load(out / f'planes_{tag}.pt', weights_only=True)
            for key in ('kv_position', 'compress_ratio', 'raw_rows', 'pooled_rows'):
                assert a['meta'][key] == b['meta'][key]
            record['planes'] = {key: delta(a[key], b[key]) for key in ('raw', 'pooled')}
            comparisons.append(record)
    assert comparisons, 'no comparisons performed: no lengths or no candidate runs'
    exact = all(s['bitwise_equal'] and s['finite'] for r in comparisons
                for s in r['steps'] + list(r['planes'].values()))
    report = {'all_exact': exact, 'comparisons': comparisons,
              'qualification': 'exact' if exact else 'UNRESOLVED: no end-to-end arithmetic bound established'}
    name = f'exact-compare-L{partial_length}-partial.json' if partial_length else 'exact-compare.json'
    (out / name).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return exact


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--partial-length', type=int)
    args = p.parse_args()
    raise SystemExit(0 if compare(args.out, args.partial_length) else 1)
