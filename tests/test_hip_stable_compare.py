import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from hip_stable_compare import delta

TESTS = Path(__file__).parent
SCRIPT = TESTS / 'hip_stable_compare.py'
CPU_ENV = dict(os.environ, CUDA_VISIBLE_DEVICES='', HIP_VISIBLE_DEVICES='', ROCR_VISIBLE_DEVICES='')


def run_compare(out):
    return subprocess.run([sys.executable, str(SCRIPT), '--out', str(out)],
                          capture_output=True, text=True, env=CPU_ENV, cwd=str(TESTS))


def build_artifacts(out, exact, lengths):
    """Minimal self-contained forced-control artifact set per the real schema."""
    out.mkdir(parents=True, exist_ok=True)
    steps = 2
    ref_tokens = [11, 22]
    ref_prefixes = ['sha-a', 'sha-b']
    correctness = {'protocol': 'forced-control',
                   'identity': {'logical_vocab': 248077, 'model_dir': '/model'},
                   'reps': 3, 'cfg_reps': 2, 'lengths': {}}
    for length in lengths:
        entry = {'reference_tokens': ref_tokens, 'reference_prefix_shas': ref_prefixes,
                 'runs': [{'config': 'baseline', 'rep': 0, 'role': 'reference-unforced'}]}
        forced_runs = [('baseline', rep) for rep in range(1, 3)] + \
                      [(name, rep) for name in ('count', 'both') for rep in range(2)]
        for name, rep in forced_runs:
            entry['runs'].append({'config': name, 'rep': rep, 'role': 'forced',
                                  'prefix_asserted': True, 'n_steps_actual': steps,
                                  'prefix_shas': ref_prefixes, 'forced_flags': [True] * steps,
                                  'steps': [{'next_token': t} for t in ref_tokens]})
        correctness['lengths'][str(length)] = entry
    (out / 'correctness.json').write_text(json.dumps(correctness, indent=2))
    base = torch.arange(248077, dtype=torch.float32)
    candidate = base.clone()
    if not exact:
        candidate[123] += 1
    planes_base = {'meta': {'kv_position': 7, 'compress_ratio': 4, 'raw_rows': 64, 'pooled_rows': 16},
                   'raw': torch.eye(4, dtype=torch.float32), 'pooled': torch.eye(2, dtype=torch.float32)}
    for length in lengths:
        for step in range(steps):
            torch.save(base, out / f'logits_baseline_rep0_L{length}_step{step}.pt')
        torch.save(planes_base, out / f'planes_baseline_rep0_L{length}.pt')
        for name, rep in forced_runs:
            tag = f'{name}_rep{rep}_L{length}'
            for step in range(steps):
                torch.save(candidate, out / f'logits_{tag}_step{step}.pt')
            torch.save(planes_base, out / f'planes_{tag}.pt')


def test_bitwise_comparison_distinguishes_signed_zero():
    result = delta(torch.tensor([0., -0.]), torch.tensor([0., 0.]))
    assert result['different'] == 0
    assert not result['bitwise_equal']


def test_bitwise_comparison_self_and_finiteness():
    a = torch.tensor([0., -0., 1.])
    result = delta(a, a.clone())
    assert result['bitwise_equal'] and result['finite']
    assert not delta(torch.tensor([float('inf')]), torch.tensor([float('inf')]))['finite']


def test_coordinate_shape_is_not_broadcast():
    with pytest.raises(AssertionError):
        delta(torch.zeros(3), torch.zeros(1, 3))


@pytest.mark.parametrize('exact', [True, False])
def test_compare_cli_exit_code_and_artifact(tmp_path, exact):
    out = tmp_path / 'out'
    build_artifacts(out, exact, [2048])
    result = run_compare(out)
    assert result.returncode == (0 if exact else 1), result.stderr
    report = json.loads((out / 'exact-compare.json').read_text())
    assert report['all_exact'] is exact


def test_compare_rejects_empty_lengths(tmp_path):
    out = tmp_path / 'out'
    build_artifacts(out, True, [])
    result = run_compare(out)
    assert result.returncode != 0
    assert 'no comparisons' in result.stderr
    assert not (out / 'exact-compare.json').exists()
