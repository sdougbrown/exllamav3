"""CPU-only contract tests for the diagnostic wrapper (no GPU, no harness load)."""
import os
import subprocess
import sys
from pathlib import Path

import pytest

TESTS = Path(__file__).parent
SCRIPT = TESTS / 'hip_stable_prefill.py'
CPU_ENV = dict(os.environ,
               CUDA_VISIBLE_DEVICES='', HIP_VISIBLE_DEVICES='', ROCR_VISIBLE_DEVICES='',
               PYTHONPATH=str(TESTS) + os.pathsep + os.environ.get('PYTHONPATH', ''),
               EXL3_VALIDATED_PREFILL_HARNESS=str(TESTS / 'no-such-harness.py'))


def run_wrapper(args):
    return subprocess.run([sys.executable, str(SCRIPT)] + args,
                          capture_output=True, text=True, env=CPU_ENV, cwd=str(TESTS))


@pytest.fixture(params=['same', 'parent', 'child', 'symlink'])
def overlapping(tmp_path, request):
    kind = request.param
    root = tmp_path / 'tree'
    qual = root / 'qual'
    qual.mkdir(parents=True)
    qual_artifact = qual / 'exact-compare.json'
    qual_artifact.write_bytes(b'qualification-artifact')
    inner = qual / 'inside'
    inner.mkdir()
    inner_artifact = inner / 'x.json'
    inner_artifact.write_bytes(b'inner')
    if kind == 'same':
        out = qual
        args = ['--out', str(qual), '--qualification', str(qual)]
        guard = [qual_artifact, inner_artifact]
    elif kind == 'parent':
        out = root
        args = ['--out', str(root), '--qualification', str(qual)]
        guard = [qual_artifact, inner_artifact]
    elif kind == 'child':
        out = inner
        args = ['--out', str(inner), '--qualification', str(qual)]
        guard = [qual_artifact, inner_artifact]
    else:
        out = qual
        link = tmp_path / 'qual-link'
        link.symlink_to(qual, target_is_directory=True)
        args = ['--out', str(qual), '--qualification', str(link)]
        guard = [qual_artifact, inner_artifact]
    return args, guard


@pytest.mark.parametrize('mode', ['correctness', 'bench'])
def test_overlapping_out_rejected_before_any_writes(overlapping, mode):
    args, guard = overlapping
    result = run_wrapper([mode] + args)
    assert result.returncode != 0
    assert 'overlaps qualification' in result.stderr
    assert 'no-such-harness' not in result.stderr, 'harness load ran before validation'
    for path in guard:
        assert path.read_bytes() == (b'qualification-artifact' if path.name == 'exact-compare.json'
                                     else b'inner')


def test_bench_capture_gdn_rejected_before_model_load_or_writes(tmp_path):
    out = tmp_path / 'out'
    qual = tmp_path / 'qual'
    out.mkdir()
    qual.mkdir()
    result = run_wrapper(['bench', '--out', str(out), '--qualification', str(qual), '--capture-gdn'])
    assert result.returncode != 0
    assert 'capture-gdn' in result.stderr and 'bench' in result.stderr
    assert list(out.iterdir()) == [], 'bench --capture-gdn wrote files before rejection'


def test_validation_precedes_harness_load(monkeypatch, tmp_path):
    import hip_stable_prefill

    def fail_if_called():
        raise AssertionError('load_harness called')

    monkeypatch.setattr(hip_stable_prefill, 'load_harness', fail_if_called)
    shared = tmp_path / 'shared'
    shared.mkdir()
    # Overlapping: rejected without touching load_harness.
    monkeypatch.setattr(sys, 'argv', ['hip_stable_prefill.py', 'correctness',
                                      '--out', str(shared), '--qualification', str(shared)])
    with pytest.raises(SystemExit):
        hip_stable_prefill.main()
    # Non-overlapping: validation passes, wrapper proceeds (next step would be load_harness).
    monkeypatch.setattr(sys, 'argv', ['hip_stable_prefill.py', 'correctness',
                                      '--out', str(shared), '--qualification', str(tmp_path / 'elsewhere')])
    with pytest.raises(AssertionError, match='load_harness called'):
        hip_stable_prefill.main()