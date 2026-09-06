from types import SimpleNamespace

import pytest

from hip_prefill_wall import run_prefill_wall


@pytest.mark.parametrize('fail', [False, True])
def test_wall_stops_after_bookkeeping_before_generation_and_restores(fail):
    events = []
    job = SimpleNamespace(chunks=0)
    job.is_prefill_done = lambda: job.chunks == 3
    generator = SimpleNamespace(draft_model=None, ngram_match_min=0, visualizer=None)

    def generate(results):
        events.append('gen_noop' if not job.is_prefill_done() else 'GENERATION_BAD')

    generator.iterate_gen = generate

    def iterate():
        events.append('start_jobs')
        job.chunks += 1
        events.extend(['prefill', 'checkpoint'])
        if fail:
            raise RuntimeError('failure')
        generator.iterate_gen([])

    generator.iterate = iterate
    ticks = iter([10., 13.])
    sync = lambda: events.append('sync_both')
    clock = lambda: next(ticks)
    if fail:
        with pytest.raises(RuntimeError, match='failure'):
            run_prefill_wall(generator, job, sync, clock)
    else:
        assert run_prefill_wall(generator, job, sync, clock) == 3.
        assert events == ['sync_both'] + ['start_jobs', 'prefill', 'checkpoint', 'gen_noop'] * 2 + [
            'start_jobs', 'prefill', 'checkpoint', 'sync_both']
    assert generator.iterate_gen is generate


@pytest.mark.parametrize('problem', ['none', 'unresolved', 'controls', 'length'])
def test_qualification_gate(tmp_path, problem):
    import json
    from hip_prefill_wall_bench import qualification_for
    controls = {'stable_router_ties': True, 'fixed_order_gdn': True}
    metadata = dict(controls)
    if problem == 'controls':
        metadata['fixed_order_gdn'] = False
    (tmp_path / 'diagnostic-controls.json').write_text(json.dumps(metadata))
    (tmp_path / 'exact-compare.json').write_text(json.dumps({'all_exact': problem != 'unresolved'}))
    (tmp_path / 'correctness.json').write_text(json.dumps({
        'identity': {'logical_vocab': 248077, 'model_dir': '/model'}, 'reps': 3, 'cfg_reps': 2,
        'lengths': {'12288': {'prompt_sha256': 'fixed-prefix'}}}))
    lengths = [65536] if problem == 'length' else [12288]
    if problem == 'none':
        assert qualification_for([tmp_path], lengths, controls)[12288]['prompt_sha256'] == 'fixed-prefix'
    else:
        with pytest.raises(AssertionError):
            qualification_for([tmp_path], lengths, controls)


def test_wall_rejects_speculation():
    generator = SimpleNamespace(draft_model=object(), ngram_match_min=0, visualizer=None)
    with pytest.raises(AssertionError):
        run_prefill_wall(generator, None, lambda: None)


@pytest.mark.parametrize('prefill_done', [False, True])
def test_wall_raises_reaped_job_error_before_success_check(prefill_done):
    events = []
    job = SimpleNamespace(chunks=0, serial_number=7)
    job.is_prefill_done = lambda: job.chunks > 0
    generator = SimpleNamespace(draft_model=None, ngram_match_min=0, visualizer=None)

    def generate(results):
        events.append('gen')

    generator.iterate_gen = generate

    def iterate():
        events.append('iterate')
        job.chunks += 1
        error = RuntimeError('reaped job failure')
        generator.iterate_gen([{'job': job, 'serial': job.serial_number, 'stage': 'error',
                                'eos': True, 'error': error}])

    generator.iterate = iterate
    with pytest.raises(RuntimeError, match='reaped job failure'):
        run_prefill_wall(generator, job, lambda: events.append('sync'), lambda: 0.)
    assert events == ['sync', 'iterate'], 'more than one iteration or generation ran'
    assert generator.iterate_gen is generate


def test_wall_propagates_original_error_object_and_restores_hook():
    job = SimpleNamespace(chunks=0, serial_number=1)
    job.is_prefill_done = lambda: False
    generator = SimpleNamespace(draft_model=None, ngram_match_min=0, visualizer=None)
    original_hook = lambda results: None
    generator.iterate_gen = original_hook
    original_error = ValueError('distinct original')

    def iterate():
        generator.iterate_gen([{'job': job, 'serial': 1, 'stage': 'error',
                                'eos': True, 'error': original_error}])

    generator.iterate = iterate
    with pytest.raises(ValueError) as exc_info:
        run_prefill_wall(generator, job, lambda: None, lambda: 0.)
    assert exc_info.value is original_error
    assert generator.iterate_gen is original_hook
