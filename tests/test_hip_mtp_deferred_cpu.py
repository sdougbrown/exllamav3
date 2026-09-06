"""CPU-only prototype contracts; never import the exllamav3 package/extension."""
import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]


def module():
    spec = importlib.util.spec_from_file_location('deferred', ROOT / 'exllamav3/generator/mtp_deferred.py')
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def supported():
    g = NS(mtp_draft=True, draft_model=type('Qwen4ExpMTPModel', (), {'loaded_tp': False})(),
           model=NS(loaded_tp=False), max_batch_size=1, active_jobs=[], pending_jobs=[],
           recurrent_cache={}, pagetable=NS(cpu_tier=None, all_pages=[NS(kv_position=0, ref_count=0)]))
    j = NS(generator=None, _mtp_deferred=None, sequences=[NS(kv_position=0, mtp_carry_hidden=None)], embeddings=None,
           banned_strings=None, filters=None, prefix_token=None, is_requeued=False,
           orig_max_rq_tokens=None, mtp_last_hidden=None)
    return g, j


def test_full_history_order_shift_and_owned_export():
    d = module().DeferredMTPPrefill(5)
    a = torch.arange(12).reshape(1, 3, 4).float()
    b = torch.arange(12, 20).reshape(1, 2, 4).float()
    expect = torch.cat((torch.zeros(1, 1, 4), a[:, :-1], a[:, -1:], b[:, :-1]), 1)
    d.append(0, 3, a)
    a.fill_(-999)
    with pytest.raises(RuntimeError, match='complete target'):
        d.drain(lambda *_: pytest.fail('early draft'))
    d.append(3, 5, b)
    b.fill_(-888)
    calls = []
    carry = d.drain(lambda start, end, hidden: calls.append((start, end, hidden.clone())))
    assert [(s, e) for s, e, _ in calls] == [(0, 3), (3, 5)]
    assert torch.equal(torch.cat([h for _, _, h in calls], 1), expect)
    assert torch.equal(carry, torch.arange(16, 20).reshape(1, 1, 4).float())
    assert d.complete and d.draft_end == 5 and not d.chunks
    assert d.retained_peak_bytes == 80
    with pytest.raises(RuntimeError):
        d.drain(lambda *_: None)


@pytest.mark.parametrize('start,end,shape', [(1, 3, (1, 2, 4)), (0, 6, (1, 6, 4)),
                                             (0, 3, (1, 2, 4)), (0, 0, (1, 0, 4)),
                                             (0, 3, (2, 3, 4))])
def test_reject_gaps_overrun_empty_and_shape(start, end, shape):
    d = module().DeferredMTPPrefill(5)
    with pytest.raises(RuntimeError):
        d.append(start, end, torch.zeros(shape))
    assert d.target_end == 0 and not d.chunks


@pytest.mark.parametrize('abort', [True, False])
def test_failure_or_cancel_invalidates_and_releases(abort):
    d = module().DeferredMTPPrefill(3)
    d.append(0, 3, torch.ones(1, 3, 4))
    if abort:
        d.abort()
    else:
        def fail(*_):
            raise ValueError('draft failure')
        with pytest.raises(ValueError, match='draft failure'):
            d.drain(fail)
    assert not d.complete and not d.chunks and d.failed
    with pytest.raises(RuntimeError):
        d.append(0, 3, torch.ones(1, 3, 4))
    with pytest.raises(RuntimeError):
        d.drain(lambda *_: None)


@pytest.mark.parametrize('problem', ['mtp', 'arch', 'tp', 'batch', 'active', 'pending', 'cache',
                                     'ref', 'tier', 'stash', 'seq', 'position', 'carry', 'mm', 'banned',
                                     'filters', 'healing', 'requeue', 'rq', 'used_job', 'used_retainer'])
def test_fail_closed_support_guards(problem):
    g, j = supported()
    if problem == 'mtp': g.mtp_draft = False
    if problem == 'arch': g.draft_model = NS(loaded_tp=False)
    if problem == 'tp': g.model.loaded_tp = True
    if problem == 'batch': g.max_batch_size = 2
    if problem == 'active': g.active_jobs = [object()]
    if problem == 'pending': g.pending_jobs = [object()]
    if problem == 'cache': g.pagetable.all_pages[0].kv_position = 1
    if problem == 'ref': g.pagetable.all_pages[0].ref_count = 1
    if problem == 'tier': g.pagetable.cpu_tier = object()
    if problem == 'stash': g.recurrent_cache = {'old': object()}
    if problem == 'seq': j.sequences *= 2
    if problem == 'position': j.sequences[0].kv_position = 1
    if problem == 'carry': j.mtp_last_hidden = torch.ones(1)
    if problem == 'mm': j.embeddings = [object()]
    if problem == 'banned': j.banned_strings = ['bad']
    if problem == 'filters': j.filters = [object()]
    if problem == 'healing': j.prefix_token = 1
    if problem == 'requeue': j.is_requeued = True
    if problem == 'rq': j.orig_max_rq_tokens = 100
    if problem == 'used_job': j.generator = g
    if problem == 'used_retainer': j._mtp_deferred = object()
    with pytest.raises(ValueError, match='deferred MTP'):
        module().validate_deferred_job(g, j)


def test_supported_job_and_nonempty_prompt():
    module().validate_deferred_job(*supported())
    with pytest.raises(ValueError):
        module().DeferredMTPPrefill(0)


def job_methods(*names):
    tree = ast.parse((ROOT / 'exllamav3/generator/job.py').read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Job')
    methods = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names]
    ns = {}
    exec(compile(ast.Module(body=methods, type_ignores=[]), 'job.py', 'exec'), ns)
    return ns


def test_readiness_requires_draft_completion():
    fn = job_methods('is_prefill_done')['is_prefill_done']
    j = NS(sequences=[NS(kv_position=3, sequence_ids=[1,2,3,4])],
           _mtp_deferred=NS(complete=False))
    assert not fn(j)
    j._mtp_deferred.complete = True
    assert fn(j)
    j._mtp_deferred = None
    assert fn(j)


def test_cleanup_clears_cache_after_release_and_aborts_history():
    fn = job_methods('deallocate_pages')['deallocate_pages']
    events = []
    page = NS(clear=lambda: events.append('clear'))
    j = NS(_mtp_deferred=NS(abort=lambda: events.append('abort')),
           _invalidate_mtp_carry=lambda: events.append('carry'),
           free_recurrent_state=lambda: events.append('free'),
           generator=NS(recurrent_cache=NS(prune_stranded=lambda: events.append('prune'))),
           pagetable=NS(deallocate_pages=lambda _: events.append('release')),
           sequences=[NS(allocated_pages=[page])])
    fn(j)
    assert events == ['abort', 'carry', 'free', 'release', 'clear', 'prune']
    assert j.sequences[0].allocated_pages == []


def test_actual_catchup_replays_full_ids_and_fences_each_host_slot():
    fn = job_methods('_catch_up_mtp_prefill')['_catch_up_mtp_prefill']
    d = module().DeferredMTPPrefill(5)
    d.append(0, 3, torch.arange(12).reshape(1,3,4).float())
    d.append(3, 5, torch.arange(12,20).reshape(1,2,4).float())
    events, pending = [], []
    ids = torch.tensor([[10,11,12,13,14,15]])
    table = object()
    seq = NS(sequence_ids=NS(torch_slice=lambda a,b: ids[:,a:b]), block_index_tensor=table)
    def stage(start):
        assert not pending, 'host staging overwritten before previous DMA fence'
        events.append(('stage',start))
        return torch.tensor([start])
    def draft(input_ids, params):
        start = int(params['cache_seqlens'][0])
        assert params['block_table'] is table
        assert torch.equal(input_ids,ids[:,start:start+input_ids.shape[1]])
        pending.append(start)
        events.append(('draft',start))
    def fence():
        events.append(('fence',pending.pop()))
    j = NS(_mtp_deferred=d, _prefill_staged_cache_seqlens=stage,
           _prefill_record_cache_seqlens_dma=fence,
           generator=NS(draft_model=NS(prefill=draft),draft_cache=object()))
    fn(j,seq)
    assert events == [('stage',0),('draft',0),('fence',0),('stage',3),('draft',3),('fence',3)]
    assert d.complete and j.mtp_last_hidden is seq.mtp_carry_hidden
    assert torch.equal(j.mtp_last_hidden,torch.arange(16,20).reshape(1,1,4).float())


@pytest.mark.parametrize('value,enabled', [(None,False),('0',False),('1',True)])
def test_generator_flag_is_default_off_and_explicit_opt_in(monkeypatch,value,enabled):
    import os
    tree = ast.parse((ROOT / 'exllamav3/generator/generator.py').read_text())
    assignment = next(n for n in ast.walk(tree) if isinstance(n,ast.Assign)
                      and any(isinstance(t,ast.Attribute) and t.attr == 'mtp_deferred_prefill' for t in n.targets))
    if value is None:
        monkeypatch.delenv('EXL3_MTP_DEFERRED_PREFILL',raising=False)
    else:
        monkeypatch.setenv('EXL3_MTP_DEFERRED_PREFILL',value)
    obj = NS()
    exec(compile(ast.Module(body=[assignment],type_ignores=[]),'generator.py','exec'), {'self':obj,'os':os})
    assert obj.mtp_deferred_prefill is enabled
