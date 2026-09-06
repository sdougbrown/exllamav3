from types import SimpleNamespace as NS
import pytest


def test_wall_includes_catchup_checkpoint_and_stops_before_first_draft():
    from hip_mtp_prefill_wall import run_mtp_prefill_wall
    events = []
    job = NS(done=False, new_tokens=0)
    job.is_prefill_done = lambda: job.done
    def draft(_):
        assert not job.done, 'draft decode entered timer'
        events.append('draft_noop')
    g = NS(mtp_draft=True, ngram_match_min=0, visualizer=None, iterate_draftmodel_mtp_gen=draft)
    def iterate():
        events.append('target')
        if events.count('target') == 2:
            events.append('catchup')
            job.done = True
        events.append('checkpoint')
        g.iterate_draftmodel_mtp_gen([])
    g.iterate = iterate
    ticks = iter([10., 15.])
    assert run_mtp_prefill_wall(g, job, lambda: events.append('sync_both'), lambda: next(ticks)) == 5.
    assert events == ['sync_both','target','checkpoint','draft_noop','target','catchup','checkpoint','sync_both']
    assert g.iterate_draftmodel_mtp_gen is draft


@pytest.mark.parametrize('done', [False, True])
def test_wall_propagates_contained_error_and_restores(done):
    from hip_mtp_prefill_wall import run_mtp_prefill_wall
    job = NS(is_prefill_done=lambda: False, new_tokens=0)
    original = lambda _: pytest.fail('draft should not run')
    g = NS(mtp_draft=True, ngram_match_min=0, visualizer=None, iterate_draftmodel_mtp_gen=original)
    error = ValueError('prefill failed')
    def iterate():
        job.is_prefill_done = lambda: done
        g.iterate_draftmodel_mtp_gen([{'stage':'error','error':error}])
    g.iterate = iterate
    with pytest.raises(ValueError) as e:
        run_mtp_prefill_wall(g, job, lambda: None)
    assert e.value is error
    assert g.iterate_draftmodel_mtp_gen is original
