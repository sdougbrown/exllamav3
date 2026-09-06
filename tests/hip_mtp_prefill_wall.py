"""Inclusive MTP prompt wall timer: stop before drafting, not just target decode."""
import time


def run_mtp_prefill_wall(generator, job, sync_all, clock=time.perf_counter):
    assert generator.mtp_draft and not generator.ngram_match_min and not generator.visualizer
    assert not job.is_prefill_done() and job.new_tokens == 0
    original = generator.iterate_draftmodel_mtp_gen

    class PrefillDone(BaseException):
        pass

    def stop_before_drafting(results):
        for result in results:
            if result.get('stage') == 'error':
                raise result['error']
        if job.is_prefill_done():
            raise PrefillDone
        return original(results)

    generator.iterate_draftmodel_mtp_gen = stop_before_drafting
    try:
        sync_all()
        start = clock()
        try:
            while not job.is_prefill_done():
                generator.iterate()
        except PrefillDone:
            pass
        sync_all()
        elapsed = clock() - start
        assert job.is_prefill_done() and job.new_tokens == 0
        return elapsed
    finally:
        generator.iterate_draftmodel_mtp_gen = original
