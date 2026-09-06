"""Prefill-only host wall driver, including ordinary per-iteration checkpoints."""
import time


def run_prefill_wall(generator, job, sync_all, clock=time.perf_counter):
    assert not generator.draft_model and not generator.ngram_match_min and not generator.visualizer
    original = generator.iterate_gen

    class PrefillDone(BaseException):
        pass

    def stop_before_generation(results):
        for result in results:
            if result.get('stage') == 'error':
                raise result['error']
        if job.is_prefill_done():
            raise PrefillDone
        return original(results)

    generator.iterate_gen = stop_before_generation
    try:
        sync_all()
        start = clock()
        try:
            while not job.is_prefill_done():
                generator.iterate()
        except PrefillDone:
            pass
        sync_all()
        return clock() - start
    finally:
        generator.iterate_gen = original
