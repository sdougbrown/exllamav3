"""Qualified, paired host-wall prefill benchmark; no model tensor instrumentation."""
import json
import os
from pathlib import Path
import statistics
import threading
import time

from hip_prefill_wall import run_prefill_wall


class DeviceMemorySampler:
    """Read-only sysfs sampling; observed peaks are not exact hardware high-water marks."""
    def __init__(self):
        self.paths = {p.parent.resolve().name: p for p in Path('/sys/class/drm').glob('card[0-9]/device/mem_info_vram_used')}
        self.peaks = {}
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def run(self):
        while not self.stop.is_set():
            with self.lock:
                for key, path in self.paths.items():
                    value = int(path.read_text())
                    self.peaks[key] = max(value, self.peaks.get(key, 0))
            self.stop.wait(.05)

    def reset(self):
        with self.lock:
            self.peaks = {k: int(p.read_text()) for k,p in self.paths.items()}

    def snapshot(self):
        with self.lock:
            return dict(self.peaks)

    def close(self):
        self.stop.set()
        self.thread.join()


def qualification_for(paths, lengths, controls):
    qualified = {}
    for path in paths:
        proof = json.loads((path / 'exact-compare.json').read_text())
        assert proof['all_exact'], path
        metadata = json.loads((path / 'diagnostic-controls.json').read_text())
        assert all(metadata[k] == controls[k] for k in ('stable_router_ties', 'fixed_order_gdn'))
        data = json.loads((path / 'correctness.json').read_text())
        assert data['identity']['logical_vocab'] == 248077
        assert data['reps'] >= 3 and data['cfg_reps'] >= 2
        for length, entry in data['lengths'].items():
            qualified[int(length)] = {'prompt_sha256': entry['prompt_sha256'], 'path': str(path),
                                      'model_dir': data['identity']['model_dir']}
    assert set(lengths) <= qualified.keys(), 'all timed lengths must be qualified'
    return qualified


def bench(b, args, qualifications, controls):
    out = Path(args.out)
    assert not (out / 'bench-wall.json').exists()
    lengths = [int(x) for x in args.lengths.split(',')]
    qualified = qualification_for(qualifications, lengths, controls)
    assert args.reps == 3 and args.chunk == 512
    assert all(Path(qualified[n]['model_dir']).resolve() == Path(args.model_dir).resolve() for n in lengths)
    cursor, cursor_error = b.journal_cursor()
    result = {'mode': 'prefill-only synchronized host wall', 'started': b.utcnow(),
              'controls': controls, 'qualification': qualified, 'chunk': args.chunk,
              'orders': b.latine_orders(args.reps), 'lengths': {},
              'journal_cursor': cursor, 'journal_cursor_error': cursor_error,
              'device_vram_metric': 'sysfs sampled peak bytes, 50 ms; includes all device users, not an exact high-water mark'}
    start = time.perf_counter()
    config, model, cache, tokenizer, generator = b.load_model(args.model_dir, args.chunk)
    result['identity'] = b._identity_block(b._EXL3_DIR, Path(b.__file__).resolve().parents[2], args.model_dir, config)
    result['identity'].update({'module_device_map': b.module_device_map(model),
                               'load_wall_s': time.perf_counter() - start,
                               'actual_vocab_size': tokenizer.actual_vocab_size,
                               'devices': {str(d): str(b.torch.cuda.get_device_properties(d)) for d in (0, 1)},
                               'torch_cpu_threads': b.torch.get_num_threads(),
                               'driver_config': {'chunk': args.chunk, 'max_output_size': 32, 'mtp': 'off',
                                   'max_batch_size': b.MAX_BATCH_SIZE, 'cache_num_tokens': b.CACHE_NUM_TOKENS,
                                   'cache_type': 'CacheLayer_quant', 'k_bits': 8, 'v_bits': 8, 'max_history': 0,
                                   'recurrent_cache_b': 4 * b.GB, 'cpu_cache_b': 0, 'gpu_split_gb': [30., 30.]},
                               'memory_after_load': b.memory_snapshot(),
                               'kfd_after_startup': b.kfd_evicted_ms(os.getpid())})
    corpus = b.build_corpus()
    sampler = DeviceMemorySampler()
    try:
        for length in lengths:
            ids = b.prompt_for(corpus, length)
            assert b.canonical_sha256(ids) == qualified[length]['prompt_sha256']
            entry = {'prompt_sha256': b.canonical_sha256(ids), 'trials': [], 'warmups': []}
            result['lengths'][str(length)] = entry
            specs = [(True, -1, name) for name in ('baseline', 'count', 'both')] + [
                (False, rep, name) for rep, order in enumerate(b.latine_orders(args.reps)) for name in order]
            for warm, rep, name in specs:
                count, uploads = b.CONFIGS[name]
                b.apply_gates(count, uploads)
                b.reset_trial_state(generator)
                trial_ids = b.prompt_for(corpus, length, offset=b.WARMUP_OFFSET) if warm else ids
                job = b._enqueue_job(generator, trial_ids, 1, f'wall_{length}_{name}_{rep}')
                before = b.kfd_evicted_ms(os.getpid())
                for d in (0, 1):
                    b.torch.cuda.reset_peak_memory_stats(d)
                sampler.reset()
                wall = run_prefill_wall(generator, job, b.sync_all)
                after = b.kfd_evicted_ms(os.getpid())
                memory = b.memory_snapshot()
                device_peak = sampler.snapshot()
                b.assert_job_uncached(job, f'wall {length} {name} {rep}')
                assert job.new_tokens == 0, 'decode entered timed region'
                record = {'config': name, 'rep': rep, 'gates': {'count': count, 'uploads': uploads},
                          'prompt_sha256': b.canonical_sha256(trial_ids), 'prefill_tokens': length - 1,
                          'prefill_wall_s': wall, 'tok_s': (length - 1) / wall,
                          'cached_pages': job.cached_pages, 'cached_tokens': job.cached_tokens,
                          'non_sequential_pages': job.non_sequential_pages, 'generated_tokens': job.new_tokens,
                          'memory': memory, 'device_vram_sampled_peak_b': device_peak,
                          'kfd_before': before, 'kfd_after': after,
                          'kfd_request_delta': {k: after[k] - before[k] for k in after.keys() & before.keys()
                                                if not k.startswith('_') and isinstance(after[k], int) and isinstance(before[k], int)}}
                generator.cancel(job)
                entry['warmups' if warm else 'trials'].append(record)
                b.log(f'wall L{length} {name} rep{rep} warm={warm}: {wall:.4f}s, {record["tok_s"]:.1f} tok/s')
                (out / 'bench-wall-partial.json').write_text(json.dumps(result, indent=2))
            assert len(entry['trials']) == 9
            entry['medians'] = {name: statistics.median(r['prefill_wall_s'] for r in entry['trials'] if r['config'] == name)
                                for name in ('baseline', 'count', 'both')}
    finally:
        sampler.close()
        result['finished'] = b.utcnow()
        result['journal_result'] = b.journal_after(cursor, out / 'kernel-wall.log') if cursor else {'skipped': cursor_error}
        (out / 'bench-wall.json').write_text(json.dumps(result, indent=2))
    return 0
