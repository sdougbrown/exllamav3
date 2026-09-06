"""Throwaway MTP qualification + inclusive wall probe; launcher-masked, sole GPU owner."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
SERVE = Path.home() / 'Serve'
HELPER = SERVE / 'hosts/rocky/bench-prefill-validated.py'


def digest(t):
    return hashlib.sha256(t.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, default=str))


def rss():
    return {k: v.strip() for k, v in (line.split(':', 1) for line in Path('/proc/self/status').read_text().splitlines())
            if k in ('VmRSS', 'VmHWM')}


def load(b):
    from exllamav3 import Config, Model, Cache, Tokenizer
    from exllamav3.cache import CacheLayer_quant
    from exllamav3.generator import Generator
    config = Config.from_directory(b.DEFAULT_MODEL_DIR)
    tokenizer = Tokenizer.from_config(config)
    model = Model.from_config(config)
    draft_config = Config.from_directory(b.DEFAULT_MODEL_DIR)
    draft = Model.from_config(draft_config, component='mtp')
    kwargs = dict(max_num_tokens=393216, layer_type=CacheLayer_quant, k_bits=8, v_bits=8,
                  max_batch_size=4, max_history=3)
    cache = Cache(model, **kwargs)
    draft_cache = Cache(draft, **kwargs)
    draft.load(use_per_device=[3., 0.], max_batch_size=4, max_chunk_size=512, max_output_size=32, verbose=True)
    model.load(use_per_device=[30., 30.], max_batch_size=4, max_chunk_size=512, max_output_size=32, verbose=True)
    generator = Generator(model, cache, tokenizer, draft_model=draft, draft_cache=draft_cache,
                          num_draft_tokens=3, dynamic_draft_tokens=False, max_batch_size=1,
                          max_chunk_size=512, recurrent_cache_size=4*b.GB, cpu_cache_size=0,
                          record_draft_stats=True)
    return config, model, cache, tokenizer, generator


class Capture:
    def __init__(self, b, g, job, directory):
        self.b, self.g, self.job, self.directory = b, g, job, directory
        directory.mkdir()
        self.meta = {'snapshots': {}, 'steps': [], 'rounds': [], 'target_chunks': [], 'draft_chunks': []}
        self.prefilling = False
        self.targets, self.drafts = [], []
        self.originals = (job.prefill, g.model.forward, g.draft_model.prefill, job.receive_sample,
                          g.iterate_draftmodel_mtp_gen)
        def keep_hidden(hidden):
            return hidden.cpu().clone() if len(job.sequences[0].sequence_ids) > 12288 else hidden.clone()
        def prefill(results):
            self.prefilling = True
            try:
                return self.originals[0](results)
            finally:
                self.prefilling = False
        def forward(input_ids, params):
            out = self.originals[1](input_ids, params)
            if self.prefilling:
                self.targets.append((int(params['cache_seqlens'][0]), input_ids.clone(), keep_hidden(params['export_states'][-1])))
            return out
        def draft_prefill(input_ids, params):
            if self.prefilling:
                self.drafts.append((int(params['cache_seqlens'][0]), input_ids.clone(), keep_hidden(params['target_hidden'])))
            return self.originals[2](input_ids, params)
        def sample(logits, next_token, next_k_tokens, next_k_probs, next_prob, results, first_sample_in_sd_batch=True):
            i = len(self.meta['steps'])
            cpu = logits[:, :, :g.tokenizer.actual_vocab_size].detach().cpu().contiguous()
            assert cpu.shape == (1, 1, 248077) and torch.isfinite(cpu).all()
            torch.save(cpu, directory / f'logits-{i}.pt')
            self.meta['steps'].append({'token': int(next_token.item()), 'logit_sha': digest(cpu),
                'prefix_sha': b.canonical_sha256(job.sequences[0].sequence_ids.torch().flatten().tolist()),
                'forced': bool(job.forced_sample), 'position': job.sequences[0].kv_position})
            return self.originals[3](logits, next_token, next_k_tokens, next_k_probs, next_prob,
                                     results, first_sample_in_sd_batch)
        def draft_round(results):
            position = job.sequences[0].kv_position
            out = self.originals[4](results)
            if out is not None:
                self.meta['rounds'].append({'position': position, 'draft_ids': out.clone().tolist()})
            return out
        job.prefill, g.model.forward, g.draft_model.prefill = prefill, forward, draft_prefill
        job.receive_sample, g.iterate_draftmodel_mtp_gen = sample, draft_round

    def prefill_snapshot(self):
        for name, chunks in [('target_chunks', self.targets), ('draft_chunks', self.drafts)]:
            end = 0
            hiddens = []
            for start, ids, hidden in chunks:
                assert start == end and ids.shape[1] == hidden.shape[1]
                end += ids.shape[1]
                cpu = hidden.cpu()
                self.meta[name].append({'start':start, 'end':end, 'ids_sha':digest(ids),
                                        'hidden_sha':digest(cpu), 'shape':list(cpu.shape)})
                hiddens.append(cpu)
            assert end == self.job.sequences[0].kv_position
            torch.save(torch.cat(hiddens, 1), self.directory / f'{name}.pt')
        target = torch.load(self.directory / 'target_chunks.pt', weights_only=True)
        draft = torch.load(self.directory / 'draft_chunks.pt', weights_only=True)
        assert torch.equal(draft[:, :1], torch.zeros_like(draft[:, :1]))
        assert torch.equal(draft[:, 1:], target[:, :-1]), 'shifted-boundary history mismatch'
        assert digest(self.job.mtp_last_hidden.cpu()) == digest(target[:, -1:])
        self.targets.clear()
        self.drafts.clear()
        self.snapshot('prefill')

    def snapshot(self, label):
        self.b.sync_all()
        seq = self.job.sequences[0]
        tensors, metadata = {}, {'position': seq.kv_position, 'recurrent_position': self.job.recurrent_state.position,
                                 'last_checkpoint': self.job.last_recurrent_checkpoint_pos}
        def add(name, tensor, save=True):
            cpu = tensor.detach().cpu().contiguous()
            if cpu.is_floating_point():
                assert torch.isfinite(cpu).all(), name
            metadata[name] = {'sha':digest(cpu), 'shape':list(cpu.shape), 'dtype':str(cpu.dtype)}
            if save:
                tensors[name] = cpu
        add('carry', self.job.mtp_last_hidden)
        if seq.mtp_carry_hidden is not None:
            add('prefill_carry', seq.mtp_carry_hidden)
        pages = seq.block_index_tensor[0].tolist()
        for kind, cache in [('target', self.g.cache), ('draft', self.g.draft_cache)]:
            for layer_idx, (key, layer) in enumerate(cache.layers.items()):
                for attr in ('qk','qv','sk','sv','raw_k','pooled'):
                    t = getattr(layer, attr, None)
                    if t is None:
                        continue
                    ratio = layer.compress_ratio if attr == 'pooled' else 1
                    parts = []
                    for i in range((seq.kv_position+255)//256):
                        n = min(256, seq.kv_position - i*256)//ratio
                        if n:
                            parts.append(t[pages[i], :n])
                    valid = torch.cat(parts, 0)
                    add(f'{kind}.{layer_idx}.{attr}', valid,
                        save=(kind == 'draft' or (layer_idx == 0 and attr in ('raw_k','pooled'))))
        slot = self.job.recurrent_state.slot
        for i, layer in enumerate(self.g.cache.recurrent_layers.values()):
            if hasattr(layer, 'recurrent_state'):
                add(f'recurrent.{i}.state', layer.recurrent_state[slot, :1])
                add(f'recurrent.{i}.conv', layer.conv_state[slot, :, :layer.module.conv_kernel_size])
            else:
                assert type(layer).__name__ == 'PLELayerState'
                add(f'recurrent.{i}.conv', layer.conv_state[slot, :, :layer.win])
                add(f'recurrent.{i}.ids', layer.id_state[slot, :layer.ctx])
        if label == 'prefill':
            for stash in self.g.recurrent_cache.values():
                for i, key in enumerate(self.g.cache.recurrent_layers):
                    for ti, t in enumerate(stash[key]):
                        add(f'checkpoint.{stash["position"]}.{i}.{ti}', t, save=False)
        self.meta['snapshots'][label] = metadata
        torch.save(tensors, self.directory / f'state-{label}.pt')
        dump(self.directory / 'capture.json', self.meta)

    def close(self):
        job, g = self.job, self.g
        job.prefill, g.model.forward, g.draft_model.prefill, job.receive_sample, g.iterate_draftmodel_mtp_gen = self.originals
        self.targets.clear()
        self.drafts.clear()
        dump(self.directory / 'capture.json', self.meta)


def compare(a, b, directory):
    differences = []
    for field in ('target_chunks','draft_chunks','snapshots','rounds'):
        if a[field] != b[field]:
            differences.append(field)
    if len(a['steps']) != len(b['steps']):
        differences.append('step_count')
    deltas = []
    for i, (sa, sb) in enumerate(zip(a['steps'], b['steps'])):
        assert (sa['token'],sa['prefix_sha'],sa['position']) == (sb['token'],sb['prefix_sha'],sb['position']), 'forced token coordinates differ'
        if sa['logit_sha'] != sb['logit_sha']:
            x = torch.load(directory[0] / f'logits-{i}.pt', weights_only=True).float()
            y = torch.load(directory[1] / f'logits-{i}.pt', weights_only=True).float()
            delta = (x-y).abs()
            deltas.append({'step':i, 'max_abs':delta.max().item(), 'unequal':int((x!=y).sum()),
                           'max_token_id':int(delta.flatten().argmax())})
    if deltas:
        differences.append('logits')
    return {'all_exact':not differences, 'different_fields':differences, 'logit_deltas':deltas}


def qualify(b, g, ids, directory, steps):
    entries, captures = {}, {}
    forced = None
    for name, deferred in [('reference', False), ('baseline', False), ('deferred', True)]:
        g.clear_queue()
        b.reset_trial_state(g)
        g.mtp_deferred_prefill = deferred
        job = b._enqueue_job(g, ids, steps + 8, name, forced=forced)
        cap = Capture(b, g, job, directory / name)
        before = b.kfd_evicted_ms(os.getpid())
        try:
            run_mtp_prefill_wall(g, job, b.sync_all)
            assert job.new_tokens == 0
            b.assert_job_uncached(job, name)
            cap.prefill_snapshot()
            nround = 0
            while job.new_tokens < steps:
                results = g.iterate()
                for r in results:
                    if r.get('stage') == 'error':
                        raise r['error']
                nround += 1
                if nround in (1,2):
                    cap.snapshot(f'round{nround}')
            # Whole rounds are retained; later forced runs use enough reference tokens to cover overshoot.
            if name == 'reference':
                forced = [s['token'] for s in cap.meta['steps']]
            else:
                assert all(s['forced'] for s in cap.meta['steps']), 'continuation escaped forcing'
                assert [s['token'] for s in cap.meta['steps']] == forced
            entries[name] = {'accepted':job.accepted_draft_tokens, 'rejected':job.rejected_draft_tokens,
                             'generated':job.new_tokens, 'draft_stats':job.draft_stats,
                             'kfd_before':before, 'kfd_after':b.kfd_evicted_ms(os.getpid()),
                             'retained_hidden_bytes':job._mtp_deferred.retained_peak_bytes if deferred else 0}
        finally:
            cap.close()
            g.cancel(job)
        captures[name] = cap.meta
        dump(directory / 'qualification-partial.json', entries)
    result = {'runs':entries, 'baseline_vs_deferred':compare(captures['baseline'], captures['deferred'],
                   (directory/'baseline',directory/'deferred')),
              'reference_vs_baseline':compare(captures['reference'], captures['baseline'],
                   (directory/'reference',directory/'baseline'))}
    dump(directory / 'qualification.json', result)
    return result


def walls(b, g, ids, directory, reps):
    from hip_prefill_wall_bench import DeviceMemorySampler
    sampler = DeviceMemorySampler()
    records = []
    specs = [(True, -1, False), (True, -1, True)] + [
        (False, rep, deferred) for rep in range(reps) for deferred in ((False,True) if rep%2==0 else (True,False))]
    try:
        for warm, rep, deferred in specs:
            g.clear_queue()
            b.reset_trial_state(g)
            g.mtp_deferred_prefill = deferred
            job = b._enqueue_job(g, ids, 32, 'wall')
            before = b.kfd_evicted_ms(os.getpid())
            b.sync_all()
            for d in (0,1): torch.cuda.reset_peak_memory_stats(d)
            sampler.reset()
            wall = run_mtp_prefill_wall(g, job, b.sync_all)
            after = b.kfd_evicted_ms(os.getpid())
            b.assert_job_uncached(job, 'wall')
            record = {'warmup':warm, 'rep':rep, 'deferred':deferred, 'prefill_wall_s':wall,
                      'actual_prefill_tokens':len(ids)-1, 'generated':job.new_tokens,
                      'cached_pages':job.cached_pages, 'cached_tokens':job.cached_tokens,
                      'retained_hidden_bytes':job._mtp_deferred.retained_peak_bytes if deferred else 0,
                      'memory':b.memory_snapshot(), 'sampled_vram_peak':sampler.snapshot(), 'rss':rss(),
                      'kfd_before':before, 'kfd_after':after,
                      'kfd_delta':{k:after[k]-before[k] for k in before if not k.startswith('_') and isinstance(before[k],int)}}
            records.append(record)
            print('WALL', json.dumps(record), flush=True)
            dump(directory / 'wall.json', records)
            g.cancel(job)
        # This extra run synchronizes the phase boundary; it is descriptive, not a paired wall trial.
        b.reset_trial_state(g)
        g.mtp_deferred_prefill = True
        job = b._enqueue_job(g, ids, 32, 'phases')
        phases = {}
        orig = job._catch_up_mtp_prefill
        start = None
        def catchup(seq):
            b.sync_all()
            phases['target_s'] = time.perf_counter()-start
            phases['target_memory'] = b.memory_snapshot()
            phases['target_rss'] = rss()
            phases['target_sampled_vram'] = sampler.snapshot()
            sampler.reset()
            t = time.perf_counter()
            orig(seq)
            b.sync_all()
            phases['catchup_s'] = time.perf_counter()-t
            phases['catchup_memory'] = b.memory_snapshot()
            phases['catchup_sampled_vram'] = sampler.snapshot()
        job._catch_up_mtp_prefill = catchup
        b.sync_all()
        sampler.reset()
        start = time.perf_counter()
        phases['inclusive_instrumented_s'] = run_mtp_prefill_wall(g, job, b.sync_all)
        phases['label'] = 'added phase synchronization and memory probes; not uninstrumented benchmark'
        dump(directory / 'phases.json', phases)
        g.cancel(job)
    finally:
        sampler.close()
    return records


def lifecycle(b, g, ids):
    from exllamav3.generator import Job
    passed = []
    def clean(job):
        b.sync_all()
        assert not b.check_reset_complete(g.pagetable)
        assert not g.recurrent_cache and not job._mtp_deferred.chunks
        assert job.mtp_last_hidden is None and job.sequences[0].mtp_carry_hidden is None
        assert len(g.cache.free_list) == g.cache.num_slots
    b.reset_trial_state(g)
    g.mtp_deferred_prefill = True
    job = b._enqueue_job(g, ids, 32, 'cancel')
    g.iterate()
    assert job._mtp_deferred.target_end == 512 and job._mtp_deferred.draft_end == 0
    try:
        g.enqueue(Job(torch.tensor([ids]),max_new_tokens=8))
    except ValueError:
        passed.append('active_second_job_rejected')
    else:
        raise AssertionError('second job accepted')
    g.cancel(job)
    clean(job)
    passed.append('mid_target_cancel_clears_pages_carry_checkpoints_and_slot')
    job = b._enqueue_job(g, ids, 32, 'draft_error')
    original = g.draft_model.prefill
    calls = []
    def fail(input_ids,params):
        calls.append(int(params['cache_seqlens'][0]))
        if len(calls) == 2:
            raise ValueError('injected draft catchup error')
        return original(input_ids,params)
    g.draft_model.prefill = fail
    try:
        try:
            run_mtp_prefill_wall(g,job,b.sync_all)
        except ValueError as e:
            assert str(e) == 'injected draft catchup error'
        else:
            raise AssertionError('error not propagated')
    finally:
        g.draft_model.prefill = original
    assert calls == [0,512]
    clean(job)
    assert not g.num_remaining_jobs()
    passed.append('partial_catchup_error_reaped_without_reusable_target_only_pages')
    job = b._enqueue_job(g,ids,32,'recovery')
    run_mtp_prefill_wall(g,job,b.sync_all)
    assert job._mtp_deferred.complete
    g.cancel(job)
    clean(job)
    passed.append('fresh_request_after_error_completes_full_catchup')
    job = b._enqueue_job(g,ids,32,'target_error')
    original_target = g.model.forward
    def target_error(input_ids,params):
        original_target(input_ids,params)
        raise ValueError('injected target error after device work')
    g.model.forward = target_error
    try:
        try:
            run_mtp_prefill_wall(g,job,b.sync_all)
        except ValueError as e:
            assert str(e) == 'injected target error after device work'
        else:
            raise AssertionError('target error not propagated')
    finally:
        g.model.forward = original_target
    assert job.sequences[0].kv_position == 0
    clean(job)
    passed.append('target_error_at_position_zero_clears_pages_state_and_slot')
    try:
        g.enqueue(job)
    except ValueError:
        passed.append('used_job_rejected_even_when_position_and_carry_are_zero')
    else:
        raise AssertionError('reused job with old DMA staging accepted')
    g.mtp_deferred_prefill = False
    job = b._enqueue_job(g,ids,32,'cached_baseline')
    run_mtp_prefill_wall(g,job,b.sync_all)
    g.cancel(job)
    g.mtp_deferred_prefill = True
    try:
        g.enqueue(Job(torch.tensor([ids]),max_new_tokens=8))
    except ValueError:
        passed.append('cached_prefix_rejected')
    else:
        raise AssertionError('prefix reuse accepted')
    b.reset_trial_state(g)
    return passed


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--lengths', default='1024,12288')
    p.add_argument('--controls', action='store_true')
    p.add_argument('--wall', action='store_true')
    p.add_argument('--lifecycle', action='store_true')
    p.add_argument('--steps', type=int, default=32)
    p.add_argument('--reps', type=int, default=3)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    owner = subprocess.run(['fuser','/dev/kfd'], capture_output=True, text=True)
    assert owner.returncode == 1 and not owner.stdout.strip(), f'GPU already owned: {owner}'
    os.environ['HIP_VISIBLE_DEVICES'] = os.environ['ROCR_VISIBLE_DEVICES'] = '0,1'
    os.environ.pop('CUDA_VISIBLE_DEVICES', None)
    os.environ['EXL3_MOE_SYNC_FREE_COUNT'] = os.environ['EXL3_PREFILL_ASYNC_UPLOADS'] = '1'
    os.environ.pop('EXL3_MTP_DEFERRED_PREFILL', None)
    if args.controls:
        os.environ['EXL3_DIAGNOSTIC_GDN_FIXED_ORDER'] = '1'
    else:
        os.environ.pop('EXL3_DIAGNOSTIC_GDN_FIXED_ORDER', None)
    spec = importlib.util.spec_from_file_location('validated', HELPER)
    b = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(b)
    global torch, run_mtp_prefill_wall
    torch = b.torch
    from hip_mtp_prefill_wall import run_mtp_prefill_wall
    if args.controls:
        from hip_stable_router_control import install
        install()
    assert torch.cuda.device_count() == 2
    cursor, error = b.journal_cursor()
    assert cursor and not error, error
    report = {'controls':args.controls, 'pid':os.getpid(), 'cursor':cursor, 'lengths':{},
              'capture_method':'qualification-only retained export clones; above 12K synchronous CPU copies avoid extra GPU history; absent from wall trials',
              'source_sha':b.git_head(ROOT), 'source_dirty':subprocess.check_output(['git','status','--short'], cwd=ROOT,text=True),
              'source_hashes':{str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest()
                               for f in [ROOT/'exllamav3/generator/job.py', ROOT/'exllamav3/generator/generator.py',
                                         ROOT/'exllamav3/generator/mtp_deferred.py', Path(__file__)]},
              'env':{k:v for k,v in os.environ.items() if k.startswith(('EXL3','HIP_','ROCR_','CUDA_','TORCH_','TRITON_','PYTORCH_','HSA_'))}}
    dump(args.out/'report.json',report)
    try:
        config, model, cache, tokenizer, g = load(b)
        report['identity'] = {'target_class':type(model).__name__, 'draft_class':type(g.draft_model).__name__,
            'target_map':b.module_device_map(model), 'draft_map':b.module_device_map(g.draft_model),
            'memory_after_load':b.memory_snapshot(), 'rss':rss(), 'kfd_after_load':b.kfd_evicted_ms(os.getpid()),
            'gpu_properties':[str(torch.cuda.get_device_properties(d)) for d in (0,1)],
            'logical_vocab':tokenizer.actual_vocab_size, 'torch':torch.__version__, 'hip':torch.version.hip,
            'config':{'cache_tokens':393216,'generator_max_batch_size':1,'load_cache_max_batch_size':4,'max_history':3,'mtp_tokens':3,'chunk':512,'target_split':[30,30],'draft_split':[3,0]}}
        corpus = b.build_corpus()
        if args.lifecycle:
            report['lifecycle'] = lifecycle(b,g,b.prompt_for(corpus,1024))
            dump(args.out/'report.json',report)
        for length in map(int,args.lengths.split(',')):
            directory = args.out / str(length)
            directory.mkdir()
            ids = b.prompt_for(corpus,length)
            torch.save(torch.tensor(ids),directory/'prompt.pt')
            result = qualify(b,g,ids,directory,args.steps)
            report['lengths'][str(length)] = result
            dump(args.out/'report.json',report)
            print('QUALIFICATION',length,json.dumps(result),flush=True)
            if not result['baseline_vs_deferred']['all_exact'] or not result['reference_vs_baseline']['all_exact']:
                report['blocked'] = 'numerical mismatch; no performance qualification'
                break
            if args.wall:
                walls(b,g,ids,directory,args.reps)
    except BaseException:
        report['error'] = traceback.format_exc()
        raise
    finally:
        report['journal'] = b.journal_after(cursor,args.out/'kernel.log')
        report['rss_final'] = rss()
        dump(args.out/'report.json',report)


if __name__ == '__main__':
    main()
