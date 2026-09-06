"""Offline artifact summary. Run with all GPU visibility blank, never beside timed trials."""
import argparse
import json
from pathlib import Path
import statistics
import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument('root',type=Path)
    args = p.parse_args()
    out = {}
    for report_path in args.root.glob('*/report.json'):
        report = json.loads(report_path.read_text())
        entry = {'controls':report['controls'], 'journal':report.get('journal'),
                 'error':report.get('error'), 'lifecycle':report.get('lifecycle'), 'lengths':{}}
        out[report_path.parent.name] = entry
        for qual_path in report_path.parent.glob('*/qualification.json'):
            root = qual_path.parent
            qual = json.loads(qual_path.read_text())
            length = root.name
            q = {'comparisons':{k:v for k,v in qual.items() if k != 'runs'},
                 'decode':{k:{n:v[n] for n in ('accepted','rejected','generated','retained_hidden_bytes')} for k,v in qual['runs'].items()}}
            entry['lengths'][length] = q
            captures = {name:json.loads((root/name/'capture.json').read_text()) for name in ('reference','baseline','deferred')}
            q['coverage'] = {'steps':len(captures['baseline']['steps']),
                'prompt_chunks':len(captures['baseline']['target_chunks']),
                'rounds':len(captures['baseline']['rounds']),
                'snapshot_tensor_counts':{k:sum(isinstance(v,dict) and 'sha' in v for v in state.values())
                                          for k,state in captures['baseline']['snapshots'].items()}}
            for left,right,key in [('reference','baseline','reference_vs_baseline'),('baseline','deferred','baseline_vs_deferred')]:
                if qual[key]['all_exact']:
                    continue
                ac,bc = captures[left],captures[right]
                assert [(c['start'],c['end'],c['ids_sha']) for c in ac['target_chunks']] == [
                    (c['start'],c['end'],c['ids_sha']) for c in bc['target_chunks']]
                first = next((i for i,(a,b) in enumerate(zip(ac['target_chunks'],bc['target_chunks'])) if a != b),None)
                detail = {'first_different_target_chunk':first,
                          'snapshot_differences':{label:[name for name in state if state[name] != bc['snapshots'][label][name]]
                                                   for label,state in ac['snapshots'].items()}}
                if first is not None:
                    start,end = ac['target_chunks'][first]['start'],ac['target_chunks'][first]['end']
                    x = torch.load(root/left/'target_chunks.pt',weights_only=True)[:,start:end].float()
                    y = torch.load(root/right/'target_chunks.pt',weights_only=True)[:,start:end].float()
                    delta = (x-y).abs()
                    coord = (x!=y).nonzero()[0].tolist()
                    detail['first_target_export_delta'] = {'start':start,'end':end,'max_abs':delta.max().item(),
                        'unequal':int((x!=y).sum()),'first_coordinate':coord,
                        'left_value':x[tuple(coord)].item(),'right_value':y[tuple(coord)].item(),
                        'before_any_draft_prefill':start == 0}
                q[key+'_isolation'] = detail
            path = root/'wall.json'
            if path.exists():
                trials = [t for t in json.loads(path.read_text()) if not t['warmup']]
                baseline = {r['rep']:r for r in trials if not r['deferred']}
                deferred = {r['rep']:r for r in trials if r['deferred']}
                assert baseline.keys() == deferred.keys()
                pairs = [{'rep':i,'baseline_s':baseline[i]['prefill_wall_s'],'deferred_s':deferred[i]['prefill_wall_s'],
                          'wall_reduction':1-deferred[i]['prefill_wall_s']/baseline[i]['prefill_wall_s'],
                          'throughput_gain':baseline[i]['prefill_wall_s']/deferred[i]['prefill_wall_s']-1,
                          'baseline_kfd_delta':baseline[i]['kfd_delta'],'deferred_kfd_delta':deferred[i]['kfd_delta']}
                         for i in baseline]
                q['wall'] = {'pairs':pairs,
                    'baseline_median_s':statistics.median(r['baseline_s'] for r in pairs),
                    'deferred_median_s':statistics.median(r['deferred_s'] for r in pairs),
                    'median_paired_wall_reduction':statistics.median(r['wall_reduction'] for r in pairs),
                    'median_paired_throughput_gain':statistics.median(r['throughput_gain'] for r in pairs),
                    'actual_prefill_tokens':int(length)-1,
                    'memory_peaks':{str(d):{name:max(r['memory'][str(d)][name] for r in trials)
                                           for name in ('torch_peak_b','torch_reserved_b')} for d in (0,1)},
                    'sampled_vram_peaks':{d:max(r['sampled_vram_peak'][d] for r in trials) for d in trials[0]['sampled_vram_peak']},
                    'rss': [r['rss'] for r in trials]}
                phase_path = root/'phases.json'
                if phase_path.exists(): q['phases'] = json.loads(phase_path.read_text())
    (args.root/'analysis.json').write_text(json.dumps(out,indent=2))
    print(json.dumps({k:{n:{'exact':q['comparisons']['baseline_vs_deferred']['all_exact'],
                           'wall':q.get('wall',{}).get('pairs')} for n,q in v['lengths'].items()} for k,v in out.items()},indent=2))


if __name__ == '__main__':
    main()
