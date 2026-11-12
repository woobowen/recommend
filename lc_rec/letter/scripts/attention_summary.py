#!/usr/bin/env python3
"""Rebuild the final lightweight summary from collected raw receipts and comparisons."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess


def main():
    p=argparse.ArgumentParser();p.add_argument('--evidence',required=True);a=p.parse_args()
    root=Path(a.evidence);jobs=root/'jobs'
    def read(path):return json.loads(path.read_text())
    report={'task':'A0 layer28 attention backward 20261112','A0_FORMAL_NORMAL':'NOT_STARTED',
            'tolerance':{'atol':1e-6,'rtol':1e-3,'exact':['scheduler','step','RNG','discrete']},
            'gpu_jobs':[],'comparisons':{},'boundary_pairs':{},'isolate':{},'installations':[],
            'source_of_claims':'raw job request/status/cleanup, unchanged comparator, boundary tensors and non-author CPU audit'}
    for d in sorted(jobs.glob('*')):
        req=read(d/'job/request.json');status=read(d/'job/status.json')
        if not req['env'].get('CUDA_VISIBLE_DEVICES'):continue
        r=read(d/'run/pipeline.json')
        row={k:r[k] for k in ['job_id','run_commit','kind','status','gpu_device_seconds','wall_seconds','exit_code']}
        row.update(runner_status=status['status'],runner_exit=status.get('exit_code'),
                   cleanup_verified=read(d/'job/cleanup.json')['verified'])
        row['ranks']=[]
        for rank in [0,1]:
            f=d/('run/training/rank%d.json'%rank)
            if f.exists():
                meta=read(f);ev=[json.loads(x) for x in (f.parent/('events.rank%d.jsonl'%rank)).read_text().splitlines()]
                row['ranks'].append({'rank':rank,'global_step':meta['global_step'],'versions':meta['versions'],
                    'peak_allocated_bytes':meta['peak_allocated_bytes'],'peak_reserved_bytes':meta['peak_reserved_bytes'],
                    'attention':meta['diagnostic_attention'],
                    'updates':[{k:x[k] for k in ['step','lr','grad_norm']} for x in ev if x['event']=='optimizer_step'],
                    'micro_loss':[x['loss'] for x in ev if x['event']=='micro_loss']})
        report['gpu_jobs'].append(row)
    for f in sorted(jobs.glob('*/run/numeric_comparison.json')):
        r=read(f);report['comparisons'][f.parents[1].name]={'status':r['status'],'ranks':[]}
        for row in r['ranks']:
            stats={'rank':row['rank'],'status':row['status'],'state_failures':len(row.get('state_failures',[])),
                   'error':row.get('error'),'all_finite':row.get('all_model_and_optimizer_tensors_finite'),
                   'tensor_comparisons':len(row.get('tensor_comparisons',[])),
                   'nonexact_tensors':sum(not x.get('exact_equal') for x in row.get('tensor_comparisons',[])),
                   'events':row.get('events')}
            report['comparisons'][f.parents[1].name]['ranks'].append(stats)
    for f in sorted(jobs.glob('*/run/boundary_comparison.json')):
        r=read(f);report['boundary_pairs'][f.parents[1].name]={'left':r['left'],'right':r['right'],'ranks':[]}
        for row in r['ranks']:
            report['boundary_pairs'][f.parents[1].name]['ranks'].append({'rank':row['rank'],'tensors':row['tensor_count'],
                'nonexact':{n:{k:v.get(k) for k in ['differing_elements','outside_fixed_tolerance_elements','max_abs','raw_witness']} for n,v in row['rows'].items() if not v['exact_equal']},
                'correspondence_exact':all(x['exact_equal'] for x in row['boundary_correspondence'].values())})
    isolated=jobs/'letter-attention-20261112-cpu-isolate/run/isolate_comparison.json'
    if isolated.exists():
        r=read(isolated)
        for rank in r['ranks']:
            modes={}
            for mode,label in [(False,'default'),(True,'det')]:
                rows=[x for proc in rank['process_reports'] for x in proc['rows'] if x['mode']==mode]
                comps=[x for x in rank['cross_process'] if x['mode']==label]
                modes[label]={'repeats':len(rows),'distinct_hashes':{k:len({x['hashes'][k] for x in rows}) for k in ['out','dQ','dK','dV']},
                    'max_dQ_abs_diff':max(x['tensors']['dQ']['max_abs'] for x in comps),
                    'dQ_violations':sum(x['tensors']['dQ']['outside_fixed_tolerance_elements'] for x in comps),
                    'median_unprofiled_backward_ms':statistics.median(x['backward_ms'] for x in rows if x['repeat']>0),
                    'peak_allocated_bytes':max(x['peak_allocated'] for x in rows),
                    'peak_reserved_bytes':max(x['peak_reserved'] for x in rows)}
            report['isolate'][str(rank['rank'])]=modes
    receipts=list(jobs.glob('*/run/final_resources.json'))
    if receipts:
        resource=read(receipts[-1]);report['gpu_device_minutes']=resource['charged_gpu_device_minutes']
        report['storage']=resource['storage'];report['final_observed_utc']=resource['utc']
    root.joinpath('final_summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'gpu_jobs':len(report['gpu_jobs']),'comparisons':{k:v['status'] for k,v in report['comparisons'].items()},'summary':str(root/'final_summary.json')}))


if __name__=='__main__':main()
