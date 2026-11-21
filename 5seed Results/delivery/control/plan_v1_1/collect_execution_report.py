"""Snapshot both rounds and validate completed result files without mutating runs."""
import csv
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time
from datetime import datetime,timezone,timedelta
import torch

ROOT=Path(__file__).resolve().parents[2]
report={'time':time.time(),'timestamp_beijing':datetime.now(timezone(timedelta(hours=8))).isoformat(timespec='seconds'),
        'jobs':[],'gpu':subprocess.check_output(['nvidia-smi',
        '--query-gpu=index,memory.used,memory.total,utilization.gpu','--format=csv,noheader'],text=True).strip()}
def read(path):
    return json.loads(path.read_text()) if path.exists() else None
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
def proc(pid):
    if not pid:return None
    try:
        s=Path(f'/proc/{pid}/stat').read_text().split()
        return dict(state=s[2],ppid=int(s[3]),sid=int(s[5]),tty_nr=int(s[6]),stdin=os.readlink(f'/proc/{pid}/fd/0'))
    except FileNotFoundError:return None
for groupname in ['first_round_20261104','second_round_20261104']:
    group=ROOT/'codex'/groupname
    if not group.exists():continue
    for status_path in sorted(group.glob('T*_status.json')):
        job=read(status_path)
        out=Path(job['output'])
        job.update(group=groupname,progress=read(out/'progress.json'),
            result=read(out/'stage_result.json'),provenance=read(out/'input_provenance.json'),
            pid_process=proc(job.get('pid')),supervisor_process=proc(job.get('supervisor_pid')))
        if out.exists():
            rows=[]
            for path in sorted(out.rglob('metrics.csv')):
                rows.extend(list(csv.DictReader(path.open())))
            vals=[r for r in rows if r.get('val/recall@5')]
            if vals:
                best=max(vals,key=lambda r:float(r['val/recall@5']))
                job['validation']={'checks_completed':len(vals),'best_step_zero_based_csv':int(best['step']),
                    'best_metrics':{k:float(v) for k,v in best.items() if k.startswith('val/') and v},
                    'latest_metrics':{k:float(v) for k,v in vals[-1].items() if k.startswith('val/') and v},
                    'checks_since_best':len(vals)-1-vals.index(best)}
            checkpoints=sorted((out/'checkpoints').glob('*.ckpt'))
            job['checkpoint_files']=[str(p) for p in checkpoints]
            source=read(out/'source_manifest.json')
            if source:
                core={p:h for p,h in source['files'].items() if p.startswith(('src/','configs/','codex/sid_order_launch/'))
                      or p in ['codex/plan_v1_1/run_task.py','codex/plan_v1_1/protocol.py','codex/plan_v1_1/attention.py']}
                assert all(sha(ROOT/p)==h for p,h in core.items())
                job['core_source_sha256']=hashlib.sha256(json.dumps(core,sort_keys=True).encode()).hexdigest()
            result=job['result']
            if result and result.get('completed'):
                assert job['state']=='completed' and job['exit_code']==0
                assert sha(Path(result['checkpoint']))==result['checkpoint_sha256']
                expected=['mlp'] if job['variant']=='mlp' else ['unshuffled','shuffle_3','shuffle_4']
                assert set(result['test_metrics'])==set(expected)
                frozen=read(out/'sample_manifest.json')
                expected_ids={u['user_id'] for u in frozen['users']}
                audits={}
                for mode in expected:
                    test=read(out/f'test_{mode}.json')
                    assert test['full_test_set'] and not test['attention_export']
                    assert test['checkpoint_sha256']==result['checkpoint_sha256']
                    assert all(math.isfinite(v) for v in test['metrics'][0].values())
                    folder=out/'attention_analysis'/mode
                    manifest=read(folder/'manifest.json')
                    assert manifest['users']==100 and manifest['full_test_subset_check']
                    assert manifest['attention_switch_prediction_check']
                    assert {r['user_id'] for r in manifest['files']}==expected_ids
                    assert len(list(folder.glob('user_*.pt.gz')))==100
                    for item in manifest['files']:
                        path=folder/item['file']
                        assert sha(path)==item['sha256']
                        with gzip.open(path,'rb') as stream:
                            data=torch.load(stream,weights_only=False,map_location='cpu')
                        assert data['user_id']==item['user_id'] and data['checkpoint_sha256']==result['checkpoint_sha256']
                        assert all(len(v)==4 for v in data['attention'].values())
                        assert all(t.shape[1]==6 and torch.isfinite(t).all() for v in data['attention'].values() for t in v)
                    audits[mode]={'users':100,'all_files_hash_and_read_verified':True}
                job['completion_audit']={'passed':True,'checkpoint_sha256_verified':True,'attention':audits}
        report['jobs'].append(job)
cores={j['core_source_sha256'] for j in report['jobs'] if 'core_source_sha256' in j}
assert len(cores)==1
report['all_started_tasks_core_source_identical']=True
target=ROOT/'codex/plan_v1_1/execution_report_latest.json'
target.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'report':str(target),'timestamp_beijing':report['timestamp_beijing'],
    'jobs':[dict(task_id=j['task_id'],state=j['state'],progress=j['progress'],
        validation=j.get('validation'),completion_audit=j.get('completion_audit')) for j in report['jobs']]},ensure_ascii=False,indent=2))
