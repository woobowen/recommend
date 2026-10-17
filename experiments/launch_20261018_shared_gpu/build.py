"""Bind a scheduling-only continuation after the original controller is empty."""
import argparse
import copy
import datetime as dt
import hashlib
import json
from pathlib import Path
import sys

from experiments.launch_20261017.queue import canonical,validate


def dump(path,value):
    data=(json.dumps(value,ensure_ascii=False,indent=2)+'\n').encode()
    if path.exists():assert path.read_bytes()==data,'immutable output differs'
    else:path.write_bytes(data)


def build(old,deployment,stop,remote_root):
    assert stop['status']=='OLD_CONTROLLER_QUIESCED' and stop['old_lock_released']
    assert not stop['after']['workgroup_live'] and stop['after']['cleanup_verified']
    config=copy.deepcopy(old)
    config['queue_id']='ga-20261018-shared-queue-v1'
    config['state_dir']=remote_root+'/queue-state'
    config['controller_snapshot']={'root':deployment['root'],'commit':deployment['manifest']['source_commit']}
    config['approved_gpu_uuids']=sorted({g for s in old['stages'] for g in s.get('gpu_uuids',[])})
    absolute=dt.datetime.fromisoformat(stop['original_absolute_queue_deadline'])
    remaining=int((absolute-dt.datetime.now(dt.timezone.utc)).total_seconds())
    assert remaining>120
    config['max_queue_seconds']=min(old['max_queue_seconds'],remaining)
    config['continuation']={k:stop[k] for k in ['old_config_path','old_config_sha256','old_queue_state_path',
        'old_queue_state_sha256','old_queue_request_path','old_queue_request_sha256','original_absolute_queue_deadline']}
    config['continuation']['old_queue_job_id']=old['queue_id']
    config['continuation']['submitted_stage_ids']=[name for name,v in stop['final_inventory'].items() if v['submitted']]
    config['storage_roots']+=list(dict.fromkeys([remote_root,old['runner']['job_root']+'/'+old['queue_id'],
        old['runner']['job_root']+'/'+config['queue_id']]))
    for s in config['stages']:
        if s['id'] in config['continuation']['submitted_stage_ids'] or not s.get('gpu_uuids'):continue
        kind=s['kind']
        peak,margin=(20480,2048) if kind in ('train','admission') else ((6144,2048) if kind=='rkmeans' else (2048,1024))
        s['resource_policy']={'allow_shared':True,'estimated_peak_mib':peak,'safety_margin_mib':margin,
            'min_runtime_free_mib':512,'max_prelaunch_utilization_percent':50,'sample_seconds':2,'required_samples':3}
        if len(s['gpu_uuids'])==1:
            preferred=s['gpu_uuids'][0]
            s['gpu_candidates']=[preferred,*[g for g in config['approved_gpu_uuids'] if g!=preferred]]
    return config


def main():
    p=argparse.ArgumentParser();p.add_argument('--old-config',type=Path,required=True)
    p.add_argument('--deployment',type=Path,required=True);p.add_argument('--stop-receipt',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--remote-root',required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    old=json.loads(args.old_config.read_text());deployment=json.loads(args.deployment.read_text());stop=json.loads(args.stop_receipt.read_text())
    assert hashlib.sha256(args.old_config.read_bytes()).hexdigest()==stop['old_config_sha256']
    config=build(old,deployment,stop,args.remote_root)
    path=args.output/'queue-config.json';dump(path,config)
    summary=validate(config,path.resolve())
    dump(args.output/'queue-validation.json',summary)
    print(json.dumps({'config':str(path),'config_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'summary':summary,
        'inherited_jobs':len(config['continuation']['submitted_stage_ids']),'original_deadline':config['continuation']['original_absolute_queue_deadline']}))


if __name__=='__main__':main()
