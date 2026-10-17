"""Compile new A2 identities from the immutable, previously reviewed protocol."""
import argparse
import copy
import datetime as dt
import hashlib
import json
from pathlib import Path

from experiments.beauty_extensions.common import REPO, sha256, write_json
from experiments.launch_20261017 import queue
from experiments.launch_20261017.run import template_plan_sha256
from .controller import STAGES

PREFIX='ga-20261018-'


def job(name):
    return PREFIX+name+'-fd-recovery-v1'


def replace(value, changes):
    if isinstance(value,str):
        for a,b in changes:
            value=value.replace(a,b)
        return value
    if isinstance(value,list):return [replace(x,changes) for x in value]
    if isinstance(value,dict):return {k:replace(v,changes) for k,v in value.items()}
    return value


def specs(deployment, scene, output, remote):
    shared=scene['config'];oldroot=shared['snapshot']['root'];newroot=deployment['root']
    original={s['id']:s for s in shared['stages']}
    oldcapture=original['sportsa2-quantize']['request']['capture_version']
    capture=deployment['manifest']['source_commit']+'/'+deployment['manifest']['snapshot_sha256']
    changes=[(oldroot,newroot),(oldcapture,capture)]
    changes += [(original[name]['request']['job_id'],job(name)) for name in STAGES]
    templates={}
    for name in sorted(STAGES):
        base=json.loads((REPO/'manifests/grid-a-launch-20261017/compiled/specs'/(name+'.json')).read_text())
        base.pop('launch_review',None)
        value=replace(base,changes)
        value['capture_version']=capture
        value['prerequisite_capture_versions']={}
        if name=='sportsa2-quantize':
            value['prerequisite_capture_versions'][original['sportsa2-encode']['request']['job_id']]=oldcapture
            value['fd_engineering']={'purpose':'A2 DataLoader IPC/FD engineering adaptation','worker_hard_seconds':3300}
        templates[name]=value
        write_json(output/'specs'/(name+'.json'),value)
    plan=dict(workflows={'Sports-A2':dict(modes=sorted({x['phase'] for x in templates.values()}),results_review='NOT_PREAPPROVED')},
              template_plan_sha256=[template_plan_sha256(x) for x in templates.values()],
              source_python_sha256={r['path']:r['sha256'] for r in deployment['manifest']['files'] if r['path'].endswith('.py')},
              only_fixed_seed=42,only_recovery=True,encode_reused=original['sportsa2-encode']['request']['job_id'],
              original_absolute_queue_deadline=shared['continuation']['original_absolute_queue_deadline'])
    write_json(output/'plan.json',plan)
    write_json(output/'spec-index.json',dict(remote_root=remote,capture_version=capture,source_deployment=deployment,
                                          specs={name:str(output/'specs'/(name+'.json')) for name in templates}))
    return templates,plan


def compile_queue(deployment, scene, output, remote, review, frozen):
    shared=scene['config'];old={s['id']:s for s in shared['stages']}
    templates={name:json.loads((output/'specs'/(name+'.json')).read_text()) for name in STAGES}
    review_record=dict(path=remote+'/launch-review.json',sha256=sha256(review))
    review_value=json.loads(review.read_text())
    assert review_value['status']=='SOURCE_AND_LAUNCH_ADMITTED_CONDITIONALLY'
    assert review_value['fd_engineering_status']=='FD_ENGINEERING_EQUIVALENCE_PASS'
    plan=json.loads((output/'plan.json').read_text())
    assert review_value['source_python_sha256']==plan['source_python_sha256']
    stages=[]
    capture=deployment['manifest']['source_commit']+'/'+deployment['manifest']['snapshot_sha256']
    changes=[(shared['snapshot']['root'],deployment['root'])]
    changes += [(old[name]['request']['job_id'],job(name)) for name in STAGES]
    for name in [s['id'] for s in shared['stages'] if s['id'] in STAGES]:
        stage=copy.deepcopy(old[name]);stage['gates']=replace(stage['gates'],changes)
        stage['depends_on']=[x for x in stage['depends_on'] if x in STAGES]
        template=templates[name]
        template['launch_review']=review_record
        write_json(output/'specs'/(name+'.json'),template)
        request=stage['request'];request.update(job_id=job(name),capture_version=capture,cwd=deployment['root'])
        request['argv']=[request['argv'][0],'-B','-m','experiments.launch_20261017.run',template['phase'],
                         '--spec',remote+'/specs/'+name+'.json']
        request['env']['PYTHONPATH']=deployment['root']
        request['inputs']=[dict(path=deployment['root']+'/DEPLOYMENT.json',sha256=deployment['manifest_sha256']),
            dict(path=remote+'/specs/'+name+'.json',sha256=sha256(output/'specs'/(name+'.json'))),review_record]
        for field in ('config','sid','checkpoint','training_protocol'):
            if isinstance(template.get(field),dict) and 'path' in template[field]:
                request['inputs'].append({k:template[field][k] for k in ('path','sha256')})
        if name=='sportsa2-quantize':
            request['timeout_seconds']=3420
            request['inputs'].append({k:scene['embedding'][k] for k in ('path','sha256')})
            stage['gates'].append({k:scene['embedding'][k] for k in ('path','sha256')})
            stage['gates'].append(dict(path=remote+'/launch-review.json',sha256=review_record['sha256'],
                json_equals={'fd_engineering_status':'FD_ENGINEERING_EQUIVALENCE_PASS'}))
        for dep in stage['depends_on']:
            root=shared['runner']['job_root']+'/'+job(dep)
            stage['gates'] += [dict(path=root+'/status.json',json_equals={'status':'SUCCEEDED','exit_code':0,'cleanup.verified':True}),
                dict(path=root+'/output/RESULT.json',json_equals={'capture_version':capture,'workflow':'Sports-A2'})]
        if name=='sportsa2-metadata':
            stage['gates'].append(dict(path=shared['runner']['job_root']+'/'+job('sportsa2-quantize')+'/output/quantizer-strict-load.json',
                json_equals={'strict_load':True,'state_exact_to_trained_model':True,'global_step':30}))
        stages.append(stage)
    config=copy.deepcopy(shared)
    config.pop('continuation',None);config.pop('controller_snapshot',None)
    config.update(queue_id='ga-20261018-a2-fd-queue-v1',state_dir=remote+'/queue-state',
                  snapshot=dict(root=deployment['root'],commit=deployment['manifest']['source_commit']),
                  blocked_workflows={x:'HISTORICAL_SUCCESS_REVIEW_ONLY_NO_NEW_COMPUTATION' for x in queue.WORKFLOWS-{'sports-a2'}},stages=stages)
    actual=scene['accounting'];config['gpu_budget_seconds']=shared['gpu_budget_seconds']-actual['used_gpu_request_seconds']
    config['budget_groups']={k:cap-actual['groups'][k] for k,cap in shared['budget_groups'].items()}
    absolute=shared['continuation']['original_absolute_queue_deadline']
    config['max_queue_seconds']=min(shared['max_queue_seconds'],int((dt.datetime.fromisoformat(absolute)-dt.datetime.now(dt.timezone.utc)).total_seconds()))
    config['storage_roots'] += [remote,shared['runner']['job_root']+'/'+config['queue_id']]
    config['storage_roots'] += [shared['runner']['job_root']+'/'+job(name) for name in STAGES]
    config['recovery']=dict(shared_config_path=scene['shared_config_path'],shared_state_path=shared['state_dir']+'/queue-status.json',
        frozen_evidence=frozen,original_absolute_queue_deadline=absolute,historical_stage_gpu_seconds=actual['used_gpu_request_seconds'],
        prior_gpu_request_seconds=shared['prior_gpu_request_usage']['seconds'],source_review_status='FD_ENGINEERING_EQUIVALENCE_PASS',
        terminal_controller_ids=['ga-20261017-queue-v1','ga-20261018-shared-queue-v1'],
        original_failure_job='ga-20261017-sportsa2-quantize-v1',
        repair_allowance='historical 3/3 exhausted; one separately authorized FD recovery, no formal retry')
    path=output/'queue-config.json';write_json(path,config)
    write_json(output/'queue-validation-local.json',queue.validate(config,path.resolve()))
    return config


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['specs','queue']);p.add_argument('--deployment',type=Path,required=True)
    p.add_argument('--scene',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--remote-root',required=True);p.add_argument('--review',type=Path);p.add_argument('--frozen',type=Path)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    d=json.loads(a.deployment.read_text());s=json.loads(a.scene.read_text())
    s['shared_config_path']='/home/bwwu/grid-audits/sid-extensions-p0/grid-a-launch-20261017/shared-20261018/compiled/queue-config-d5528b2c185395bd.json'
    if a.mode=='specs':specs(d,s,a.output,a.remote_root)
    else:compile_queue(d,s,a.output,a.remote_root,a.review,json.loads(a.frozen.read_text()))


if __name__=='__main__':main()
