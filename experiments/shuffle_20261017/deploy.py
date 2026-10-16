"""Task-local requests using the accepted runner and immutable formal snapshot."""
import argparse
import json
from pathlib import Path
import sys

from experiments.formal_20261015 import deploy as base
from experiments.beauty_extensions.common import sha256,write_json

SUPPORT=Path('/home/addaswsw/recommend-support/grid-shuffle-20261017')
REMOTE='/home/bwwu/grid-audits/sid-extensions-p0/grid-shuffle-20261017'
base.SUPPORT=SUPPORT
base.REMOTE=REMOTE
base.SOURCE_DIRS=[*base.SOURCE_DIRS,'experiments/shuffle_20261017']


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['preflight','evaluate','attention','identity-check'])
    parser.add_argument('--spec',required=True,type=Path)
    parser.add_argument('--job',required=True)
    parser.add_argument('--gpu',type=int,choices=[0,1])
    parser.add_argument('--timeout',required=True,type=int)
    args=parser.parse_args()
    assert args.job.startswith('gs-20261017-')
    cap={'preflight':900,'evaluate':1800,'attention':1200,'identity-check':120}[args.mode]
    assert 60<=args.timeout<=cap
    spec=json.loads(args.spec.read_text())
    assert spec['domain'] in ('toys','sports') and spec['route'] in ('A','B')
    assert spec['shuffle_n'] in (0,3,4) and spec['generation_k']==10
    sys.path.insert(0,str(base.TRANSPORT))
    from transport import Transport
    import gridctl
    t=Transport('new-server');t.identity();gpu=''
    if args.mode!='identity-check':
        assert args.gpu is not None;gpu=base.GPUS[args.gpu]
        observation=json.loads(t.python("import subprocess,json,shutil;print(json.dumps({'apps':subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader'],text=True),'devices':subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.free','--format=csv,noheader,nounits'],text=True),'disk_free':shutil.disk_usage('/home/bwwu').free}))"))
        write_json(SUPPORT/(args.job+'-resource.json'),observation)
        assert not any(row.split(',')[0].strip()==gpu for row in observation['apps'].splitlines()),'GPU occupied'
        assert next(int(row.split(',')[1]) for row in observation['devices'].splitlines() if row.split(',')[0].strip()==gpu)>=14336
        assert observation['disk_free']>=20*1024**3
    root,manifest,manifest_sha=base.snapshot(t)
    remote_spec=REMOTE+'/specs/'+sha256(args.spec)+'.json';t.put(args.spec,remote_spec)
    inputs=[{'path':remote_spec,'sha256':sha256(args.spec)},{'path':root+'/DEPLOYMENT.json','sha256':manifest_sha}]
    for key in ('config','sid','checkpoint','test_index','attention_cohort','nonauthor_gate','evaluation_result','evaluation_status'):
        if isinstance(spec.get(key),dict) and set(spec[key])>={'path','sha256'}:
            inputs.append({k:spec[key][k] for k in ('path','sha256')})
    module='experiments.shuffle_20261017.preflight' if args.mode=='preflight' else 'experiments.formal_20261015.run'
    argv=['/data/bwwu/envs/grid-20261009-e61baf3/bin/python','-B','-m',module]
    if args.mode!='preflight':argv.append(args.mode)
    argv+=['--spec',remote_spec]
    request={'schema':1,'job_id':args.job,'host':t.profile['hostname'],'user':t.profile['user'],'uid':t.profile['uid'],
        'namespace':t.profile['namespace'],'deployment_sha256':t.profile['deployment_sha256'],
        'capture_version':manifest['source_commit']+'/'+manifest['snapshot_sha256'],
        'argv':argv,'cwd':root,'timeout_seconds':args.timeout,
        'env':{'CUDA_VISIBLE_DEVICES':gpu,'OMP_NUM_THREADS':'4','TF_CPP_MIN_LOG_LEVEL':'3',
            'TF_NUM_INTRAOP_THREADS':'2','TF_NUM_INTEROP_THREADS':'2','PYTHONDONTWRITEBYTECODE':'1',
            'PYTHONPATH':root,'HF_HUB_OFFLINE':'1','WANDB_MODE':'disabled','TOKENIZERS_PARALLELISM':'false'},
        'inputs':inputs}
    path=SUPPORT/(args.job+'-request.json')
    if path.exists():assert json.loads(path.read_text())==request,'Original ID must be queried'
    write_json(path,request);receipt=gridctl.submit(t,path)
    write_json(SUPPORT/(args.job+'-submission.json'),receipt)
    print(json.dumps({'deployment':root,'receipt':receipt},ensure_ascii=False))


if __name__=='__main__':main()
