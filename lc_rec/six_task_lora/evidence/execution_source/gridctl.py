"""Explicit-host client for task-local deployments; old clients are untouched."""
import argparse
import datetime as dt
from zoneinfo import ZoneInfo
import fcntl
import hashlib
import json
from pathlib import Path
import re
import shlex
import tarfile
import transport

ROOT=transport.ROOT

def valid(job):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,90}',job): raise ValueError('invalid job ID')
    return job

def state_update(target,job,result):
    with (ROOT/'.state.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        state=json.loads((ROOT/'STATE.json').read_text())
        state['jobs'][target+'/'+job]=dict(result,target=target,job_id=job)
        state['last_updated']=dt.datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()
        transport.atomic(ROOT/'STATE.json',state)

def deploy(t):
    identity=t.identity();profile=t.profile
    if not profile.get('remote_support'): raise ValueError('remote task root not yet verified/set')
    base=profile['remote_support']
    deployment=dict(hostname=identity['hostname'],uid=identity['uid'],user=identity['user'],
        namespace=ROOT.name+'/'+t.target,job_root=base+'/jobs')
    path=ROOT/'evidence'/'deployments'/(t.target+'.json');transport.atomic(path,deployment)
    runner=ROOT/'scripts'/'runner.py'; runner_sha=transport.sha(runner)
    remote_runner=base+'/scripts/runner-'+runner_sha[:16]+'.py'
    t.put(path,remote_runner+'.deployment.json')
    t.put(runner,remote_runner)
    profile.update(identity,runner=remote_runner,runner_sha256=runner_sha,job_root=deployment['job_root'],
        namespace=deployment['namespace'],deployment_sha256=transport.sha(path),status='DEPLOYED_NOT_ACCEPTED')
    all_profiles=json.loads((ROOT/'profiles.json').read_text());all_profiles[t.target]=profile
    transport.atomic(ROOT/'profiles.json',all_profiles)
    return profile

def initialize(t):
    identity=t.identity();profile=t.profile
    if not Path(identity['home']).is_absolute() or '..' in Path(identity['home']).parts:
        raise ValueError('unexpected home path')
    profile.update(identity)
    if not profile.get('remote_support'):
        profile['remote_support']=str(Path(identity['home'])/'grid-audits'/ROOT.name)
    if not profile.get('runner'):profile['status']='IDENTITY_CONFIRMED_NOT_DEPLOYED'
    profiles=json.loads((ROOT/'profiles.json').read_text());profiles[t.target]=profile
    transport.atomic(ROOT/'profiles.json',profiles)
    transport.atomic(ROOT/'evidence'/('initialized-'+t.target+'.json'),profile)
    return profile

def command(t,mode,job,extra=None):
    valid(job);p=t.profile
    if not p.get('runner'):raise ValueError('target runner not deployed')
    args=[p['python'],'-I','-S','-B',p['runner'],mode]
    args += [p['job_root'],extra] if mode=='submit' else [p['job_root']+'/'+job]
    return shlex.join(args)

def query(t,job):
    result=json.loads(t.ssh(command(t,'query',job)))
    p=ROOT/'jobs'/t.target/job
    local_request=p/'request.json'
    if local_request.exists():
        request=json.loads(local_request.read_text())
        fingerprint=hashlib.sha256(json.dumps(request,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
        if result['request_sha256']!=fingerprint:raise ValueError('queried request identity differs from local request')
        result['local_request_verified']=True
    else:result['local_request_verified']=False
    stamp=dt.datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%dT%H%M%S%f%z')
    transport.atomic(p/('query-'+stamp+'.json'),result);transport.atomic(p/'last-query.json',result)
    state_update(t.target,job,result);return result

def submit(t,path):
    request=json.loads(Path(path).read_text());job=valid(request['job_id']);p=t.profile
    for k in ['host','user','uid','namespace','deployment_sha256']:
        expected=p['hostname'] if k=='host' else p[k]
        if request[k]!=expected: raise ValueError('request does not match explicit target: '+k)
    local=ROOT/'jobs'/t.target/job/'request.json';local.parent.mkdir(parents=True,exist_ok=True)
    with (local.parent/'.request.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if local.exists() and json.loads(local.read_text())!=request: raise ValueError('local request conflict')
        transport.atomic(local,request)
    state_update(t.target,job,{'status':'PREPARED','request':str(local)})
    remote=p['remote_support']+'/requests/'+job+'.json';t.put(local,remote)
    result=json.loads(t.ssh(command(t,'submit',job,remote)))
    transport.atomic(local.parent/'receipt.json',result);state_update(t.target,job,result);return result

def fetch(t,job):
    query(t,job)
    meta=json.loads(t.ssh(command(t,'export',job),timeout=120))
    p=ROOT/'jobs'/t.target/job;archive=p/'evidence.tar.gz'
    result=t.get_verified(meta['path'],archive,meta)
    target=p/'retrieved';target.mkdir(exist_ok=True)
    with tarfile.open(archive,'r:gz') as tf:
        members=tf.getmembers()
        names=[m.name for m in members]
        if len(names)!=len(set(names)):raise ValueError('duplicate exported paths')
        for m in members:
            path=Path(m.name)
            if path.is_absolute() or '..' in path.parts or not m.isfile():raise ValueError('unsafe exported member')
            if target.resolve() not in (target/path).resolve().parents:raise ValueError('export escape')
        tf.extractall(target,filter='data')
    manifest=json.loads((target/'export-manifest.json').read_text())
    listed={f['path'] for f in manifest['files']}
    if listed|{'export-manifest.json'}!=set(names):raise ValueError('export manifest path mismatch')
    for f in manifest['files']:
        item=target/f['path']
        if item.stat().st_size!=f['bytes'] or transport.sha(item)!=f['sha256']:raise ValueError('export member mismatch')
    request=json.loads((target/'request.json').read_text())
    if request!=json.loads((p/'request.json').read_text()):raise ValueError('retrieved request mismatch')
    transport.atomic(p/'retrieval.json',dict(result,manifest_verified=True,request_verified=True))
    return dict(result,manifest_verified=True,request_verified=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--target',required=True,choices=['A800','new-server'])
    sub=parser.add_subparsers(dest='mode',required=True)
    sub.add_parser('identity');sub.add_parser('initialize');sub.add_parser('deploy')
    p=sub.add_parser('submit');p.add_argument('request')
    for name in ['query','logs','fetch']:
        p=sub.add_parser(name);p.add_argument('job_id')
    n=parser.parse_args();t=transport.Transport(n.target)
    if n.mode=='identity': result=t.identity()
    elif n.mode=='initialize':result=initialize(t)
    elif n.mode=='deploy':result=deploy(t)
    elif n.mode=='submit':result=submit(t,n.request)
    elif n.mode=='query':result=query(t,n.job_id)
    elif n.mode=='fetch':result=fetch(t,n.job_id)
    else:
        valid(n.job_id);query(t,n.job_id)
        print(t.ssh(shlex.join(['tail','-n','100',t.profile['job_root']+'/'+n.job_id+'/work.log'])),end='');raise SystemExit(0)
    print(json.dumps(result,ensure_ascii=False,indent=2))
