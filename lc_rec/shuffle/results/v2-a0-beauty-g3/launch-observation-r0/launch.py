"""Materialize the reviewed G3 formal parent request using the existing transport."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

from lc_rec.shuffle_v2 import g3_control as gc, g3_formal as gf, g3_assets as ga

REVIEWED = '910034add71fe9f8f988c052b52cb484c47f0bf5'
PLAN_FILE = 'lc_rec/shuffle/results/v2-a0-beauty-g3/preflight/formal-plan.json'
PLAN_SHA = 'cdf5754c1eeb6c22836f0b4a5914c1fb126f0d4bc6ff6c3139ef09446e5b04e2'
GPU_VERSION = '86660be5a62c8325caf74cf2810581850ca12803'
CONTROLLER_VERSION = 'cb3b1f36d69ef0c1a7f554ef5455026523635f78'
PARENT = 'g3-beauty-v2-group-formal-r0'

FIELD_CHECK = """
import hashlib, importlib.metadata, json, os, socket, subprocess
from datetime import datetime,timezone
from pathlib import Path
from lc_rec.shuffle_v2 import g3_assets as ga, g3_formal as gf

p = ga.verified(PLAN_PATH, PLAN_SHA)
plan = ga.read(p)
gf.validate(plan)
assert socket.gethostname() == plan['parent']['request']['host'] and os.getuid() == plan['parent']['request']['uid']
jobs, requests = Path(plan['job_root']), Path(plan['requests_root'])
root = Path(plan['parent']['request']['argv'][-1]).parent
wanted_ids = [plan['parent']['name'], plan['paired']['name']] + [job['name'] for pair in plan['modes'].values() for job in pair.values()]
formal = {'jobs': [name for name in wanted_ids if (jobs/name).exists()],
          'requests': [name for name in wanted_ids if (requests/(name+'.json')).exists()],
          'outputs': [path.name for path in root.iterdir() if 'formal' in path.name or 'group-series' in path.name]}
assert not any(formal.values()), formal
ga.verified(plan['runner'], plan['runner_sha256'])
runner_deploy = str(plan['runner']) + '.deployment.json'
ga.verified(runner_deploy, plan['parent']['request']['deployment_sha256'])
for version, files in plan['release_inputs'].items():
 for member in files:
  ga.verified(member['path'],member['sha256'])
for pair in [*plan['modes'].values(), {'final': plan['paired']}]:
 for phase in pair.values():
  for asset in phase['request']['inputs']:
   if asset['sha256'] != gf.APPROVAL_SHA: ga.verified(asset['path'],asset['sha256'])
config, original, rank, adapter, known_files = ga.load()
indices, catalog, users, identities = original.catalog_and_users(config['data_root'])
assert (len(indices),len(catalog),len(users))==(12101,12101,22363)
weights=ga.verified(adapter/'adapter_model.bin',config['adapter_sha256'])
assert rank['model']['revision']==Path(config['model']).name
model_files=rank['model']['files']
for name, row in model_files.items():
 path=Path(config['model'])/name
 assert path.is_file() and path.stat().st_size==row['bytes']
 if path.stat().st_size<32*1024**2: ga.verified(path,row['sha256'])
assert rank['tokenizer_identity']['vocab_size']==32786 and rank['tokenizer_identity']['new_token_count']==786
normal_path=Path(config['formal_root'])/'attempts/normal/a0/test/predictions.rank0.jsonl'
assert normal_path.is_file() and normal_path.stat().st_size==400029854
versions={k:importlib.metadata.version(k) for k in rank['versions']}
assert versions==rank['versions']
def smi(query): return subprocess.check_output(['nvidia-smi',*query,'--format=csv,noheader,nounits'],text=True).splitlines()
gpus=smi(['--query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu'])
apps=smi(['--query-compute-apps=gpu_uuid,pid,used_gpu_memory'])
chosen=[row.split(', ') for row in gpus if plan['card'] in row]
assert len(chosen)==1 and chosen[0][2]=='NVIDIA GeForce RTX 4090' and int(chosen[0][3])<100
assert int(chosen[0][4])>=24000 and not any(plan['card'] in row for row in apps)
space=os.statvfs(root)
free=space.f_bavail*space.f_frsize
host_free=int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))*1024
assert free>=20*1024**3 and host_free>=40*1024**3
prereq={}
for name in plan['prerequisite_jobs']:
 row=json.loads(subprocess.check_output(['/usr/bin/python3','-I','-S','-B',plan['runner'],'query',str(jobs/name)],text=True))
 assert gf.success(row) and not row['supervisor_alive'],name
 prereq[name]={key: row[key] for key in ('status','exit_code','request_sha256','cleanup_verified')}
for request in [plan['parent']['request'],*(phase['request'] for pair in plan['modes'].values() for phase in pair.values()),plan['paired']['request']]:
 assert Path(request['argv'][0]).is_file() and os.access(request['argv'][0],os.X_OK)
 for key in ('CUDA_HOME','LD_LIBRARY_PATH'):
  assert all(Path(path).is_dir() for path in request['env'][key].split(':')), key
print(json.dumps(dict(
 observed_utc=datetime.now(timezone.utc).isoformat(),host=socket.gethostname(),formal_inventory=formal,
 prereq=prereq,device=chosen[0],all_gpus=gpus,compute_apps=apps,free_output_bytes=free,
 host_memavailable_bytes=host_free,versions=versions,interpreter=request['argv'][0],
 release_counts={k:len(v) for k,v in plan['release_inputs'].items()},asset_identities=identities,
 adapter={'path':str(weights),'sha256':config['adapter_sha256'],'identity_sha256':config['model_identity_sha256']},
 tokenizer=rank['tokenizer_identity'],model=rank['model'],normal={'path':str(normal_path),'sha256_from_accepted_receipt':config['normal_predictions_sha256']},
 prior_export_transfer_exists=(root/'cpu-review-export-r0/cpu-review-export-r0.tar.gz').exists()
),ensure_ascii=False))
"""


def dump(path, value):
    ga_path = Path(path)
    raw = (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n').encode()
    with ga_path.open('xb') as stream:
        stream.write(raw)


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--output', required=True, type=Path)
    options = parser.parse_args()
    output = options.output
    output.mkdir(parents=True, exist_ok=False)
    raw = subprocess.check_output(['git', 'show', f'{REVIEWED}:{PLAN_FILE}'], cwd=gc.control.REPO)
    assert hashlib.sha256(raw).hexdigest() == PLAN_SHA
    plan = json.loads(raw)
    gf.validate(plan)
    assert (plan['gpu_commit'], plan['evaluation_and_controller_commit']) == (GPU_VERSION, CONTROLLER_VERSION)
    target, _, _ = gc.connection()
    assert (plan['runner'], plan['runner_sha256']) == (target.profile['runner'], target.profile['runner_sha256'])
    current = gc.control.job_states(target, gc.PREFIX)
    if PARENT in current:
        dump(output/'existing-parent.json', gc.control.gridctl.query(target, PARENT))
        raise SystemExit('Existing parent queried. Continue with the original identity.')
    assert not any('formal' in name or 'group-series-' in name for name in current)
    for job in plan['prerequisite_jobs']:
        assert current[job]['status']=='SUCCEEDED' and current[job]['exit_code']==0
    limits = ga.read(ga.CONFIG)['budget']
    history = {'preflight_gpu_seconds':gc.usage(current,'gpu-preflight-'),'cpu_worker_seconds':gc.usage(current,'cpu-'),
               'formal_gpu_seconds':gc.usage(current,'gpu-formal-')}
    future_cpu = sum(x['cpu']['request']['timeout_seconds'] for x in plan['modes'].values()) + plan['paired']['request']['timeout_seconds']
    future_gpu = sum(x['gpu']['request']['timeout_seconds']+15 for x in plan['modes'].values())
    assert history['preflight_gpu_seconds'] < limits['preflight_device_seconds']
    assert history['cpu_worker_seconds'] + future_cpu + 60 <= limits['cpu_wall_seconds']
    assert history['formal_gpu_seconds'] + future_gpu <= limits['formal_device_seconds']
    dump(output/'budget-before.json',{'past':history,'limits':limits,'reserved_gpu_seconds_with_cleanup':future_gpu,
                                     'reserved_cpu_worker_seconds':future_cpu,'parent_wall_limit_seconds':126000})
    plan_path = output/'formal-plan.json'
    plan_path.write_bytes(raw)
    approval = {'scope':'G3_BEAUTY_SHUFFLE_V2_FORMAL','authorized_by':'user','web_review_pass':True,
                'run_commit':GPU_VERSION,'plan_sha256':PLAN_SHA,'experiment':'GRID_G3_BEAUTY_LETTER_LC_REC_LLAMA_SHUFFLE_V2',
                'source':'本对话当前用户消息「# 任务：正式启动第三组 GRID G3 Beauty Shuffle V2，并完成 30 分钟启动观察、内部审核和完整阶段交付」第一节明确授权',
                'authorization_excerpt':'我现在明确授权你：1. 按已审方案正式启动第三组整套计算链',
                'allowed':['S2','S3a','S3b','S4','four_full_CPU_evaluations','final_five_paired_comparisons'],
                'single_gpu_uuid':plan['card'],'recorded_at_utc':datetime.now(timezone.utc).isoformat(),
                'web_review_status':'THIRD_GROUP_PREFLIGHT_COMPLETE_WEB_REVIEW_PASS',
                'web_review_sync_commit':'a5698277b70054527fdb54ae69bd65d73ff8f4e5',
                'observation_minutes':30}
    auth_path = output/'authorization.json'
    dump(auth_path,approval)
    approval_sha = ga.sha(auth_path)
    template = plan['parent']['request']
    parent = json.loads(json.dumps(template).replace(gf.APPROVAL_SHA,approval_sha))
    parent['inputs'] = [*plan['release_inputs'][CONTROLLER_VERSION],*parent['inputs'],
                        {'path':plan['server_plan_path'],'sha256':PLAN_SHA},
                        {'path':plan['approval_path'],'sha256':approval_sha}]
    assert parent['job_id']==PARENT and parent['capture_version']==CONTROLLER_VERSION
    assert gf.APPROVAL_SHA in template['argv'] and gf.APPROVAL_SHA not in parent['argv']
    request = output/'parent-request.json'
    dump(request,parent)
    plan_receipt = target.put(plan_path,plan['server_plan_path'])
    src=FIELD_CHECK.replace('PLAN_PATH',repr(plan['server_plan_path'])).replace('PLAN_SHA',repr(PLAN_SHA))
    argv=[parent['argv'][0],'-B','-c',src]
    command='cd '+shlex.quote(parent['cwd'])+' && '+shlex.join(argv)
    check=json.loads(target.ssh(command,timeout=90))
    dump(output/'prelaunch.json',check)
    auth_receipt=target.put(auth_path,plan['approval_path'])
    dump(output/'upload-receipts.json',{'plan':plan_receipt,'authorization':auth_receipt})
    try:
        response=gc.control.gridctl.submit(target,request)
    except (OSError,RuntimeError,subprocess.SubprocessError,ValueError) as exc:
        dump(output/'submission-response-error.json',{'exception':repr(exc),'action':'query original ID'})
        response=gc.control.gridctl.query(target,PARENT)
    dump(output/'submission-response.json',response)
    print(json.dumps({key: response.get(key) for key in ('job_id','status','submission','request_sha256','received_at','supervisor_alive')}),flush=True)


if __name__=='__main__':
    main()
