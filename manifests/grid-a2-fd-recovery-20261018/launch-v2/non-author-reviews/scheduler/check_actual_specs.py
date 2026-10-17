"""Independently bind the released source and real eleven templates, CPU only."""
from pathlib import Path
import copy,hashlib,json,subprocess,sys
sys.dont_write_bytecode=True
ROOT=Path('/home/addaswsw/recommend-worktrees/sid-extensions-p0');sys.path.insert(0,str(ROOT))
from experiments.a2_fd_recovery_20261018 import build
from experiments.launch_20261017.run import template_plan_sha256
OUT=Path(__file__).resolve().parent
SUPPORT=OUT.parents[1]
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
d=read(SUPPORT/'source-deployment.json');scene=read(SUPPORT/'initial-scene.json')
actual=SUPPORT/'compiled-c31d2d3eaaafbd18'
index=read(actual/'spec-index.json');plan=read(actual/'plan.json')
assert d['manifest']['source_commit']=='0d18820025e7a28675ed7112be0b1b4ff0110025'
source={r['path']:r for r in d['manifest']['files']}
review=read(OUT/'scheduler_cpu_review.json')
for f in review['source_files']:assert f['sha256']==source[f['path']]['sha256'] and sha(ROOT/f['path'])==f['sha256']
checked=[]
for f in d['manifest']['files']:
    b=subprocess.run(['git','show',d['manifest']['source_commit']+':'+f['path']],cwd=ROOT,check=True,capture_output=True).stdout
    assert hashlib.sha256(b).hexdigest()==f['sha256'];checked.append(f['path'])
assert len(checked)==262
assert plan['source_python_sha256']=={p:r['sha256'] for p,r in source.items() if p.endswith('.py')}
reproduced=OUT/'actual-spec-recompile';reproduced.mkdir(exist_ok=True)
templates,newplan=build.specs(d,scene,reproduced,index['remote_root'])
assert plan==newplan and sha(actual/'plan.json')==sha(reproduced/'plan.json')
receipts=[]
for name,expected in templates.items():
    p=actual/'specs'/(name+'.json');value=read(p)
    value.pop('launch_review',None)
    assert value==expected
    assert template_plan_sha256(value) in plan['template_plan_sha256']
    receipts.append({'name':name,'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p),'template_plan_sha256':template_plan_sha256(value)})
assert len(receipts)==11
out={'status':'ACTUAL_RELEASED_SOURCE_AND_ELEVEN_SPECS_PASS','reviewer':'/root/sports_a1_result_audit','source_commit':d['manifest']['source_commit'],'snapshot_root':d['root'],'snapshot_sha256':d['manifest']['snapshot_sha256'],'manifest_sha256':d['manifest_sha256'],'manifest_git_files_byte_checked':len(checked),'reviewed_scheduler_source_bytes_unchanged':True,'plan':{'path':str(actual/'plan.json'),'sha256':sha(actual/'plan.json')},'specs':receipts,'full_queue_binding':'PENDING_ACTUAL_CONFIG','real_runner_submissions':0}
(OUT/'actual_specs_review.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
