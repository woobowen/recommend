"""Bind unchanged approved Beauty plans to independently reviewed repaired bytes."""
import hashlib
import json
from pathlib import Path
import shutil
import sys

REPO = Path('/home/addaswsw/recommend-worktrees/sid-extensions-p0')
SUPPORT = Path('/home/addaswsw/recommend-support/grid-formal-20261015')
sys.path[:0] = [str(REPO), '/home/addaswsw/recommend-support/sid-extensions-p0/scripts']
from transport import Transport
from experiments.formal_20261015.deploy import REMOTE, source_records
from experiments.beauty_extensions.common import sha256, write_json
from experiments.beauty_extensions.authorization import canonical_plan


def main():
    t = Transport('new-server')
    t.identity()
    cpu = Path('/home/addaswsw/recommend-support/sid-extensions-p0/jobs/new-server/gf-20261015-config-roundtrip-r4/retrieved')
    status = json.loads((cpu/'status.json').read_text());result = json.loads((cpu/'output/RESULT.json').read_text())
    assert status['status'] == 'SUCCEEDED' and status['exit_code'] == 0 and status['cleanup']['verified']
    assert result['status'] == 'PASS' and result['model_constructions'] == result['GPU_model_calls'] == 0
    independent_cpu = SUPPORT/'config-roundtrip-runtime-review.json'
    review_cpu = json.loads(independent_cpu.read_text())
    assert review_cpu['status'] == 'CPU_CONFIGURATION_REGRESSION_PASS'
    cpu_remote = REMOTE+'/approvals/'+sha256(independent_cpu)+'/cpu-config-review.json'
    t.put(independent_cpu,cpu_remote)
    code = {r['path']:r['sha256'] for r in source_records() if r['path'].endswith('.py')}
    publication = REPO/'manifests/grid-formal-20261015/approval-r4'
    publication.mkdir(exist_ok=True)
    records=[]
    for route in ('a','b'):
        for n in (0,3,4):
            name=f'beauty-{route}-{n}'
            path=REPO/'experiments/formal_20261015/specs'/(name+'.json')
            spec=json.loads(path.read_text())
            review=SUPPORT/'results-review-r4'/(name+'-gate-review.json')
            v=json.loads(review.read_text())
            plan=hashlib.sha256(json.dumps(canonical_plan(spec),sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
            assert v['status']=='PREFLIGHT_GATE_PASS' and v['plan_sha256']==plan and v['source_python_sha256']==code
            remote=REMOTE+'/approvals/'+sha256(review)+'/review.json'
            t.put(review,remote)
            token={'kind':'FORMAL_RUN_AUTHORIZED',
                'explicit_user_approval_reference':'2026-10-15 user explicitly authorized Beauty A/B Normal/Shuffle3/Shuffle4 formal evaluation; separately approved this one strictly scoped configuration serialization repair and CPU recheck',
                'plan_sha256':plan,'source_python_sha256':code,
                'assets':{key:spec[key] for key in ('config','sid','checkpoint')},
                'independent_review':{'path':remote,'sha256':sha256(review)},
                'independent_cpu_config_review':{'path':cpu_remote,'sha256':sha256(independent_cpu)},
                'cpu_config_job':'gf-20261015-config-roundtrip-r4',
                'cpu_config_result':{'path':'/home/bwwu/grid-audits/sid-extensions-p0/jobs/gf-20261015-config-roundtrip-r4/output/RESULT.json','sha256':sha256(cpu/'output/RESULT.json')},
                'original_three_correction_rounds_not_reset':True,'additional_exception_rounds':1}
            local=SUPPORT/(name+'-authorization-r4.json');write_json(local,token)
            remote_token=REMOTE+'/approvals/'+sha256(local)+'/authorization.json';t.put(local,remote_token)
            spec['nonauthor_gate']={'path':remote_token,'sha256':sha256(local)};write_json(path,spec)
            shutil.copyfile(review,publication/(name+'-gate-review.json'))
            shutil.copyfile(local,publication/(name+'-authorization.json'))
            records.append({'name':name,'spec_sha256':sha256(path),'gate':spec['nonauthor_gate'],'plan_sha256':plan})
    write_json(SUPPORT/'beauty-six-authorized-inputs-r4.json',{'records':records,'source_python_sha256':code})
    print(json.dumps({'bound_plans':len(records),'source_python_files':len(code),'research_plans_changed':False}))


if __name__=='__main__':main()
