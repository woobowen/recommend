"""Bind real nonauthor receipts to user-authorized immutable formal plans."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

REPO=Path('/home/addaswsw/recommend-worktrees/sid-extensions-p0')
SUPPORT=Path('/home/addaswsw/recommend-support/grid-shuffle-20261017')
sys.path[:0]=[str(REPO),'/home/addaswsw/recommend-support/sid-extensions-p0/scripts']
from transport import Transport
from experiments.shuffle_20261017.deploy import REMOTE,base
from experiments.beauty_extensions.common import sha256,write_json
from experiments.beauty_extensions.authorization import canonical_plan


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--spec',type=Path,required=True)
    p.add_argument('--review',type=Path,required=True);args=p.parse_args()
    spec=json.loads(args.spec.read_text());review=json.loads(args.review.read_text())
    code={r['path']:r['sha256'] for r in base.source_records() if r['path'].endswith('.py')}
    plan=hashlib.sha256(json.dumps(canonical_plan(spec),sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
    assert review['status']=='PREFLIGHT_GATE_PASS' and review['plan_sha256']==plan
    assert review['source_python_sha256']==code
    t=Transport('new-server');t.identity()
    if 'evaluation_independent_review' in spec:
        receipt=SUPPORT/'results-audit'/f"{spec['domain']}-b-{spec['shuffle_n']}-evaluation-review.json"
        value=json.loads(receipt.read_text())
        assert value['status']=='FORMAL_EVALUATION_ACCEPTANCE_PASS' and value['job_id']==spec['evaluation_job_id']
        assert (value['domain'],value['route'],value['shuffle_n'])==(spec['domain'],spec['route'],spec['shuffle_n'])
        assert all(value[k]==spec[k] for k in ('config','sid','checkpoint'))
        assert sha256(receipt)==spec['evaluation_independent_review']['sha256']
        t.put(receipt,spec['evaluation_independent_review']['path'])
    remote_review=REMOTE+'/approvals/'+sha256(args.review)+'/review.json'
    t.put(args.review,remote_review)
    gate={'kind':'FORMAL_RUN_AUTHORIZED','explicit_user_approval_reference':spec['user_authorization_reference'],
        'plan_sha256':plan,'source_python_sha256':code,
        'assets':{k:spec[k] for k in ('config','sid','checkpoint')},
        'independent_review':{'path':remote_review,'sha256':sha256(args.review)},
        'task_correction_round':2,'prior_Beauty_correction_history_unchanged':True}
    local=SUPPORT/'approvals'/(args.spec.stem+'-authorization.json');local.parent.mkdir(exist_ok=True)
    if local.exists():assert json.loads(local.read_text())==gate
    else:write_json(local,gate)
    remote_gate=REMOTE+'/approvals/'+sha256(local)+'/authorization.json';t.put(local,remote_gate)
    spec['nonauthor_gate']={'path':remote_gate,'sha256':sha256(local)};write_json(args.spec,spec)
    publication=REPO/'manifests/grid-shuffle-20261017/approvals';publication.mkdir(exist_ok=True)
    shutil.copyfile(args.review,publication/(args.spec.stem+'-gate-review.json'))
    shutil.copyfile(local,publication/(args.spec.stem+'-authorization.json'))
    print(json.dumps({'spec':str(args.spec),'plan_sha256':plan,'source_python_files':len(code),'reviewer':review.get('reviewer_task')}))


if __name__=='__main__':main()
