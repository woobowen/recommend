"""Bind the user's approved domain training plans to independent exact admission."""
import hashlib
import json
from pathlib import Path
import shutil
import sys

REPO=Path('/home/addaswsw/recommend-worktrees/sid-extensions-p0')
SUPPORT=Path('/home/addaswsw/recommend-support/grid-formal-20261015')
sys.path[:0]=[str(REPO),'/home/addaswsw/recommend-support/sid-extensions-p0/scripts']
from transport import Transport
from experiments.formal_20261015.deploy import REMOTE,source_records
from experiments.beauty_extensions.common import sha256,write_json
from experiments.beauty_extensions.authorization import canonical_plan


def main():
    t=Transport('new-server');t.identity()
    code={r['path']:r['sha256'] for r in source_records() if r['path'].endswith('.py')}
    publication=REPO/'manifests/grid-formal-20261015/mlp-admission';publication.mkdir(exist_ok=True)
    for domain in ('toys','sports'):
        path=REPO/f'experiments/formal_20261015/specs/{domain}-mlp.json'
        spec=json.loads(path.read_text());review=SUPPORT/'mlp-admission'/f'{domain}-gate-review.json'
        value=json.loads(review.read_text())
        plan=hashlib.sha256(json.dumps(canonical_plan(spec),sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
        assert value['status']=='PREFLIGHT_GATE_PASS' and value['plan_sha256']==plan and value['source_python_sha256']==code
        assert not spec.get('checkpoint') and not spec.get('resume_checkpoint') and spec['hard_cap_hours']==12
        remote_review=REMOTE+'/approvals/'+sha256(review)+'/review.json';t.put(review,remote_review)
        token={'kind':'FORMAL_RUN_AUTHORIZED',
            'explicit_user_approval_reference':'2026-10-15 user explicitly approved both option1 protocols, scratch full-parameter training, each domain twelve-hour total hard allowance including complete testing; later approved one scoped configuration serialization repair only',
            'plan_sha256':plan,'source_python_sha256':code,
            'assets':{name:spec[name] for name in ('config','sid')},
            'independent_review':{'path':remote_review,'sha256':sha256(review)},
            'hard_cap_seconds_including_full_test':43200,
            'original_three_correction_rounds_not_reset':True,'additional_scoped_exception_rounds':1}
        local=SUPPORT/f'{domain}-formal-authorization.json';write_json(local,token)
        remote_token=REMOTE+'/approvals/'+sha256(local)+'/authorization.json';t.put(local,remote_token)
        spec['nonauthor_gate']={'path':remote_token,'sha256':sha256(local)};write_json(path,spec)
        shutil.copyfile(review,publication/review.name);shutil.copyfile(local,publication/local.name)
    for name in ('review_metadata.py','reviewer-invocation.json'):
        shutil.copyfile(SUPPORT/'mlp-admission'/name,publication/name)
    print(json.dumps({'domains_authorized':2,'Py_source_files':len(code),'scratch_only':True,'hard_limit_each_including_full_test':43200}))


if __name__=='__main__':main()
