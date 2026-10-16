"""Prepare exact formal plans only from successful task-local raw evidence."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

REPO=Path('/home/addaswsw/recommend-worktrees/sid-extensions-p0')
SUPPORT=Path('/home/addaswsw/recommend-support/grid-shuffle-20261017')
JOBS=Path('/home/addaswsw/recommend-support/sid-extensions-p0/jobs/new-server')
REMOTE_JOBS='/home/bwwu/grid-audits/sid-extensions-p0/jobs'
REMOTE='/home/bwwu/grid-audits/sid-extensions-p0/grid-shuffle-20261017'
SPECS=REPO/'experiments/shuffle_20261017/specs'


def read(path):return json.loads(path.read_bytes())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def record(path):return {'path':str(path),'sha256':sha(path)}


def write_new(path,value):
    if path.exists():assert read(path)==value,'Refuse changed plan identity';return
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')


def evaluation(domain):
    job=f'gs-20261017-{domain}-b-preflight-r2'
    root=JOBS/job/'retrieved';status=read(root/'status.json')
    result=read(root/'output/RESULT.json')
    assert status['status']=='SUCCEEDED' and status['exit_code']==0 and status['cleanup']['verified']
    assert result['status']=='PASS' and not result['formal'] and result['engineering_only']
    source=read(SPECS/(domain+'-b-preflight.json'))
    base={k:source[k] for k in ('route','domain','seed','generation_k','data_dir','config','sid','checkpoint')}
    for name,filename in [('test_index','test_target_index.jsonl'),('attention_cohort','fixed_attention_samples.json')]:
        expected={'path':REMOTE_JOBS+'/'+job+'/output/'+filename,'sha256':sha(root/'output'/filename)}
        assert result[name]==expected
        base[name]=expected
    base.update(stage='grid-shuffle-20261017',formal_approved=True,
        purpose='User-authorized complete test under inherited native model/beam/evaluator; within-route control only',
        preflight_job_id=job,
        user_authorization_reference='2026-10-17 one-time Toys/Sports Shuffle + corresponding fixed32 attention, analysis, independent review, wbw publication and GPT review ZIP; no new training or SID generation')
    for n in (0,3,4):write_new(SPECS/f'{domain}-b-{n}.json',{**base,'shuffle_n':n})


def attention(domain):
    for n,label in [(0,'normal'),(3,'shuffle3'),(4,'shuffle4')]:
        job=f'gs-20261017-{domain}-b-{label}-v1'
        root=JOBS/job/'retrieved';status=read(root/'status.json')
        result=read(root/'output/RESULT.json')
        review_path=SUPPORT/'results-audit'/f'{domain}-b-{n}-evaluation-review.json'
        review=read(review_path)
        assert status['status']=='SUCCEEDED' and status['exit_code']==0 and status['cleanup']['verified']
        assert result['status']=='COMPLETE' and result['formal'] and result['test_complete']
        assert review['status']=='FORMAL_EVALUATION_ACCEPTANCE_PASS' and review['job_id']==job
        spec=read(SPECS/f'{domain}-b-{n}.json');spec.pop('nonauthor_gate')
        assert (review['domain'],review['route'],review['shuffle_n'])==(domain,'B',n)
        for key in ('checkpoint','sid','config'):assert result[key]==review[key]==spec[key]
        spec.update(purpose='Formal fixed32 last-layer decoder cross-attention bound to accepted complete evaluation; original observer and beam paths',
            evaluation_result={'path':REMOTE_JOBS+'/'+job+'/output/RESULT.json','sha256':sha(root/'output/RESULT.json')},
            evaluation_status={'path':REMOTE_JOBS+'/'+job+'/status.json','sha256':sha(root/'status.json')},
            evaluation_job_id=job,
            evaluation_independent_review={'path':REMOTE+'/approvals/'+sha(review_path)+'/formal-evaluation-review.json',
                'sha256':sha(review_path)})
        write_new(SPECS/f'attention-{domain}-b-{n}.json',spec)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('phase',choices=['evaluation','attention'])
    p.add_argument('--domain',choices=['toys','sports'],required=True);args=p.parse_args()
    {'evaluation':evaluation,'attention':attention}[args.phase](args.domain)
    print(json.dumps({'prepared':args.phase,'domain':args.domain,'model_calls':0,'argv':sys.argv}))


if __name__=='__main__':main()
