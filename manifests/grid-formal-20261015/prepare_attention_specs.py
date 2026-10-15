"""Prepare separate fixed32 capture plans bound to accepted formal evaluations."""
import hashlib
import json
from pathlib import Path
import sys

REPO=Path('/home/addaswsw/recommend-worktrees/sid-extensions-p0')
SUPPORT=Path('/home/addaswsw/recommend-support/grid-formal-20261015')
JOBS=Path('/home/addaswsw/recommend-support/sid-extensions-p0/jobs/new-server')
REMOTE='/home/bwwu/grid-audits/sid-extensions-p0/grid-formal-20261015'
sys.path.insert(0,'/home/addaswsw/recommend-support/sid-extensions-p0/scripts')
from transport import Transport


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    t=Transport('new-server');t.identity()
    destination=SUPPORT/'attention-specs';destination.mkdir(exist_ok=True)
    for route in ('a','b'):
        for n in (0,3,4):
            condition='normal' if n==0 else f'shuffle{n}'
            job=f'gf-20261015-{route}-{condition}-v1'
            base=JOBS/job/'retrieved';result=json.loads((base/'output/RESULT.json').read_text())
            status=json.loads((base/'status.json').read_text())
            assert status['status']=='SUCCEEDED' and status['exit_code']==0 and status['cleanup']['verified']
            assert result['status']=='COMPLETE' and result['formal'] and result['test_complete']
            assert result['metrics']['samples']==22363 and result['metrics']['coverage_exact']
            review=SUPPORT/'results-review-r4'/f'{route}-{condition}-formal-review.json'
            audited=json.loads(review.read_text());assert audited['status']=='FORMAL_EVALUATION_ACCEPTANCE_PASS'
            remote_review=REMOTE+'/approvals/'+sha(review)+'/formal-evaluation-review.json';t.put(review,remote_review)
            spec=json.loads((REPO/f'experiments/formal_20261015/specs/beauty-{route}-{n}.json').read_text())
            spec.pop('nonauthor_gate')
            assert all(result[k]==spec[k] for k in ('checkpoint','config','sid')) and result['shuffle_n']==n
            spec.update({'purpose':'Separately authorized formal fixed32 attention after accepted complete evaluation; no model re-evaluation',
                'phase':'formal_attention_capture',
                'evaluation_result':{'path':f'/home/bwwu/grid-audits/sid-extensions-p0/jobs/{job}/output/RESULT.json','sha256':sha(base/'output/RESULT.json')},
                'evaluation_status':{'path':f'/home/bwwu/grid-audits/sid-extensions-p0/jobs/{job}/status.json','sha256':sha(base/'status.json')},
                'evaluation_independent_review':{'path':remote_review,'sha256':sha(review)},
                'attention_protocol':{'cohort':'existing fixed32, evaluation split, selected before model outputs',
                    'kind':'cross-attention','rows':'four autoregressive SID generation queries',
                    'columns':'encoder SID and SEP positions','layer':'last decoder layer',
                    'display_heads':'mean','retained_heads':'all raw heads of last layer',
                    'paths':['actual generation final Top-1 ancestry','local first parent per step'],
                    'scientific_similarity_claim':None},
                'baseline_role':'current main working baseline' if (route,n)==('b',0) else ('route-specific Normal' if n==0 else 'Shuffle condition')})
            path=destination/f'beauty-{route}-{n}.json'
            data=json.dumps(spec,ensure_ascii=False,indent=2)+'\n'
            if path.exists():assert path.read_text()==data
            else:path.write_text(data)
    print(json.dumps({'separate_attention_specs':6,'canonical_plan_gate_pending':True,'models_or_GPU_called':0}))


if __name__=='__main__':main()
