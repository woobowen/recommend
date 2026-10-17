"""Archive immutable admitted configuration and completed CPU result reviews."""
import hashlib
import json
from pathlib import Path

B=Path(__file__).resolve().parent
R=Path('/home/addaswsw/recommend-worktrees/sid-extensions-p0')
T=R/'manifests/grid-a2-fd-recovery-20261018/launch-v2'
T.mkdir(parents=True,exist_ok=True)
d=json.loads((B/'queue-deployment.json').read_text());c=Path(d['local_root'])
for p in [B/'reviews/fd/final-request-admission-v2.json',B/'reviews/scheduler/final-request-review-v2.json']:
 v=json.loads(p.read_text());assert v['formal_submission_admitted'] and not v['future_results_approved']
 assert v['config_sha256']==d['config_sha256']
sources={p:T/'compiled'/p.relative_to(c) for p in c.rglob('*') if p.is_file()}
for name in ['source-deployment.json','queue-deployment.json','source-github-readback-v2.json','source-github-verified-v2.json']:
 sources[B/name]=T/name
for name in ['submit_recovery.py','monitor.py','prepare_formal_v2.py','archive_launch.py']:
 sources[B/name]=T/'local-invocation-scripts'/name
for directory in ['fd','scheduler']:
 for p in (B/'reviews'/directory).iterdir():
  if p.is_file() and p.suffix in ['.json','.py','.md']:
   sources[p]=T/'non-author-reviews'/directory/p.name
for p in [B/'reviews/toys/TOYS_INDEPENDENT_REVIEW.md',B/'reviews/toys/toys-independent-review.json',
          B/'reviews/toys/toys-supplemental-review.json',B/'reviews/sports_a1/REPORT.md',
          B/'reviews/sports_a1/audit_result.json',B/'reviews/sports_a1/checkpoint_direct_audit.json',
          B/'reviews/sports_a1/cohort_process_direct_audit.json']:
 sources[p]=T/'completed-result-reviews'/p.parent.name/p.name
for name in ['cross-domain-evidence-index.json','cross-domain-original-metrics.json','cross-domain-original-metrics.csv',
             'SOURCE_NOTES.md','fixed-source-manifest.json','index-validation.json']:
 p=B/'reviews/cross_domain'/name;sources[p]=T/'cross-domain-inheritance'/name
records=[]
for p,q in sources.items():
 data=p.read_bytes();q.parent.mkdir(parents=True,exist_ok=True)
 if q.exists():assert q.read_bytes()==data,'do not overwrite archived launch artifact: '+str(q)
 else:q.write_bytes(data)
 records.append(dict(path=q.relative_to(T).as_posix(),bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),source=str(p)))
(T/'copy-manifest.json').write_text(json.dumps(records,indent=2)+'\n')
print(json.dumps(dict(target=str(T),members=len(records),bytes=sum(x['bytes'] for x in records),submitted=False)))
