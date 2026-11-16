import hashlib, pathlib
assert hashlib.sha256(pathlib.Path('/data/bwwu/lc-rec/a0-beauty-20261115/production-closure-20261115/formal-candidate-5b2917d/independent-prelaunch-review.json').read_bytes()).hexdigest()=='3a18da80530451d72efde8aed46e7d51ad44b5bcd4b3396e579fab328de53a97'
import hashlib, json, subprocess
from pathlib import Path
from production_identity import identity
from production_control import approval_documents, validate_plan
from formal_control import digest
base=Path('/data/bwwu/lc-rec/a0-beauty-20261115/production-closure-20261115/formal-candidate-5b2917d')
candidate=base/'formal-candidate-v2.json'
budget=base/'budget-user-approved.json'
review=base/'independent-prelaunch-review.json'
assert hashlib.sha256(candidate.read_bytes()).hexdigest()=='cd5e7ab6b15f27a506079fea8786d6e6e1dbfdcd3bfb6c5a49ddaf9f43f1ec5e'
assert hashlib.sha256(budget.read_bytes()).hexdigest()=='2bcd27d8b7c6dbdfe373e3fe97d9bbcc8ca9b2626e54d482241fea46f7936deb'
c=json.loads(candidate.read_text()); b=json.loads(budget.read_text()); r=json.loads(review.read_text())
assert b['budget']==c['budget']['budget'] and b['stage_seconds']==c['budget']['stage_seconds']
assert r['candidate_plan_sha256']=='cd5e7ab6b15f27a506079fea8786d6e6e1dbfdcd3bfb6c5a49ddaf9f43f1ec5e'
args=c['seal_argv_after_real_independent_review']; precision=Path(args[args.index('--precision-approval')+1]); release=Path(args[args.index('--deployment')+1]).parent
assert hashlib.sha256(precision.read_bytes()).hexdigest()==c['precision_approval_sha256']
pr={'prelaunch_review':str(review),'precision_approval':str(precision),'approval_files':{'prelaunch_review':identity(review),'precision_approval':identity(precision)},'run_commit':c['engineering_commit'],'config_sha256':c['effective_config_sha256'],'release':str(release)}
review_sha=pr['approval_files']['prelaunch_review']['sha256']
approval_documents(pr,require_pass=True)
args[args.index('--budget')+1]=str(budget)
subprocess.run(args,check=True)
plan_path=Path(args[args.index('--root')+1])/'plan.json'; raw=plan_path.read_bytes(); plan=json.loads(raw)
validate_plan(plan); approval_documents(plan,require_pass=True)
assert plan['approval_files']['prelaunch_review']['sha256']==review_sha
assert plan['budget']==b['budget'] and plan['stage_seconds']==b['stage_seconds']
print(json.dumps({'status':'SEALED_NOT_SUBMITTED','path':str(plan_path),'file_sha256':hashlib.sha256(raw).hexdigest(),'canonical_plan_digest':digest(plan),'review_sha256':review_sha}))
