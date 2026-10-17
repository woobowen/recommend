"""One immutable queue ID; re-entry queries it instead of restarting."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
sys.path.insert(0,'/home/addaswsw/recommend-support/sid-extensions-p0/scripts')
from transport import Transport,atomic,sha
import gridctl

parser=argparse.ArgumentParser()
parser.add_argument('--index',type=Path,required=True)
parser.add_argument('--queue-review',type=Path,required=True)
parser.add_argument('--published-commit')
parser.add_argument('--prepare-only',action='store_true')
args=parser.parse_args();index=json.loads(args.index.read_text());local=Path(index['local_root'])
deployment=json.loads((local/'queue-deployment.json').read_text())
review=json.loads(args.queue_review.read_text())
assert 'PASS' in review['status'] or 'ADMISSIBLE' in review['status']
assert review['future_results_approved'] is False
assert review['queue_config']['sha256']==deployment['config_sha256']
repo=Path('/home/addaswsw/recommend-worktrees/sid-extensions-p0')
assert subprocess.check_output(['git','branch','--show-current'],cwd=repo,text=True).strip()=='wbw'
if not args.prepare_only:
 assert args.published_commit
 remote=subprocess.check_output(['git','ls-remote','origin','refs/heads/wbw'],cwd=repo,text=True).split()[0]
 assert remote==args.published_commit
 for name in ('queue.py','queue_runner.py','run.py','train.py','a2_upstream.py','build.py'):
  path='experiments/launch_20261017/'+name
  data=subprocess.check_output(['git','show',remote+':'+path],cwd=repo)
  assert data==(repo/path).read_bytes()
t=Transport('new-server');t.identity()
review_remote=index['remote_root']+'/reviews/'+sha(args.queue_review)+'-final-queue-review.json'
t.put(args.queue_review,review_remote)
request=json.loads((local/'queue-supervisor-request.json').read_text())
request['inputs'].append({'path':review_remote,'sha256':sha(args.queue_review)})
path=local/'queue-supervisor-request-final.json'
if path.exists():assert json.loads(path.read_text())==request
atomic(path,request)
if args.prepare_only:
 print(json.dumps({'prepared_request':str(path),'sha256':sha(path),'submitted':False}));sys.exit(0)
published_request=subprocess.check_output(['git','show',remote+':manifests/grid-a-launch-20261017/compiled/queue-supervisor-request-final.json'],cwd=repo)
assert published_request==path.read_bytes()
job=request['job_id']
existing=json.loads(t.python('from pathlib import Path;import json;p=Path('+repr(t.profile['job_root']+'/'+job+'/accepted.json')+');print(json.dumps({"accepted":p.exists()}))'))
if existing['accepted']:
 receipt=gridctl.query(t,job)
 expected=hashlib.sha256(json.dumps(request,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
 assert receipt['request_sha256']==expected
else:
 t.profile['runner']=deployment['outer_runner']
 receipt=gridctl.submit(t,path)
atomic(local/'queue-submission-receipt.json',receipt)
atomic(local/'submission-github-readback.json',{'published_commit':remote,
 'runtime_source_commit':index['source_deployment']['manifest']['source_commit'],
 'source_bytes_equal':True,'formal_queue_job_id':job,'submission_status':receipt['status']})
print(json.dumps(receipt,indent=2))
