"""Single submission after both concrete nonauthor admission receipts pass."""
import json
from pathlib import Path
import shlex
import sys
sys.path.insert(0,'/home/addaswsw/recommend-support/sid-extensions-p0/scripts')
import gridctl
from transport import Transport,atomic,sha

B=Path(__file__).resolve().parent
d=json.loads((B/'queue-deployment.json').read_text())
root=Path(d['local_root']);request_path=root/'supervisor-request.json'
request=json.loads(request_path.read_text())
for path in [B/'reviews/fd/final-request-admission-v2.json',B/'reviews/scheduler/final-request-review-v2.json']:
    review=json.loads(path.read_text())
    assert review['formal_submission_admitted'] is True,path
    assert review['future_results_approved'] is False,path
    assert review['config_sha256']==d['config_sha256'],path
    assert review['outer_request_file_sha256']==sha(request_path),path
assert request['job_id']=='ga-20261018-a2-fd-queue-v1'
t=Transport('new-server');t.identity()
exists=json.loads(t.python('import pathlib,json;print(json.dumps(pathlib.Path('+repr(t.profile['job_root']+'/'+request['job_id'])+').exists()))'))
if exists:raise RuntimeError('New controller ID already exists. Query this exact ID; never submit/restart blindly.')
validation=t.ssh(shlex.join(['env','PYTHONPATH='+d['snapshot'],request['argv'][0],'-B','-m',
    'experiments.a2_fd_recovery_20261018.controller','validate','--config',d['remote_config']]),timeout=60)
atomic(root/'immediate-presubmit-validation.json',json.loads(validation))
assert sha(root/'queue-config.json')==d['config_sha256']
# The existing child runner is unchanged. Only this CPU controller uses the
# separately audited exact-ID long-supervision derivative.
t.profile['runner']=d['outer_runner']
receipt=gridctl.submit(t,request_path)
atomic(root/'controller-submission.json',receipt)
print(json.dumps(receipt,indent=2))
