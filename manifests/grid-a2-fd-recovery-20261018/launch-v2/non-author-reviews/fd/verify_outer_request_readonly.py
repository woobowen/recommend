"""Call only the accepted remote runner's read-only verify(request) function."""
import datetime as dt
import hashlib
import json
from pathlib import Path
import sys

OUT = Path(__file__).resolve().parent
BASE = OUT.parents[1]
request = json.loads((BASE / 'compiled-c31d2d3eaaafbd18/supervisor-request.json').read_text())
deployment = json.loads((BASE / 'queue-deployment.json').read_text())
sys.path.insert(0, '/home/addaswsw/recommend-support/sid-extensions-p0/scripts')
from transport import Transport

t = Transport('new-server')
t.identity()
code = '''import runpy,json,datetime,hashlib,pathlib,traceback
request=REQUEST
runner=RUNNER
source=pathlib.Path(runner)
module=runpy.run_path(runner,run_name='independent_readonly_verify')
try:
 module['verify'](request)
 outcome=dict(verify_pass=True,error=None,failed_line=None)
except BaseException as error:
 frames=traceback.extract_tb(error.__traceback__)
 outcome=dict(verify_pass=False,error=repr(error),failed_line=dict(file=frames[-1].filename,line=frames[-1].lineno,code=frames[-1].line))
print(json.dumps(dict(at=datetime.datetime.now(datetime.timezone.utc).isoformat(),runner=runner,runner_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),request=request,job_directory_exists=(module['JOB_ROOT']/request['job_id']).exists(),verification_only_no_submission=True,**outcome)))
'''.replace('REQUEST', repr(request)).replace('RUNNER', repr(deployment['outer_runner']))
result = json.loads(t.python(code, timeout=60))
result['reviewer'] = '/root/fd_repair_reviewer'
path = OUT / ('outer-request-readonly-verify-pass.json' if result['verify_pass'] else 'outer-request-readonly-verify-blocked.json')
assert not path.exists(), 'Preserve original verifier receipt'
path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
print(json.dumps(dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),verify_pass=result['verify_pass'],failed_line=result['failed_line'],job_directory_exists=result['job_directory_exists'])))
