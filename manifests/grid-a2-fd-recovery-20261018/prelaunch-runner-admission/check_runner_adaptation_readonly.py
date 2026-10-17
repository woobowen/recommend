"""Independent positive/negative checks of the narrow outer-runner adaptation.

Only the actual candidate verify() function is compiled into the old accepted
runner's namespace. Remote files stay read only. Changed config negative cases
use explicitly labelled in-memory read fixtures; submit is never called.
"""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys

REPO = Path('/home/addaswsw/recommend-worktrees/sid-extensions-p0')
BASE = Path('/home/addaswsw/recommend-support/grid-a2-fd-recovery-20261018')
OUT = Path(__file__).resolve().parent

path = REPO / 'experiments/launch_20261017/queue_runner.py'
current = path.read_text()
old = subprocess.check_output(['git', 'show', '0d18820025e7a28675ed7112be0b1b4ff0110025:experiments/launch_20261017/queue_runner.py'], cwd=REPO).decode()
addition = '''    a2_fd_continuation = (r['job_id'] == 'ga-20261018-a2-fd-queue-v1' and
        len(r['argv']) == 7 and r['argv'][1:6] ==
        ['-B', '-m', 'experiments.a2_fd_recovery_20261018.controller', 'supervise', '--config'])
    if a2_fd_continuation:
        config_path = Path(r['argv'][6])
        assert config_path.is_absolute()
        config = read(config_path)
        assert config['queue_id'] == r['job_id'] and config['snapshot']['root'] == r['cwd']
        assert config['recovery']['original_absolute_queue_deadline'] == '2026-10-19T05:40:30.012007+00:00'
        assert r['timeout_seconds'] <= config['max_queue_seconds']
        assert {item['path']: item['sha256'] for item in r['inputs']}[str(config_path)] == sha(config_path)
'''
assert current.count(addition) == 1
restored = current.replace(addition, '', 1)
assert restored.count('original_queue or shared_continuation or a2_fd_continuation') == 1
restored = restored.replace('original_queue or shared_continuation or a2_fd_continuation', 'original_queue or shared_continuation', 1)
assert restored == old
tree = ast.parse(current)
verify_node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'verify')
verify_source = ast.get_source_segment(current, verify_node)
request = json.loads((BASE / 'compiled-c31d2d3eaaafbd18/supervisor-request.json').read_text())
deployment = json.loads((BASE / 'queue-deployment-c31d2d3eaaafbd18.json').read_text())
sys.path.insert(0, '/home/addaswsw/recommend-support/sid-extensions-p0/scripts')
from transport import Transport
t = Transport('new-server')
t.identity()
code = '''import runpy,json,datetime,hashlib,pathlib,copy,traceback
base=__A2_AUDIT_REQUEST__
outer=__A2_AUDIT_OUTER__
original=runpy.run_path(outer,run_name='independent_readonly_runner_test')
scope=dict(original)
exec(compile(__A2_AUDIT_VERIFY__,'candidate_queue_runner.verify','exec'),scope)
old_verify=original['verify'];new_verify=scope['verify'];original_read=scope['read']
config_path=pathlib.Path(base['argv'][6]);config=original_read(config_path)
new_job=original['JOB_ROOT']/base['job_id']
assert not new_job.exists()
cases=[]
def check(name,request,expected,old=False,config_override=None):
 scope['read']=lambda p:copy.deepcopy(config_override) if config_override is not None and pathlib.Path(p)==config_path else original_read(p)
 try:
  (old_verify if old else new_verify)(request)
  actual=True;error=None;line=None
 except BaseException as ex:
  actual=False;error=repr(ex);frame=traceback.extract_tb(ex.__traceback__)[-1]
  line=dict(file=frame.filename,line=frame.lineno,code=frame.line)
 assert actual==expected,(name,actual,error,line)
 cases.append(dict(case=name,expected_pass=expected,actual_pass=actual,error=error,failed_line=line,config_fixture='in-memory read override only' if config_override is not None else 'unchanged actual hash-bound server config'))
check('old_unchanged_runner_rejects_new_long_controller',copy.deepcopy(base),False,old=True)
check('new_exact_CPU_controller_accepted',copy.deepcopy(base),True)
q=copy.deepcopy(base);q['env']['CUDA_VISIBLE_DEVICES']='GPU-dfff0e95-1db6-e2f4-3431-55bb852fa0ae';check('non_CPU_long_controller_rejected',q,False)
q=copy.deepcopy(base);q['job_id']='ga-20261018-a2-fd-queue-v2';check('wrong_ID_long_controller_rejected',q,False)
q=copy.deepcopy(base);q['argv'][3]='experiments.launch_20261017.queue';check('wrong_module_long_controller_rejected',q,False)
q=copy.deepcopy(base);q['argv'].append('--extra');check('extra_argv_long_controller_rejected',q,False)
q=copy.deepcopy(base);q['inputs']=[x for x in q['inputs'] if x['path']!=str(config_path)];check('unbound_config_rejected',q,False)
q=copy.deepcopy(base);next(x for x in q['inputs'] if x['path']==str(config_path))['sha256']='0'*64;check('mismatched_config_hash_rejected',q,False)
q=copy.deepcopy(base);q['timeout_seconds']=config['max_queue_seconds']+1;check('above_config_wall_rejected',q,False)
q=copy.deepcopy(config);q['recovery']['original_absolute_queue_deadline']='2026-10-20T05:40:30.012007+00:00';check('changed_absolute_deadline_rejected',copy.deepcopy(base),False,config_override=q)
q=copy.deepcopy(config);q['queue_id']='unapproved-controller';check('config_queue_ID_mismatch_rejected',copy.deepcopy(base),False,config_override=q)
q=copy.deepcopy(config);q['snapshot']['root']='/home/bwwu/recommend';check('config_snapshot_root_mismatch_rejected',copy.deepcopy(base),False,config_override=q)
for name in ['ga-20261017-queue-v1','ga-20261018-shared-queue-v1']:
 q=original_read(original['JOB_ROOT']/name/'request.json')
 check('old_controller_unchanged_accepted_old_runner_'+name,copy.deepcopy(q),True,old=True)
 check('old_controller_unchanged_accepted_new_runner_'+name,copy.deepcopy(q),True)
scope['read']=original_read
assert not new_job.exists()
print(json.dumps(dict(status='NARROW_OUTER_RUNNER_READONLY_POSITIVE_NEGATIVE_PASS',at=datetime.datetime.now(datetime.timezone.utc).isoformat(),original_runner=outer,original_runner_sha256=hashlib.sha256(pathlib.Path(outer).read_bytes()).hexdigest(),candidate_verify_source_sha256=hashlib.sha256(__A2_AUDIT_VERIFY__.encode()).hexdigest(),cases=cases,case_count=len(cases),new_job_directory_absent_before_after=True,submission_called=False,source_execution_scope='Only candidate verify() AST function in accepted old runner namespace; other code proven byte-identical. Config mutation negative fixtures override read in memory only; no remote file or job mutation.')))
'''.replace('__A2_AUDIT_REQUEST__', repr(request)).replace('__A2_AUDIT_OUTER__', repr(deployment['outer_runner'])).replace('__A2_AUDIT_VERIFY__', repr(verify_source))
compile(code, 'independent_remote_runner_cases', 'exec')
result = json.loads(t.python(code, timeout=60))
result.update(reviewer='/root/fd_repair_reviewer',candidate_source_path=str(path),candidate_full_source_sha256=hashlib.sha256(current.encode()).hexdigest(),source_diff_only_exact_new_CPU_controller_long_wall_gate=True,old_runner_full_source_sha256=hashlib.sha256(old.encode()).hexdigest(),no_global_limits_changed=True,no_GPU_execution=True)
target = OUT / 'runner-adaptation-readonly-tests.json'
assert not target.exists()
target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
print(json.dumps(dict(path=str(target),sha256=hashlib.sha256(target.read_bytes()).hexdigest(),status=result['status'],cases=result['case_count'],candidate_sha256=result['candidate_full_source_sha256'])))
