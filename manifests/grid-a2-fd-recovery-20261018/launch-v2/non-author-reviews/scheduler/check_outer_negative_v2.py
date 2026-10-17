"""Read-only negative tests against the actual deployed outer verify function."""
from pathlib import Path
import json,sys
sys.path.insert(0,'/home/addaswsw/recommend-support/sid-extensions-p0/scripts')
from transport import Transport
P=Path(__file__).resolve().parent;S=P.parents[1]
qd=json.loads((S/'queue-deployment.json').read_text());request=json.loads((Path(qd['local_root'])/'supervisor-request.json').read_text())
code='''import copy,importlib.util,json
from pathlib import Path
spec=importlib.util.spec_from_file_location('independent_negative_outer_verify',__OUTER_PATH__)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
request=__REQUEST_OBJECT__
module.verify(request)
rows=[{'case':'actual_long_CPU_controller_request','status':'PASS'}]
cases=[
('nonempty_CUDA_VISIBLE_DEVICES',lambda r:r['env'].__setitem__('CUDA_VISIBLE_DEVICES','0')),
('unlisted_long_controller_ID',lambda r:r.__setitem__('job_id','ga-20261018-a2-fd-queue-other-v1')),
('wrong_module',lambda r:r['argv'].__setitem__(3,'experiments.launch_20261017.run')),
('extra_argv',lambda r:r['argv'].append('--extra')),
('wall_above_bound_config',lambda r:r.__setitem__('timeout_seconds',json.loads(Path(r['argv'][6]).read_text())['max_queue_seconds']+1)),
('config_input_hash_changed',lambda r:next(x for x in r['inputs'] if x['path']==r['argv'][6]).__setitem__('sha256','0'*64)),
]
for name,change in cases:
 r=copy.deepcopy(request);change(r)
 try:module.verify(r)
 except (AssertionError,ValueError,KeyError) as error:rows.append({'case':name,'status':'EXPECTED_REJECTION','exception_type':type(error).__name__})
 else:raise AssertionError('incorrectly admitted: '+name)
original=module.read
for name,change in [
('changed_original_deadline',lambda c:c['recovery'].__setitem__('original_absolute_queue_deadline','2026-10-20T05:40:30.012007+00:00')),
('config_queue_id_mismatch',lambda c:c.__setitem__('queue_id','ga-20261018-a2-fd-other-v1')),
('config_snapshot_root_mismatch',lambda c:c['snapshot'].__setitem__('root','/home/bwwu/other-snapshot')),
]:
 def mock_read(path):
  value=original(path)
  if str(path)==request['argv'][6]:change(value)
  return value
 module.read=mock_read
 try:module.verify(request)
 except (AssertionError,ValueError,KeyError) as error:rows.append({'case':name,'status':'EXPECTED_REJECTION','exception_type':type(error).__name__})
 else:raise AssertionError('incorrectly admitted: '+name)
 finally:module.read=original
print(json.dumps({'status':'ACTUAL_OUTER_PRECISE_CPU_LONG_WALL_GATES_PASS','source':'actual deployed outer runner verify; in-memory request/config mutations only','checks':rows,'filesystem_mutations':0,'submit_calls':0,'GPU_or_model_calls':0}))
'''.replace('__OUTER_PATH__',repr(qd['outer_runner'])).replace('__REQUEST_OBJECT__',repr(request))
r=json.loads(Transport('new-server').python(code,timeout=45))
(P/'final-outer-negative-gates-v2.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
