"""Conservative update accounting with one independently certified failed-LR0 exception.

Never modifies/deletes the append-only attempt ledger. Every ambiguous attempt
still consumes quota; only the explicitly frozen, hash-checked failed job below
can be classified as zero completed global updates.
"""
import json
from pathlib import Path
from common import sha


def charged_attempts(entries, task=None, jobs=None):
 task=Path(task) if task else Path(__file__).parents[1]
 jobs=Path(jobs) if jobs else Path('/home/bwwu/grid-audits/sid-extensions-p0/jobs')
 proof_path=task/'config/update_reconciliations.json'
 if not proof_path.exists():return entries,[]
 proof=json.loads(proof_path.read_text());assert len(proof['certified_failed_zero_update_jobs'])==1
 checked=[];excluded=set()
 for row in proof['certified_failed_zero_update_jobs']:
  assert (row['job_id'],row['key'],row['route'])==('qwen-20261110-q17-f-r0','q17','F')
  report=task/row['independent_review_path'];assert sha(report)==row['independent_review_sha256']
  root=jobs/row['job_id']
  for name,digest in row['raw_files_sha256'].items():assert sha(root/name)==digest
  status=json.loads((root/'status.json').read_text());cleanup=json.loads((root/'cleanup.json').read_text())
  assert status['status']=='FAILED' and status['exit_code']==1 and cleanup['verified']
  for rank in [0,1]:
   r=json.loads((root/'output/probe'/f'rank{rank}.json').read_text());current=r['current_update']
   assert r['run_commit']==row['run_commit'] and r['status']=='BLOCKED' and r['updates']==[]
   assert current['index']==0 and current['engine_step_before']==0 and current['lr_before']==[0.,0.]
   assert len(current['microsteps'])==7 and all(x['global_steps_after']==0 and not x['boundary'] for x in current['microsteps'])
   assert 'torch._foreach_sqrt(device_exp_avg_sqs)' in r['traceback'] and 'torch.OutOfMemoryError' in r['traceback']
  matches=[i for i,x in enumerate(entries) if x['job_id']==row['job_id']]
  assert len(matches)==1
  entry=entries[matches[0]];assert entry['index_in_job']==0 and entry['run_commit']==row['run_commit'] and entry['key']=='q17' and entry['route']=='F'
  excluded.add(matches[0]);checked.append({'job_id':row['job_id'],'failed_boundary_attempts':1,'completed_global_updates':0,'original_attempt_retained':True,'partial_optimizer_state_discarded':True,'gpu_device_seconds_never_excluded':True,'independent_review_sha256':row['independent_review_sha256']})
 return [row for i,row in enumerate(entries) if i not in excluded],checked
