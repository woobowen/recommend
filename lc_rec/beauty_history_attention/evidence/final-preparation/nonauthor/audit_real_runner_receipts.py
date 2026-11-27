"""Independent runner terminal and actual GPU budget check."""
import datetime
import json
from pathlib import Path

jobs=Path('/home/bwwu/grid-audits/sid-extensions-p0/jobs')
prefix='letter-beauty-v2-history-attn-final-'
fixed_GPU='GPU-59cac3bd-3645-668b-79c8-839e0d878893'
final={}
tot=0.
for dir in sorted(jobs.glob(prefix+'*')):
    ident=dir.name.removeprefix(prefix)
    def read(name): return json.loads((dir/name).read_text())
    status=read('status.json')
    clean=read('cleanup.json')
    request=read('request.json')
    begin=datetime.datetime.fromisoformat(status['started_at'])
    end=datetime.datetime.fromisoformat(status['ended_at'])
    seconds=(end-begin).total_seconds()
    is_gpu=request['env']['CUDA_VISIBLE_DEVICES'] != ''
    if is_gpu:
        assert request['env']['CUDA_VISIBLE_DEVICES']==fixed_GPU
        tot+=seconds
    assert status['status']=='SUCCEEDED' and status['exit_code']==0
    assert not status['worker_may_be_live'] and clean['verified'] and clean['last_observation']['verified_empty']
    assert request['job_id']==status['job_id']==dir.name
    final[ident]=dict(job_id=dir.name,status=status['status'],exit_code=0,verified_cleanup=True,seconds=seconds,gpu=is_gpu,
                      request_timeout_seconds=request['timeout_seconds'],commit=request['capture_version'])
assert {'contract-r0','old-causal-r0','b-mask-r0','generate-stop-r0','generate-resume-r0','metrics-r0'} <= set(final)
assert tot < 60*60
formal=[d.name for d in jobs.glob('letter-beauty-v2-history-attn-formal-b-*')]
assert not formal, formal
print(json.dumps(dict(gpu_device_seconds_upper_bound=tot,uses_one_rtx4090_uuid=fixed_GPU,
                      complete_stage_receipts=final,matching_formal_b_jobs=formal),indent=2,allow_nan=False))
