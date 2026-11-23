"""Read-only final verification of stopped formal runs and isolated diagnosis."""
from pathlib import Path
import hashlib,json,subprocess
root=Path('/home/jcy/GRID_wmh_Qwen_TIGER_20261009')
plan=json.loads((root/'codex/yelp_aligned_v1/experiment_plan.json').read_text())
changed=[]
for filename,digest in plan['original_code_sha256'].items():
    if hashlib.sha256(Path(filename).read_bytes()).hexdigest()!=digest:
        changed.append(filename)
processes=subprocess.check_output(['ps','-u','jcy','-o','pid,ppid,args'],text=True).splitlines()
formal=[p for p in processes if any(x in p for x in ('codex/yelp_aligned_v1/run_task.py','codex/yelp_aligned_v1/launch_training.py'))]
diagnostic=root/'experiments/yelp_diagnostic_patience40_seed43'
report=dict(formal_plan_state=plan['state'],auto_launch=plan['auto_launch'],
    formal_processes=formal,original_code_changed=changed,
    tasks=[dict(id=t['id'],status=t['status']) for t in plan['tasks']],
    diagnostic_status=json.loads((root/'codex/yelp_diagnostics/extended_seed43_immediate/status.json').read_text()),
    latest_convergence=json.loads((diagnostic/'latest_convergence.json').read_text()),
    configuration_changes=json.loads((diagnostic/'configuration_comparison.json').read_text()))
assert plan['state']=='stopped_by_user' and not plan['auto_launch'] and not formal and not changed
print(json.dumps(report,indent=2))
