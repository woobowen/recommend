"""Bind an independent bounded resume to its own successful short-run output."""
import argparse
import json
from pathlib import Path
import sys

from experiments.beauty_extensions.common import write_json
from .deploy import SUPPORT, TRANSPORT_SUPPORT
from .run import require_preflight


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--spec', type=Path, required=True)
    p.add_argument('--short-job', required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    spec = json.loads(a.spec.read_text()); require_preflight(spec)
    assert spec['route'] == 'MLP' and spec['domain'] in ('toys', 'sports')
    assert a.short_job.startswith('gsp-20261015-') and '/' not in a.short_job and '..' not in a.short_job
    assert not a.output.exists()
    sys.path.insert(0, str(TRANSPORT_SUPPORT / 'scripts'))
    from transport import Transport
    import gridctl
    t = Transport('new-server')
    state = gridctl.query(t, a.short_job)
    assert state['status'] == 'SUCCEEDED' and state['cleanup_verified'] and not state['workgroup_live']
    root = t.profile['job_root'] + '/' + a.short_job + '/output'
    code = '''import pathlib,json,hashlib
root=pathlib.Path(ROOT)
inv=json.loads((root/'invocation.json').read_text())
result=json.loads((root/'RESULT.json').read_text())
short=json.loads((root/'short_training.json').read_text())
assert result['status']=='PASS' and result['phase']=='mlp-short' and result['engineering_only'] and not result['formal_approved']
assert short['status']=='PASS' and short['engineering_only'] and short['initial_step']==0 and short['final_step']==2 and short['new_optimizer_updates']==2
checkpoint=short['checkpoint'];path=pathlib.Path(checkpoint['path'])
assert path.is_file() and not path.is_symlink() and path.parent==root/'checkpoints'
assert path.stat().st_size==checkpoint['bytes'] and hashlib.sha256(path.read_bytes()).hexdigest()==checkpoint['sha256']
pred=root/'after-training-test/predictions.jsonl';assert pred.is_file() and not pred.is_symlink()
print(json.dumps({'source_spec':inv['spec'],'source_code_identity':inv['code_identity'],
 'checkpoint':checkpoint,'prior_predictions':{'path':str(pred),'bytes':pred.stat().st_size,'sha256':hashlib.sha256(pred.read_bytes()).hexdigest()}}))
'''.replace('ROOT', repr(root))
    observed = json.loads(t.python(code, timeout=60))
    assert observed['source_spec'] == spec
    spec['resume_checkpoint'] = observed['checkpoint']
    spec['prior_predictions'] = observed['prior_predictions']
    spec['own_short_job'] = a.short_job
    require_preflight(spec)
    write_json(a.output, spec)
    write_json(SUPPORT / (a.short_job + '-resume-plan-receipt.json'), {
        'short_job': a.short_job, 'independent_resume_spec': str(a.output.resolve()),
        'verified_short_terminal_state': state['status'], 'observed': observed,
        'formal_approved': False, 'formal_state_reuse_allowed': False})
    print(json.dumps({'output': str(a.output), 'short_job': a.short_job,
        'checkpoint_sha256': observed['checkpoint']['sha256'], 'formal_approved': False}))


if __name__ == '__main__':
    main()
