"""Bind six separately reviewed formal fixed32 plans without model execution."""
import hashlib
import json
from pathlib import Path
import shutil
import sys

REPO = Path('/home/addaswsw/recommend-worktrees/sid-extensions-p0')
SUPPORT = Path('/home/addaswsw/recommend-support/grid-formal-20261015')
sys.path[:0] = [str(REPO), '/home/addaswsw/recommend-support/sid-extensions-p0/scripts']
from transport import Transport
from experiments.beauty_extensions.authorization import canonical_plan
from experiments.beauty_extensions.common import sha256, write_json
from experiments.formal_20261015.deploy import REMOTE, source_records


def main():
    t = Transport('new-server')
    t.identity()
    code = {r['path']: r['sha256'] for r in source_records() if r['path'].endswith('.py')}
    publication = REPO/'manifests/grid-formal-20261015/attention-admission'
    publication.mkdir(exist_ok=True)
    for route in ('a', 'b'):
        for n in (0, 3, 4):
            source = SUPPORT/'attention-specs'/f'beauty-{route}-{n}.json'
            spec = json.loads(source.read_text())
            review = SUPPORT/'attention-admission'/f'beauty-{route}-{n}-gate-review.json'
            audited = json.loads(review.read_text())
            digest = hashlib.sha256(json.dumps(canonical_plan(spec), sort_keys=True,
                separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()
            assert audited['status'] == 'PREFLIGHT_GATE_PASS'
            assert audited['plan_sha256'] == digest and audited['source_python_sha256'] == code
            assert spec['phase'] == 'formal_attention_capture'
            remote_review = REMOTE+'/approvals/'+sha256(review)+'/review.json'
            t.put(review, remote_review)
            token = {'kind': 'FORMAL_RUN_AUTHORIZED',
                'explicit_user_approval_reference': 'User formal-stage instructions explicitly authorize '
                    'Beauty baseline and A/B formal fixed32 attention after corresponding complete '
                    'formal evaluation and independent identity acceptance. B-Normal serves both '
                    'current main working baseline and route B Normal without duplicate capture.',
                'plan_sha256': digest, 'source_python_sha256': code,
                'assets': {key: spec[key] for key in ('config', 'sid', 'checkpoint')},
                'independent_review': {'path': remote_review, 'sha256': sha256(review)},
                'original_three_correction_rounds_not_reset': True,
                'additional_scoped_configuration_exception_rounds': 1,
                'unapplied_resource_sharing_proposal': True}
            local = SUPPORT/'attention-admission'/f'beauty-{route}-{n}-authorization.json'
            write_json(local, token)
            remote_token = REMOTE+'/approvals/'+sha256(local)+'/authorization.json'
            t.put(local, remote_token)
            spec['nonauthor_gate'] = {'path': remote_token, 'sha256': sha256(local)}
            destination = REPO/'experiments/formal_20261015/specs'/f'attention-beauty-{route}-{n}.json'
            assert not destination.exists(), destination
            write_json(destination, spec)
            for item in (review, local):
                shutil.copyfile(item, publication/item.name)
    index = SUPPORT/'attention-admission/beauty-six-attention-gate-review-index.json'
    shutil.copyfile(index, publication/index.name)
    print(json.dumps({'separate_attention_plans_bound': 6, 'source_Python_files': len(code),
        'model_or_GPU_calls': 0, 'duplicate_baseline_capture': False}))


if __name__ == '__main__':
    main()
