"""Bind independent exact MLP fixed32 admission to the user's continuation."""
import hashlib
import json
from pathlib import Path
import shutil
import sys

REPO = Path('/home/addaswsw/recommend-worktrees/sid-extensions-p0')
SUPPORT = Path('/home/addaswsw/recommend-support/grid-formal-20261015')
sys.path[:0] = [str(REPO), '/home/addaswsw/recommend-support/sid-extensions-p0/scripts']
from transport import Transport
from experiments.formal_20261015.deploy import REMOTE, source_records
from experiments.beauty_extensions.authorization import canonical_plan
from experiments.beauty_extensions.common import sha256, write_json


def main():
    t = Transport('new-server')
    t.identity()
    code = {r['path']: r['sha256'] for r in source_records() if r['path'].endswith('.py')}
    publication = REPO/'manifests/grid-formal-20261015/mlp-attention-admission'
    publication.mkdir(exist_ok=True)
    for domain in ('toys', 'sports'):
        spec = json.loads((SUPPORT/'mlp-attention-specs'/f'{domain}.json').read_text())
        review = SUPPORT/'mlp-attention-admission'/f'{domain}-gate-review.json'
        audited = json.loads(review.read_text())
        digest = hashlib.sha256(json.dumps(canonical_plan(spec), sort_keys=True,
            separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()
        assert audited['status'] == 'PREFLIGHT_GATE_PASS' and audited['plan_sha256'] == digest
        assert audited['source_python_sha256'] == code and spec['attention_wall_seconds_limit'] == 1200
        allowance = spec['resource_allowance']
        assert allowance['maximum_cumulative_worker_wall_with_capture_seconds'] < 43200
        remote_review = REMOTE+'/approvals/'+sha256(review)+'/review.json'
        t.put(review, remote_review)
        token = {'kind': 'FORMAL_RUN_AUTHORIZED',
            'explicit_user_approval_reference': 'User explicitly resumed formal task on 2026-10-16, '
                'adopting CODEX_GRID_FORMAL_CONTINUATION_FINAL_20261016.txt in full; section4.2 '
                'authorizes each successful Toys/Sports MLP best fixed32 capture within original '
                'remaining domain resources. This plan uses at most1200s and no training or full-test repetition.',
            'plan_sha256': digest, 'source_python_sha256': code,
            'assets': {key: spec[key] for key in ('config', 'sid', 'checkpoint')},
            'independent_review': {'path': remote_review, 'sha256': sha256(review)},
            'resource_allowance': allowance,
            'original_three_correction_rounds_not_reset': True,
            'single_scoped_configuration_exception_already_complete': True}
        local = SUPPORT/'mlp-attention-admission'/f'{domain}-authorization.json'
        write_json(local, token)
        remote_token = REMOTE+'/approvals/'+sha256(local)+'/authorization.json'
        t.put(local, remote_token)
        spec['nonauthor_gate'] = {'path': remote_token, 'sha256': sha256(local)}
        destination = REPO/f'experiments/formal_20261015/specs/attention-{domain}-mlp.json'
        assert not destination.exists(), destination
        write_json(destination, spec)
        for source in (review, local):
            shutil.copyfile(source, publication/source.name)
    print(json.dumps({'domains_bound': 2, 'source_Python_files': len(code), 'GPU_calls': 0}))


if __name__ == '__main__':
    main()
