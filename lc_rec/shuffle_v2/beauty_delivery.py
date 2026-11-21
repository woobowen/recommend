"""Collect the evidence for one Beauty ZIP using explicit, immutable file paths."""
import argparse
import fcntl
from datetime import datetime
import json
from pathlib import Path
import subprocess
import tarfile
import time

from lc_rec.shuffle_v2 import beauty, beauty_control as bc, control, core
from lc_rec.shuffle_v2.beauty_independent import accepted_stage
from lc_rec.shuffle_v2.beauty_package import PREFLIGHT_READY, verify
from lc_rec.shuffle_v2.beauty_prep_ops import remaining_cpu

FROZEN_FILES = {
    'lc_rec/a0_beauty/formal_evaluate.py',
    'lc_rec/a0_beauty/shuffle_entry.py',
    'lc_rec/a0_beauty/uneven_sampler.py',
    'lc_rec/a0_beauty/offline_metrics.py',
    'lc_rec/a0_beauty/preflight_users.py',
    'lc_rec/a0_beauty/config/preflight_users_seed42.json',
    'lc_rec/letter/review/final_audit_20261113/external/official_cce.py',
}
RUN_FILES = {
    'lc_rec/shuffle_v2/' + file for file in (
        'beauty.py', 'a0_beauty_letter.json', 'beauty_control.py', 'beauty_evaluate.py', 'control.py',
        'core.py', 'evaluate.py', 'generate.py', 'reference.py', 'replay.py', 'runtime.py', 'series.py', 'statistics.py',
        'test_protocol.py', 'beauty_runner.py', 'beauty_lifecycle.py', 'collect.py')
} | {'lc_rec/shuffle/core.py', 'lc_rec/letter/scripts/formal_evaluate.py',
     'lc_rec/letter/review/final_audit_20261113/statistics_verify.py'}
DELIVERY_FILES = {
    'lc_rec/shuffle_v2/' + file for file in (
        'beauty_delivery.py', 'beauty_independent.py', 'beauty_observe.py', 'beauty_package.py',
        'beauty_prelaunch.py', 'independent_beauty_review.py', 'beauty_preflight_eval.py', 'beauty_prep_ops.py', 'beauty_assessment.py')
}
JOB_RECORDS = ('request.json', 'accepted.json', 'attempt.json', 'status.json', 'cleanup.json', 'work.log', 'supervisor.log')


def git_bytes(*args):
    return subprocess.check_output(['git', *args], cwd=control.REPO)


def materialize(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        stream.write(data)


def terminal(states):
    return {name: row for name, row in states.items() if row.get('ended_at')}


def collect(config, states, replay, preflight, completed):
    base = config['baseline']
    paths = {'execution/runner.py': f'{control.BASE}/scripts/runner-a1-96h-{control.RUNNER_SHA[:16]}.py'}
    for kind in ('index', 'inter', 'item'):
        paths[f'data/Beauty.{kind}.json'] = f"{base['data-root']}/Beauty/Beauty.{kind}.json"

    def take(area, folder, names):
        paths.update({area + '/' + name: folder + '/' + name for name in names})

    normal_test = ('predictions.rank0.jsonl', 'predictions.rank1.jsonl', 'rank0.json', 'rank1.json', 'metrics.json')
    primary = ('metrics.json', 'per_user_metrics.npz', 'per_user.csv')
    take('conditions/normal/test', base['normal_test_dir'], normal_test)
    take('conditions/normal/primary', config['normal_primary'], primary)
    take('archive', str(Path(base['normal_stage_a']).parent), ('stage_a.json', 'evaluation_config.json'))
    take('archive', str(Path(base['adapter']).parent), ('best_selection.json', 'trainer_state.json',
         'checkpoint_receipts/step-6160-letter-a0-beauty-formal-beauty-a0-seed42-formal-train-a0.json'))
    take('archive/tokenizer', base['adapter'], ('adapter_config.json', 'added_tokens.json', 'tokenizer.model',
                                             'tokenizer_config.json', 'special_tokens_map.json'))
    take('archive/base-config', base['model'], ('config.json', 'generation_config.json'))
    if replay:
        take('replay', f'{bc.ROOT}/{replay}', ('audit.json', 'inputs.jsonl', 'assets.json', 'examples.json',
                                             'samples.json', 'single-shard.json'))
    if preflight:
        take('preflight', f'{bc.ROOT}/{preflight}', ('summary.json',))
        for mode in core.MODES:
            folder = f'{bc.ROOT}/{preflight}'
            take(f'preflight/{mode}', f'{folder}/{mode}', ('rank0.json', 'predictions.rank0.jsonl', 'v2-binding.rank0.json'))
            for name in (f'{mode}.request.json', f'{mode}.exit.json', f'{mode}.log', f'{mode}.resources.jsonl'):
                paths[f'preflight/{mode}/{name}'] = f'{folder}/{name}'
        take('preflight/derived', f'{bc.ROOT}/{preflight}/shuffle2', ('derived.rank0.diff', 'derived.rank0.py'))
    normal_jobs = [name for name in states if name.startswith(bc.PREFIX + 'cpu-eval-normal-') and states[name]['status'] == 'SUCCEEDED']
    if normal_jobs:
        assert len(normal_jobs) == 1
        where = f"{bc.ROOT}/{normal_jobs[0].removeprefix(bc.PREFIX)}"
        take('conditions/normal/recomputed', where, ('summary.json', 'evaluation-config.json', 'offline_metrics.diff'))
        take('evaluation', where, ('offline_metrics.derived.py',))
    if completed:
        series = accepted_stage(states, 'group-series')
        pair = accepted_stage(states, 'cpu-summary')
        take('paired', f'{bc.ROOT}/{pair}', ('summary.json', 'metrics.csv', 'paired.csv', 'changes_from_normal.csv', 'arrays.npz'))
        take('execution', f'{bc.ROOT}/{series}', ('summary.json',))
        for mode in core.MODES:
            gpu = f"{bc.ROOT}/{accepted_stage(states, 'gpu-formal-' + mode)}"
            cpu = f"{bc.ROOT}/{accepted_stage(states, 'cpu-eval-' + mode)}"
            take(f'conditions/{mode}/test', f'{gpu}/test', ('predictions.rank0.jsonl', 'metrics.json', 'rank0.json', 'v2-binding.rank0.json'))
            take(f'conditions/{mode}/primary', f'{cpu}/primary', primary)
            take(f'conditions/{mode}/evaluation', cpu, ('summary.json', 'evaluation-config.json'))
            take(f'conditions/{mode}/execution', gpu, ('summary.json', f'{mode}.request.json', f'{mode}.exit.json',
                                                      f'{mode}.resources.jsonl', f'{mode}.log'))
        first_gpu = f"{bc.ROOT}/{accepted_stage(states, 'gpu-formal-shuffle2')}/test"
        if not preflight:
            take('execution/derived', first_gpu, ('derived.rank0.diff', 'derived.rank0.py'))
    for name in terminal(states):
        take(f'jobs/{name}', f'{control.BASE}/jobs/{name}', JOB_RECORDS)
    for name in terminal(states):
        if not name.startswith(bc.PREFIX + 'cpu-lifecycle-'):
            continue
        local = name.removeprefix(bc.PREFIX)
        leaves = ('worker.json',)
        if '-decoy-' in local:
            leaves += ('survived.json',)
        elif '-orphan-' in local or '-timeout-' in local:
            leaves += ('child.json',)
        take(f'execution/lifecycle/{local}', f'{bc.ROOT}/{local}', leaves)
    return paths


def runner_diff(commit, legacy, current):
    import difflib
    old = git_bytes('show', f'{commit}:{legacy}').decode().splitlines(True)
    new = git_bytes('show', f'{commit}:{current}').decode().splitlines(True)
    return ''.join(difflib.unified_diff(old, new, fromfile=legacy, tofile=current)).encode()


def add_research_documents(build, commit):
    import posixpath
    import re
    from urllib.parse import urlsplit

    status = 'lc_rec/shuffle/results/v2-a0-beauty-letter/README.md'
    reproduction = 'lc_rec/shuffle_v2/beauty_reproduction.md'
    mapping = {status: 'research/status.md', reproduction: 'research/beauty_reproduction.md'}
    archive_evidence = {
        'preflight-summary.json': 'preflight/summary.json',
        'preflight-evaluation.json': 'preflight-evaluation/summary.json',
        'assessment.json': 'execution/assessment.json',
        'formal-plan.json': 'execution/formal-plan.json',
        'lifecycle.json': 'execution/lifecycle.json',
        'nonauthor-research.json': 'reviews/research-review.json',
        'nonauthor-statistics.json': 'reviews/statistics-review.json',
        'nonauthor-execution.md': 'reviews/execution-review.md',
        'prelaunch-gate.json': 'reviews/prelaunch.json',
        'preflight-r1-status.json': 'jobs/letter-beauty-v2-gpu-preflight-r1/status.json',
        'formal-inventory.json': 'execution/formal-inventory.json'}
    mapping.update({posixpath.join(posixpath.dirname(status), 'prep', filename): stored
                    for filename, stored in archive_evidence.items()})
    for name in RUN_FILES | DELIVERY_FILES:
        mapping[name] = 'source/' + name
    mapping['lc_rec/shuffle_v2/beauty_control.py'] = 'source/operations/beauty_control.py'

    for source in (status, reproduction):
        target = mapping[source]
        original = git_bytes('show', f'{commit}:{source}').decode()

        def replace(match):
            href = match.group(1)
            url = urlsplit(href)
            if url.scheme or not url.path:
                return match.group(0)
            linked = posixpath.normpath(posixpath.join(posixpath.dirname(source), url.path))
            assert linked in mapping, (source, href)
            relative = posixpath.relpath(mapping[linked], posixpath.dirname(target))
            return '](' + relative + ('#' + url.fragment if url.fragment else '') + ')'

        projected = re.sub(r'\]\(([^)]+)\)', replace, original)
        materialize(build / target, projected.encode())


def publication_inputs(args, config, states, build):
    for name in sorted(RUN_FILES):
        materialize(build / 'source' / name, git_bytes('show', f'{args.run_commit}:{name}'))
    for name in sorted(DELIVERY_FILES):
        materialize(build / 'source' / name, git_bytes('show', f'{args.delivery_commit}:{name}'))
    materialize(build / 'source/beauty-changes.diff',
                git_bytes('diff', 'ed90c9d43747b590027382bf96bee0f5989494ec', args.run_commit, '--', *sorted(RUN_FILES)))
    for name in sorted(FROZEN_FILES):
        materialize(build / 'archive/source' / name, git_bytes('show', f'{beauty.LETTER_RUN}:{name}'))
    for name, record in config['baseline']['author_sources'].items():
        path = Path(record['path'])
        blob = git_bytes('show', f'{beauty.LETTER_RUN}:lc_rec/letter/vendor/LETTER/LETTER-LC-Rec/{name}')
        materialize(build / 'archive/source/LETTER-LC-Rec' / path.name, blob)
    materialize(build / 'archive/source/LETTER-LC-Rec/utils.py',
                git_bytes('show', f'{beauty.LETTER_RUN}:lc_rec/letter/vendor/LETTER/LETTER-LC-Rec/utils.py'))
    materialize(build / 'archive/model-identity.json',
                git_bytes('show', f'{args.run_commit}:lc_rec/letter/evidence/formal_det_20261112/model_identity.json'))
    materialize(build / 'offline_review.sh', git_bytes('show', f'{args.delivery_commit}:lc_rec/shuffle_v2/beauty_offline_review.sh'))
    materialize(build / 'README.md', args.readme.read_bytes())
    if args.status == PREFLIGHT_READY:
        add_research_documents(build, args.delivery_commit)
    operation = 'lc_rec/shuffle_v2/beauty_control.py'
    materialize(build / 'source/operations/beauty_control.py', git_bytes('show', f'{args.delivery_commit}:{operation}'))
    materialize(build / 'source/operations/beauty_control.diff',
                git_bytes('diff', args.run_commit, args.delivery_commit, '--', operation))
    for archive_path, filename in args.local_file:
        materialize(build / archive_path, Path(filename).read_bytes())
    support = Path('/home/addaswsw/recommend-support/sid-extensions-p0/scripts')
    for name in ('gridctl.py', 'transport.py'):
        materialize(build / 'source/support' / name, (support / name).read_bytes())
    legacy_runner = 'lc_rec/a1_qwen3_06b/scripts/runner.py'
    beauty_runner = bc.RUNNER_SOURCE
    materialize(build / 'execution/runner-beauty.diff',
                runner_diff(args.run_commit, legacy_runner, beauty_runner))
    for name, path in (('publication/run.json', bc.SUPPORT / (args.run_commit + '.deployment.json')),
                       ('publication/readback.json', args.readback), ('reviews/prelaunch.json', args.prelaunch),
                       ('reviews/final.json', args.final_review),
                       ('execution/observations.jsonl', bc.SUPPORT / 'observations.jsonl')):
        if path and path.is_file():
            materialize(build / name, path.read_bytes())
    for memo in args.memo:
        materialize(build / 'reviews' / memo.name, memo.read_bytes())
    if args.status != PREFLIGHT_READY:
        for plan in bc.SUPPORT.glob('*.plan.json'):
            materialize(build / 'execution' / plan.name, plan.read_bytes())
    elapsed = {'preflight_device_seconds': 0.0, 'formal_device_seconds': 0.0, 'cpu_worker_wall_seconds': 0.0}
    for name, row in terminal(states).items():
        stage = name.removeprefix(bc.PREFIX)
        category = ('cpu_worker_wall_seconds' if stage.startswith('cpu-') else
                    'preflight_device_seconds' if stage.startswith('gpu-preflight-') else
                    'formal_device_seconds' if stage.startswith('gpu-formal-') else None)
        if category and row.get('started_at'):
            elapsed[category] += (datetime.fromisoformat(row['ended_at']) - datetime.fromisoformat(row['started_at'])).total_seconds()
    materialize(build / 'execution/ledger.json', json.dumps({'totals': elapsed, 'jobs': states}, indent=2).encode())


def build_package(connection, args):
    config = json.loads(beauty.CONFIG.read_text())
    states = control.job_states(connection, bc.PREFIX)
    completed = args.status == 'SECOND_GROUP_COMPLETE_PENDING_WEB_REVIEW'
    if args.status == PREFLIGHT_READY:
        assert args.replay and args.preflight and args.prelaunch and args.readback
        assert not any(name.startswith((bc.PREFIX + 'gpu-formal-', bc.PREFIX + 'group-series-')) for name in states)
    if completed:
        assert args.replay and args.preflight and args.prelaunch and args.final_review and args.readback
        assert all(row.get('ended_at') and row['cleanup']['verified'] for row in states.values())
    build = bc.SUPPORT / 'bundles' / args.destination.stem
    build.mkdir(parents=True, exist_ok=False)
    overlay = build / 'overlay'
    publication_inputs(args, config, states, overlay)
    paths = collect(config, states, args.replay, args.preflight, completed)
    for name, path in args.extra:
        assert name not in paths
        paths[name] = path
    namespace = f'{control.BASE}/transfers/{bc.NAMESPACE}/{args.destination.stem}'
    unpacked = f'{namespace}/overlay'
    upload = build / 'overlay.tar.gz'
    with tarfile.open(upload, 'x:gz') as archive:
        for path in sorted(overlay.rglob('*')):
            if path.is_file():
                member = path.relative_to(overlay).as_posix()
                archive.add(path, member)
                assert member not in paths
                paths[member] = f'{unpacked}/{member}'
    specification = build / 'package-spec.json'
    materialize(specification, json.dumps({'status': args.status, 'files': paths}, indent=2).encode())
    tar_receipt = connection.put(upload, namespace + '/overlay.tar.gz')
    spec_receipt = connection.put(specification, namespace + '/package-spec.json')
    output = namespace + '/' + args.destination.name
    from shlex import join
    connection.ssh(join(['/usr/bin/python3', '-I', '-S', '-B', '-c',
                        f"""import hashlib,tarfile
from pathlib import Path
source=Path({tar_receipt['path']!r})
assert hashlib.sha256(source.read_bytes()).hexdigest()=={tar_receipt['sha256']!r}
root=Path({unpacked!r}); root.mkdir()
with tarfile.open(source,'r:gz') as tar: tar.extractall(root,filter='data')
"""]))
    # The audit export is an independently supervised, budgeted CPU job.
    delivery_release, _ = bc.deployment(args.delivery_commit)
    payload = bc.request(connection, config, args.delivery_commit, 'cpu-review-export-' + args.export_attempt,
                         'beauty_package', 1200, ['--spec', spec_receipt['path']],
                         [{'path': spec_receipt['path'], 'sha256': spec_receipt['sha256']}])
    assert payload['cwd'] == delivery_release
    argv = payload['argv']
    index = argv.index('--output')
    argv[index:index + 2] = ['--zip', output]
    with (bc.SUPPORT / 'submit.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        current = control.job_states(connection, bc.PREFIX)
        assert remaining_cpu(current, config) >= 1215
        if args.status == PREFLIGHT_READY:
            assert not any(name.startswith((bc.PREFIX + 'gpu-formal-', bc.PREFIX + 'group-series-')) for name in current)
        reply = bc.submit(connection, payload)
    while reply['status'] in ('ACCEPTED', 'RUNNING'):
        time.sleep(10)
        reply = control.gridctl.query(connection, payload['job_id'])
    assert reply['status'] == 'SUCCEEDED' and reply['exit_code'] == 0 and reply['cleanup_verified'], reply
    remote_log = f"{connection.profile['job_root']}/{payload['job_id']}/work.log"
    receipt = json.loads(control.remote_python(connection,
        f'from pathlib import Path; print(Path({remote_log!r}).read_text())'))
    connection.get_verified(output, args.destination, receipt, timeout=900)
    verified = verify(args.destination)
    assert verified['sha256'] == receipt['sha256'] and verified['bytes'] == receipt['bytes']
    verified['export_job'] = payload['job_id']
    verified['export_wall_seconds'] = (datetime.fromisoformat(reply['ended_at']) -
                                      datetime.fromisoformat(reply['started_at'])).total_seconds()
    print(json.dumps(verified, indent=2))


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--status', choices=(PREFLIGHT_READY, 'SECOND_GROUP_COMPLETE_PENDING_WEB_REVIEW', 'PARTIAL_BLOCKED'), required=True)
    parser.add_argument('--run-commit', required=True)
    parser.add_argument('--delivery-commit', required=True)
    parser.add_argument('--destination', type=Path, required=True)
    parser.add_argument('--readme', type=Path, required=True)
    parser.add_argument('--readback', type=Path)
    parser.add_argument('--replay')
    parser.add_argument('--preflight')
    parser.add_argument('--prelaunch', type=Path)
    parser.add_argument('--final-review', type=Path)
    parser.add_argument('--memo', type=Path, action='append', default=[])
    parser.add_argument('--extra', action='append', nargs=2, default=[], metavar=('ARCHIVE_PATH', 'SERVER_PATH'))
    parser.add_argument('--local-file', action='append', nargs=2, default=[], metavar=('ARCHIVE_PATH', 'LOCAL_PATH'))
    parser.add_argument('--export-attempt', type=bc.attempt_id, default='r0')
    args = parser.parse_args()
    from lc_rec.shuffle_v2.beauty_package import safe_name
    assert all(safe_name(name) for name, _ in args.local_file)
    build_package(bc.connect(bc.runner_sha_at(args.delivery_commit)), args)


if __name__ == '__main__':
    main()
