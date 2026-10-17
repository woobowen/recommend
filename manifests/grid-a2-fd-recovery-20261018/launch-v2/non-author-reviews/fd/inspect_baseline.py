"""Independent read-only review of preserved A2 EMFILE baseline.

Writes only a review receipt beside this script. Does not import research code,
change an environment, contact a server, train a model, or modify original data.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path('/home/addaswsw/recommend-worktrees/sid-extensions-p0')
PACKAGE = Path('/home/addaswsw/recommend-support/grid-a-launch-20261017/deliveries/a2-emfile-capture-20261017T175451Z/package')
OUT = Path(__file__).resolve().parent


def identity(path):
    raw = path.read_bytes()
    return {'path': str(path), 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


def main():
    failure = PACKAGE / 'a2-failure'
    status = json.loads((failure / 'status.json').read_text())
    trace = json.loads((failure / 'original-traceback-and-process-fd.json').read_text())
    config = json.loads((failure / 'output/used_config.json').read_text())
    log = (failure / 'work.log').read_text()
    assert log.count('Too many open files') == trace['EMFILE_count'] == 24
    assert status['status'] == 'INTERRUPTED' and status['exit_code'] == -9
    assert status['cleanup']['verified'] and status['cleanup']['last_observation']['verified_empty']
    assert status['worker_may_be_live'] is False
    assert hashlib.sha256((failure / 'work.log').read_bytes()).hexdigest() == trace['full_log_sha256']
    n = trace['before_prefix_bytes']
    assert hashlib.sha256((failure / 'work.log').read_bytes()[:n]).hexdigest() == trace['before_prefix_sha256']
    refs = [
        'references/wmh-original/GRID/src/data/loading/components/collate_functions.py',
        'references/wmh-original/GRID/src/data/loading/components/interfaces.py',
        'references/wmh-original/GRID/src/data/loading/datamodules/sequence_datamodule.py',
        'references/wmh-original/GRID/src/data/loading/components/dataloading.py',
        'references/wmh-original/GRID/src/data/loading/components/pre_processing.py',
        'references/wmh-original/GRID/src/modules/clustering/residual_quantization.py',
        'experiments/launch_20261017/a2_upstream.py',
        'experiments/launch_20261017/run.py',
    ]
    files = [identity(failure / name) for name in ['status.json', 'work.log', 'original-traceback-and-process-fd.json', 'output/used_config.json', 'request.json']]
    comparisons = []
    for name in refs:
        local = identity(ROOT / name)
        frozen = PACKAGE / 'a2-code/scientific-C2' / name
        record = {'relative_path': name, 'local': local}
        if frozen.exists():
            record['preserved_C2'] = identity(frozen)
            record['exact_bytes'] = record['local']['sha256'] == record['preserved_C2']['sha256']
            assert record['exact_bytes']
        comparisons.append(record)
    groups = {}
    for p in trace['processes']:
        groups.setdefault(p['role'], []).append({k: p.get(k) for k in ['pid', 'fd_count_observed', 'RLIMIT_NOFILE_soft', 'RLIMIT_NOFILE_hard', 'PPID', 'role_basis']})
    loader = {}
    for key, value in config['data_loading']['datamodule'].items():
        if isinstance(value, dict):
            loader[key] = {field: value.get(field) for field in ['batch_size_per_device', 'num_workers', 'should_shuffle_rows', 'assign_all_files_per_worker', 'pin_memory', 'persistent_workers', 'timeout', 'drop_last', 'collate_fn']}
    result = {
        'reviewer': '/root/fd_repair_reviewer',
        'observed_at': dt.datetime.now(dt.timezone.utc).isoformat(),
        'review_kind': 'independent preserved raw evidence inspection, not current live observation',
        'status': 'BASELINE_EVIDENCE_VERIFIED_NEW_REPAIR_ADMISSION_PENDING',
        'scientific_result': 'SCIENTIFIC_RESULT_NOT_YET_AVAILABLE',
        'old_job': {k: status[k] for k in ['job_id', 'status', 'exit_code', 'started_at', 'ended_at', 'worker_may_be_live']},
        'old_cleanup': {k: status['cleanup'][k] for k in ['verified', 'reason', 'ended_at']},
        'EMFILE_count': trace['EMFILE_count'],
        'traceback': trace['first_traceback_stack'],
        'old_role_counts': dict(Counter(p['role'] for p in trace['processes'])),
        'old_FD_observations': groups,
        'important_original_vs_prompt_discrepancy': 'Original rank0 worker candidates have soft=hard=524288, not prompt table soft1024. Their FD89..133 are original observations; rank membership remains argv-derived without saved PPID/worker_id.',
        'root_cause_boundary': 'Tensor IPC fragmentation is strongly consistent with traceback and prefetch FD scale; saved data do not classify FD targets or establish the unique root cause.',
        'model': config['model'], 'optimizer': config['optim'], 'trainer': config['trainer'],
        'loader': loader,
        'frozen_file_identities': files,
        'local_vs_preserved_C2': comparisons,
        'no_installation': True,
    }
    target = OUT / 'baseline-inspection.json'
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'path': str(target), 'sha256': identity(target)['sha256'], 'status': result['status']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
