"""Continue the user-authorized single-replica experiment through both pilots.

The already running first discovery job is adopted by PID and Linux start time.
This controller lives outside the frozen scientific source directory.
"""
import argparse
from datetime import datetime
import fcntl
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
from zoneinfo import ZoneInfo

ROOT = Path('/home/jcy/GRID_wmh_Qwen_TIGER_20261009')
SCIENCE = ROOT/'codex/sid_prefix_semantics_v1'
sys.path.insert(0, str(SCIENCE))
from common import sha, write_json
from run_stage import read_complete_labels

OUT = ROOT/'experiments/sid_prefix_semantics_v1'
CONTROL = OUT/'single_replica_control'


def proc_info(pid):
    try:
        raw = Path(f'/proc/{pid}/stat').read_text()
        fields = raw[raw.rfind(')')+2:].split()
        return fields[0], fields[19]
    except FileNotFoundError:
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--adopt-beauty-discovery-pid', type=int, required=True)
    ap.add_argument('--adopt-log', required=True)
    ap.add_argument('--gpus', default='0,1')
    ap.add_argument('--execute', action='store_true')
    args = ap.parse_args()
    if not args.execute:
        ap.error('Controller not started; --execute requires prior user authorization')
    CONTROL.mkdir(parents=True, exist_ok=True)
    lock = (CONTROL/'controller.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    ready = json.loads((SCIENCE/'preparation/readiness.json').read_text())
    assert ready['ready'] and ready['config_sha256'] == sha(SCIENCE/'config.json')
    for name, digest in ready['source_sha256'].items():
        assert sha(SCIENCE/name) == digest, name
    cfg = json.loads((SCIENCE/'config.json').read_text())
    adopted = args.adopt_beauty_discovery_pid
    identity = proc_info(adopted)
    if identity and identity[0] != 'Z':
        cmd = Path(f'/proc/{adopted}/cmdline').read_bytes().replace(b'\0', b' ').decode()
        assert 'run_stage.py discover --dataset beauty ' in cmd, cmd
    else:
        assert (OUT/'beauty/taxonomy_draft.json').exists(), 'Discovery exited without taxonomy'
    record = {'state': 'running', 'controller_pid': __import__('os').getpid(),
        'physical_gpus': [int(x) for x in args.gpus.split(',')], 'replicas': 1,
        'user_authorized': True, 'automatic_restart': False,
        'stops_after_both_pilots_for_quality_review': True,
        'controller_sha256': sha(__file__), 'config_sha256': sha(SCIENCE/'config.json'),
        'completed': [], 'queue': [('beauty', s) for s in ['draft', 'revise', 'pilot']] +
                                   [('toys', s) for s in ['discover', 'draft', 'revise', 'pilot']]}
    shutil.copy2(__file__, CONTROL/'controller_source_snapshot.py')
    write_json(CONTROL/'authorization_and_plan.json', dict(record,
        adopted_beauty_discovery_pid=adopted,
        authorization='User confirmed starting the experiment in the current chat on 2026-11-09.'))

    def update(**values):
        record.update(values)
        record['updated_at'] = datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()
        write_json(CONTROL/'status.json', record)

    def verify_stage(dataset, stage):
        if stage in ['discover', 'revise']:
            path = OUT/dataset/('taxonomy_draft.json' if stage == 'discover' else 'taxonomy_frozen.json')
            assert json.loads(path.read_text())['categories'], str(path)
        else:
            path = OUT/dataset/f'{stage}_shard0of1.jsonl'
            complete = json.loads(path.with_suffix('.complete.json').read_text())
            assert sha(path) == complete['sha256']
            if stage == 'draft':
                read_complete_labels(OUT/dataset, 'draft', cfg['datasets'][dataset]['expected_items'])
            else:
                assert complete['rows'] == cfg['review_items']
                assert (OUT/dataset/'pilot_manual_review.csv').exists()

    try:
        update(current_dataset='beauty', current_stage='discover', current_pid=adopted,
               current_log=args.adopt_log)
        while identity and identity[0] != 'Z':
            time.sleep(15)
            current = proc_info(adopted)
            if not current or current[1] != identity[1] or current[0] == 'Z':
                break
            update()
        verify_stage('beauty', 'discover')
        record['completed'].append(['beauty', 'discover'])
        for dataset, stage in list(record['queue']):
            log_path = CONTROL/f'{dataset}_{stage}.log'
            command = [sys.executable, '-u', str(SCIENCE/'run_stage.py'), stage,
                       '--dataset', dataset, '--gpus', args.gpus,
                       '--shard', '0', '--shards', '1', '--execute']
            with log_path.open('xb') as log:
                child = subprocess.Popen(command, cwd=SCIENCE, stdin=subprocess.DEVNULL,
                                         stdout=log, stderr=subprocess.STDOUT)
                update(current_dataset=dataset, current_stage=stage,
                       current_pid=child.pid, current_log=str(log_path))
                code = child.wait()
            if code != 0:
                raise RuntimeError(f'{dataset}/{stage} exited with code {code}; see {log_path}')
            verify_stage(dataset, stage)
            record['completed'].append([dataset, stage])
            record['queue'].pop(0)
            update()
        update(state='ready_for_pilot_quality_review', current_pid=None,
               review_files=[str(OUT/d/'pilot_manual_review.csv') for d in ['beauty','toys']],
               final_annotation_started=False, evaluation_started=False)
    except Exception as exc:
        update(state='failed', error=repr(exc), traceback=traceback.format_exc())
        raise


if __name__ == '__main__':
    main()
