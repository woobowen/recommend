"""User-authorized serial pipeline: pilots continue directly to final labels and evaluation."""
import argparse
from collections import Counter
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
from zoneinfo import ZoneInfo

ROOT = Path('/home/jcy/GRID_wmh_Qwen_TIGER_20261009')
SCIENCE = ROOT/'codex/sid_prefix_semantics_v7_compact'
sys.path.insert(0, str(SCIENCE))
from common import sha, write_json, read_jsonl
from run_stage import read_complete_labels

OUT = ROOT/'experiments/sid_prefix_semantics_v1'
CONTROL = OUT/'single_replica_control'


def stage_plan(datasets):
    return [(d, stage) for d in datasets for stage in ['discover', 'draft', 'revise', 'pilot', 'final']] + [('both', 'evaluate')]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--gpus', default='0,1')
    ap.add_argument('--execute', action='store_true')
    args = ap.parse_args()
    if not args.execute:
        ap.error('No pipeline started; --execute is required')
    CONTROL.mkdir(parents=True, exist_ok=True)
    lock = (CONTROL/'controller.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    ready = json.loads((SCIENCE/'preparation/readiness.json').read_text())
    assert ready['ready'] and ready['config_sha256'] == sha(SCIENCE/'config.json')
    for name, digest in ready['source_sha256'].items():
        assert sha(SCIENCE/name) == digest, name
    assert json.loads((SCIENCE/'preparation/structured_decoding_verified.json').read_text())['passed']
    cfg = json.loads((SCIENCE/'config.json').read_text())
    plan = stage_plan(cfg['datasets'])
    record = {'state': 'running', 'controller_pid': os.getpid(),
        'physical_gpus': [int(x) for x in args.gpus.split(',')], 'replicas': 1,
        'user_authorized': True, 'automatic_restart': False,
        'stops_after_both_pilots_for_quality_review': False,
        'pilot_followed_automatically_by_final': True, 'automatic_evaluation': True,
        'human_review_claimed': False, 'controller_sha256': sha(__file__),
        'config_sha256': sha(SCIENCE/'config.json'), 'completed': [], 'queue': plan.copy(),
        'science_version': SCIENCE.name, 'beauty_preserved_from_v1': True,
        'consolidation_method': 'local_keyed_functional_normalization_complete_mappings_v6'}
    snapshot = CONTROL/'source_snapshot_compact_v7'
    snapshot.mkdir(exist_ok=True)
    for path in list(SCIENCE.glob('*.py'))+[SCIENCE/'config.json', Path(__file__)]:
        shutil.copy2(path, snapshot/path.name)
    write_json(CONTROL/'authorization_and_plan.json', dict(record,
        authorization='User explicitly instructed: after pilot annotation, enter full classification and evaluation; no further confirmation required.'))

    def update(**values):
        record.update(values)
        record['updated_at'] = datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()
        write_json(CONTROL/'status.json', record)

    def completed(dataset, stage):
        if stage == 'evaluate':
            path = OUT/'evaluation/results.json'
            return path.exists() and (OUT/'evaluation/summary.csv').exists()
        if stage in ['discover', 'revise']:
            path = OUT/dataset/('taxonomy_draft.json' if stage == 'discover' else 'taxonomy_frozen.json')
            return path.exists()
        return (OUT/dataset/f'{stage}_shard0of1.complete.json').exists()

    def verify(dataset, stage):
        if stage == 'evaluate':
            data = json.loads((OUT/'evaluation/results.json').read_text())
            assert data['records'] and {r['dataset'] for r in data['records']} == set(cfg['datasets'])
            assert data['provenance']['config_sha256'] == sha(SCIENCE/'config.json')
        elif stage in ['discover', 'revise']:
            path = OUT/dataset/('taxonomy_draft.json' if stage == 'discover' else 'taxonomy_frozen.json')
            assert json.loads(path.read_text())['categories']
        else:
            path = OUT/dataset/f'{stage}_shard0of1.jsonl'
            receipt = json.loads(path.with_suffix('.complete.json').read_text())
            assert receipt['sha256'] == sha(path)
            assert receipt['config_sha256'] == sha(SCIENCE/'config.json')
            if stage in ['draft','final']:
                read_complete_labels(OUT/dataset, stage, cfg['datasets'][dataset]['expected_items'])
            else:
                rows = read_jsonl(path)
                assert len(rows) == cfg['review_items'] == receipt['rows']
                assert len({r['item_id'] for r in rows}) == len(rows)
                assert (OUT/dataset/'pilot_manual_review.csv').exists()
                report = {
                    'rows': len(rows), 'unknown_items': sum(r['category_id']=='UNKNOWN' for r in rows),
                    'needs_review_items': sum(r['needs_review'] for r in rows),
                    'category_counts': dict(Counter(r['category_id'] for r in rows)),
                    'schema_and_id_checks_passed': True, 'human_review_performed': False,
                    'accuracy_measured': False, 'automatic_continuation_authorized_by_user': True}
                if not (OUT/dataset/'pilot_quality_report.json').exists():
                    write_json(OUT/dataset/'pilot_quality_report.json', report)

    try:
        for dataset, stage in plan:
            update(current_dataset=dataset, current_stage=stage)
            if completed(dataset, stage):
                verify(dataset, stage)
                record['completed'].append([dataset, stage])
                record['queue'].pop(0)
                update()
                continue
            if stage == 'evaluate':
                command = [sys.executable, '-u', str(SCIENCE/'evaluate.py'), '--execute']
            else:
                command = [sys.executable, '-u', str(SCIENCE/'run_stage.py'), stage,
                           '--dataset', dataset, '--gpus', args.gpus,
                           '--shard', '0', '--shards', '1', '--execute']
            log_path = CONTROL/f'{dataset}_{stage}_{time.time_ns()}.log'
            with log_path.open('xb') as log:
                child = subprocess.Popen(command, cwd=SCIENCE, stdin=subprocess.DEVNULL,
                                         stdout=log, stderr=subprocess.STDOUT)
                update(current_pid=child.pid, current_log=str(log_path))
                while child.poll() is None:
                    try:
                        child.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        pass
                    progress = {}
                    if stage == 'discover':
                        progress['completed_discovery_batches'] = len(list((OUT/dataset/'discovery_batches').glob('*.json')))
                    elif stage in ['draft', 'pilot', 'final']:
                        labels = OUT/dataset/f'{stage}_shard0of1.jsonl'
                        if labels.exists():
                            with labels.open() as stream:
                                progress['completed_items'] = sum(1 for line in stream if line.strip())
                    update(progress=progress)
                code = child.returncode
            if code != 0:
                raise RuntimeError(f'{dataset}/{stage} exited with code {code}; see {log_path}')
            verify(dataset, stage)
            record['completed'].append([dataset, stage])
            record['queue'].pop(0)
            update(current_pid=None, progress={})
        update(state='completed', current_pid=None, final_annotation_completed=True, evaluation_completed=True,
               results=str(OUT/'evaluation/results.json'), summary=str(OUT/'evaluation/summary.csv'))
    except Exception as exc:
        update(state='failed', error=repr(exc), traceback=traceback.format_exc())
        raise


if __name__ == '__main__':
    main()
