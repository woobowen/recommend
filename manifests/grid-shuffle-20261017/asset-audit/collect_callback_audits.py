"""Collect terminal original callback audit jobs without restarting calculation."""
import argparse
from pathlib import Path
import json
import shutil
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, '/home/addaswsw/recommend-support/sid-extensions-p0/scripts')
from transport import Transport, sha
import gridctl


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suffix', required=True)
    parser.add_argument('--expected-status', choices=['FAILED', 'SUCCEEDED'], required=True)
    args = parser.parse_args()
    transport = Transport('new-server')
    receipts = []
    for domain in ['toys', 'sports']:
        job = f'gs-20261017-{domain}-b-callback-audit-{args.suffix}'
        query = gridctl.query(transport, job)
        assert query['status'] == args.expected_status
        retrieval = gridctl.fetch(transport, job)
        original = gridctl.ROOT / 'jobs/new-server' / job / 'retrieved'
        dest = HERE / 'callback-results' / job
        shutil.copytree(original, dest, dirs_exist_ok=True)
        status = json.loads((dest / 'status.json').read_text())
        assert status['cleanup']['verified'] and not status['worker_may_be_live']
        record = {'domain': domain, 'job_id': job, 'status': status['status'],
                  'exit_code': status['exit_code'], 'cleanup_verified': True,
                  'retrieval': retrieval, 'evidence_directory': str(dest)}
        if args.expected_status == 'SUCCEEDED':
            result = json.loads((dest / 'output/RESULT.json').read_text())
            metadata = json.loads((dest / 'output/checkpoint_callback_metadata.json').read_text())
            assert metadata['status'] == result['status'] == 'PASS'
            assert metadata['embedded_sid_exact'] and metadata['csv_selection_crosscheck']['exact_match']
            assert metadata['model_constructions'] == metadata['GPU_model_calls'] == 0
            assert metadata['all_state_tensors_CPU'] and not metadata['cuda_initialized']
            record['metadata'] = {'path': str(dest / 'output/checkpoint_callback_metadata.json'),
                                  'sha256': sha(dest / 'output/checkpoint_callback_metadata.json')}
        receipts.append(record)
    output = HERE / f'callback-{args.suffix}-collection.json'
    output.write_text(json.dumps({'expected_terminal_status': args.expected_status, 'receipts': receipts}, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'receipt': str(output), 'sha256': sha(output), 'jobs': [{'job_id': r['job_id'], 'status': r['status']} for r in receipts]}))


if __name__ == '__main__':
    main()
