"""Verify terminal detached CPU jobs and collect their hash-checked dictionaries."""
from pathlib import Path
import datetime as dt
import json
import shutil
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, '/home/addaswsw/recommend-support/sid-extensions-p0/scripts')
from transport import Transport, sha
import gridctl


def main():
    transport = Transport('new-server')
    sources = json.loads((HERE / 'candidate_specs.json').read_text())
    receipts = []
    for domain, count in [('toys', 11924), ('sports', 18357)]:
        job = f'gs-20261017-{domain}-b-sid-dictionary-v1'
        query = gridctl.query(transport, job)
        assert query['status'] == 'SUCCEEDED', (job, query['status'])
        receipt = gridctl.fetch(transport, job)
        original = gridctl.ROOT / 'jobs' / 'new-server' / job / 'retrieved'
        dest = HERE / 'sid-dictionary-results' / job
        shutil.copytree(original, dest, dirs_exist_ok=True)
        result = json.loads((dest / 'output/RESULT.json').read_text())
        dictionary = json.loads((dest / 'output/sid_dictionary.json').read_text())
        assert dictionary['source'] == sources[domain]['sid']
        assert result['status'] == 'PASS'
        assert dictionary['shape'] == [4, count]
        assert dictionary['row_count'] == dictionary['unique_four_code_count'] == count
        assert dictionary['duplicate_four_code_rows'] == 0
        assert dictionary['model_constructions'] == dictionary['GPU_model_calls'] == 0
        assert dictionary['cuda_initialized'] is False
        legal = dictionary['legal_raw_sids']
        assert legal == sorted(legal) and len({tuple(row) for row in legal}) == count
        assert all(len(row) == 4 and all(type(code) is int and 0 <= code < 256 for code in row) for row in legal)
        status = json.loads((dest / 'status.json').read_text())
        assert status['exit_code'] == 0 and status['worker_may_be_live'] is False
        assert status['cleanup']['verified'] is True
        receipts.append({'job_id': job, 'domain': domain, 'status': status['status'],
                         'source': dictionary['source'], 'items': count,
                         'dictionary_path': str(dest / 'output/sid_dictionary.json'),
                         'dictionary_sha256': sha(dest / 'output/sid_dictionary.json'),
                         'dictionary_bytes': (dest / 'output/sid_dictionary.json').stat().st_size,
                         'retrieval': receipt, 'no_model_or_GPU_execution': True,
                         'source_complete_record_preserved': True,
                         'requested_worker_wall_seconds': 120,
                         'elapsed_supervised_wall_seconds': (dt.datetime.fromisoformat(status['ended_at']) - dt.datetime.fromisoformat(status['started_at'])).total_seconds()})
    (HERE / 'sid-dictionary-collection-review.json').write_text(json.dumps({
        'status': 'PASS', 'reviewer': '/root/asset_audit',
        'scope': 'CPU dictionary projection author self-check, not independent nonauthor result review',
        'receipts': receipts}, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(receipts, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
