"""Submit two hash-bound read-only CPU audits of original checkpoint callbacks."""
from pathlib import Path
import json
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, '/home/addaswsw/recommend-support/sid-extensions-p0/scripts')
from transport import Transport, sha
import gridctl


def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')


def main():
    script = HERE / 'inspect_checkpoint_callbacks.py'
    script_sha = sha(script)
    previous = json.loads((HERE.parent / 'gs-20261017-toys-b-preflight-r2-request.json').read_text())
    assets = json.loads((HERE / 'candidate_specs.json').read_text())
    source_review = json.loads((HERE / 'asset-source-review.json').read_text())
    deployment = next(row for row in previous['inputs'] if row['path'].endswith('/DEPLOYMENT.json'))
    transport = Transport('new-server')
    transport.identity()
    base = transport.profile['remote_support'] + '/grid-shuffle-20261017/asset-audit'
    remote_script = base + '/scripts/' + script_sha + '/inspect_checkpoint_callbacks.py'
    save(HERE / 'callback-script-deployment.json', transport.put(script, remote_script))
    for domain in ['toys', 'sports']:
        selection = source_review['B'][domain]['checkpoint_selection']
        spec = {
            'domain': domain, 'route': 'B', **assets[domain],
            'expected_selected_step': selection['selected_step'],
            'expected_best_validation_recall5': float(selection['independent_csv_maximum']['maximum']['val/recall@5']),
            'source_commit': previous['capture_version'].split('/')[0],
            'snapshot': {'path': previous['cwd'], 'capture_version': previous['capture_version'],
                         'deployment_sha256': deployment['sha256']},
            'script': {'path': remote_script, 'sha256': script_sha, 'bytes': script.stat().st_size},
        }
        local_spec = HERE / 'callback-specs' / (domain + '.json')
        save(local_spec, spec)
        spec_sha = sha(local_spec)
        remote_spec = base + '/specs/' + spec_sha + '.json'
        save(HERE / 'callback-specs' / (domain + '-deployment.json'), transport.put(local_spec, remote_spec))
        request = {key: previous[key] for key in ['schema', 'host', 'user', 'uid', 'namespace', 'deployment_sha256']}
        request.update({
            'job_id': f'gs-20261017-{domain}-b-callback-audit-v1',
            'capture_version': previous['capture_version'] + '/checkpoint-callback/' + script_sha + '/' + spec_sha,
            'argv': ['/data/bwwu/envs/grid-20261009-e61baf3/bin/python', '-B', remote_script, '--spec', remote_spec],
            'cwd': previous['cwd'], 'timeout_seconds': 120,
            'env': {'CUDA_VISIBLE_DEVICES': '', 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
                    'TF_CPP_MIN_LOG_LEVEL': '3', 'TF_NUM_INTRAOP_THREADS': '1', 'TF_NUM_INTEROP_THREADS': '1',
                    'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONUNBUFFERED': '1',
                    'PYTHONPATH': previous['cwd'], 'HF_HUB_OFFLINE': '1',
                    'WANDB_MODE': 'disabled', 'TOKENIZERS_PARALLELISM': 'false'},
            'inputs': [spec['script'], {'path': remote_spec, 'sha256': spec_sha}, deployment,
                       spec['config'], spec['sid'], spec['checkpoint']],
        })
        local_request = HERE / 'callback-requests' / (request['job_id'] + '.json')
        save(local_request, request)
        save(HERE / 'callback-requests' / (request['job_id'] + '-submission.json'), gridctl.submit(transport, local_request))
        query = gridctl.query(transport, request['job_id'])
        save(HERE / 'callback-requests' / (request['job_id'] + '-startup-query.json'), query)
        print(json.dumps({'job_id': request['job_id'], 'script_sha256': script_sha, 'startup_status': query['status']}))


if __name__ == '__main__':
    main()
