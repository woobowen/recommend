"""Independently compare CPU dictionary exports to original small SID payloads.

Uses a restricted non-torch tensor metadata reader plus raw ZIP storage bytes;
only known tensor rebuild metadata and int storage types are accepted. It never
constructs a model or initializes CUDA. Optional fetch is exact hash-bound SCP.
"""
import argparse
import collections
import datetime as dt
import hashlib
import io
import json
from pathlib import Path
import pickle
import sys
import zipfile

import numpy as np


ROOT = Path('/home/addaswsw/recommend-support/grid-shuffle-20261017')
ASSETS = ROOT/'asset-audit'


def identity(path):
    path = Path(path); raw = path.read_bytes()
    return {'path': str(path), 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


def read(path):
    return json.loads(Path(path).read_bytes())


def tensor_metadata(storage, offset, size, stride, *unused):
    assert isinstance(storage, dict) and storage['kind'] == 'storage'
    assert isinstance(offset, int) and isinstance(size, tuple) and isinstance(stride, tuple)
    assert len(size) == len(stride) == 2
    return {'storage': storage, 'offset': offset, 'size': size, 'stride': stride}


class MetadataUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch._utils' and name in ('_rebuild_tensor_v2', '_rebuild_tensor'):
            return tensor_metadata
        if module == 'torch' and name in ('LongStorage', 'IntStorage'):
            return {'LongStorage': 'i8', 'IntStorage': 'i4'}[name]
        if module == 'collections' and name == 'OrderedDict':
            return collections.OrderedDict
        raise ValueError(f'Unexpected global in SID tensor metadata: {module}.{name}')

    def persistent_load(self, value):
        assert len(value) == 5 and value[0] == 'storage' and value[1] in ('i8', 'i4')
        assert isinstance(value[2], str) and isinstance(value[3], str) and isinstance(value[4], int)
        return {'kind': 'storage', 'dtype': value[1], 'key': value[2], 'device': value[3], 'elements': value[4]}


def read_sid(path):
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        assert len(names) == len(set(names))
        metadata_names = [n for n in names if n.endswith('/data.pkl')]
        assert len(metadata_names) == 1
        base = metadata_names[0].removesuffix('data.pkl')
        metadata = MetadataUnpickler(io.BytesIO(archive.read(metadata_names[0]))).load()
        assert metadata['size'][0] == 4 and all(v > 0 for v in metadata['size'])
        byteorder = archive.read(base+'byteorder').decode() if base+'byteorder' in names else 'little'
        assert byteorder in ('little', 'big')
        storage = metadata['storage']
        dtype = np.dtype(('<' if byteorder == 'little' else '>')+storage['dtype'])
        raw = archive.read(base+'data/'+storage['key'])
        assert len(raw) == storage['elements']*dtype.itemsize
        values = np.frombuffer(raw, dtype=dtype)
        positions = metadata['offset'] + np.arange(metadata['size'][0])[:, None]*metadata['stride'][0] + np.arange(metadata['size'][1])[None, :]*metadata['stride'][1]
        assert positions.min() >= 0 and positions.max() < len(values)
        tensor = values[positions]
        assert tensor.dtype.kind == 'i' and (tensor >= 0).all()
        return tensor, {'shape': list(metadata['size']), 'stride': list(metadata['stride']), 'storage_offset': metadata['offset'],
                        'storage_dtype': str(dtype), 'storage_elements': storage['elements'], 'byteorder': byteorder,
                        'container_members': len(names), 'restricted_metadata_unpickler': True}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fetch', action='store_true')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    assert not a.output.exists()
    candidates = read(ASSETS/'candidate_specs.json')
    source_script = ASSETS/'export_sid_dictionary.py'
    original_calculation = Path('/home/addaswsw/recommend-worktrees/sid-extensions-p0/manifests/grid-formal-20261015/export_sid_dictionary.py')
    if a.fetch:
        sys.path.insert(0, '/home/addaswsw/recommend-support/sid-extensions-p0/scripts')
        from transport import Transport
        transport = Transport('new-server')
        remote_identity = transport.identity()
    else:
        remote_identity = None
    reviews = []
    for domain in ('toys', 'sports'):
        sid_asset = candidates[domain]['sid']
        local_sid = a.output.parent/'raw-sid-for-audit'/f'{domain}-{sid_asset["sha256"][:16]}.pt'
        if a.fetch:
            transport.get_verified(sid_asset['path'], local_sid, sid_asset, timeout=45)
        sid_identity = identity(local_sid)
        assert all(sid_identity[name] == sid_asset[name] for name in ('bytes', 'sha256'))
        sid, tensor_receipt = read_sid(local_sid)
        independent_legal = sorted({tuple(map(int, row)) for row in sid.T})
        job = f'gs-20261017-{domain}-b-sid-dictionary-v1'
        directory = ASSETS/'sid-dictionary-results'/job
        manifest = read(directory/'export-manifest.json')
        names = {v['path'] for v in manifest['files']}
        assert len(names) == len(manifest['files'])
        assert {str(v.relative_to(directory)) for v in directory.rglob('*') if v.is_file()} == names|{'export-manifest.json'}
        for row in manifest['files']:
            member = directory/row['path']
            assert not member.is_symlink()
            assert all(identity(member)[n] == row[n] for n in ('bytes', 'sha256'))
        request, accepted, status, invocation, result, dictionary = [read(directory/name) for name in ('request.json', 'accepted.json', 'status.json', 'output/invocation.json', 'output/RESULT.json', 'output/sid_dictionary.json')]
        spec_path = ASSETS/'sid-dictionary-specs'/f'{domain}.json'
        spec = read(spec_path)
        fingerprint = hashlib.sha256(json.dumps(request, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()
        assert accepted['request_sha256'] == fingerprint
        assert status['status'] == 'SUCCEEDED' and status['exit_code'] == 0
        assert status['cleanup']['verified'] and status['cleanup']['last_observation']['verified_empty'] and not status['worker_may_be_live']
        assert status['job_id'] == invocation['job_id'] == dictionary['identity']['job_id'] == job
        assert request['env']['CUDA_VISIBLE_DEVICES'] == invocation['CUDA_VISIBLE_DEVICES'] == ''
        assert dictionary['model_constructions'] == dictionary['GPU_model_calls'] == 0 and not dictionary['cuda_initialized']
        assert result['model_constructions'] == result['GPU_model_calls'] == 0 and not result['cuda_initialized']
        assert result['status'] == 'PASS' and result['domain'] == spec['domain'] == domain
        assert dictionary['source'] == result['source'] == spec['sid'] == sid_asset
        assert dictionary['shape'] == list(sid.shape) == [4, spec['expected_item_count']]
        assert dictionary['row_count'] == result['row_count'] == sid.shape[1]
        assert list(map(tuple, dictionary['legal_raw_sids'])) == independent_legal
        assert dictionary['unique_four_code_count'] == result['unique_four_code_count'] == len(independent_legal)
        assert dictionary['duplicate_four_code_rows'] == sid.shape[1]-len(independent_legal)
        assert dictionary['calculation_source']['sha256'] == identity(original_calculation)['sha256']
        assert dictionary['identity']['script'] == invocation['script'] == spec['script']
        assert identity(source_script)['sha256'] == spec['script']['sha256']
        assert dictionary['identity']['spec_sha256'] == invocation['spec']['sha256'] == identity(spec_path)['sha256']
        inputs = {row['path']: row for row in request['inputs']}
        assert all(inputs[sid_asset['path']][n] == sid_asset[n] for n in ('bytes', 'sha256'))
        assert inputs[spec['script']['path']]['sha256'] == identity(source_script)['sha256']
        assert inputs[invocation['spec']['path']]['sha256'] == identity(spec_path)['sha256']
        assert invocation['interpreter'] == request['argv'][0] == '/data/bwwu/envs/grid-20261009-e61baf3/bin/python'
        assert invocation['argv'] == request['argv'][2:]
        assert request['capture_version'] == spec['source_commit']+'/sid-dictionary/'+identity(source_script)['sha256']+'/'+identity(spec_path)['sha256']
        dictionary_identity = identity(directory/'output/sid_dictionary.json')
        assert all(dictionary_identity[n] == result['dictionary'][n] for n in ('bytes', 'sha256'))
        artifacts = {row['path']: row for row in status['artifacts']}
        for name in ('output/RESULT.json', 'output/invocation.json', 'output/sid_dictionary.json'):
            assert all(identity(directory/name)[n] == artifacts[name][n] for n in ('bytes', 'sha256'))
        wall = (dt.datetime.fromisoformat(status['ended_at'])-dt.datetime.fromisoformat(status['started_at'])).total_seconds()
        assert 0 < wall < request['timeout_seconds'] <= 120
        reviews.append({'domain': domain, 'job_id': job, 'status': 'INDEPENDENT_RAW_SID_DICTIONARY_PROJECTION_PASS',
            'exact_raw_storage_four_code_projection': True, 'source': sid_asset, 'dictionary': dictionary_identity,
            'raw_SID_local': sid_identity, 'raw_tensor_reader': tensor_receipt, 'unique_legal_four_code_rows': len(independent_legal),
            'all_export_members_sha256_verified': len(manifest['files']), 'accepted_request_sha256': fingerprint,
            'source_commit': spec['source_commit'], 'worker_wall_seconds_CPU_only': wall,
            'runtime_CUDA_not_initialized_and_no_model_calls': True, 'terminal_cleanup_verified': True,
            'inputs': [identity(directory/n) for n in ('request.json','accepted.json','status.json','export-manifest.json','output/RESULT.json','output/invocation.json')]+[identity(spec_path)]})
    receipt = {'status': 'INDEPENDENT_SID_DICTIONARY_PROJECTIONS_PASS', 'reviewer': '/root/results_audit',
        'reviewed_at_utc': dt.datetime.now(dt.timezone.utc).isoformat(), 'program': identity(Path(__file__)),
        'producer_program': identity(source_script), 'calculation_source': identity(original_calculation), 'argv': sys.argv,
        'reviews': reviews, 'raw_SID_transfer_bytes_total': sum(v['raw_SID_local']['bytes'] for v in reviews),
        'remote_identity_if_fetched': remote_identity, 'new_model_or_GPU_calls': 0,
        'original_payload_or_source_writes': 0, 'Git_operations': 0,
        'scope_limit': 'This certifies exact dictionary projection from matching raw SID payloads and original detached receipts; historical checkpoint/SID semantic pairing is covered by the separate asset audit, not inferred from this projection.'}
    a.output.write_text(json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'review': identity(a.output), 'status': receipt['status'], 'unique_rows': {v['domain']: v['unique_legal_four_code_rows'] for v in reviews}}, ensure_ascii=False))


if __name__ == '__main__':
    main()
