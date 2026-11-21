"""Build and replay an exact, compressed GRID prediction evidence archive."""

import argparse
from array import array
import concurrent.futures
import hashlib
import json
import lzma
import math
from pathlib import Path
import shutil
import struct
import sys
import tempfile
import zipfile

from recheck_saved_grid import check, read_json


HERE = Path(__file__).resolve().parent
CROSS = Path('manifests/grid-a2-fd-recovery-20261018/final-results/cross-domain/cross-domain-evidence-index-with-a2.json')
A2_ARCHIVE = Path('grid-a2-fd-recovery-20261018/deliveries/GRID_A2_FD_RECOVERY_FINAL_20261018T045438Z.zip')
MLP_JOBS = {
    'beauty-mlp-formal-v1': 'Beauty',
    'gf-20261015-toys-mlp-formal-v1': 'Toys',
    'gf-20261015-sports-mlp-formal-v1': 'Sports',
}
CANDIDATES_PER_ROW = 10
TOKENS_PER_SID = 4
AXIS_ROW_BYTES = 16
ROW_FIELDS = ['sample_key', 'sids', 'scores', 'target']


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def file_identity(path):
    with path.open('rb') as stream:
        return {'bytes': path.stat().st_size, 'sha256': hashlib.file_digest(stream, 'sha256').hexdigest()}


def json_line(row):
    return (json.dumps(row) + '\n').encode('ascii')


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')


def byte_planes(raw, width):
    return b''.join(raw[plane::width] for plane in range(width))


def interleave(raw, width):
    values, remainder = divmod(len(raw), width)
    assert remainder == 0
    restored = bytearray(len(raw))
    for plane in range(width):
        restored[plane::width] = raw[plane * values:(plane + 1) * values]
    return restored


def compress(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(lzma.compress(data, preset=6))


def write_axis(package, axes, axis):
    identity = sha256(axis)
    member = f'axes/{identity}.xz'
    path = package / member
    if identity not in axes:
        compress(path, axis)
        axes[identity] = {'member': member, 'uncompressed_bytes': len(axis), **file_identity(path)}
    return identity


def root_relative(path, root):
    return path.relative_to(root).as_posix()


def prepare_cases(repo, source_root, support_root):
    cross = read_json(repo / CROSS)
    cases = {}
    with zipfile.ZipFile(support_root / A2_ARCHIVE) as recovery:
        recovered = {
            Path(member).parts[2]: member
            for member in recovery.namelist()
            if member.startswith('raw/') and member.endswith('/predictions.jsonl')
        }
    for number, route in enumerate(cross['routes']):
        for condition in route['conditions']:
            job = condition['job_id']
            if number < 4:
                previous_source = {'path_relative_to_recommend_support': f'sid-extensions-p0/jobs/new-server/{job}/retrieved/output/test/predictions.jsonl'}
            else:
                previous_source = {'archive_relative_to_recommend_support': A2_ARCHIVE.as_posix(), 'archive_member': recovered[job]}
            cases[job] = {
                'job': job, 'domain': route['domain'], 'route': route['route_id'],
                'condition': condition['condition'], 'shuffle_n': condition['shuffle_n'],
                'published_result': condition.get('evaluation_result_source'),
                'published_review': condition.get('evaluation_review_source'),
                'scientific_evaluation_execution': condition['scientific_evaluation_execution'],
                'previous_source': previous_source,
                **{key: condition[key] for key in ('samples', 'unique_users', 'prediction_sha256', 'duplicate_candidate_count', 'invalid_candidate_count', 'sid')},
                'metrics': condition['metrics_original'],
            }
    for job, domain in MLP_JOBS.items():
        report = read_json(source_root / job / 'recomputed_metrics.json')
        if domain == 'Beauty':
            previous_source = {'archive_relative_to_delivery_root': 'review/beauty-mlp-fixed-originals.tar.gz', 'archive_member': f'{job}/output/test/predictions.jsonl'}
        else:
            previous_source = {'path_relative_to_recommend_support': f'grid-formal-20261015/collected-9a64fd31218d0c68/{job}/output/test/predictions.jsonl'}
        cases[job] = {
            'job': job, 'domain': domain, 'route': f'{domain} MLP', 'condition': 'MLP',
            'previous_source': previous_source,
            **{key: report[key] for key in ('samples', 'unique_users', 'prediction_sha256', 'duplicate_candidate_count', 'invalid_candidate_count', 'metrics')},
        }
    return cases


def extract_case(package, source_root, job, case):
    source = source_root / job / 'predictions.jsonl'
    digest = hashlib.sha256()
    axis = bytearray()
    sid_rows = bytearray()
    sid_values = set()
    scores = array('f')
    original_bytes = 0
    count = 0
    with source.open('rb') as stream:
        for line in stream:
            raw = json.loads(line)
            assert list(raw) == ROW_FIELDS and json_line(raw) == line, (source, count, 'JSON encoding mismatch')
            key = bytes.fromhex(raw['sample_key'])
            assert len(key) == 12 and key.hex() == raw['sample_key'], (source, count, 'sample key')
            assert len(raw['target']) == TOKENS_PER_SID and len(raw['sids']) == len(raw['scores']) == CANDIDATES_PER_ROW
            score_bytes = struct.pack('<10f', *raw['scores'])
            assert all(math.isfinite(score) for score in raw['scores'])
            assert struct.pack('<10d', *struct.unpack('<10f', score_bytes)) == struct.pack('<10d', *raw['scores']), (source, count, 'non-exact float32')
            target = bytes(raw['target'])
            axis.extend(key + target)
            for sid in raw['sids']:
                assert len(sid) == TOKENS_PER_SID
                code = bytes(sid)
                sid_values.add(code)
                sid_rows.extend(code)
            scores.extend(raw['scores'])
            digest.update(line)
            original_bytes += len(line)
            count += 1
    assert case['samples'] == count and digest.hexdigest() == case['prediction_sha256'], source
    assert len(sid_values) < 65536, source
    assert scores.itemsize == 4
    if sys.byteorder != 'little':
        scores.byteswap()
    vocabulary = sorted(sid_values)
    by_code = {code: index for index, code in enumerate(vocabulary)}
    ids = array('H', (by_code[bytes(sid_rows[offset:offset + 4])] for offset in range(0, len(sid_rows), 4)))
    assert ids.itemsize == 2
    if sys.byteorder != 'little':
        ids.byteswap()
    destination = package / 'cases' / job
    candidates = destination / 'candidates.xz'
    packed_scores = destination / 'scores.xz'
    compress(candidates, struct.pack('<I', len(vocabulary)) + b''.join(vocabulary) + byte_planes(ids.tobytes(), 2))
    compress(packed_scores, byte_planes(scores.tobytes(), 4))
    case['previous_raw'] = {
        'path_relative_to_delivery_root': f'payload/grid/predictions/{job}/predictions.jsonl',
        'bytes': original_bytes, 'sha256': digest.hexdigest(),
    }
    case['columns'] = {
        'candidates': {'member': root_relative(candidates, package), **file_identity(candidates)},
        'scores': {'member': root_relative(packed_scores, package), **file_identity(packed_scores)},
    }
    case['candidate_dictionary_rows'] = len(vocabulary)
    print(f'{job}: {count} rows, {original_bytes} bytes, source hash verified', flush=True)
    return job, case, axis


def preserve_other_evidence(package, source_root):
    originals = []
    for path in sorted(source_root.glob('*/*')):
        if path.name == 'predictions.jsonl' or not path.is_file():
            continue
        relative = root_relative(path, source_root)
        copy = package / 'original' / relative
        copy.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == '.jsonl':
            copy = copy.with_name(copy.name + '.xz')
            compress(copy, path.read_bytes())
        else:
            shutil.copyfile(path, copy)
        originals.append({'previous_raw_relative_to_predictions_dir': relative,
                          'source_identity': file_identity(path), 'member': root_relative(copy, package),
                          **file_identity(copy)})
    return originals


def build(args):
    package = args.package
    if (package / 'manifest.json').exists():
        raise ValueError('Choose a fresh package path; existing manifest is protected.')
    package.mkdir(parents=True, exist_ok=True)
    cases = prepare_cases(args.repo, args.source_root, args.support_root)
    assert set(cases) == {path.parent.name for path in args.source_root.glob('*/predictions.jsonl')}
    assert len(cases) == 27 and sum(case['samples'] for case in cases.values()) == 706641
    axes = {}
    tasks = ((package, args.source_root, job, case) for job, case in cases.items())
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        for job, case, axis in pool.map(extract_case_with_args, tasks):
            case['axis_sha256'] = write_axis(package, axes, axis)
            cases[job] = case
    manifest = {'schema': 'grid-exact-predictions-v1', 'cases': dict(sorted(cases.items())),
                'axes': axes, 'unchanged_original_bytes': preserve_other_evidence(package, args.source_root)}
    write_json(package / 'manifest.json', manifest)


def extract_case_with_args(args):
    return extract_case(*args)


def read_member(package, record):
    path = package / record['member']
    assert file_identity(path) == {'bytes': record['bytes'], 'sha256': record['sha256']}, path
    return lzma.decompress(path.read_bytes()) if path.suffix == '.xz' else path.read_bytes()


def restore_case(package, manifest, job, stream):
    case = manifest['cases'][job]
    count = case['samples']
    axis = read_member(package, manifest['axes'][case['axis_sha256']])
    assert len(axis) == count * AXIS_ROW_BYTES and sha256(axis) == case['axis_sha256']
    packed = read_member(package, case['columns']['candidates'])
    dictionary_size, = struct.unpack_from('<I', packed)
    assert dictionary_size == case['candidate_dictionary_rows']
    dictionary_end = 4 + dictionary_size * 4
    dictionary = [list(packed[start:start + 4]) for start in range(4, dictionary_end, 4)]
    ids = interleave(packed[dictionary_end:], 2)
    scores = interleave(read_member(package, case['columns']['scores']), 4)
    assert len(ids) == count * CANDIDATES_PER_ROW * 2 and len(scores) == count * CANDIDATES_PER_ROW * 4
    digest = hashlib.sha256()
    restored_bytes = 0
    for index in range(count):
        row_axis = index * AXIS_ROW_BYTES
        row = {'sample_key': axis[row_axis:row_axis + 12].hex(),
               'sids': [dictionary[key] for key in struct.unpack_from('<10H', ids, index * 20)],
               'scores': list(struct.unpack_from('<10f', scores, index * 40)),
               'target': list(axis[row_axis + 12:row_axis + AXIS_ROW_BYTES])}
        line = json_line(row)
        stream.write(line)
        digest.update(line)
        restored_bytes += len(line)
    assert restored_bytes == case['previous_raw']['bytes'] and digest.hexdigest() == case['prediction_sha256'], job


def extract(args):
    manifest = read_json(args.package / 'manifest.json')
    with args.output.open('xb') as stream:
        restore_case(args.package, manifest, args.job, stream)


def replay_case(args):
    package, manifest, job = args
    with tempfile.TemporaryDirectory(prefix=f'grid-prediction-{job}-') as workspace:
        restored = Path(workspace) / 'predictions.jsonl'
        with restored.open('wb') as stream:
            restore_case(package, manifest, job, stream)
        return job, check(restored, manifest['cases'][job])


def replay(args):
    manifest = read_json(args.package / 'manifest.json')
    for record in manifest['unchanged_original_bytes']:
        data = read_member(args.package, record)
        assert {'bytes': len(data), 'sha256': sha256(data)} == record['source_identity'], record['member']
    cases = manifest['cases']
    tasks = ((args.package, manifest, job) for job in cases)
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = dict(pool.map(replay_case, tasks))
    summary = {'cases': len(results), 'rows': sum(row['samples'] for row in results.values()),
               'max_absolute_difference': max(max(row['diff_abs'].values()) for row in results.values()),
               'prediction_jsonl_bytes_restored': sum(case['previous_raw']['bytes'] for case in cases.values()),
               'canonical_original_jsonl_sha256_matches_every_case': True,
               'unchanged_original_artifacts': len(manifest['unchanged_original_bytes']),
               'results': results}
    assert summary['cases'] == 27 and summary['rows'] == 706641
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    tools = parser.add_subparsers(dest='command', required=True)
    encoder = tools.add_parser('build')
    encoder.add_argument('--source-root', type=Path, required=True)
    encoder.add_argument('--repo', type=Path, required=True)
    encoder.add_argument('--support-root', type=Path, required=True)
    encoder.add_argument('--package', type=Path, default=HERE)
    encoder.add_argument('--workers', type=int, default=4)
    encoder.set_defaults(execute=build)
    decoder = tools.add_parser('extract')
    decoder.add_argument('--package', type=Path, default=HERE)
    decoder.add_argument('--job', required=True)
    decoder.add_argument('--output', type=Path, required=True)
    decoder.set_defaults(execute=extract)
    checker = tools.add_parser('replay')
    checker.add_argument('--package', type=Path, default=HERE)
    checker.add_argument('--workers', type=int, default=2)
    checker.set_defaults(execute=replay)
    args = parser.parse_args()
    args.execute(args)


if __name__ == '__main__':
    main()
