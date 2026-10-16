"""Independent read-only delivery ZIP audit; stdlib only, no unsafe extraction.

Checks every ZIP/NPZ CRC, byte count and SHA, safe/unique names, numeric NPY
layout, all record-to-NPZ array references, and obvious credential signatures.
An external member inventory is required for the final audit to compare the
archive against the collector's intended bytes. The audit report stays outside
the reviewed ZIP; no self-referential report/hash is manufactured.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import datetime as dt
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import stat
import struct
import zipfile

TEXT_SUFFIXES = {'.json', '.jsonl', '.yaml', '.yml', '.md', '.txt', '.py', '.sh', '.log', '.csv'}
SECRET_PATTERNS = {
    'private_key_block': re.compile(rb'-----BEGIN (?:OPENSSH|RSA|EC|DSA|ENCRYPTED)? ?PRIVATE KEY-----'),
    'github_pat': re.compile(rb'(?<![A-Za-z0-9])(?:gh[pousr]_[A-Za-z0-9]{30,255}|github_pat_[A-Za-z0-9_]{50,255})(?![A-Za-z0-9])'),
    'aws_access_key': re.compile(rb'(?<![A-Za-z0-9])(?:AKIA|ASIA)[A-Z0-9]{16}(?![A-Za-z0-9])'),
    'hf_access_token': re.compile(rb'(?<![A-Za-z0-9])hf_[A-Za-z0-9]{30,255}(?![A-Za-z0-9])'),
    'openai_token': re.compile(rb'(?<![A-Za-z0-9])sk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{40,255}(?![A-Za-z0-9])'),
    'literal_bearer_token': re.compile(rb'(?i)authorization[\s\"\x27:=]+bearer\s+[A-Za-z0-9._~-]{24,}'),
}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def safe_name(name):
    p = PurePosixPath(name)
    return (bool(name) and '\\' not in name and not p.is_absolute()
            and '..' not in p.parts and '.' not in p.parts
            and not re.match(r'^[A-Za-z]:', name) and '\x00' not in name
            and str(p) == name.rstrip('/'))


def scan_secret(name, data):
    """Return locations/signature types only; never emit secret contents."""
    findings = []
    if PurePosixPath(name).suffix.lower() in TEXT_SUFFIXES:
        for kind, regex in SECRET_PATTERNS.items():
            for match in regex.finditer(data):
                findings.append({'member': name, 'kind': kind,
                                 'line': data.count(b'\n', 0, match.start()) + 1})
    base = PurePosixPath(name).name.lower()
    if base in {'id_rsa', 'id_ed25519', 'id_ecdsa', 'credentials', '.env', 'known_hosts'} or base.endswith(('.pem', '.key')):
        findings.append({'member': name, 'kind': 'credential_like_filename'})
    return findings


def inspect_npy(data):
    assert data[:6] == b'\x93NUMPY', 'NPY magic'
    major, minor = data[6:8]
    if major == 1:
        header_size, offset, encoding = struct.unpack('<H', data[8:10])[0], 10, 'latin1'
    elif major in (2, 3):
        header_size, offset, encoding = struct.unpack('<I', data[8:12])[0], 12, ('utf8' if major == 3 else 'latin1')
    else:
        raise AssertionError('unsupported NPY version')
    assert header_size < 1024 * 1024
    header = ast.literal_eval(data[offset:offset + header_size].decode(encoding).strip())
    assert set(header) == {'descr', 'fortran_order', 'shape'}
    assert type(header['fortran_order']) is bool
    shape = header['shape']
    assert isinstance(shape, tuple) and all(type(n) is int and n >= 0 for n in shape)
    dtype = header['descr']
    assert isinstance(dtype, str) and re.fullmatch(r'[<>=|][biuf][0-9]+', dtype), 'numeric/bool arrays only'
    length = 1
    for dimension in shape:
        length *= dimension
    item_size = int(dtype[2:])
    assert item_size in (1, 2, 4, 8, 16)
    assert len(data) == offset + header_size + item_size * length, 'NPY storage size mismatch'
    return {'shape': list(shape), 'descr': dtype, 'fortran_order': header['fortran_order'],
            'storage_bytes': item_size * length, 'sha256': digest(data)}


def inspect_npz(data):
    arrays = {}
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        infos = archive.infolist()
        assert len({i.filename for i in infos}) == len(infos), 'duplicate NPZ member'
        assert archive.testzip() is None, 'NPZ CRC'
        for info in infos:
            assert safe_name(info.filename) and not info.is_dir() and info.filename.endswith('.npy')
            assert not (info.flag_bits & 1), 'encrypted NPZ member'
            arrays[info.filename[:-4]] = inspect_npy(archive.read(info))
    return arrays


def normalize_dtype(dtype):
    code = {'bool': 'b1', 'float16': 'f2', 'float32': 'f4', 'float64': 'f8',
            'int8': 'i1', 'int16': 'i2', 'int32': 'i4', 'int64': 'i8',
            'uint8': 'u1', 'uint16': 'u2', 'uint32': 'u4', 'uint64': 'u8'}
    return code[dtype]


def array_refs(value):
    if isinstance(value, dict):
        if '__array__' in value:
            yield value
        else:
            for child in value.values():
                yield from array_refs(child)
    elif isinstance(value, list):
        for child in value:
            yield from array_refs(child)


def load_inventory(path):
    raw = json.loads(path.read_text())
    rows = raw['files'] if isinstance(raw, dict) else raw
    rows = [{k: row[k] for k in ('path', 'bytes', 'sha256')} for row in rows]
    assert len({row['path'] for row in rows}) == len(rows), 'duplicate intended inventory path'
    assert all(safe_name(row['path']) and type(row['bytes']) is int and row['bytes'] >= 0
               and re.fullmatch('[0-9a-f]{64}', row['sha256']) for row in rows)
    return rows


def audit(args):
    zip_data = args.zip.read_bytes()
    members, directories, npz, records, secrets = [], [], {}, {}, []
    inner_manifest = None
    with zipfile.ZipFile(io.BytesIO(zip_data)) as archive:
        infos = archive.infolist()
        names = [info.filename for info in infos]
        assert len(set(names)) == len(names), 'duplicate outer ZIP members'
        assert all(safe_name(name) for name in names), 'unsafe outer ZIP path'
        assert archive.testzip() is None, 'outer ZIP CRC'
        for info in infos:
            mode = info.external_attr >> 16
            assert not stat.S_ISLNK(mode), 'symlink ZIP entry'
            assert not (info.flag_bits & 1), 'encrypted ZIP entry'
            if info.is_dir():
                directories.append(info.filename)
                continue
            data = archive.read(info)
            assert len(data) == info.file_size
            members.append({'path': info.filename, 'bytes': len(data), 'sha256': digest(data)})
            secrets.extend(scan_secret(info.filename, data))
            if info.filename == 'MANIFEST.json':
                inner_manifest = json.loads(data)
            if info.filename.endswith('.npz'):
                npz[info.filename] = inspect_npz(data)
            if PurePosixPath(info.filename).name == 'record.json':
                document = json.loads(data)
                if document.get('schema') == 'grid-beauty-cross-attention-v1':
                    records[info.filename] = document
    assert not secrets, 'obvious secret signature locations (no values): ' + json.dumps(secrets)
    inventory_match = None
    if args.inventory:
        expected = load_inventory(args.inventory)
        assert sorted(expected, key=lambda row: row['path']) == sorted(members, key=lambda row: row['path']), 'ZIP inventory differs from intended files'
        inventory_match = {'path': str(args.inventory), 'sha256': digest(args.inventory.read_bytes()), 'files': len(expected)}
    if args.expected_npz is not None:
        assert len(npz) == args.expected_npz, 'wrong complete raw NPZ count'
        assert len(records) == args.expected_npz, 'wrong complete raw record count'
    ref_count = 0
    by_name = {row['path']: row for row in members}
    if inner_manifest is not None:
        inner_rows = [{key: row[key] for key in ('path', 'bytes', 'sha256')}
                      for row in inner_manifest['files']]
        assert len({row['path'] for row in inner_rows}) == len(inner_rows)
        assert sorted(inner_rows, key=lambda row: row['path']) == sorted(
            [row for row in members if row['path'] != 'MANIFEST.json'], key=lambda row: row['path']), 'inner manifest inventory mismatch'
        if args.content_commit:
            assert inner_manifest['commit'] == args.content_commit, 'inner manifest content commit mismatch'
    for name, document in records.items():
        assert document['arrays_file'] == 'arrays.npz'
        neighbor = str(PurePosixPath(name).parent / document['arrays_file'])
        assert neighbor in npz and by_name[neighbor]['sha256'] == document['arrays_sha256']
        references = list(array_refs(document['bundle']))
        assert {ref['__array__'] for ref in references} == set(npz[neighbor]), 'unreferenced/missing NPZ array'
        for ref in references:
            value = npz[neighbor][ref['__array__']]
            assert value['shape'] == ref['shape'] and value['descr'][1:] == normalize_dtype(ref['dtype']), 'record/NPY layout mismatch'
        ref_count += len(references)
    for required in args.require:
        assert any(re.search(required, row['path']) for row in members), 'required member pattern missing: ' + required
    return {'status': 'PASS', 'reviewer_task': '/root/delivery_audit',
            'reviewed_at_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
            'zip': {'path': str(args.zip), 'bytes': len(zip_data), 'sha256': digest(zip_data)},
            'files': len(members), 'directories': len(directories),
            'uncompressed_bytes': sum(row['bytes'] for row in members),
            'extension_counts': dict(Counter(PurePosixPath(row['path']).suffix for row in members)),
            'all_outer_and_nested_NPZ_CRC_PASS': True, 'all_members_read_and_hashed': True,
            'safe_unique_unencrypted_regular_members': True, 'obvious_secret_signatures': secrets,
            'numeric_NPZ_count': len(npz), 'verified_record_count': len(records),
            'verified_array_references': ref_count, 'intended_inventory_match': inventory_match,
            'inner_manifest_members_checked': len(inner_manifest['files']) if inner_manifest else None,
            'inner_manifest_content_commit': inner_manifest.get('commit') if inner_manifest else None,
            'members': members,
            'scope_limit': 'Byte/format/package review; model/beam/head/statistical semantics separately reviewed from original evidence. Signature scan is not proof of absence of every possible secret.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--zip', required=True, type=Path)
    parser.add_argument('--inventory', type=Path)
    parser.add_argument('--expected-npz', type=int)
    parser.add_argument('--content-commit')
    parser.add_argument('--require', action='append', default=[])
    parser.add_argument('--report', required=True, type=Path)
    args = parser.parse_args()
    assert args.report.resolve() != args.zip.resolve()
    try:
        report = audit(args)
    except Exception as error:
        report = {'status': 'FAIL', 'reviewer_task': '/root/delivery_audit',
                  'reviewed_at_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
                  'zip': str(args.zip), 'exception': type(error).__name__, 'reason': str(error)}
    report['program'] = {'path': str(Path(__file__).resolve()), 'sha256': digest(Path(__file__).read_bytes())}
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({k: report.get(k) for k in ('status', 'zip', 'files', 'numeric_NPZ_count', 'reason')}, ensure_ascii=False))
    raise SystemExit(0 if report['status'] == 'PASS' else 1)


if __name__ == '__main__':
    main()
