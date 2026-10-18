"""Local synthetic package-contract tests; never use a real secret or experiment.

Fixtures are review-only artifacts and must not be copied into the final bundle.
The token below is deliberately synthetic and built from repeated characters.
"""
import datetime as dt
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import zipfile
import zlib

OUT = Path(__file__).resolve().parent
BASE = OUT.parents[1]
source = BASE / 'package_delivery.py'
module_spec = importlib.util.spec_from_file_location('independent_delivery_package', source)
package = importlib.util.module_from_spec(module_spec)
module_spec.loader.exec_module(package)
stamp = dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
root = OUT / 'fixtures' / stamp
root.mkdir(parents=True)


def stage(name):
    path = root / name
    (path / 'review').mkdir(parents=True)
    (path / 'review/final-internal-review.json').write_text(json.dumps({'status': 'PARTIAL', 'scope': 'SYNTHETIC_PACKAGE_FIXTURE_ONLY'}) + '\n')
    return path


def test(name, function):
    try:
        value = function()
        return dict(case=name, accepted=True, detail=value)
    except BaseException as error:
        return dict(case=name, accepted=False, error=repr(error))


normal = stage('normal')
(normal / 'rows.json').write_text('{"fixture": true}\n')
normal_result = test('normal_PARTIAL_package', lambda: package.build(normal, root / 'normal.zip'))
assert normal_result['accepted'] and normal_result['detail']['scientific_review_status'] == 'PARTIAL'

alias_path = root / 'canonical-alias.zip'
payload = {'evidence/a.txt': b'first', 'evidence/./a.txt': b'second'}
records = [dict(path=k, bytes=len(v), sha256=hashlib.sha256(v).hexdigest(), crc32=f'{zlib.crc32(v) & 0xffffffff:08x}') for k, v in payload.items()]
with zipfile.ZipFile(alias_path, 'x') as archive:
    for name, data in payload.items():
        archive.writestr(name, data)
    archive.writestr('manifest.json', json.dumps(dict(files=records, scientific_review_status='PARTIAL', fixture_only=True)))
alias_result = test('reject_path_alias_evidence_dot_a_vs_canonical', lambda: package.verify(alias_path))

extensionless = stage('extensionless-synthetic-token')
token = b'ghp_' + b'A' * 40
assert package.SECRET_PATTERNS['github_token'].search(token)
(extensionless / 'NOTES').write_bytes(b'FAKE_TEST_VALUE=' + token + b'\n')
extensionless_result = test('reject_supported_secret_pattern_in_extensionless_payload', lambda: package.build(extensionless, root / 'extensionless.zip'))

dot_env = stage('env-variant-synthetic-token')
(dot_env / '.env.local').write_bytes(b'FAKE_TEST_VALUE=' + token + b'\n')
env_result = test('reject_env_variant_payload_with_supported_secret_pattern', lambda: package.build(dot_env, root / 'env-variant.zip'))

changed_crc = root / 'modified-payload.zip'
with zipfile.ZipFile(changed_crc, 'x') as archive:
    archive.writestr('rows.json', b'modified payload')
    archive.writestr('manifest.json', json.dumps(dict(files=[dict(path='rows.json', bytes=1, sha256='0' * 64, crc32='00000000')], scientific_review_status='PARTIAL')))
crc_result = test('reject_hash_size_manifest_mismatch', lambda: package.verify(changed_crc))
assert not crc_result['accepted']

secret_zip = root / 'verify-secret-bytes.zip'
content = b'FAKE_TEST_VALUE=' + token + b'\n'
with zipfile.ZipFile(secret_zip, 'x') as archive:
    archive.writestr('payload/NOTES', content)
    archive.writestr('manifest.json', json.dumps(dict(files=[dict(path='payload/NOTES', bytes=len(content), sha256=hashlib.sha256(content).hexdigest(), crc32=f'{zlib.crc32(content) & 0xffffffff:08x}')], scientific_review_status='PARTIAL')))
verify_secret_result = test('verify_independently_rejects_supported_secret_even_with_valid_hashes', lambda: package.verify(secret_zip))
assert not verify_secret_result['accepted']

corrupt = root / 'real-CRC-corruption.zip'
with zipfile.ZipFile(corrupt, 'x', compression=zipfile.ZIP_STORED) as archive:
    archive.writestr('rows.json', b'{"fixture":true}')
    archive.writestr('manifest.json', json.dumps(dict(files=[dict(path='rows.json', bytes=16, sha256='0'*64, crc32='00000000')], scientific_review_status='PARTIAL')))
with zipfile.ZipFile(corrupt) as archive:
    offset = archive.getinfo('rows.json').header_offset
data = bytearray(corrupt.read_bytes())
name_length, extra_length = struct.unpack_from('<HH', data, offset + 26)
payload_offset = offset + 30 + name_length + extra_length
data[payload_offset] ^= 1
corrupt.write_bytes(data)
real_crc_result = test('reject_actual_stored_ZIP_payload_CRC_corruption', lambda: package.verify(corrupt))
assert not real_crc_result['accepted']

result = dict(
    reviewer='/root/fd_repair_reviewer', at=dt.datetime.now(dt.timezone.utc).isoformat(),
    package_tool_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
    status='PACKAGE_TOOL_CONTRACT_GAPS_REPRODUCED' if any(r['accepted'] for r in [alias_result, extensionless_result, env_result]) else 'PACKAGE_TOOL_FIXTURES_PASS',
    cases=[normal_result, alias_result, extensionless_result, env_result, crc_result, verify_secret_result, real_crc_result],
    fixtures_path=str(root), fixture_secret_is_synthetic=True, no_real_secret_used=True,
    production_artifact_or_experiment_modified=False, GPU_requested=0,
    packaging_note='Review fixtures intentionally include invalid paths and synthetic secret patterns. Exclude the fixtures subtree from the final evidence staging; retain this summary and test source as the verification record.',
)
target = OUT / ('package-fixture-review-' + stamp + '.json')
target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
print(json.dumps(dict(path=str(target),sha256=hashlib.sha256(target.read_bytes()).hexdigest(),status=result['status'],cases=[dict(case=r['case'],accepted=r['accepted']) for r in result['cases']])))
