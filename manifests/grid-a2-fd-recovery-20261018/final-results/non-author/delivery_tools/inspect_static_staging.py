"""Read-only local audit of static copies; no scientific recomputation/import."""
import ast
import collections
import datetime as dt
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess
import sys
import unicodedata

OUT = Path(__file__).resolve().parent
BASE = OUT.parents[1]
REPO = Path('/home/addaswsw/recommend-worktrees/sid-extensions-p0')
STAGE = Path(sys.argv[1]) if len(sys.argv) > 1 else BASE / 'deliveries/final-evidence-staging-v2'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def identity(path):
    return dict(path=str(path), bytes=path.stat().st_size, sha256=sha(path))


manifest_path = STAGE / 'assembly/static-inputs.json'
manifest = json.loads(manifest_path.read_text())
assert manifest['status'] == 'STATIC_REVIEW_INPUTS_STAGED_NOT_FINAL_DELIVERY'
assert manifest['scientific_task_final_review_pass'] is False and manifest['a2_final_outputs_not_staged']
assert manifest['final_zip_not_built'] and manifest['remote_operations'] == 0
records = manifest['files']
assert manifest['members'] == len(records) == len({r['path'] for r in records})
assert sum(r['bytes'] for r in records) == manifest['bytes']
paths = {r['path'] for r in records}
actual = []
for path in STAGE.rglob('*'):
    assert not path.is_symlink(), str(path)
    if path.is_file():
        actual.append(path.relative_to(STAGE).as_posix())
assert set(actual) == paths | {'assembly/static-inputs.json'}
assert len(actual) == len(set(actual)) == len({r.casefold() for r in actual}) == len(records) + 1
assert not (STAGE / 'manifest.json').exists()
assert not (STAGE / 'review/final-internal-review.json').exists()
assert not list((BASE / 'deliveries').glob('*.zip'))
for name in actual:
    rel = PurePosixPath(name)
    assert name == rel.as_posix() == unicodedata.normalize('NFC', name)
    assert not rel.is_absolute() and '..' not in rel.parts
    assert not any(c in name for c in ['\\', ':', '\x00', '\n', '\r'])
    assert not any(part in {'fixtures', '__pycache__', '.ssh', '.env', '.netrc', '.npmrc', '.pypirc', 'id_rsa', 'id_ed25519', 'credentials.json'} or part.startswith(('.env.', 'fixture-')) for part in rel.parts)
    assert not name.endswith(('.ckpt', '.tar.gz', '.tgz', '.tar', '.zip', '.pyc', '.tfrecord', '.tfrecord.gz', '.safetensors', '.bin'))
    assert rel.name != 'item_embeddings.pt'
local_sources = 0
git_sources = 0
source_changes = []
same_inode = []
dest_sizes = []
for row in records:
    destination = STAGE / row['path']
    assert destination.resolve().is_relative_to(STAGE.resolve())
    assert destination.stat().st_size == row['bytes'] and sha(destination) == row['sha256'], row['path']
    dest_sizes.append((row['bytes'], row['path']))
    provenance = row['source']
    if provenance.startswith('git:'):
        _, commit, member = provenance.split(':', 2)
        data = subprocess.check_output(['git', 'show', commit + ':' + member], cwd=REPO)
        assert len(data) == row['bytes'] and hashlib.sha256(data).hexdigest() == row['sha256']
        git_sources += 1
    else:
        source = Path(provenance)
        assert source.is_absolute() and source.is_file() and not source.is_symlink()
        if source.stat().st_size != row['bytes'] or sha(source) != row['sha256']:
            source_changes.append(dict(staged_path=row['path'], source=str(source), staged_sha256=row['sha256'], current_sha256=sha(source)))
        if (source.stat().st_dev, source.stat().st_ino) == (destination.stat().st_dev, destination.stat().st_ino):
            same_inode.append(row['path'])
        local_sources += 1
assert not same_inode
# This reviewer adapted its own inspector for the v2 member count while the
# author copied v2. Its staged bytes remain exactly manifest-bound historical
# evidence; no execution/source/result payload is allowed to diverge.
assert all(row['staged_path'] == 'review/delivery_tools/inspect_static_staging.py'
           and row['staged_sha256'] == '2074fa02f32d2a365fe53234afa0525046eac6acdb607cd4f7b7ae1a31a1f710'
           for row in source_changes), source_changes
deployment = json.loads((STAGE / 'execution/source-deployment.json').read_text())
commit = deployment['manifest']['source_commit']
new_prefix = 'source/a2-execution-' + commit + '/'
old_commit = 'b44399926c98369191c2ff5c2425377c8c07d3c8'
old_prefix = 'source/original-science-' + old_commit + '/'
new = {r['path'][len(new_prefix):]: r for r in records if r['path'].startswith(new_prefix)}
old = {r['path'][len(old_prefix):]: r for r in records if r['path'].startswith(old_prefix)}
assert len(new) == manifest['actual_A2_execution_source_members'] == 262
assert len(old) == manifest['selected_original_science_members'] == 244
assert {k: v['sha256'] for k, v in new.items()} == {r['path']: r['sha256'] for r in deployment['manifest']['files']}
assembler = BASE / 'assemble_static_inputs.py'
ast.parse(assembler.read_text(), filename=str(assembler))
assert 'never add an A2 final result or final PASS' in assembler.read_text()
assert "scientific_task_final_review_pass=False" in assembler.read_text()
assert "assert not target.exists()" in assembler.read_text()
assert "'fixture-'" in assembler.read_text()
result = dict(
    reviewer='/root/fd_repair_reviewer', reviewed_at=dt.datetime.now(dt.timezone.utc).isoformat(),
    status='STATIC_STAGING_COPY_IDENTITY_AND_EXCLUSIONS_PASS',
    scientific_task_final_review_pass=False, actual_final_ZIP_review_pass=False,
    staging=str(STAGE), copy_manifest=identity(manifest_path),
    copy_records=len(records), static_metadata_members=1, actual_staged_members=len(actual),
    payload_bytes=manifest['bytes'], all_payload_size_SHA256_matches_copy_manifest=True,
    copied_absolute_sources_independently_rehashed=local_sources,
    immutable_Git_member_sources_independently_reread=git_sources,
    current_source_content_divergences=source_changes,
    source_divergence_scope='Only this independent inspection script was adapted after its historical copy was made. Its staged content still matches the copy manifest. All scientific, execution, source-map and result originals match their current source bytes.',
    duplicate_names=0, casefold_collisions=0, unsafe_paths=0, symlinks=0, copied_source_shared_inode_count=0,
    excluded_checkpoints_transfer_archives_fixtures_pycache_TFRecords_T5weights_absent=True,
    original_failure_full_package_copy_present=any(p.startswith('history/original-emfile-review/a2-failure/') for p in paths),
    actual_A2_source_commit=commit, actual_A2_source_members=262,
    actual_A2_source_complete_against_deployment_manifest=True,
    original_science_commit=old_commit, original_science_members=244,
    old_source_scope='Selected244 old Git files overlapping the actual new262-source map, not a claim of a complete old server snapshot. Original scientific C2, shared-controller S1 and new85b77 source identities remain separate.',
    historical_receipt_scope='Historical PARTIAL/PENDING/NOT_YET_AVAILABLE entries remain original event receipts with their timestamps and review scope. This static copy review does not promote them to current final scientific status.',
    final_status_and_artifact_absence=dict(final_review_file_absent=True,final_package_manifest_absent=True,final_ZIP_absent=True,A2_later_final_outputs_not_staged=True),
    assembler_review=dict(**identity(assembler),closed_known_local_sources=True,actual_git_bytes_used=True,no_server_GPU_model_operation=True,no_overwrite_of_changed_existing_member=True),
    largest_payload_members=[dict(bytes=n,path=p) for n,p in sorted(dest_sizes,reverse=True)[:10]],
    required_before_actual_delivery=['Finish real A2 downstream experiments and result reviews; stage their final raw products and source-bound receipts separately.', 'Write genuine final result/review navigation and use event time/scope to distinguish historical receipts.', 'Produce a new complete final ZIP manifest including later additions; do not treat this static copy manifest as the final package manifest.', 'Run actual final package hash/CRC/path/duplicate/sensitive-pattern verification and independent final delivery review.'],
    remote_operations=0,GPU_requested=0,scientific_recomputation=False,dependencies_installed=[],GPT_UPLOAD_REQUIRED='YES',
)
target = OUT / 'static-staging-review-v2.json'
assert not target.exists()
target.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(dict(path=str(target),sha256=sha(target),status=result['status'],payload_records=len(records),payload_bytes=manifest['bytes'],absolute_source_checks=local_sources,Git_source_checks=git_sources,final_scientific_pass=False)))
