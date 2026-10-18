"""Independent production ZIP and fixed scientific publication review; never rebuild."""
import collections
import datetime as dt
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import unicodedata
import zipfile
import zlib

OUT = Path(__file__).resolve().parent
BASE = OUT.parents[1]
REPO = Path('/home/addaswsw/recommend-worktrees/sid-extensions-p0')
PUB = '42ac9acf23ae0bdc65ed6505089e1b88101755cd'
EXEC = '85b77f5e2a5b497fdc1f620649ba488fc783a245'


def read(path): return json.loads(Path(path).read_text())
def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024*1024),b''): h.update(block)
    return h.hexdigest()
def identity(path):
    path=Path(path); return dict(path=str(path),bytes=path.stat().st_size,sha256=sha(path))
def git_file(commit,path): return subprocess.check_output(['git','show',commit+':'+path],cwd=REPO)
def exact_bytes(data,row):
    assert len(data)==row['bytes'] and hashlib.sha256(data).hexdigest()==row['sha256']


request=read(BASE/'deliveries/package-build-request.json')
archive_path=Path(request['output']);stage=Path(request['staging'])
build_receipt_path=Path(str(archive_path)+'.verification.json')
receipt=read(build_receipt_path)
assert receipt['status']=='PACKAGE_INTEGRITY_PASS'
assert request['scientific_review_status']==receipt['scientific_review_status']=='SCIENTIFIC_INTERNAL_REVIEW_PASS'
assert request['global_PASS_not_yet_claimed'] and request['production_ZIP_count_before']==0
assert len(list((BASE/'deliveries').glob('*.zip')))==1
assert sha(BASE/'package_delivery_final.py')==request['tool_sha256']=='2fdb67d726f7652f00c3d57bfd73fb3b6f255f0e0bd666e86c2540f591eda53c'
zip_sha=sha(archive_path)
assert zip_sha==receipt['sha256']=='2f90e35df3db126099096dd58bc08df4c3784471042e23019706bb8aef2265d8'
assert archive_path.stat().st_size==receipt['bytes']==154094202
external_sha=Path(str(archive_path)+'.sha256').read_text().split()
assert external_sha==[zip_sha,archive_path.name]

# Fresh real connector reads, independent of the author's readback receipts.
ref_response=read(OUT/'final-github-ref-raw.json')
assert not ref_response.get('isError')
remote_ref=json.loads(ref_response['structuredContent']['content'])
assert remote_ref['ref']=='refs/heads/wbw' and remote_ref['object']['sha']==PUB
fresh_files=[]
for key,path in [('adapter','experiments/a2_fd_recovery_20261018/adapter.py'),
                 ('final_report','manifests/grid-a2-fd-recovery-20261018/final-results/FINAL_RESULTS.md'),
                 ('metrics','manifests/grid-a2-fd-recovery-20261018/final-results/original-metrics.csv')]:
    raw_path=OUT/('final-github-'+key+'-raw.json');response=read(raw_path)
    assert not response.get('isError');value=response['structuredContent']
    assert value['encoding']=='utf-8' and '/blob/'+PUB+'/' in value['display_url']
    data=value['content'].encode();assert data==git_file(PUB,path)
    blob_sha=hashlib.sha1(('blob '+str(len(data))+'\0').encode()+data).hexdigest()
    assert value['sha']==blob_sha
    fresh_files.append(dict(path=path,bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),Git_blob_sha1=blob_sha,raw_connector_receipt=identity(raw_path)))
anonymous=read(BASE/'publication/scientific/anonymous-readback.json')
assert anonymous['commit']==PUB and anonymous['branch']=='wbw' and len(anonymous['files'])==11
for row in anonymous['files']:
    data=git_file(PUB,row['path']);exact_bytes(data,row)
    assert row['HTTP_status']==200 and row['Git_blob_bytes_exact'] and row['anonymously_accessible_without_credential']
source=read(BASE/'source-deployment.json')['manifest']
assert source['source_commit']==EXEC and len(source['files'])==262
for row in source['files']:
    assert git_file(PUB,row['path'])==git_file(EXEC,row['path'])
    exact_bytes(git_file(EXEC,row['path']),row)

patterns={
 'private_key':re.compile(rb'-----BEGIN (?:RSA |OPENSSH |EC |DSA |ENCRYPTED )?PRIVATE KEY-----'),
 'github_token':re.compile(rb'(?<![A-Za-z0-9])gh[pousr]_[A-Za-z0-9]{30,}'),
 'github_pat':re.compile(rb'github_pat_[A-Za-z0-9_]{40,}'),
 'hf_token':re.compile(rb'(?<![A-Za-z0-9])hf_[A-Za-z0-9]{30,}'),
 'aws_access_key':re.compile(rb'(?<![A-Z0-9])(?:AKIA|ASIA)[A-Z0-9]{16}(?![A-Z0-9])'),
}
members={};byte_total=0
with zipfile.ZipFile(archive_path,'r') as archive:
    entries=archive.infolist();names=[entry.filename for entry in entries]
    assert len(names)==len(set(names))==len({name.casefold()for name in names})==3153
    assert archive.comment==b'' and archive.testzip() is None
    manifest_bytes=archive.read('manifest.json');manifest=json.loads(manifest_bytes)
    records={row['path']:row for row in manifest['files']}
    assert len(records)==len(manifest['files'])==3152
    assert set(names)==set(records)|{'manifest.json'}
    assert manifest['scientific_review_status']=='SCIENTIFIC_INTERNAL_REVIEW_PASS'
    assert hashlib.sha256(manifest_bytes).hexdigest()==receipt['manifest_sha256']
    stage_paths={p.relative_to(stage).as_posix()for p in stage.rglob('*')if p.is_file()}
    assert set(names)==stage_paths
    for entry in entries:
        name=entry.filename;rel=PurePosixPath(name)
        assert name==rel.as_posix()==unicodedata.normalize('NFC',name) and not rel.is_absolute() and '..'not in rel.parts
        assert not any(c in name for c in ['\\',':','\0','\n','\r'])
        assert not entry.is_dir() and not stat.S_ISLNK(entry.external_attr>>16) and not entry.flag_bits&1
        assert not entry.comment and entry.file_size<=100000000
        assert not any(part in {'fixtures','__pycache__','.ssh','.env','.netrc','.npmrc','.pypirc','id_rsa','id_ed25519','credentials.json'}or part.startswith(('fixture-','.env.'))for part in rel.parts)
        assert not name.endswith(('.ckpt','.bin','.safetensors','.tfrecord','.tfrecord.gz','.pyc','.tar.gz','.tgz','.tar','.zip'))
        assert rel.name!='item_embeddings.pt'
        data=archive.read(entry);byte_total+=len(data)
        assert not any(pattern.search(data)for pattern in patterns.values()),name
        assert zip_sha.encode()not in data,'Current ZIP SHA must remain outside its own payload'
        value=dict(path=name,bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),crc32=f'{zlib.crc32(data)&0xffffffff:08x}')
        assert len(data)==entry.file_size and value['crc32']==f'{entry.CRC:08x}'
        assert not (stage/name).is_symlink() and (stage/name).read_bytes()==data
        if name!='manifest.json':assert all(value[k]==records[name][k]for k in ['bytes','sha256','crc32'])
        members[name]=value
    old_members=read(OUT/'prepublication-staging-members.json')['members']
    assert len(old_members)==2911
    for row in old_members:assert row==members[row['path']],row['path']
    late=set(records)-{row['path']for row in old_members}
    assert len(late)==241
    preappend=json.loads(archive.read('assembly/prepublication-addendum.json'))
    pubappend=json.loads(archive.read('assembly/publication-addendum.json'))
    covered=set()
    for row in preappend['files']+pubappend['files']:
        if row.get('root',str(stage))!=str(stage):continue
        assert all(members[row['path']][key]==row[key]for key in ['bytes','sha256'])
        covered.add(row['path'])
        if row['source'].startswith('git:'):
            _,commit,path=row['source'].split(':',2);exact_bytes(git_file(commit,path),row)
        else:exact_bytes(Path(row['source']).read_bytes(),row)
    generated=set(pubappend['generated_members'])|{'assembly/prepublication-addendum.json','assembly/publication-addendum.json'}
    # The single production driver appends its exact request and own bytes
    # immediately before serialization, after assembly/publication-addendum.
    # Independently bind these two known construction-time copies as well.
    construction_metadata=[]
    for name,path in [('execution/package-generation-request.json',BASE/'deliveries/package-build-request.json'),
                      ('tools/final-operations/build_final_package.py',BASE/'build_final_package.py')]:
        assert archive.read(name)==path.read_bytes()
        construction_metadata.append(dict(member=name,source=identity(path)))
    assert late==covered|generated|{row['member']for row in construction_metadata}
    version=json.loads(archive.read('publication/version-map.json'))
    assert version['scientific_release_commit']==PUB and version['scientific_execution_commit']==EXEC
    assert version['source_members_exact_to_execution']==262 and version['post_ZIP_delivery_receipt_commit'] is None
    internal=json.loads(archive.read('review/final-internal-review.json'))
    assert internal['status']=='SCIENTIFIC_INTERNAL_REVIEW_PASS' and internal['global_final_task_pass'] is False
    assert internal['serialized_ZIP_integrity_not_yet_approved'] is True and internal['webpage_GPT_second_review_not_yet_performed'] is True
    map_document=json.loads(archive.read('assembly/reviewer-input-path-map.json'))
    assert len(map_document['inputs'])==363
    for row in map_document['inputs']:
        assert row['canonical_ZIP_member'] in row['all_existing_equal_members']
        for name in row['all_existing_equal_members']:assert all(members[name][key]==row[key]for key in ['bytes','sha256'])
    publication_names=[name for name in names if name.startswith('publication/scientific-release-tree/')]
    assert len(publication_names)==222
    for name in publication_names:
        rel=name.removeprefix('publication/scientific-release-tree/')
        assert archive.read(name)==git_file(PUB,'manifests/grid-a2-fd-recovery-20261018/final-results/'+rel)
    raw_npz=[name for name in names if name.startswith(('raw/toys/','raw/sports_a1/','raw/a2/'))and '-attention-'in name and name.endswith('.npz')]
    raw_png=[name for name in names if name.startswith(('raw/toys/','raw/sports_a1/','raw/a2/'))and '-attention-'in name and name.endswith('.png')]
    assert len(raw_npz)==48 and len(raw_png)==384
    original_predictions=[name for name in names if name.startswith(('raw/toys/','raw/sports_a1/','raw/a2/'))and '-eval-'in name and name.endswith('/test/predictions.jsonl')]
    assert len(original_predictions)==12
    assert sum(sum(1 for line in archive.read(name).splitlines()if line.strip())for name in original_predictions)==330060
    assert archive.read('FINAL_RESULTS.md')==git_file(PUB,'manifests/grid-a2-fd-recovery-20261018/final-results/FINAL_RESULTS.md')
    assert json.loads(archive.read('assets-not-copied.json'))['summary']['dataset_members']==1293
assert byte_total<=660000000
windows_path=subprocess.check_output(['wslpath','-w',str(archive_path)],text=True).strip()
assert windows_path.startswith('\\\\wsl.') and archive_path.is_file()
report=dict(
 reviewer='/root/fd_repair_reviewer',reviewed_at=dt.datetime.now(dt.timezone.utc).isoformat(),
 status='FINAL_PACKAGE_AND_SCIENTIFIC_PUBLICATION_NONAUTHOR_REVIEW_PASS',
 actual_package_and_published_scientific_evidence_pass=True,
 final_whole_task_delivery_receipt_commit_not_yet_approved=True,
 zip=dict(path=str(archive_path),Windows_UNC=windows_path,bytes=archive_path.stat().st_size,sha256=zip_sha,
          total_members=3153,payload_members=3152,total_uncompressed_bytes=byte_total,
          manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest()),
 all_actual_member_sizes_SHA256_and_CRC32_verified=True,ZIP_testzip_CRC_pass=True,
 all_serialized_members_exact_to_actual_final_staging=True,
 prepublication_reviewed_members_preserved_exactly=2911,late_members_source_bound_and_verified=241,
 construction_time_metadata_provenance=construction_metadata,
 construction_time_scope='Two known driver/request copies were created after the assembly addendum; their actual ZIP bytes are directly matched to the saved single-build driver and request. This external receipt supplies that final provenance without modifying the original ZIP.',
 full_current_ZIP_SHA_is_external_and_absent_from_own_contents=True,
 safe_canonical_NFC_paths=True,casefold_collisions=0,duplicate_members=0,symlinks=0,encrypted_members=0,
 common_sensitive_pattern_scan=dict(status='PASS',scope='Every actual uncompressed member including manifest',patterns=list(patterns),bytes_scanned=byte_total),
 prohibited_fixtures_weight_payloads_TFRecords_transfer_archives_pycache_absent=True,
 complete_A_route_raw_attention=dict(NPZ=48,PNG=384),complete_original_evaluation=dict(conditions=12,prediction_rows=330060),
 raw_reviewer_input_mappings_exact=363,publication_tree_files_exact_to_fixed_Git=222,
 publication_commit=PUB,scientific_execution_commit=EXEC,execution_source_files_unchanged_at_publication=262,
 fresh_independent_remote_ref_receipt=identity(OUT/'final-github-ref-raw.json'),fresh_independent_remote_key_files=fresh_files,
 existing_anonymous_readback_receipt=identity(BASE/'publication/scientific/anonymous-readback.json'),existing_anonymous_key_files_rechecked_against_Git=11,
 scientific_cross_review=identity(OUT/'scientific-results-cross-review.json'),prepublication_review=identity(OUT/'prepublication-final-staging-review.json'),
 author_actual_package_build_receipt=identity(build_receipt_path),
 package_internal_status='SCIENTIFIC_INTERNAL_REVIEW_PASS',
 state_scope='The package correctly preserves pre-ZIP scientific/publication/staging PASS with global=false. This new external review approves the actual one serialized ZIP and fixed scientific publication; the later final whole-task receipt Git commit/ref has not been fabricated or pre-approved.',
 limitations=['Toys upstream producer remains non-unique; both candidates stay reported.',
             'New Sports A1/A2 are method reproductions with separate scratch generator training, not recovered author weights or one-factor causal SID controls.',
             'Single seed42 and preselected evaluation fixed32 descriptive statistics do not prove strict invariance or target-prefix nonuse.',
             'Full environment tree/member count/logical size/content image are unverified; only recorded interpreter/requirements and nine critical package identities were checked.',
             'Large assets stay on server with explicit inherited source hashes and fresh stat limits; no fresh full corpus or environment rehash is claimed.',
             'Webpage GPT second review is still to be performed by the user with this one ZIP.'],
 remaining='Parent must publish only actual final package/review receipts to wbw, then verify the new real remote commit/ref and files before its final user delivery. No new experiments or cleanup.',
 production_archives_created_by_reviewer=0,archive_or_stage_mutations=0,Git_mutations=0,new_server_calls=0,GPU_calls=0,model_loads=0,dependencies_installed=[],GPT_UPLOAD_REQUIRED='YES')
target=OUT/'final-package-nonauthor-review.json';assert not target.exists()
target.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(dict(path=str(target),sha256=sha(target),status=report['status'],ZIP_sha256=zip_sha,ZIP_bytes=archive_path.stat().st_size,
                     members=3153,actual_uncompressed_bytes=byte_total,Windows_UNC=windows_path,late_receipt_commit_not_yet_approved=True)))
