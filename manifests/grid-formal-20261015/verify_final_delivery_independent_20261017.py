#!/usr/bin/env python3
"""Non-author package verification. Reads saved results; never computes research statistics."""
import argparse
import collections
import datetime
import gzip
import hashlib
import io
import itertools
import json
import pathlib
import re
import stat
import subprocess
import urllib.request
import zipfile

P = argparse.ArgumentParser()
P.add_argument('package')
P.add_argument('--commit', required=True)
P.add_argument('--phase', choices=['candidate', 'final'], required=True)
P.add_argument('--receipt', required=True)
A = P.parse_args()
package = pathlib.Path(A.package)
repo = pathlib.Path('/home/addaswsw/recommend-worktrees/sid-extensions-p0')
support = pathlib.Path('/home/addaswsw/recommend-support/grid-formal-20261015')
old = support / 'deliveries/grid-formal-20261016-gpt-audit-snapshot.zip'
checks = []
def check(name, condition, details=None):
    checks.append({'check': name, 'passed': bool(condition), 'details': details})
def sha(b):
    return hashlib.sha256(b).hexdigest()
def git(*args, input=None):
    return subprocess.run(['git', *args], cwd=repo, input=input, check=True, capture_output=True).stdout

with zipfile.ZipFile(package) as z:
    infos = z.infolist()
    names = z.namelist()
    name_set = set(names)
    check('no_duplicate_member_paths', len(names) == len(name_set))
    bad_paths = [n for n in names if n.startswith('/') or '\\' in n or '..' in pathlib.PurePosixPath(n).parts or re.match(r'^[A-Za-z]:', n)]
    links = [i.filename for i in infos if stat.S_ISLNK(i.external_attr >> 16)]
    check('no_absolute_parent_drive_or_backslash_paths', not bad_paths, bad_paths)
    check('no_symbolic_links', not links, links)
    check('CRC_all_members', z.testzip() is None)
    bodies = {n: z.read(n) for n in names}
    check('all_members_fully_read', len(bodies) == len(names))
    manifest = json.loads(bodies['MANIFEST.json'])
    files = manifest['files']
    expected = {x['path']: x for x in files}
    check('manifest_unique_entries', len(expected) == len(files))
    check('manifest_exact_member_set_excluding_self', set(expected) == name_set - {'MANIFEST.json'})
    check('manifest_declared_count', manifest['member_count_including_manifest'] == len(names))
    mismatches = [n for n, e in expected.items() if n not in bodies or len(bodies[n]) != e['bytes'] or sha(bodies[n]) != e['sha256']]
    check('manifest_all_member_bytes_and_SHA256', not mismatches, mismatches)
    check('manifest_fixed_release_commit', manifest['commit'] == A.commit)
    check('old_stage_archive_preserved', sha(old.read_bytes()) == 'b3627f2acaf70406da399f531fefb10471eb91b06c017049e0e81e26dd274f58')
    with zipfile.ZipFile(old) as prior:
        old_mismatch = [n for n in prior.namelist() if 'original-evidence/' + n not in bodies or prior.read(n) != bodies['original-evidence/' + n]]
        check('all_617_approved_stage_members_preserved_byte_for_byte', not old_mismatch, old_mismatch)
    current = [n for n in names if n.startswith('current/')]
    batch_input = ''.join(A.commit + ':' + n[len('current/'):] + '\n' for n in current).encode()
    stream = io.BytesIO(git('cat-file', '--batch', input=batch_input))
    git_mismatch = []
    for n in current:
        header = stream.readline().decode().strip().split()
        if len(header) != 3 or header[1] != 'blob':
            git_mismatch.append(n)
            continue
        data = stream.read(int(header[2]))
        stream.read(1)
        if data != bodies[n]:
            git_mismatch.append(n)
    check('every_current_member_exact_fixed_Git_blob', not git_mismatch, {'members_checked': len(current), 'mismatches': git_mismatch})
    patterns = {
        'private_key': re.compile(rb'-----BEGIN (?:OPENSSH |RSA |EC |DSA |ENCRYPTED )?PRIVATE KEY-----'),
        'github_token': re.compile(rb'\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{50,})\b'),
        'openai_key': re.compile(rb'\bsk-(?:proj-)?[A-Za-z0-9_-]{40,}\b'),
        'aws_access_key': re.compile(rb'\b(?:AKIA|ASIA)[A-Z0-9]{16}\b'),
        'credential_URL': re.compile(rb'https?://[^\s/\"\x27:]+:[^\s/\"\x27@]+@'),
        'live_Bearer': re.compile(rb'\bBearer\s+[A-Za-z0-9_-]{25,}(?:\.[A-Za-z0-9_-]{10,})*'),
    }
    secrets = []
    prohibited = []
    for n, body in bodies.items():
        if re.search(r'(^|/)(?:id_rsa|id_ed25519|\.env|credentials|token)(?:$|\.)', n, re.I):
            prohibited.append(n)
        if n.endswith(('.json', '.jsonl', '.md', '.txt', '.yaml', '.yml', '.py', '.log')):
            scan = body
        elif n.endswith('.jsonl.gz'):
            scan = gzip.decompress(body)
        else:
            continue
        for label, pattern in patterns.items():
            if pattern.search(scan):
                secrets.append({'member': n, 'type': label})
    check('no_obvious_secret_or_auth_material_matches', not secrets and not prohibited, {'matches_without_secret_values': secrets, 'sensitive_names': prohibited, 'scope': 'all textual members and gzip JSONL, high-confidence key/token/credential patterns; not a claim to prove absence of every possible secret'})
    base = 'current/manifests/grid-formal-20261015/'
    cond = ['a-normal','a-shuffle3','a-shuffle4','b-normal','b-shuffle3','b-shuffle4']
    attention = ['gf-20261015-' + c + '-attention-v1' for c in cond] + ['gf-20261015-' + d + '-mlp-attention-v1' for d in ['toys','sports']]
    core = ['gf-20261015-' + c + '-v1' for c in cond] + ['gf-20261015-' + d + '-mlp-formal-v1' for d in ['toys','sports']] + attention + ['beauty-mlp-formal-v1','beauty-mlp-formal-attention32-v1']
    version_map = json.loads(bodies[base + 'FINAL_VERSION_MAP.json'])
    mapping = version_map['job_map']
    check('18_core_jobs_complete_unique_mapping', len(mapping) == 18 and {x['job_id'] for x in mapping} == set(core))
    job_status_checks = []
    for job_id in core:
        prefix = 'original-evidence/jobs/' + job_id + '/'
        needed = [prefix + 'request.json', prefix + 'status.json', prefix + 'cleanup.json', prefix + 'output/RESULT.json']
        missing = [n for n in needed if n not in bodies]
        state = json.loads(bodies[prefix+'status.json']) if not missing else {}
        check('job_original_identity_and_success:' + job_id, not missing and state.get('status') == 'SUCCEEDED' and state.get('exit_code') == 0, {'missing':missing,'status':state.get('status'),'exit_code':state.get('exit_code')})
    capture_versions = {x['capture_version'] for x in mapping}
    check('distinct_historical_source_snapshots_retained', len(capture_versions) == 6, {'distinct_commit_snapshot_pairs': sorted(capture_versions)})
    stats_name = base + 'attention-statistics/attention-descriptive-statistics.json'
    stats = json.loads(bodies[stats_name])
    check('8_main_statistics_conditions_no_engineering_only', len(stats['jobs']) == 8 and {j['job_id'] for j in stats['jobs']} == set(attention))
    pair_total = 0
    expected_pairs = set(itertools.combinations(range(1,5),2))
    raw_names = []
    for j in stats['jobs']:
        jid = j['job_id']
        rows = j['per_sample_query_pairs']
        pair_total += len(rows)
        keys = collections.defaultdict(set)
        for r in rows:
            keys[(r['sample_key'],r['path'])].add(tuple(r['query_pair']))
        check('32users_2paths_6pairs:' + jid, j['samples'] == 32 and j['cohort_keys_exact'] and len(rows) == 384 and len(keys) == 64 and all(v == expected_pairs for v in keys.values()) and {p for _,p in keys} == {'final_top1_cross','local_first_cross'})
        inp_mismatch = []
        for f in j['input_files']:
            rel = f['path'].split('/retrieved/',1)[1]
            n = 'original-evidence/jobs/' + jid + '/' + rel
            if n not in bodies or len(bodies[n]) != f['bytes'] or sha(bodies[n]) != f['sha256']:
                inp_mismatch.append(n)
        check('statistics_raw_input_identity:' + jid, len(j['input_files']) == 8 and not inp_mismatch, inp_mismatch)
        for b in range(4):
            prefix = 'original-evidence/jobs/' + jid + '/output/batch-' + format(b,'02d') + '/'
            needed = [prefix+'bundle/arrays.npz',prefix+'bundle/record.json',prefix+'figures/plot_receipt.json',prefix+'off_on_checks.json']
            check('raw_batch_contract:' + jid + ':' + str(b), all(n in bodies for n in needed))
            raw_names.append(needed[0])
            with zipfile.ZipFile(io.BytesIO(bodies[needed[0]])) as npz:
                check('NPZ_container_CRC:' + jid + ':' + str(b), npz.testzip() is None)
                for item in npz.namelist():
                    npz.read(item)
        review_name = base + 'attention-results-review/' + jid + '-review.json'
        check('original_technical_audit_present:' + jid, review_name in bodies)
    check('3072_main_pair_path_records', pair_total == 3072)
    check('32_main_raw_NPZ_members', len(raw_names) == 32 and len(set(raw_names)) == 32)
    runtime = json.loads(bodies[base+'attention-results-review/attention-statistics-runtime-review.json'])
    provenance = json.loads(bodies[base+'attention-results-review/attention-statistics-source-provenance-review-20261017.json'])
    check('actual_independent_statistics_receipts_bind_packaged_report', runtime['report']['sha256'] == sha(bodies[stats_name]) == provenance['actual_report']['sha256'] and runtime['status'] == 'INDEPENDENT_ATTENTION_DESCRIPTIVE_STATISTICS_PASS' and provenance['status'] == 'INDEPENDENT_ATTENTION_STATISTICS_SOURCE_AND_RUNTIME_PASS' and len(runtime['jobs']) == 8)
    pngs = [n for n in names if n.endswith('.png')]
    png_hashes = {sha(bodies[n]) for n in pngs}
    inventory = json.loads(bodies['references/FINAL_PNG_INVENTORY.json'])
    check('all_256_original_PNG_in_inventory', len(inventory) == 256 and collections.Counter(v['job_id'] for v in inventory) == {jid:32 for jid in attention})
    png_missing = [v['remote_path'] for v in inventory if v['included_in_final_package'] and v['sha256'] not in png_hashes]
    check('every_declared_selected_PNG_in_package', not png_missing, {'selected_main':sum(v['included_in_final_package'] for v in inventory),'missing':png_missing})
    least_missing = []
    for j in stats['jobs']:
        for path, items in j['least_similar_three_under_declared_mean_cosine'].items():
            for item in items:
                rel = item['plot_relative'].split('/')
                matches = [v for v in inventory if v['job_id']==j['job_id'] and v['batch']==int(rel[1].split('-')[1]) and v['file']==rel[-1]]
                if len(matches)!=1 or matches[0]['sha256'] not in png_hashes:
                    least_missing.append([j['job_id'],path,item['sample_key']])
    check('both_paths_all_declared_least_three_original_PNG_available', not least_missing, least_missing)
    for n in pngs:
        check('PNG_signature:' + n, bodies[n].startswith(b'\x89PNG\r\n\x1a\n'))
    summary = json.loads(bodies[base+'EXECUTION_SUMMARY.json'])
    check('scientific_work_complete_flags', all(summary[k] for k in ['formal_results_complete','attention_capture_and_technical_review_complete','attention_statistics_complete','attention_statistics_independent_review_complete']))
    if A.phase == 'final':
        check('final_report_READY_FOR_REVIEW', summary['status'] == 'READY_FOR_REVIEW' and not summary['remaining_work'] and b'READY_FOR_REVIEW' in bodies[base+'FINAL_RECEIPT.md'])
    else:
        check('candidate_no_premature_READY_claim', summary['status'] == 'COMPLETED_PENDING_DELIVERY_REVIEW')
    reference = json.loads(bodies['references/FIXED_GITHUB_REFERENCES.json'])
    check('all_current_GitHub_references_fixed_commit', all(r['commit'] == A.commit for r in reference))
    raw_readback = []
    for rel in ['FINAL_RECEIPT.md','EXECUTION_SUMMARY.json','FINAL_VERSION_MAP.json','attention-statistics/attention-descriptive-statistics.json','attention-results-review/attention-statistics-runtime-review.json','final-integration-review-20261017.json']:
        path = 'manifests/grid-formal-20261015/' + rel
        url = 'https://raw.githubusercontent.com/woobowen/recommend/' + A.commit + '/' + path
        with urllib.request.urlopen(url, timeout=25) as response:
            data = response.read()
            code = response.status
        n = 'current/' + path
        raw_readback.append({'path':path,'http_status':code,'bytes':len(data),'sha256':sha(data),'matches_package':data==bodies[n]})
    check('independent_full_body_fixed_GitHub_GETs', all(x['matches_package'] and x['http_status']==200 for x in raw_readback), raw_readback)
    remote = git('ls-remote','origin','refs/heads/wbw').decode().strip().split()[0]
    check('actual_origin_wbw_ref', remote == A.commit, remote)

failures = [x for x in checks if not x['passed']]
receipt = {
    'schema':'grid-final-delivery-nonauthor-review-v1',
    'status': 'INDEPENDENT_FINAL_DELIVERY_PACKAGE_PASS' if not failures else 'INDEPENDENT_FINAL_DELIVERY_PACKAGE_FAILED',
    'phase':A.phase,
    'reviewer_task':'/root/delivery_review',
    'reviewer_is_nonauthor_of_statistics_models_results_and_packager':True,
    'reviewed_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'package':{'path':str(package),'bytes':package.stat().st_size,'members':len(names),'sha256':sha(package.read_bytes())},
    'published_commit':A.commit,
    'checks':checks,
    'checks_passed':len(checks)-len(failures),
    'checks_failed':len(failures),
    'scope':'Independent package/identity/GitHub delivery checks; reads complete matrices as container bytes, does not rerun research statistics or models; reuses already actual nonauthor numeric review by report hash.',
    'delivery_decision': 'Candidate contents accepted; only READY/delivery metadata may change before final publication and a full final package verification' if not failures and A.phase=='candidate' else ('Final delivery accepted for GPT webpage external review' if not failures else 'Not accepted until listed delivery failures are resolved'),
    'limitations':['No new model, checkpoint parameter or scientific arithmetic rerun; inherited technical audits reused by identity and scope','No claim that GPT webpage final external review has passed','Secret scan checks high confidence obvious credentials, not every possible hidden encoding'],
    'verification_program':{'path':str(pathlib.Path(__file__)),'sha256':sha(pathlib.Path(__file__).read_bytes())},
    'new_research_or_model_calls':0,
    'original_evidence_or_Git_writes':0,
}
out=pathlib.Path(A.receipt)
if out.exists():
    raise FileExistsError('Will not overwrite independent review receipt: '+str(out))
out.write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'receipt':str(out),'status':receipt['status'],'checks':len(checks),'failed':failures,'package':receipt['package']},ensure_ascii=False))
raise SystemExit(bool(failures))
