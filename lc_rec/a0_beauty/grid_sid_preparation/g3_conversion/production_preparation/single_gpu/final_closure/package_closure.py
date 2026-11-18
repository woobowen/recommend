#!/usr/bin/env python3
"""Read fixed GitHub bytes and package the G3 final-review delta."""
import argparse
import concurrent.futures
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import urllib.request
import zipfile

HERE = Path(__file__).resolve().parent
REPO = next(p for p in HERE.parents if (p/'.git').exists())
PRIOR_PREFIX = HERE.parent.relative_to(REPO).as_posix()+'/'
PREFIX = PRIOR_PREFIX+'final_closure/'
README = (HERE.parent.parent/'README.md').relative_to(REPO).as_posix()
RUN = '21d495b0be5b606853ec7cec68a3eaa1b58acbd7'
PRIOR = '6db8ab4ec98f5692014eeac02b03f3f55015d1eb'
RAW = 'https://raw.githubusercontent.com/woobowen/recommend/'
BLOB = 'https://github.com/woobowen/recommend/blob/'
SOURCES = {
    PRIOR: [README, *(PRIOR_PREFIX+name for name in (
        'acceptance.json','final_job_ledger.json','formal_single_gpu_budget_proposal.json',
        'run_deployment.json','server_readback.json',
        'jobs/g3-lcrec-single-cpu-20261118-r3-final/output/cpu/result.json',
        'jobs/g3-lcrec-single-train-20261118-r3/output/g3_preflight.json',
        'jobs/g3-lcrec-single-compare-20261118-r3/output/boundary_compare.json',
        'jobs/g3-lcrec-single-evaluation-20261118-r3/output/evaluation/summary.json'))],
    RUN: ['lc_rec/a0_beauty/'+name for name in (
        'formal_train.py','formal_evaluate.py','formal_control.py','production_control.py',
        'production_pipeline.py','prepare_production_plan.py','g3_identity.py',
        'config/formal_g3_beauty_seed42.json')],
}
SECRET_PATTERNS = [rb'-----BEGIN [A-Z ]*PRIVATE KEY-----', rb'gh[pousr]_[A-Za-z0-9]{24,}',
                   rb'github_pat_[A-Za-z0-9_]{30,}', rb'sk-[A-Za-z0-9_-]{32,}',
                   rb'AKIA[A-Z0-9]{16}', rb'hf_[A-Za-z0-9]{30,}']


def sha(data):
    return hashlib.sha256(data).hexdigest()


def encoded(value):
    return (json.dumps(value,ensure_ascii=False,indent=2)+'\n').encode()


def git(*argv, text=False):
    return subprocess.check_output(['git',*argv],cwd=REPO,text=text,timeout=45)


def blob(commit, name):
    return git('show',commit+':'+name)


def check(condition, message):
    if not condition:
        raise ValueError(message)


def verify(item):
    commit, name, expected = item
    url = RAW+commit+'/'+name
    request = urllib.request.Request(url,headers={'User-Agent':'G3-final-closure-verification'})
    with urllib.request.urlopen(request,timeout=30) as result:
        raw = result.read()
        check(result.status==200 and raw==expected,'fixed GitHub content mismatch '+url)
    return dict(commit=commit,path=name,url=BLOB+commit+'/'+name,http=200,bytes=len(raw),sha256=sha(raw))


def safe_name(name):
    p = PurePosixPath(name)
    return (not p.is_absolute() and all(part not in ('.','..') for part in p.parts)
            and re.fullmatch(r'[A-Za-z0-9_.\-/]+',name) and p.as_posix()==name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--commit',required=True)
    parser.add_argument('--output',type=Path,required=True)
    a = parser.parse_args()
    commit = git('rev-parse',a.commit+'^{commit}',text=True).strip()
    output = a.output.resolve()
    check(not output.exists(),'preserve prior package')
    ref = git('ls-remote','origin','refs/heads/wbw-lc-rec',text=True).strip()
    check(ref.split()==[commit,'refs/heads/wbw-lc-rec'],'GitHub branch ref mismatch')
    manifest_name = PRIOR_PREFIX+'run_deployment.json'
    manifest = json.loads(blob(PRIOR,manifest_name))
    differences = []
    for row in manifest['files']:
        if sha(blob(commit,row['relative']))!=row['sha256']:
            differences.append(row['relative'])
    check(len(manifest['files'])==539 and differences==['lc_rec/a0_beauty/g3_remote.py'],
          'unexpected frozen runtime source change')
    check(blob(commit,differences[0])==blob(PRIOR,differences[0]),'new collection source change')
    file_names = git('ls-tree','-r','--name-only',commit,'--',PREFIX,README,text=True).splitlines()
    files = {n:blob(commit,n) for n in file_names}
    check(README in files and len(files)>12,'review entry missing')
    checks = [(commit,n,v) for n,v in files.items()]
    for old_commit, names in SOURCES.items():
        for name in names:
            checks.append((old_commit,name,blob(old_commit,name)))
    summary = json.loads(files[PREFIX+'final_plan_summary.json'])
    plan_bytes = files[PREFIX+'final_plan.json']
    canonical = json.dumps(json.loads(plan_bytes), sort_keys=True, separators=(',',':'),
                           ensure_ascii=False, allow_nan=False).encode()
    check(sha(plan_bytes)==summary['plan']['sha256'] and
          sha(canonical)==summary['canonical_digest'], 'plan bytes or canonical digest')
    check(summary['status']=='READY_FOR_FINAL_REVIEW' and
          json.loads(files[PREFIX+'non_author_review.json'])['status'].startswith('PASS'),
          'required direct review or acceptance missing')
    prep = summary['preparation_script_commit']
    for name in ('final_plan_check.py','closure_cpu_client.py','approved_budget.json',
                 'budget_approval.json','external_technical_review.json','web_technical_review_source.md'):
        rel = PREFIX+name
        value = blob(prep,rel)
        check(value==files[rel],'delivery changed prepared source '+rel)
        checks.append((prep,rel,value))
    for attempt in summary['cpu_jobs']:
        attempt_commit = attempt['capture_version']
        if attempt_commit != prep:
            for name in ('final_plan_check.py','closure_cpu_client.py'):
                rel = PREFIX+name
                checks.append((attempt_commit,rel,blob(attempt_commit,rel)))
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        verified = list(pool.map(verify,checks))
    receipt = dict(status='PASS', utc=datetime.now(timezone.utc).isoformat(), final_delivery_commit=commit,
                   model_run_commit=RUN,prior_web_review_commit=PRIOR,preparation_script_commit=prep,
                   remote_ref=ref,runtime_files_compared=len(manifest['files']),
                   sole_inherited_difference=differences[0], fixed_github_files=verified)
    files['FIXED_GITHUB_READBACK.json'] = encoded(receipt)
    files['REVIEW_NAVIGATION.json'] = encoded(dict(
        scope='Final plan and incremental prelaunch closure', entry=README,
        delivery_commit=commit, previous_technical_review_commit=PRIOR, runtime_commit=RUN,
        final_plan=PREFIX+'final_plan.json', summary=PREFIX+'final_plan_summary.json',
        current_delivery_web_review='PENDING', formal_launch_authorized=False,
        formal_experiment_started=False))
    rows = []
    for name, value in sorted(files.items()):
        check(safe_name(name) and len(value)<2*1024**2,'unsafe review member '+name)
        check(not any(re.search(p,value) for p in SECRET_PATTERNS),'secret pattern in '+name)
        rows.append(dict(path=name,bytes=len(value),sha256=sha(value)))
    files['MEMBER_SHA256.json'] = encoded(dict(members=rows,self_excluded=True))
    with zipfile.ZipFile(output,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as archive:
        for name, value in sorted(files.items()):
            archive.writestr(name,value)
    with zipfile.ZipFile(output) as archive:
        names = archive.namelist()
        check(len(names)==len(set(names)) and all(safe_name(n) for n in names)
              and archive.testzip() is None,'ZIP CRC/member paths')
        check(set(names)==set(files) and all(archive.read(n)==files[n] for n in names),
              'ZIP member identity')
    windows = subprocess.check_output(['wslpath','-w',str(output)],text=True).strip()
    powershell = '/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe'
    ps = "$p='"+windows.replace("'","''")+"'; [PSCustomObject]@{Length=(Get-Item -LiteralPath $p).Length;SHA256=(Get-FileHash -LiteralPath $p -Algorithm SHA256).Hash}|ConvertTo-Json -Compress"
    actual = json.loads(subprocess.check_output([powershell,'-NoLogo','-NoProfile','-NonInteractive',
                                                '-Command',ps],text=True,timeout=45).lstrip('\ufeff'))
    archive_hash = sha(output.read_bytes())
    check(actual['Length']==output.stat().st_size and actual['SHA256'].lower()==archive_hash,
          'actual Windows UNC read differs')
    verification = dict(status='PASS', zip_path=str(output), bytes=output.stat().st_size,
        sha256=archive_hash, members=len(files), crc='PASS', member_bytes_and_hash='PASS',
        duplicates=0, dangerous_paths=0, git_refs_http200_exact=len(verified),
        secret_scan='Pattern scan of all member bytes for PEM private keys and common GitHub, API, AWS and HF token formats',
        windows_path=windows,windows_unc_read='PASS',final_delivery_commit=commit,
        all_members=rows+[dict(path='MEMBER_SHA256.json',bytes=len(files['MEMBER_SHA256.json']),
                                sha256=sha(files['MEMBER_SHA256.json']))])
    output.with_suffix('.zip.sha256').write_text(archive_hash+'  '+output.name+'\n')
    output.with_suffix('.zip.verification.json').write_bytes(encoded(verification))
    output.with_suffix('.github-readback.json').write_bytes(encoded(receipt))
    print(encoded({k:v for k,v in verification.items() if k!='all_members'}).decode(),end='')


if __name__=='__main__':
    main()
