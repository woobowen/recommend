#!/usr/bin/env python3
"""Build and verify a lightweight, fixed-Git review ZIP, with external hash."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import zipfile


def main():
    p=argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument('--commit',required=True)
    p.add_argument('--output',required=True,type=Path)
    a=p.parse_args();repo=Path(__file__).resolve().parents[2]
    a.commit=subprocess.check_output(['git','rev-parse','--verify',a.commit+'^{commit}'],cwd=repo,text=True).strip()
    names=subprocess.check_output(['git','ls-tree','-r','--name-only',a.commit,'--','lc_rec/a0_beauty','lc_rec/letter/config','lc_rec/letter/scripts','lc_rec/shuffle'],cwd=repo,text=True).splitlines()
    files={}
    for name in names:
        wanted=(name.startswith('lc_rec/a0_beauty/grid_sid_preparation/g3_conversion/') or
            name.startswith('lc_rec/a0_beauty/grid_sid_preparation/identity_audit_abfe/') or
            name.startswith('lc_rec/a0_beauty/evidence/g3-production-20261118/') or
            name.startswith('lc_rec/letter/scripts/') or name.startswith('lc_rec/shuffle/') or
            (name.startswith('lc_rec/a0_beauty/') and len(Path(name).parts)==3 and name.endswith('.py')) or
            name.startswith('lc_rec/a0_beauty/config/') or name.startswith('lc_rec/letter/config/'))
        if not wanted or '.pyc' in name or '__pycache__' in name:continue
        value=subprocess.check_output(['git','show',a.commit+':'+name],cwd=repo)
        if len(value)>8*1024**2:raise ValueError('large review member '+name)
        files[name]=value
    for name in ('lc_rec/letter/evidence/formal_20261111/model_identity_preflight.json',
                 'lc_rec/letter/evidence/formal_20261111/precision_approval.json',
                 'lc_rec/letter/review/final_audit_20261113/external/official_cce.py',
                 'lc_rec/letter/review/final_audit_20261113/statistics_verify.py'):
        files[name]=subprocess.check_output(['git','show',a.commit+':'+name],cwd=repo)
    vendor='lc_rec/letter/vendor/LETTER/LETTER-LC-Rec/'
    for name in ('data.py','collator.py','evaluate.py','test_ddp.py','prompt.py','utils.py','modeling_letter.py','config/ds_z2_bf16.json'):
        files[vendor+name]=subprocess.check_output(['git','show',a.commit+':'+vendor+name],cwd=repo)
    references={'delivery_commit':a.commit,'repository':'woobowen/recommend','branch':'wbw-lc-rec',
        'fixed_github':'https://github.com/woobowen/recommend/tree/'+a.commit,
        'base_conversion_commit':'8f36720a024be671a3ba893370a98126bc0f3b4a',
        'upstream_audit_commit':'e60c46b6c550da5b5d04aa0f0484a99043f4d40d',
        'large_assets':'Remain on new-server; see the pinned run requests and input identity receipts. No model/checkpoint is included.',
        'formal_experiment_started':False,'formal_launch_authorized':False}
    files['REVIEW_REFERENCES.json']=(json.dumps(references,indent=2)+'\n').encode()
    rows=[];suspicious=[]
    secret_patterns=[rb'-----BEGIN (?:OPENSSH|RSA|EC|DSA) PRIVATE KEY-----',rb'gh[pousr]_[A-Za-z0-9]{24,}',rb'github_pat_[A-Za-z0-9_]{30,}',rb'sk-[A-Za-z0-9]{32,}']
    for name,value in files.items():
        path=PurePosixPath(name)
        if path.is_absolute() or '..' in path.parts or '\\' in name:raise ValueError('unsafe path '+name)
        if any(re.search(pattern,value) for pattern in secret_patterns):suspicious.append(name)
        rows.append({'path':name,'bytes':len(value),'sha256':hashlib.sha256(value).hexdigest()})
    if suspicious:raise ValueError('potential secret in '+repr(suspicious))
    files['MEMBER_SHA256.json']=(json.dumps({'algorithm':'SHA-256','members':rows,
        'manifest_note':'This manifest excludes itself; whole ZIP SHA is outside the archive.'},indent=2)+'\n').encode()
    if a.output.exists():raise FileExistsError(a.output)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(a.output,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for name,value in sorted(files.items()):z.writestr(name,value)
    with zipfile.ZipFile(a.output) as z:
        if z.testzip() is not None:raise ValueError('CRC failure')
        if len(z.namelist())!=len(set(z.namelist())):raise ValueError('duplicate ZIP members')
        for row in rows:
            value=z.read(row['path'])
            if len(value)!=row['bytes'] or hashlib.sha256(value).hexdigest()!=row['sha256']:raise ValueError('member identity mismatch')
    result={'zip_path':str(a.output.resolve()),'bytes':a.output.stat().st_size,
        'sha256':hashlib.sha256(a.output.read_bytes()).hexdigest(),'members':len(files),
        'delivery_commit':a.commit,'crc':'PASS','member_sha256':'PASS','duplicate_members':0,
        'dangerous_paths':0,'sensitive_content_scan':'PASS: private keys and common GitHub/API token patterns',
        'scope':'Source, fixed small SID/bridge assets, raw CPU/dry-run and non-author evidence; no GPU PASS implied.',
        'all_members_including_manifest':[{'path':n,'bytes':len(v),'sha256':hashlib.sha256(v).hexdigest()} for n,v in sorted(files.items())]}
    a.output.with_suffix(a.output.suffix+'.sha256').write_text(result['sha256']+'  '+a.output.name+'\n')
    a.output.with_suffix(a.output.suffix+'.verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='all_members_including_manifest'},indent=2))


if __name__=='__main__':main()
