#!/usr/bin/env python3
"""Build and fully verify a lightweight review ZIP from a fixed Git commit."""
import argparse
import hashlib
import json
from pathlib import Path,PurePosixPath
import re
import shutil
import subprocess
import tempfile
import zipfile


def main():
    p=argparse.ArgumentParser();p.add_argument('--commit',required=True);p.add_argument('--publication',required=True)
    p.add_argument('--output',required=True);p.add_argument('--windows',required=True);a=p.parse_args()
    repo=Path(__file__).resolve().parents[3]
    commit=subprocess.check_output(['git','rev-parse',a.commit],cwd=repo,text=True).strip()
    paths=subprocess.check_output(['git','ls-tree','-r','--name-only',commit,'lc_rec/letter'],cwd=repo,text=True).splitlines()
    chosen=[]
    for name in paths:
        if name.startswith(('lc_rec/letter/scripts/','lc_rec/letter/config/','lc_rec/letter/evidence/attention_20261112/','lc_rec/letter/review/attention_20261112/')):
            chosen.append(name)
        elif name in ['lc_rec/letter/ATTENTION_BACKWARD_DIAGNOSIS_REVIEW_ENTRY.md','lc_rec/letter/RESUME_DIAGNOSIS_REVIEW_ENTRY.md',
                      'lc_rec/letter/evidence/resume_20261112/earliest_trace_raw_events.json','lc_rec/letter/evidence/resume_20261112/full_trace_comparison.json',
                      'lc_rec/letter/review/resume_20261112/final_review.md']:
            chosen.append(name)
        elif name.startswith('lc_rec/letter/vendor/LETTER/LETTER-LC-Rec/') and name.endswith(('.py','.sh')):
            chosen.append(name)
    output=Path(a.output);output.parent.mkdir(parents=True,exist_ok=True)
    assert not output.exists(), 'do not overwrite an existing delivery'
    patterns=[rb'-----BEGIN (?:OPENSSH |RSA |EC )?PRIVATE KEY-----',rb'\bgh[pousr]_[A-Za-z0-9]{25,}\b',
              rb'\bgithub_pat_[A-Za-z0-9_]{30,}\b',rb'\bsk-(?:proj-)?[A-Za-z0-9_-]{35,}\b']
    rows=[];names=set()
    with zipfile.ZipFile(output,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        payload={n:subprocess.check_output(['git','show',commit+':'+n],cwd=repo) for n in chosen}
        payload['PUBLICATION_READBACK.json']=Path(a.publication).read_bytes()
        payload['FIXED_IDENTITY.json']=(json.dumps({'DELIVERY_COMMIT':commit,'source':'git show fixed commit; external final readback is separate'},indent=2)+'\n').encode()
        for name,data in payload.items():
            path=PurePosixPath(name);assert not path.is_absolute() and '..' not in path.parts and name not in names
            assert not name.endswith(('.pt','.bin','.safetensors','.pth')) and len(data)<16*1024**2
            assert not any(re.search(pat,data) for pat in patterns), 'sensitive material '+name
            names.add(name);rows.append({'path':name,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()})
            archive.writestr(name,data)
        manifest={'coverage':'Every payload member except this manifest itself; manifest SHA is recorded in external validation receipt','files':rows}
        archive.writestr('SHA256_MANIFEST.json',json.dumps(manifest,indent=2)+'\n')
    with zipfile.ZipFile(output) as archive:
        assert archive.testzip() is None
        members=archive.infolist();assert len(members)==len({m.filename for m in members})
        for m in members:
            path=PurePosixPath(m.filename);assert not path.is_absolute() and '..' not in path.parts
            assert (m.external_attr>>16)&0o170000 != 0o120000
        with tempfile.TemporaryDirectory(prefix='attention-unzip-',dir=output.parent) as td:
            archive.extractall(td)
            for row in rows:
                extracted=Path(td)/row['path'];assert extracted.stat().st_size==row['bytes']
                assert hashlib.sha256(extracted.read_bytes()).hexdigest()==row['sha256']
        manifest_sha=hashlib.sha256(archive.read('SHA256_MANIFEST.json')).hexdigest()
    windows=Path(a.windows);windows.parent.mkdir(parents=True,exist_ok=True);assert not windows.exists()
    shutil.copyfile(output,windows)
    digest=hashlib.sha256(output.read_bytes()).hexdigest();assert hashlib.sha256(windows.read_bytes()).hexdigest()==digest
    winpath=subprocess.check_output(['wslpath','-w',str(windows)],text=True).strip()
    receipt={'DELIVERY_COMMIT':commit,'wsl':str(output.resolve()),'windows':winpath,'windows_mount':str(windows),
             'bytes':output.stat().st_size,'sha256':digest,'members':len(rows)+1,
             'uncompressed_payload_bytes':sum(x['bytes'] for x in rows),'manifest_sha256':manifest_sha,
             'checks':{'CRC':True,'full_extraction':True,'member_sizes_hashes':True,'unique_safe_paths':True,
                       'sensitive_scan':True,'windows_copy_hash_access':True}}
    output.with_suffix(output.suffix+'.validation.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()
