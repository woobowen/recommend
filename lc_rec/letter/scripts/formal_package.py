#!/usr/bin/env python3
"""Build and actually unpack/hash-check the lightweight A0 audit backup.

Includes fixed Git content and explicitly supplied publication/deployment
receipts. Never includes model, adapter, checkpoint or full raw predictions.
"""
import argparse
import hashlib
import json
from pathlib import Path,PurePosixPath
import re
import subprocess
import tempfile
import zipfile


def digest(data):return hashlib.sha256(data).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--commit',required=True);p.add_argument('--output',required=True)
    p.add_argument('--receipts',required=True);a=p.parse_args()
    repo=Path(__file__).resolve().parents[3]
    commit=subprocess.check_output(['git','rev-parse',a.commit],cwd=repo,text=True).strip()
    paths=subprocess.check_output(['git','ls-tree','-r','--name-only',commit,'lc_rec/letter'],cwd=repo,text=True).splitlines()
    members={}
    for name in paths:
        relative=name.removeprefix('lc_rec/letter/')
        include=(relative=='NORMAL_REVIEW_ENTRY.md' or relative.startswith(('evidence/formal_20261111/','review/formal_20261111/'))
                 or relative.startswith('scripts/formal_') or relative in ['scripts/compat.py','scripts/test_formal_replay.py','scripts/test_formal_evaluate.py','scripts/runner_timeout_fixture.py','config/formal_instruments_seed42.json','config/author_instruments.json']
                 or relative.startswith('vendor/LETTER/LETTER-LC-Rec/') and (relative.endswith('.py') or '/config/' in relative))
        if not include:continue
        if relative.endswith(('.bin','.pt','.pth','.safetensors','.model')) or 'predictions.rank' in relative:raise ValueError('payload selected: '+relative)
        data=subprocess.check_output(['git','show',commit+':'+name],cwd=repo)
        assert len(data)<=8*1024**2,name
        members[name]=data
    entry='lc_rec/letter/NORMAL_REVIEW_ENTRY.md'
    assert entry in members,'one review entry is required'
    for f in sorted(Path(a.receipts).rglob('*')):
        if f.is_file() and not f.is_symlink():
            assert f.stat().st_size<=8*1024**2
            members['final_receipts/'+str(f.relative_to(a.receipts))]=f.read_bytes()
    assert any(n.startswith('final_receipts/') for n in members)
    manifest={'DELIVERY_COMMIT':commit,'entry':entry,'files':[
       {'path':n,'bytes':len(b),'sha256':digest(b)} for n,b in sorted(members.items())]}
    members['MEMBER_SHA256.json']=(json.dumps(manifest,indent=2)+'\n').encode()
    secret_patterns=[rb'-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----',
                     rb'gh[pousr]_[A-Za-z0-9]{30,}',rb'github_pat_[A-Za-z0-9_]{40,}',rb'\bhf_[A-Za-z0-9]{30,}\b',rb'\bAKIA[0-9A-Z]{16}\b']
    for n,b in members.items():
        q=PurePosixPath(n)
        assert not q.is_absolute() and '..' not in q.parts and '\\' not in n
        assert not any(re.search(pattern,b) for pattern in secret_patterns),'possible credential in '+n
    output=Path(a.output);output.parent.mkdir(parents=True,exist_ok=True)
    assert not output.exists(),'preserve previous package; choose a distinct output name'
    with zipfile.ZipFile(output,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for n,b in sorted(members.items()):z.writestr(n,b)
    with zipfile.ZipFile(output) as z:
        assert z.testzip() is None
        assert len(z.namelist())==len(set(z.namelist()))==len(members)
        assert set(z.namelist())==set(members)
        with tempfile.TemporaryDirectory(prefix='letter-a0-zip-verify-') as temp:
            root=Path(temp)
            z.extractall(root)
            for n,b in members.items():
                path=root/n
                assert path.is_file() and not path.is_symlink() and path.stat().st_size==len(b) and digest(path.read_bytes())==digest(b)
    windows=subprocess.check_output(['wslpath','-w',str(output.resolve())],text=True).strip()
    # Check real Windows filesystem access, not merely WSL path conversion.
    command=['/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe','-NoProfile','-NonInteractive','-Command',
             "$p='"+windows.replace("'","''")+"'; if (!(Test-Path -LiteralPath $p -PathType Leaf)) {exit 2}; (Get-Item -LiteralPath $p).Length"]
    win_bytes=int(subprocess.check_output(command,text=True).strip())
    assert win_bytes==output.stat().st_size
    result={'status':'PASS','DELIVERY_COMMIT':commit,'wsl_path':str(output.resolve()),'windows_path':windows,
            'bytes':output.stat().st_size,'sha256':digest(output.read_bytes()),'members':len(members),
            'validation':['exists','ZIP_CRC','actual_full_extract','all_member_sizes_SHA256','no_duplicate_members','no_dangerous_paths','credential_pattern_scan','Windows_Test-Path_and_size'],
            'limitations':'Pattern scan complements scoped membership; no claim to detect every possible secret.'}
    output.with_suffix(output.suffix+'.sha256').write_text(result['sha256']+'  '+output.name+'\n')
    output.with_suffix(output.suffix+'.verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
