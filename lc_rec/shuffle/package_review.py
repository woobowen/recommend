"""Build and validate a lightweight review ZIP from a fixed delivered commit."""
import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path,PurePosixPath
import re
import shutil
import subprocess
import zipfile
from control import REPO,SUPPORT
from collect_results import RUN


def digest(raw):return hashlib.sha256(raw).hexdigest()
def file_digest(path):return digest(path.read_bytes())
def main():
    p=argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument('--commit',required=True);p.add_argument('--readback',type=Path,required=True)
    a=p.parse_args();commit=subprocess.check_output(['git','rev-parse',a.commit],cwd=REPO,text=True).strip()
    readback=json.loads(a.readback.read_text())
    assert readback['status']=='PASS' and readback['DELIVERY_COMMIT']==commit
    assert readback['SHUFFLE_RUN_COMMIT']==RUN
    selections=['lc_rec/shuffle','lc_rec/a1_qwen3_06b/A1_NORMAL_FINAL_REVIEW_ENTRY.md',
      'lc_rec/a1_qwen3_06b/review/final_20261114','lc_rec/letter/A0_NORMAL_FINAL_REVIEW_ENTRY.md',
      'lc_rec/letter/EVALUATION_PROTOCOL_V1.md','lc_rec/letter/config/a0_evaluation_v1.json',
      'lc_rec/letter/scripts/formal_evaluate.py','lc_rec/a1_qwen3_06b/scripts/evaluate.py',
      'lc_rec/qwen_preflight_20261110/scripts/common.py','lc_rec/a1_qwen3_06b/scripts/offline_metrics.py',
      'lc_rec/a1_qwen3_06b/scripts/runner.py',
      'tools/sid-extensions-prep/transport.py','tools/sid-extensions-prep/gridctl.py',
      'lc_rec/a1_qwen3_06b/review/independent_extended.py',
      'lc_rec/letter/review/formal_det_20261112/independent_metrics.py',
      'lc_rec/letter/review/final_audit_20261113/statistics_verify.py',
      'lc_rec/letter/review/final_audit_20261113/external/official_cce.py',
      'lc_rec/letter/review/final_audit_20261113/external/LICENSE',
      'lc_rec/letter/review/final_audit_20261113/external/source_identity.json',
      'lc_rec/letter/review/identity/IDENTITY_AUDIT.md',
      'lc_rec/letter/evidence/formal_det_20261112/model_identity.json',
      'lc_rec/letter/vendor/LETTER/data/Instruments/Instruments.index.json',
      'lc_rec/letter/vendor/LETTER/data/Instruments/Instruments.inter.json']
    for name in ['data.py','collator.py','prompt.py','utils.py','evaluate.py','test_ddp.py']:
        selections.append('lc_rec/letter/vendor/LETTER/LETTER-LC-Rec/'+name)
    for prefix in [
      'lc_rec/letter/evidence/final_audit_20261113/jobs/letter-a0-20261112-final-audit-20261113-offline-r1/results/primary',
      'lc_rec/a1_qwen3_06b/evidence/status_checks/20261114T034030Z/existing_results/offline_evaluation/primary']:
        selections += [prefix+'/'+n for n in ['metrics.json','groups.json','bootstrap.json']]
    tree=subprocess.check_output(['git','ls-tree','-r','-z',commit,'--',*selections],cwd=REPO)
    members={}
    for row in tree.split(b'\0'):
        if not row:continue
        identity,name=row.split(b'\t',1);mode,kind,blob=identity.decode().split();name=name.decode()
        assert kind=='blob' and mode in ['100644','100755']
        assert not re.search(r'(?:adapter_model|model-\d|model\.safetensors|predictions\.rank)',name)
        raw=subprocess.check_output(['git','cat-file','blob',blob],cwd=REPO)
        assert len(raw)<12*1024**2
        members['repo/'+name]=raw
    assert 'repo/lc_rec/shuffle/FINAL_REVIEW_ENTRY.md' in members
    assert b'FINAL AUDIT: PASS' in members['repo/lc_rec/shuffle/FINAL_REVIEW_ENTRY.md']
    for model,prefix in [('a0','letter-a0-20261112-'),('a1','letter-a1-q06-20261113-')]:
        for n in [3,4]:
            base='repo/lc_rec/shuffle/results/'+prefix+f'shuffle{n}-instruments-seed42-20261114-r0/'
            for path in ['job/status.json','evaluation/primary/metrics.json','evaluation/primary/groups.json',
                         'evaluation/primary/bootstrap.json','evaluation/paired/paired_comparison.json',
                         'evaluation/independent_inputs.json','evaluation/independent_extended.json',
                         'evaluation/independent_statistics.json','evaluation/prediction_manifest.json']:
                assert base+path in members,base+path
    members['external/GITHUB_FINAL_READBACK.json']=a.readback.read_bytes()
    entry=f'''# LETTER A0/A1 Shuffle review package

FINAL DELIVERY COMMIT: {commit}
GPU / CPU EVALUATION RUN: {RUN}

Start: https://github.com/woobowen/recommend/blob/{commit}/lc_rec/shuffle/FINAL_REVIEW_ENTRY.md

This package contains the actual source, configuration, all four result and independent-check receipts, frozen Normal tables, source/deployment identities, and GitHub readback. Large model/adapter/checkpoint payloads and full prediction JSONL stay on new-server, located by the recorded paths, byte sizes and SHA-256. No training or extra shuffle realization is included.

MEMBERS_SHA256.json hashes every other member; it excludes itself to avoid circular identity. The whole ZIP SHA-256 and validation receipt are outside the ZIP. Original fixed evidence and review limitations remain in the repository reports.
'''
    members['REVIEW_ENTRY.md']=entry.encode()
    assert len(members)==len({n.casefold() for n in members})
    for name in members:
        parts=PurePosixPath(name)
        assert not parts.is_absolute() and '..' not in parts.parts and '\\' not in name and ':' not in name
        assert not any(ord(c)<32 for c in name)
    secrets=[rb'-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----',rb'gh[pousr]_[A-Za-z0-9]{30,}',
             rb'github_pat_[A-Za-z0-9_]{30,}',rb'AKIA[0-9A-Z]{16}']
    for name,raw in members.items():
        assert not any(re.search(pattern,raw) for pattern in secrets),'Potential secret in '+name
    manifest={'DELIVERY_COMMIT':commit,'SHUFFLE_RUN_COMMIT':RUN,
      'self_excluded':'MEMBERS_SHA256.json','members':[{'path':n,'bytes':len(raw),'sha256':digest(raw)} for n,raw in sorted(members.items())]}
    members['MEMBERS_SHA256.json']=(json.dumps(manifest,indent=2)+'\n').encode()
    assert sum(map(len,members.values()))<100*1024**2
    out=SUPPORT/'delivery';out.mkdir(parents=True,exist_ok=True)
    path=out/('LETTER_A0_A1_Shuffle_Final_'+commit[:12]+'.zip')
    assert not path.is_symlink()
    if not path.exists():
        stamp=int(subprocess.check_output(['git','show','-s','--format=%ct',commit],cwd=REPO,text=True))
        timestamp=dt.datetime.fromtimestamp(stamp,dt.timezone.utc).timetuple()[:6]
        with zipfile.ZipFile(path,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
            for name,raw in sorted(members.items()):
                info=zipfile.ZipInfo(name,timestamp);info.compress_type=zipfile.ZIP_DEFLATED
                info.external_attr=0o100644<<16;z.writestr(info,raw)
    with zipfile.ZipFile(path) as z:
        assert z.testzip() is None
        assert len(z.namelist())==len(set(z.namelist()))==len(members)
        assert set(z.namelist())==set(members)
        for name,raw in members.items():assert z.read(name)==raw
        unpacked=out/('verified-unpacked-'+commit[:12])
        assert not unpacked.is_symlink()
        if not unpacked.exists():unpacked.mkdir();z.extractall(unpacked)
        for name,raw in members.items():
            target=unpacked/name
            assert not target.is_symlink() and target.resolve().is_relative_to(unpacked.resolve())
            assert target.is_file() and target.read_bytes()==raw
    whole=file_digest(path);windows=Path('/mnt/c/Users/lenovo/Downloads')/path.name
    assert windows.parent.is_dir()
    assert not windows.is_symlink()
    if windows.exists():assert file_digest(windows)==whole
    else:
        with path.open('rb') as src,windows.open('xb') as dest:shutil.copyfileobj(src,dest)
    assert windows.stat().st_size==path.stat().st_size and file_digest(windows)==whole
    windows_name=subprocess.check_output(['wslpath','-w',str(windows)],text=True).strip()
    powershell='/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe'
    command="$p='"+windows_name.replace("'","''")+"';$f=Get-Item -LiteralPath $p;$h=Get-FileHash -LiteralPath $p -Algorithm SHA256;[pscustomobject]@{Path=$f.FullName;Bytes=$f.Length;SHA256=$h.Hash}|ConvertTo-Json -Compress"
    native=json.loads(subprocess.check_output([powershell,'-NoProfile','-NonInteractive','-Command',command],encoding='utf-8-sig'))
    assert native['SHA256'].lower()==whole and native['Bytes']==path.stat().st_size
    receipt={'status':'PASS_ZIP_VALIDATION','DELIVERY_COMMIT':commit,'SHUFFLE_RUN_COMMIT':RUN,
      'WSL_path':str(path),'Windows_path':windows_name,'bytes':path.stat().st_size,'sha256':whole,
      'members':len(members),'uncompressed_bytes':sum(map(len,members.values())),
      'member_sha256_count':len(manifest['members']),'CRC':'PASS','full_extraction_and_byte_comparison':'PASS',
      'duplicate_and_unsafe_paths':'PASS','secret_pattern_scan':'PASS','Windows_native_readback':native,
      'user_controlled_retained_backup':windows_name,'verified_unpacked_path':str(unpacked),
      'validation_utc':dt.datetime.now(dt.timezone.utc).isoformat()}
    (out/(path.name+'.sha256')).write_text(whole+'  '+path.name+'\n')
    (out/'ZIP_VALIDATION.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()
