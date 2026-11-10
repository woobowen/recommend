#!/usr/bin/env python3
"""Build and actually extract a fixed Git delivery plus external readback receipts."""
import argparse,hashlib,json,pathlib,re,subprocess,tempfile,zipfile,shutil

def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
    return h.hexdigest()
def safe(n):
    p=pathlib.PurePosixPath(n)
    return n and not p.is_absolute() and '..' not in p.parts and '\\' not in n and ':' not in n and str(p)==n
def scan(name,b):
    for p in [rb'-----BEGIN (?:OPENSSH|RSA|EC|DSA) PRIVATE KEY-----',rb'gh[pousr]_[A-Za-z0-9]{25,}',rb'github_pat_[A-Za-z0-9_]{20,}',rb'hf_[A-Za-z0-9]{25,}',rb'AKIA[A-Z0-9]{16}']:
        if re.search(p,b):raise ValueError('Secret-like payload '+name)
    if pathlib.PurePosixPath(name).suffix in ['.safetensors','.bin','.whl','.pt','.pth','.ckpt'] or pathlib.PurePosixPath(name).name in ['id_rsa','id_ed25519','hosts.yml']:raise ValueError('Forbidden artifact '+name)
def main():
    p=argparse.ArgumentParser();p.add_argument('--repo',required=True,type=pathlib.Path);p.add_argument('--receipts',required=True,type=pathlib.Path);p.add_argument('--output',required=True,type=pathlib.Path);p.add_argument('--windows-copy',required=True,type=pathlib.Path);a=p.parse_args()
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=a.repo,text=True).strip()
    selected=['lc_rec/six_task_lora','lc_rec/vendor/original','lc_rec/evidence/official_manifest.json','lc_rec/evidence/aux_pairing/output/auxiliary_pairing.json','lc_rec/evidence/cpu_input_asset_hashes.json','lc_rec/offload/evidence/preparation/model_identity_readback_20261109.json']
    tree=subprocess.check_output(['git','ls-tree','-r','-z',commit,*selected],cwd=a.repo)
    files={};blobs={}
    for item in tree.split(b'\0'):
        if not item:continue
        meta,name=item.split(b'\t',1);mode,kind,blob=meta.decode().split();name=name.decode()
        assert kind=='blob' and mode in ['100644','100755'];files[name]=a.repo/name;blobs[name]=blob
    for q in sorted(a.receipts.rglob('*')):
        if q.is_file():files['final_receipts/'+str(q.relative_to(a.receipts))]=q
    assert files and len(files)==len(set(files))
    payloads={};members=[]
    for name,q in sorted(files.items()):
        assert safe(name) and q.is_file() and not q.is_symlink()
        b=q.read_bytes();scan(name,b);assert len(b)<64*1024**2
        if name in blobs:
            actual=hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest();assert actual==blobs[name],name+' differs from DELIVERY_COMMIT'
        payloads[name]=b;members.append({'path':name,'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest(),'git_blob_sha1':blobs.get(name)})
    assert sum(len(b) for b in payloads.values())<160*1024**2
    manifest={'DELIVERY_COMMIT':commit,'files':members,'manifest_excludes_itself':True}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(a.output,'x',zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for n,b in payloads.items():z.writestr(n,b)
        z.writestr('MEMBER_SHA256.json',json.dumps(manifest,indent=2)+'\n')
    with zipfile.ZipFile(a.output) as z:
        names=z.namelist();assert len(names)==len(set(names)) and all(safe(n) for n in names)
        assert z.testzip() is None
        assert set(names)==set(files)|{'MEMBER_SHA256.json'}
        with tempfile.TemporaryDirectory(prefix='lc-six-real-extract-') as t:
            z.extractall(t)
            for m in members:
                q=pathlib.Path(t)/m['path'];assert q.stat().st_size==m['bytes'] and digest(q)==m['sha256'];scan(m['path'],q.read_bytes())
    assert a.windows_copy.parent.is_dir() and not a.windows_copy.exists()
    shutil.copyfile(a.output,a.windows_copy);assert digest(a.output)==digest(a.windows_copy)
    win=subprocess.check_output(['wslpath','-w',str(a.windows_copy)],text=True).strip()
    receipt={'DELIVERY_COMMIT':commit,'wsl_path':str(a.output.resolve()),'windows_path':win,'windows_wsl_path':str(a.windows_copy),'bytes':a.output.stat().st_size,'sha256':digest(a.output),'member_count':len(names),'crc_verified':True,'full_extraction_verified':True,'every_member_size_sha256_verified':True,'duplicate_members':False,'dangerous_paths':False,'secret_pattern_scan':'PASS','windows_copy_sha256_verified':True,'scan_limit':'Pattern scan and review of included paths; not a universal proof about all possible secrets.'}
    for q in [a.output,a.windows_copy]:
        q.with_suffix(q.suffix+'.sha256').write_text(receipt['sha256']+'  '+q.name+'\n')
        q.with_suffix(q.suffix+'.receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2))
if __name__=='__main__':main()
