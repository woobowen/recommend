#!/usr/bin/env python3
"""Package only the reviewed LC-Rec subtree and external final receipts, then extract and verify."""
import argparse,hashlib,json,pathlib,re,stat,subprocess,tempfile,zipfile

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def safe_name(name):
    p=pathlib.PurePosixPath(name)
    return bool(name) and not p.is_absolute() and '..' not in p.parts and '\\' not in name and ':' not in name and str(p)==name
def secret_scan(p):
    data=p.read_bytes()
    for pattern in [rb'-----BEGIN (?:OPENSSH|RSA|EC|DSA) PRIVATE KEY-----',rb'gh[pousr]_[A-Za-z0-9]{25,}',rb'github_pat_[A-Za-z0-9_]{20,}',rb'hf_[A-Za-z0-9]{25,}',rb'AKIA[A-Z0-9]{16}']:
        if re.search(pattern,data):raise ValueError('secret-like content: '+str(p))
    if p.suffix.lower() in ['.safetensors','.bin','.whl','.pt','.pth','.ckpt'] or p.name in ['Instruments.zip','hosts.yml','id_rsa','id_ed25519']:
        raise ValueError('excluded asset/credential: '+str(p))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--repo',type=pathlib.Path,required=True);ap.add_argument('--receipts',type=pathlib.Path,required=True);ap.add_argument('--output',type=pathlib.Path,required=True);a=ap.parse_args()
    a.output.parent.mkdir(parents=True,exist_ok=True)
    tree=subprocess.check_output(['git','ls-tree','-r','-z','HEAD','lc_rec'],cwd=a.repo)
    expected={}
    for row in tree.split(b'\0'):
        if not row:continue
        meta,name=row.split(b'\t',1);mode,kind,blob=meta.decode().split()
        assert kind=='blob' and mode in ['100644','100755']
        expected[name.decode()]=blob
    files={name:a.repo/name for name in expected}
    for p in sorted(a.receipts.rglob('*')):
        if p.is_file():files['final_receipts/'+str(p.relative_to(a.receipts))]=p
    assert files and len(files)==len(set(files))
    manifest=[];payloads={}
    for name,p in sorted(files.items()):
        assert safe_name(name) and p.is_file() and not p.is_symlink()
        secret_scan(p);data=p.read_bytes();payloads[name]=data
        blob=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
        if name in expected:assert blob==expected[name], 'working bytes differ from declared HEAD: '+name
        manifest.append({'path':name,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),'git_blob_sha1':blob if name in expected else None})
    manifest_name='MEMBER_SHA256.json';assert manifest_name not in files
    body=(json.dumps({'files':manifest,'manifest_excludes_itself':True},ensure_ascii=False,indent=2)+'\n').encode()
    with zipfile.ZipFile(a.output,'x',zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for name,data in sorted(payloads.items()):z.writestr(name,data)
        z.writestr(manifest_name,body)
    with zipfile.ZipFile(a.output) as z:
        members=z.infolist();znames=[m.filename for m in members]
        assert len(znames)==len(set(znames)) and all(safe_name(n) for n in znames)
        assert z.testzip() is None
        assert set(znames)==set(files)|{manifest_name}
        assert sum(m.file_size for m in members)<50*1024**2
        with tempfile.TemporaryDirectory(prefix='lc-rec-zip-check-') as temp:
            z.extractall(temp)
            extracted=pathlib.Path(temp)
            parsed=json.loads((extracted/manifest_name).read_text())
            for row in parsed['files']:
                p=extracted/row['path'];assert p.stat().st_size==row['bytes'] and digest(p)==row['sha256']
                secret_scan(p)
    receipt={'zip_path':str(a.output.resolve()),'bytes':a.output.stat().st_size,'sha256':digest(a.output),'members':len(members),'file_payloads':len(manifest),'crc_verified':True,'actual_extract_and_member_hash_verification':True,'safe_paths':True,'secret_scan':'PASS (credential patterns and excluded asset types; human review also required)','source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=a.repo,text=True).strip()}
    a.output.with_suffix(a.output.suffix+'.sha256').write_text(receipt['sha256']+'  '+a.output.name+'\n')
    a.output.with_suffix(a.output.suffix+'.receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))
if __name__=='__main__':main()
