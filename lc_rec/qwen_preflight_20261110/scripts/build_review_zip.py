"""One lightweight review ZIP from a fixed Git tree and external final receipts."""
import argparse,hashlib,json,re,shutil,subprocess,tempfile,zipfile
from pathlib import Path,PurePosixPath

REPO=Path(__file__).resolve().parents[3]
PREFIX='lc_rec/qwen_preflight_20261110/'

def digest(data):return hashlib.sha256(data).hexdigest()
def scan(name,data):
 p=PurePosixPath(name);assert not p.is_absolute() and '..' not in p.parts and '\\' not in name
 assert not name.endswith(('.whl','.bin','.safetensors','.pt','.pth','.pem','.key','.p12','.tar','.gz','.zip'))
 assert len(data)<8*1024**2,'No large model, adapter or unrelated log'
 text=data.decode('utf8','replace')
 patterns=[r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----',r'\bghp_[A-Za-z0-9]{25,}\b',r'\bgithub_pat_[A-Za-z0-9_]{30,}\b',r'\bhf_[A-Za-z0-9]{25,}\b',r'\bAKIA[A-Z0-9]{16}\b']
 for pattern in patterns:assert not re.search(pattern,text),'Secret-like payload in '+name

def main():
 p=argparse.ArgumentParser();p.add_argument('--commit',required=True);p.add_argument('--receipts',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--windows-copy',type=Path,required=True);a=p.parse_args()
 commit=subprocess.check_output(['git','rev-parse',a.commit],cwd=REPO,text=True).strip();assert len(commit)==40
 tree=subprocess.check_output(['git','ls-tree','-r','--full-tree',commit,'--',PREFIX],cwd=REPO,text=True).splitlines();members={};sources={}
 for line in tree:
  meta,name=line.split('\t',1);mode,kind,blob=meta.split();assert mode=='100644' and kind=='blob';assert name.startswith(PREFIX)
  rel=name[len(PREFIX):];data=subprocess.check_output(['git','show',commit+':'+name],cwd=REPO);scan(rel,data);members[rel]=data;sources[rel]={'git_path':name,'git_blob':blob,'commit':commit}
 assert 'REVIEW_ENTRY.md' in members,'Exactly one review entry required'
 basis=json.loads(members['metadata/method_basis.json'])
 for row in basis['frozen_reference_files']:
  assert row['path'].startswith('lc_rec/letter/') and len(row['commit'])==40
  data=subprocess.check_output(['git','show',row['commit']+':'+row['path']],cwd=REPO);assert digest(data)==row['sha256']
  rel='reference_source/'+row['path'];assert rel not in members;scan(rel,data);members[rel]=data;sources[rel]={'git_path':row['path'],'commit':row['commit'],'frozen_reference':True}
 for path in sorted(a.receipts.rglob('*')):
  if not path.is_file():continue
  assert not path.is_symlink();rel='final_receipts/'+str(path.relative_to(a.receipts));assert rel not in members;data=path.read_bytes();scan(rel,data);members[rel]=data;sources[rel]={'external_receipt':str(path)}
 assert sum(map(len,members.values()))<64*1024**2
 manifest={'delivery_commit':commit,'files':[{'path':name,'bytes':len(data),'sha256':digest(data),'source':sources[name]} for name,data in sorted(members.items())]}
 members['SHA256_MANIFEST.json']=(json.dumps(manifest,indent=2)+'\n').encode()
 a.output.parent.mkdir(parents=True,exist_ok=True)
 assert not a.output.exists(),'Do not overwrite an existing delivered ZIP'
 with zipfile.ZipFile(a.output,'w',zipfile.ZIP_DEFLATED,compresslevel=9) as z:
  for name,data in sorted(members.items()):z.writestr(name,data)
 with zipfile.ZipFile(a.output) as z:
  names=z.namelist();assert len(names)==len(set(names)) and set(names)==set(members);assert z.testzip() is None
  assert names.count('REVIEW_ENTRY.md')==1
  with tempfile.TemporaryDirectory(prefix='qwen-review-verify-') as temporary:
   root=Path(temporary)
   for name in names:
    info=z.getinfo(name);assert not ((info.external_attr>>16)&0o170000)==0o120000
    scan(name,z.read(name));assert info.file_size==len(members[name])
   z.extractall(root)
   for row in manifest['files']:
    value=(root/row['path']).read_bytes();assert len(value)==row['bytes'] and digest(value)==row['sha256']
   assert (root/'SHA256_MANIFEST.json').read_bytes()==members['SHA256_MANIFEST.json']
 zip_hash=digest(a.output.read_bytes());a.windows_copy.parent.mkdir(parents=True,exist_ok=True);assert not a.windows_copy.exists();shutil.copyfile(a.output,a.windows_copy);assert digest(a.windows_copy.read_bytes())==zip_hash
 windows_path=subprocess.check_output(['wslpath','-w',str(a.windows_copy)],text=True).strip()
 ps="""$qwenZip='%s';$qwenItem=Get-Item -LiteralPath $qwenZip;[pscustomobject]@{bytes=$qwenItem.Length;sha256=(Get-FileHash -Algorithm SHA256 -LiteralPath $qwenZip).Hash}|ConvertTo-Json -Compress"""%windows_path.replace("'","''")
 output=subprocess.check_output(['/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe','-NoProfile','-NonInteractive','-Command',ps],text=True)
 checked=json.loads(output);assert checked['bytes']==a.output.stat().st_size and checked['sha256'].lower()==zip_hash
 receipt={'delivery_commit':commit,'wsl_path':str(a.output.resolve()),'windows_copy_wsl_path':str(a.windows_copy.resolve()),'windows_path':windows_path,'bytes':a.output.stat().st_size,'sha256':zip_hash,'members':len(members),'crc_pass':True,'full_unpack_all_member_size_sha_pass':True,'duplicate_and_path_safety_pass':True,'secret_pattern_scan_pass':True,'windows_powershell_actual_file_hash':checked}
 a.output.with_suffix(a.output.suffix+'.validation.json').write_text(json.dumps(receipt,indent=2)+'\n');a.output.with_suffix(a.output.suffix+'.sha256').write_text(zip_hash+'  '+a.output.name+'\n');print(json.dumps(receipt,indent=2))
if __name__=='__main__':main()
