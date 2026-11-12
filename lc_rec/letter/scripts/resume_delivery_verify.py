#!/usr/bin/env python3
"""Read-only delivery checks, including historical evidence preservation."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import subprocess

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
    repo=Path(__file__).resolve().parents[3];root=repo/'lc_rec/letter'
    old='2c13d30c61337be961dca3d3d4cc05a63a463117'
    unchanged=0
    for path in subprocess.check_output(['git','ls-tree','-r','--name-only',old,'lc_rec/letter/evidence','lc_rec/letter/review','lc_rec/letter/vendor'],cwd=repo,text=True).splitlines():
        data=subprocess.check_output(['git','show',old+':'+path],cwd=repo)
        assert (repo/path).read_bytes()==data,path;unchanged+=1
    checked=[]
    names=subprocess.check_output(['git','diff','--name-only',old],cwd=repo,text=True).splitlines()
    names+=subprocess.check_output(['git','ls-files','--others','--exclude-standard','lc_rec/letter'],cwd=repo,text=True).splitlines()
    patterns=[rb'-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----',rb'gh[pousr]_[A-Za-z0-9]{30,}',rb'github_pat_[A-Za-z0-9_]{40,}',rb'\bhf_[A-Za-z0-9]{30,}\b',rb'\bAKIA[0-9A-Z]{16}\b']
    for name in sorted(set(names)):
        assert name.startswith('lc_rec/letter/'),name
        path=repo/name
        if not path.is_file():continue
        assert not path.is_symlink();data=path.read_bytes()
        assert len(data)<8*1024**2 and not any(re.search(x,data) for x in patterns),name
        if path.suffix=='.py':ast.parse(data,filename=name);compile(data,name,'exec')
        if path.suffix=='.json':json.loads(data)
        checked.append(name)
    links=[];entry=root/'RESUME_DIAGNOSIS_REVIEW_ENTRY.md'
    if entry.exists():
        for link in re.findall(r'\]\(([^)]+)\)',entry.read_text()):
            if '://' in link or link.startswith('#'):continue
            dest=(entry.parent/link.split('#')[0]).resolve();assert dest.is_relative_to(repo) and dest.exists(),link;links.append(link)
    result={'status':'PASS_DELIVERY_INTEGRITY_ONLY','historical_blobs_unchanged':unchanged,'checked_new_or_changed_files':len(checked),'local_links':len(links),'secret_pattern_scan':'PASS','no_model_or_GPU_execution':True}
    Path(a.output).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))

if __name__=='__main__':main()
