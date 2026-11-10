#!/usr/bin/env python3
"""Verify the continuation evidence without rerunning data, model, or GPU work."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import subprocess

from build_review_zip import secret_scan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    repo = root.parents[1]
    base = '61854bb1df0f26a71e68d95ab8eb564f7c762f76'
    assert subprocess.check_output(['git','branch','--show-current'],cwd=repo,text=True).strip()=='wbw-lc-rec'
    # Frozen history is checked by blob identity, not rerunning its CPU audit.
    tree = subprocess.check_output(['git','ls-tree','-rz',base,'lc_rec/letter/evidence','lc_rec/letter/review'],cwd=repo)
    historical = 0
    for row in tree.split(b'\0'):
        if not row:
            continue
        meta,name = row.split(b'\t',1)
        mode,kind,oid = meta.decode().split()
        assert kind=='blob' and mode in ('100644','100755')
        data=(repo/name.decode()).read_bytes()
        assert hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()==oid,name.decode()
        historical+=1
    manifests=[]
    for path in sorted((root/'evidence/autograd_20261110/jobs').glob('*/light-export-manifest.json')):
        manifest=json.loads(path.read_text())
        assert len(manifest['files'])==len({x['path'] for x in manifest['files']})
        for item in manifest['files']:
            rel=Path(item['path']);assert not rel.is_absolute() and '..' not in rel.parts
            p=path.parent/rel;data=p.read_bytes()
            assert not p.is_symlink() and len(data)==item['bytes'] and hashlib.sha256(data).hexdigest()==item['sha256'],str(p)
            secret_scan(p)
        state=json.loads((path.parent/'controller-query.json').read_text())
        assert state['cleanup_verified'] and not state['worker_alive'] and not state['workgroup_live']
        manifests.append({'job_id':path.parent.name,'status':state['status'],'exit_code':state['exit_code'],'members':len(manifest['files']),'cleanup_verified':True})
    changed=subprocess.check_output(['git','diff','--name-only',base],cwd=repo,text=True).splitlines()
    untracked=subprocess.check_output(['git','ls-files','--others','--exclude-standard','lc_rec/letter'],cwd=repo,text=True).splitlines()
    checked=[]
    for name in sorted(set(changed+untracked)):
        assert name.startswith('lc_rec/letter/'),name
        p=repo/name
        if not p.is_file():
            continue
        secret_scan(p)
        if p.suffix=='.py':
            ast.parse(p.read_text(),filename=str(p));compile(p.read_text(),str(p),'exec')
        elif p.suffix=='.json':
            json.loads(p.read_text())
        checked.append(name)
    links=[]
    entry=root/'AUTOGRAD_REVIEW_ENTRY.md'
    for link in re.findall(r'\]\(([^)]+)\)',entry.read_text()):
        if '://' in link or link.startswith('#'):
            continue
        p=(entry.parent/link.split('#',1)[0]).resolve()
        assert p.is_relative_to(repo) and p.exists(),link
        links.append(link)
    rounds=json.loads((root/'evidence/autograd_20261110/correction_rounds.json').read_text())
    assert rounds['prior_correction_rounds_used']==3 and 0<=rounds['current_correction_rounds_used']<=3
    result={'status':'PASS_DELIVERY_CONSISTENCY_ONLY','historical_evidence_blobs_unchanged':historical,
            'light_export_manifests':manifests,'changed_and_new_files_checked':len(checked),
            'local_review_links_checked':len(links),'secret_pattern_scan':'PASS',
            'no_model_GPU_or_data_audit_execution':True,'branch':'wbw-lc-rec'}
    Path(args.output).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
