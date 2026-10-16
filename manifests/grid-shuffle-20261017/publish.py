"""Serial wbw publication with actual GitHub bytes and server snapshot readback."""
import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
from urllib.parse import quote

REPO=Path('/home/addaswsw/recommend-worktrees/sid-extensions-p0')
SUPPORT=Path('/home/addaswsw/recommend-support/grid-shuffle-20261017')
PREFIX='manifests/grid-shuffle-20261017/'
ALLOWED=['experiments/shuffle_20261017/',PREFIX]
DOCS=['docs/sid-extensions/experiments/shuffle.md','docs/sid-extensions/experiments/attention.md']
SECRET=re.compile(rb'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----|gh[pousr]_[A-Za-z0-9]{25,}|github_pat_[A-Za-z0-9_]{30,}|sk-[A-Za-z0-9]{30,}')


def git(*args):return subprocess.check_output(['git',*args],cwd=REPO,text=True).strip()
def identity(data):return {'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('phase',choices=['content','delivery']);args=p.parse_args()
    assert git('branch','--show-current')=='wbw' and not git('diff','--cached','--name-only')
    before=git('rev-parse','HEAD');remote=git('ls-remote','origin','refs/heads/wbw').split()[0]
    subprocess.run(['git','merge-base','--is-ancestor',remote,before],cwd=REPO,check=True)
    paths=sorted(set(git('diff','--name-only').splitlines()+git('ls-files','--others','--exclude-standard','--',*ALLOWED,*DOCS).splitlines()))
    assert paths and all(n.startswith(tuple(ALLOWED)) or n in DOCS for n in paths),paths
    if args.phase=='delivery':
        assert all(n==PREFIX+'DELIVERY_RECEIPT.json' or n.startswith(PREFIX+'delivery-audit/') for n in paths)
    for name in paths:
        f=REPO/name;assert f.is_file() and not f.is_symlink() and not SECRET.search(f.read_bytes()),name
    subprocess.run(['git','add','--',*paths],cwd=REPO,check=True)
    assert set(git('diff','--cached','--name-only').splitlines())==set(paths)
    subprocess.run(['git','diff','--cached','--check'],cwd=REPO,check=True)
    message='Publish verified Toys Sports Shuffle tests attention and explicit A gaps' if args.phase=='content' else 'Record verified Shuffle review package and final delivery audit'
    subprocess.run(['git','commit','-m',message],cwd=REPO,check=True);commit=git('rev-parse','HEAD')
    subprocess.run(['git','push','origin','HEAD:refs/heads/wbw'],cwd=REPO,check=True,timeout=120)
    assert git('ls-remote','origin','refs/heads/wbw').split()[0]==commit
    ref=json.loads(subprocess.check_output(['gh','api','repos/woobowen/recommend/git/ref/heads/wbw'],cwd=REPO))
    assert ref['object']['sha']==commit
    remote_commit=json.loads(subprocess.check_output(['gh','api','repos/woobowen/recommend/git/commits/'+commit],cwd=REPO))
    assert remote_commit['tree']['sha']==git('rev-parse',commit+'^{tree}')
    critical=[PREFIX+'FINAL_REPORT.md',PREFIX+'EXECUTION_SUMMARY.json','experiments/formal_20261015/run.py',
        'experiments/shuffle_20261017/preflight.py','experiments/shuffle_20261017/deploy.py',
        PREFIX+'attention-statistics/attention-descriptive-statistics.json',PREFIX+'results-audit/toys-sports-paired-shuffle-analysis.json']
    critical.extend(f'experiments/shuffle_20261017/specs/{d}-b-{n}.json' for d in ['toys','sports'] for n in [0,3,4])
    critical.extend(f'experiments/shuffle_20261017/specs/attention-{d}-b-{n}.json' for d in ['toys','sports'] for n in [0,3,4])
    if args.phase=='delivery':critical.extend([PREFIX+'DELIVERY_RECEIPT.json',PREFIX+'delivery-audit/zip-independent-review.json'])
    blobs=[]
    for name in critical:
        data=subprocess.check_output(['gh','api','-H','Accept: application/vnd.github.raw+json',
            'repos/woobowen/recommend/contents/'+quote(name,safe='/')+'?ref='+commit],cwd=REPO,timeout=30)
        expected=subprocess.check_output(['git','show',commit+':'+name],cwd=REPO)
        assert data==expected,name
        blobs.append({'path':name,**identity(data)})
    summary=json.loads((REPO/PREFIX/'EXECUTION_SUMMARY.json').read_text())
    roots=sorted({r['snapshot_root'] for r in summary['jobs'] if r['snapshot_sha256']})
    sys.path.insert(0,'/home/addaswsw/recommend-support/sid-extensions-p0/scripts')
    from transport import Transport
    t=Transport('new-server');t.identity()
    code="""from pathlib import Path
import json,hashlib
results=[]
for name in ROOTS:
 p=Path(name);mf=p/'DEPLOYMENT.json';d=json.loads(mf.read_text())
 for row in d['files']:
  f=p/row['path'];assert f.is_file() and not f.is_symlink()
  assert f.stat().st_size==row['bytes'] and hashlib.sha256(f.read_bytes()).hexdigest()==row['sha256']
 results.append({'root':name,'manifest_sha256':hashlib.sha256(mf.read_bytes()).hexdigest(),'manifest':d})
print(json.dumps(results))
""".replace('ROOTS',repr(roots))
    snapshots=json.loads(t.python(code,timeout=60))
    by_root={item['root']:item for item in snapshots}
    for job in summary['jobs']:
        if not job['snapshot_sha256']:continue
        item=by_root[job['snapshot_root']];manifest=item['manifest']
        request=json.loads((Path(job['local_evidence'])/'request.json').read_text())
        assert request['cwd']==item['root']
        assert request['capture_version']==manifest['source_commit']+'/'+manifest['snapshot_sha256']
        assert {r['path']:r['sha256'] for r in request['inputs']}[item['root']+'/DEPLOYMENT.json']==item['manifest_sha256']
    for item in snapshots:
        manifest=item['manifest'];source=manifest['source_commit']
        subprocess.run(['git','merge-base','--is-ancestor',source,commit],cwd=REPO,check=True)
        for row in manifest['files']:
            actual=subprocess.check_output(['git','show',source+':'+row['path']],cwd=REPO)
            assert identity(actual)=={k:row[k] for k in ['bytes','sha256']},row['path']
    receipt={'phase':args.phase,'commit':commit,'parent_commit':before,'branch':'wbw',
        'observed_at_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'remote_ref_verified':True,
        'GitHub_ref_and_tree_verified':True,'key_file_bytes_verified':True,'actual_GitHub_file_readbacks':blobs,
        'changed_paths':paths,'server_snapshot_mapping_verified':True,'server_snapshots':snapshots,
        'server_has_git':False,'other_branches_written':False,'force_push':False}
    out=SUPPORT/('publication-'+args.phase+'.json')
    with out.open('x') as f:json.dump(receipt,f,indent=2);f.write('\n')
    print(json.dumps({'commit':commit,'phase':args.phase,'key_files_read_back':len(blobs),'server_snapshots_verified':len(snapshots)}))


if __name__=='__main__':main()
