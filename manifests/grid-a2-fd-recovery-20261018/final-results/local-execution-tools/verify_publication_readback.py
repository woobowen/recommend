#!/usr/bin/env python3
"""Verify the actual remote ref and anonymous GitHub bytes at a fixed commit."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import argparse
import datetime as dt
import hashlib
import json
import re
import subprocess
import urllib.request

BASE = Path(__file__).resolve().parent
REPO = Path('/home/addaswsw/recommend-worktrees/sid-extensions-p0')
PREFIX = 'manifests/grid-a2-fd-recovery-20261018/final-results/'

def git(*args):
    return subprocess.run(['git', *args], cwd=REPO, check=True, stdout=subprocess.PIPE).stdout

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('commit'); parser.add_argument('label', choices=['scientific','delivery'])
    args = parser.parse_args(); assert re.fullmatch('[0-9a-f]{40}', args.commit)
    assert git('branch','--show-current').decode().strip() == 'wbw'
    assert git('remote','get-url','origin').decode().strip() == 'https://github.com/woobowen/recommend.git'
    remote = git('ls-remote','origin','refs/heads/wbw').decode().strip()
    assert remote.split() == [args.commit, 'refs/heads/wbw'], remote
    paths = ['experiments/a2_fd_recovery_20261018/adapter.py', 'experiments/launch_20261017/a2_upstream.py',
             'manifests/grid-a2-fd-recovery-20261018/launch-v2/compiled/queue-config.json',
             PREFIX+'FINAL_RESULTS.md', PREFIX+'original-metrics.csv',
             PREFIX+'non-author/a2_outputs/A2_OUTPUTS_NONAUTHOR_REVIEW.json',
             PREFIX+'non-author/delivery_tools/prepublication-final-staging-review.json',
             PREFIX+'terminal-scene.json', PREFIX+'source-deployment.json',
             PREFIX+'cross-domain/cross-domain-evidence-index-with-a2.json',
             PREFIX+'non-author/asset_inventory/assets-not-copied.json']
    if args.label == 'delivery':
        paths += [PREFIX+'delivery/DELIVERY.json', PREFIX+'delivery/final-package-nonauthor-review.json']
    def fetch(path):
        expected = git('show', args.commit+':'+path)
        url = 'https://raw.githubusercontent.com/woobowen/recommend/'+args.commit+'/'+path
        request = urllib.request.Request(url, headers={'User-Agent':'GRID-authorized-result-byte-verification'})
        with urllib.request.urlopen(request, timeout=45) as response:
            actual = response.read(); status = response.status; headers = dict(response.headers)
        assert status == 200 and actual == expected, path
        return dict(path=path, URL=url, HTTP_status=status, bytes=len(actual),
                    sha256=hashlib.sha256(actual).hexdigest(), Git_blob_bytes_exact=True,
                    anonymously_accessible_without_credential=True,
                    content_type=headers.get('Content-Type'), ETag=headers.get('ETag'))
    with ThreadPoolExecutor(max_workers=4) as pool:
        files = list(pool.map(fetch, paths))
    destination = BASE/'publication'/args.label; destination.mkdir(parents=True,exist_ok=True)
    path = destination/'anonymous-readback.json'; assert not path.exists()
    report = dict(status='FIXED_GITHUB_REF_AND_ANONYMOUS_CRITICAL_BYTES_PASS',
                  checked_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                  repository='woobowen/recommend', branch='wbw', commit=args.commit,
                  actual_remote_ref=remote, scientific_execution_commit='85b77f5e2a5b497fdc1f620649ba488fc783a245',
                  files=files, final_package_integrity_not_inferred_from_Git=True)
    path.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(status=report['status'], commit=args.commit,
                         critical_files=len(files), path=str(path)),ensure_ascii=False,indent=2))

if __name__=='__main__':main()
