"""Publish only wbw and read back GitHub bytes plus the fixed server export."""
import base64
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from urllib.parse import quote

from experiments.beauty_extensions.common import REPO, write_json
from .deploy import SUPPORT, TRANSPORT_SUPPORT, snapshot


def command(argv):
    return subprocess.check_output(argv,cwd=REPO,text=True).strip()


def main():
    assert command(['git','branch','--show-current'])=='wbw'
    assert not command(['git','diff','--name-only']) and not command(['git','diff','--cached','--name-only'])
    commit=command(['git','rev-parse','HEAD'])
    subprocess.run(['git','push','origin','HEAD:refs/heads/wbw'],cwd=REPO,check=True)
    remote=command(['git','ls-remote','origin','refs/heads/wbw']).split()[0];assert remote==commit
    ref=json.loads(command(['gh','api','repos/woobowen/recommend/git/ref/heads/wbw']));assert ref['object']['sha']==commit
    critical=['experiments/beauty_extensions/cli.py','experiments/beauty_extensions/authorization.py',
        'experiments/preflight_20261015/run.py','experiments/preflight_20261015/specs/beauty-b.json',
        'experiments/preflight_20261015/specs/toys-mlp.json','experiments/preflight_20261015/specs/sports-mlp.json',
        'manifests/grid-shuffle-mlp-preflight-20261015/EXECUTION_SUMMARY.json',
        'manifests/grid-shuffle-mlp-preflight-20261015/shuffle-a-execution-review.json',
        'manifests/grid-shuffle-mlp-preflight-20261015/shuffle-b-review.json',
        'manifests/grid-shuffle-mlp-preflight-20261015/source-conditions-review.json',
        'docs/sid-extensions/experiments/shuffle.md','docs/sid-extensions/experiments/mlp.md']
    blobs=[]
    for name in critical:
        record=json.loads(command(['gh','api','repos/woobowen/recommend/contents/'+quote(name,safe='/')+'?ref='+commit]))
        data=base64.b64decode(record['content']);expected=subprocess.check_output(['git','show',commit+':'+name],cwd=REPO)
        assert data==expected
        blobs.append({'path':name,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),'github_blob_sha':record['sha']})
    sys.path.insert(0,str(TRANSPORT_SUPPORT/'scripts'))
    from transport import Transport
    t=Transport('new-server');t.identity();root,manifest,manifest_sha=snapshot(t)
    for record in manifest['files']:
        data=subprocess.check_output(['git','show',commit+':'+record['path']],cwd=REPO)
        assert len(data)==record['bytes'] and hashlib.sha256(data).hexdigest()==record['sha256']
    result={'commit':commit,'branch':'wbw','remote_ref_verified':True,'github_ref_verified':True,
        'github_actual_blob_readbacks':blobs,'server_export':root,'server_has_git':False,
        'server_snapshot_sha256':manifest['snapshot_sha256'],'server_deployment_manifest_sha256':manifest_sha,
        'server_all_source_config_files_match_fixed_commit':True,'server_source_config_file_count':len(manifest['files']),
        'identity_kind':'fixed Git commit plus verified export manifest; not a server Git checkout',
        'formal_runs_started':False,'other_branches_written':False}
    write_json(SUPPORT/'publication.json',result);print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
