"""Bind a later report release to unchanged actual execution source bytes."""
from pathlib import Path
import hashlib
import json
import subprocess

BASE=Path(__file__).resolve().parent
REPO=Path('/home/addaswsw/recommend-worktrees/sid-extensions-p0')
COMMIT='42ac9acf23ae0bdc65ed6505089e1b88101755cd'

def main():
    manifest=json.loads((BASE/'source-deployment.json').read_text())['manifest']
    records=[]
    for entry in manifest['files']:
        data=subprocess.run(['git','show',COMMIT+':'+entry['path']],cwd=REPO,check=True,stdout=subprocess.PIPE).stdout
        assert len(data)==entry['bytes'] and hashlib.sha256(data).hexdigest()==entry['sha256']
        records.append(entry)
    report=dict(status='FINAL_SCIENTIFIC_RELEASE_262_EXECUTION_SOURCE_BYTES_EXACT',publication_commit=COMMIT,
                execution_commit=manifest['source_commit'],source_files=records,
                scientific_source_modified_by_result_publication=False)
    path=BASE/'publication/scientific/source-map-at-release.json'
    path.parent.mkdir(parents=True,exist_ok=True)
    assert not path.exists();path.write_text(json.dumps(report,indent=2)+'\n')
    fetched=json.loads((BASE/'publication/scientific/github-connector-readback.json').read_text())
    assert fetched['commit']==COMMIT
    checked=[]
    for item in fetched['files']:
        assert item['status']=='fulfilled' and not item['value']['isError']
        actual=item['value']['structuredContent']['content'].encode()
        expected=subprocess.run(['git','show',COMMIT+':'+item['path']],cwd=REPO,check=True,stdout=subprocess.PIPE).stdout
        assert actual==expected
        checked.append(dict(path=item['path'],bytes=len(actual),sha256=hashlib.sha256(actual).hexdigest(),GitHub_connector_bytes_exact=True))
    path=BASE/'publication/scientific/github-connector-verified.json';assert not path.exists()
    path.write_text(json.dumps(dict(commit=COMMIT,files=checked),indent=2)+'\n')
    print(json.dumps(dict(status=report['status'],execution_source_files=len(records),actual_GitHub_connector_files_verified=len(checked))))

if __name__=='__main__':main()
