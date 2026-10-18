"""Append the reviewed delivery semantics and review receipt; no git refs change."""
from pathlib import Path
import datetime as dt
import hashlib
import json

BASE=Path(__file__).resolve().parent
STAGE=BASE/'deliveries/final-evidence-staging-v2'
PUB=Path('/home/addaswsw/recommend-worktrees/sid-extensions-p0/manifests/grid-a2-fd-recovery-20261018/final-results')

def main():
    inputs=[
        (BASE/'package_delivery_final.py','tools/final-operations/package_delivery_final.py','local-execution-tools/package_delivery_final.py'),
        (BASE/'verify_publication_readback.py','tools/final-operations/verify_publication_readback.py','local-execution-tools/verify_publication_readback.py'),
        (BASE/'package-status-adaptation.json','execution/package-status-adaptation.json','local-execution-tools/package-status-adaptation.json'),
    ]
    for name in ['prepublication-final-staging-review.json','final-package-status-tool-review.json']:
        inputs.append((BASE/'reviews/delivery_tools'/name,'review/final/delivery_tools/'+name,'non-author/delivery_tools/'+name))
    inputs.append((Path(__file__),'tools/final-operations/'+Path(__file__).name,'local-execution-tools/'+Path(__file__).name))
    records=[]
    for source,stage_relative,pub_relative in inputs:
        data=source.read_bytes()
        for root,relative in [(STAGE,stage_relative),(PUB,pub_relative)]:
            target=root/relative;target.parent.mkdir(parents=True,exist_ok=True)
            assert not target.exists();target.write_bytes(data)
            records.append(dict(root=str(root),path=relative,source=str(source),bytes=len(data),sha256=hashlib.sha256(data).hexdigest()))
    report=dict(status='REVIEWED_PREPUBLICATION_ADDENDUM_COPIED',
                copied_utc=dt.datetime.now(dt.timezone.utc).isoformat(),files=records,
                source_execution_modified=False,Git_refs_or_index_modified=False,
                production_ZIP_not_built=True)
    for target in [STAGE/'assembly/prepublication-addendum.json',PUB/'copy-manifest-prepublication-addendum.json']:
        assert not target.exists();target.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(status=report['status'],copied_members=len(records),actual_remote_publication_not_yet_claimed=True)))

if __name__=='__main__':main()
