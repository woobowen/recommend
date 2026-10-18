#!/usr/bin/env python3
"""Serialize exactly one production ZIP after the scientific/publication gates."""
from pathlib import Path
import datetime as dt
import hashlib
import json
import subprocess

BASE=Path(__file__).resolve().parent
STAGE=BASE/'deliveries/final-evidence-staging-v2'

def main():
    review=json.loads((STAGE/'review/final-internal-review.json').read_text())
    assert review['status']=='SCIENTIFIC_INTERNAL_REVIEW_PASS' and not review['global_final_task_pass']
    assert not list((BASE/'deliveries').glob('GRID_A2_FD_RECOVERY_FINAL_*.zip'))
    stamp=dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    output=BASE/'deliveries'/('GRID_A2_FD_RECOVERY_FINAL_'+stamp+'.zip')
    tool=BASE/'package_delivery_final.py';assert hashlib.sha256(tool.read_bytes()).hexdigest()=='2fdb67d726f7652f00c3d57bfd73fb3b6f255f0e0bd666e86c2540f591eda53c'
    argv=['python3',str(tool),'build',str(STAGE),str(output)]
    request=dict(created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),argv=argv,cwd=str(BASE),
                 tool_sha256=hashlib.sha256(tool.read_bytes()).hexdigest(),staging=str(STAGE),output=str(output),
                 scientific_review_status=review['status'],global_PASS_not_yet_claimed=True,
                 production_ZIP_count_before=0,server_or_research_mutations=0)
    for path in [BASE/'deliveries/package-build-request.json',STAGE/'execution/package-generation-request.json']:
        assert not path.exists();path.write_text(json.dumps(request,indent=2)+'\n')
    source=STAGE/'tools/final-operations/build_final_package.py';assert not source.exists();source.write_bytes(Path(__file__).read_bytes())
    log=BASE/'deliveries/package-build.log';assert not log.exists()
    with log.open('wb') as stream:
        completed=subprocess.run(argv,cwd=BASE,stdout=stream,stderr=subprocess.STDOUT)
    assert completed.returncode==0, str(log)
    verification=json.loads(output.with_suffix('.zip.verification.json').read_text())
    assert verification['status']=='PACKAGE_INTEGRITY_PASS' and verification['scientific_review_status']=='SCIENTIFIC_INTERNAL_REVIEW_PASS'
    print(json.dumps(dict(status=verification['status'],ZIP=str(output),bytes=verification['bytes'],
                         members=verification['members'],sha256=verification['sha256'],
                         scientific_review_status=verification['scientific_review_status'],
                         actual_nonauthor_ZIP_review_still_required=True),indent=2))

if __name__=='__main__':main()
