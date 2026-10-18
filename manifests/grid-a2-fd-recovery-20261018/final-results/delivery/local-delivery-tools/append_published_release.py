"""Append already published immutable bytes and a scoped scientific review."""
from pathlib import Path
import datetime as dt
import hashlib
import json
import subprocess

BASE=Path(__file__).resolve().parent
REPO=Path('/home/addaswsw/recommend-worktrees/sid-extensions-p0')
STAGE=BASE/'deliveries/final-evidence-staging-v2'
COMMIT='42ac9acf23ae0bdc65ed6505089e1b88101755cd'
PREFIX='manifests/grid-a2-fd-recovery-20261018/final-results/'

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path):return json.loads(path.read_text())

def main():
    assert not (STAGE/'assembly/publication-addendum.json').exists()
    public=read(BASE/'publication/scientific/anonymous-readback.json')
    assert public['commit']==COMMIT and len(public['files'])==11 and all(r['Git_blob_bytes_exact'] for r in public['files'])
    connector=read(BASE/'publication/scientific/github-connector-verified.json');assert connector['commit']==COMMIT and len(connector['files'])==4
    source=read(BASE/'publication/scientific/source-map-at-release.json');assert len(source['source_files'])==262
    records=[]
    def copy(path,relative,data=None,provenance=None):
        data=path.read_bytes() if data is None else data
        target=STAGE/relative;target.parent.mkdir(parents=True,exist_ok=True)
        assert not target.exists();target.write_bytes(data)
        records.append(dict(path=relative,source=provenance or str(path),bytes=len(data),sha256=hashlib.sha256(data).hexdigest()))
    for path in sorted((BASE/'publication').rglob('*')):
        if path.is_file():copy(path,'publication/readback/'+path.relative_to(BASE/'publication').as_posix())
    names=subprocess.run(['git','ls-tree','-r','--name-only',COMMIT,'--',PREFIX],cwd=REPO,check=True,stdout=subprocess.PIPE,text=True).stdout.splitlines()
    assert len(names)==222
    for name in names:
        data=subprocess.run(['git','show',COMMIT+':'+name],cwd=REPO,check=True,stdout=subprocess.PIPE).stdout
        copy(REPO/name,'publication/scientific-release-tree/'+name[len(PREFIX):],data=data,provenance='git:'+COMMIT+':'+name)
    for name in ['verify_source_at_release.py','append_published_release.py']:
        copy(BASE/name,'tools/final-operations/'+name)
    scene=read(BASE/'final-scene/scene-20261018T043428336611Z.json')
    mapping=dict(scientific_release_commit=COMMIT,scientific_execution_commit=scene['source_commit'],
                 snapshot_root=scene['source_snapshot'],snapshot_sha256=scene['source_snapshot_sha256'],
                 prior_delivery_commit='9ab3d0ea5d32db00e278f74c599a1943885453db',
                 actual_GitHub_ref_and_key_bytes_read=True,anonymous_readback_files=11,connector_readback_files=4,
                 source_members_exact_to_execution=262,
                 post_ZIP_delivery_receipt_commit=None,
                 late_delivery_receipt_note='After this one ZIP is serialized and independently checked, wbw receives only its hashes/review receipts. That later commit cannot be prewritten here or treated as execution source.')
    (STAGE/'publication/version-map.json').write_text(json.dumps(mapping,indent=2)+'\n')
    expected_reviews={
      'review/final/fd/quantization-result-review.json':'QUANTIZATION_AND_SID_INDEPENDENT_REVIEW_PASS',
      'review/final/a2_training/final_training_review.json':'A2_TRAINING_NONAUTHOR_REVIEW_PASS',
      'review/final/a2_outputs/A2_OUTPUTS_NONAUTHOR_REVIEW.json':'A2_EVALUATION_ATTENTION_STATISTICS_NONAUTHOR_REVIEW_PASS',
      'review/final/delivery_tools/scientific-results-cross-review.json':'ALL_AUTHORIZED_SCIENTIFIC_RESULTS_NONAUTHOR_CROSS_REVIEW_PASS',
    }
    proofs=[]
    for name,status in expected_reviews.items():
        path=STAGE/name;assert read(path)['status']==status
        proofs.append(dict(path=name,status=status,bytes=path.stat().st_size,sha256=sha(path)))
    for name in ['review/final/delivery_tools/prepublication-final-staging-review.json',
                 'review/final/delivery_tools/final-package-status-tool-review.json']:
        path=STAGE/name;proofs.append(dict(path=name,status=read(path)['status'],bytes=path.stat().st_size,sha256=sha(path)))
    review=dict(status='SCIENTIFIC_INTERNAL_REVIEW_PASS',reviewed_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        author='/root',actual_non_author_reviewers=['/root/fd_repair_reviewer','/root/sports_a1_result_audit','/root/toys_result_audit'],
        scope='All authorized scientific execution/results, real scoped/cross reviews, immutable source/data identities, terminal cleanup/budget, published wbw scientific report and reviewed evidence staging.',
        global_final_task_pass=False,serialized_ZIP_integrity_not_yet_approved=True,
        mandatory_remaining_delivery_gates=['Actual one final serialized ZIP: all member SHA/CRC/size/path/secret checks',
                                         'Actual nonauthor final package and version/publication cross review',
                                         'Post-ZIP fixed wbw delivery-receipt publication and real readback'],
        expected_calculations=dict(new_A2_stages_succeeded=11,old34_history_unchanged=True,original_failed_quantizer_preserved=True,
                                   new_formal_quantizer_attempts=1,original_three_correction_rounds_not_reset=True),
        verified_outputs=dict(original_A_route_prediction_rows=330060,A_route_conditions=12,full_Sports_targets_per_condition=35598,
                              full_Toys_targets_per_condition=19412,attention_NPZ=48,attention_PNG=384,
                              attention_query_pair_rows=4608,cross_domain_routes=8,cross_domain_conditions=24),
        resource_totals=scene['resources'],activity=scene['activity'],version=mapping,non_author_evidence=proofs,
        limits=['Toys producer is not uniquely established','A2 is method reproduction with a new scratch-trained generator',
                'Single seed and fixed32 evaluation cohort support descriptive claims only',
                'Whole environment tree size/member/hash not captured; byte-identical environment restore not assured',
                'FD168 is a saved sampled maximum, not a continuous kernel-level bound'],
        webpage_GPT_second_review_not_yet_performed=True,GPT_UPLOAD_REQUIRED='YES')
    (STAGE/'review/final-internal-review.json').write_text(json.dumps(review,ensure_ascii=False,indent=2)+'\n')
    audit=dict(status='PUBLISHED_SCIENTIFIC_RELEASE_AND_SCOPED_REVIEW_APPENDED',files=records,
               generated_members=['publication/version-map.json','review/final-internal-review.json'],
               actual_source_COMMIT=scene['source_commit'],scientific_publication_COMMIT=COMMIT,
               prepackage_status=review['status'],global_final_PASS_not_claimed=True)
    (STAGE/'assembly/publication-addendum.json').write_text(json.dumps(audit,indent=2)+'\n')
    files=[p for p in STAGE.rglob('*') if p.is_file()]
    print(json.dumps(dict(status=audit['status'],published_files=len(names),total_payload_members=len(files),
                         total_payload_bytes=sum(p.stat().st_size for p in files),prepackage_status=review['status']),indent=2))

if __name__=='__main__':main()
