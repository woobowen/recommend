"""Apply nonauthor-reported namespace/historical-field clarification only."""
import json
import hashlib
from pathlib import Path
R=Path('/home/addaswsw/recommend-worktrees/sid-extensions-p0')
S=Path('/home/addaswsw/recommend-support/grid-formal-20261015')
F=R/'manifests/grid-formal-20261015'
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
summary=json.loads((F/'EXECUTION_SUMMARY.json').read_text())
summary['historical_mlp_admission_snapshot']=summary.pop('mlp_admission')
summary['historical_independent_review_scope']=summary.pop('independent_review_scope')
summary['mlp_admission']={'GPU_training_started':True,'formal_training_and_complete_test_accepted':True,'conditions_and_budgets_unchanged':True,'original_admission_values_preserved_in':'historical_mlp_admission_snapshot','current_accepted_review':'mlp-results-review/mlp-formal-combined-review.json'}
summary['independent_review_scope']={'runtime_formal_PASS_claimed':True,'formal_Beauty_results':'results-review/beauty-six-formal-results-independent-summary.json','formal_domain_MLP':'mlp-results-review/mlp-formal-combined-review.json','inherited_Beauty_MLP':'../beauty-extensions/final-review.json and b-attention-review.json#/formal_mlp_attention32_review','eight_attention_technical_reviews':'attention-results-review/gf-*-attention-v1-review.json','actual_statistics':'attention-results-review/attention-statistics-runtime-review.json','statistics_source_provenance':'attention-results-review/attention-statistics-source-provenance-review-20261017.json','final_integration':'pending actual final-integration-review-20261017.json; not presigned','final_package':'pending actual independent delivery review; not presigned','historical_scope_preserved_in':'historical_independent_review_scope'}
save(F/'EXECUTION_SUMMARY.json',summary)
mapping=json.loads((F/'FINAL_VERSION_MAP.json').read_text())
mapping['job_map_reference_namespace']='stage_snapshot_zip_and_final_zip_original-evidence member paths; repository paths are separately specified below'
def repo_review(path):
    for a,b in [('reviews/formal-results/','manifests/grid-formal-20261015/results-review/'),('reviews/mlp-results/','manifests/grid-formal-20261015/mlp-results-review/'),('reviews/attention/','manifests/grid-formal-20261015/attention-results-review/'),('reviews/inherited-beauty/','manifests/beauty-extensions/')]:
        if path.startswith(a):return b+path[len(a):]
    return None
for j in mapping['job_map']:
    old=j['inherited'];base=('manifests/beauty-extensions/runs/' if old else 'manifests/grid-formal-20261015/runs/')+j['job_id']
    j['reference_namespace']='stage_snapshot_zip; final ZIP original-evidence/ prefix'
    j['repository_request']=base+'/request.json'
    j['repository_result']=base+('/RESULT.json' if old else '/output/RESULT.json')
    j['repository_review_references']=[x for x in (repo_review(p) for p in j['review_references']) if x and (R/x).is_file()]
    j['permanent_remote_job_directory']='/home/bwwu/grid-audits/sid-extensions-p0/jobs/'+j['job_id']
    assert (R/j['repository_request']).is_file() and (R/j['repository_result']).is_file()
save(F/'FINAL_VERSION_MAP.json',mapping)
source=Path(__file__);dest=F/source.name
assert not dest.exists();dest.write_bytes(source.read_bytes())
print(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [F/'EXECUTION_SUMMARY.json',F/'FINAL_VERSION_MAP.json']}))
