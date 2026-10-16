"""Exercise the independent paired-review program on manual synthetic cases."""
import ast
import datetime
import hashlib
import importlib.util
import json
from pathlib import Path

BASE = Path(__file__).parent
program = BASE/'review_actual_paired_analysis.py'
spec = importlib.util.spec_from_file_location('nonauthor_paired_review_under_test', program)
review = importlib.util.module_from_spec(spec); spec.loader.exec_module(review)
source = (BASE/'review_analysis_adaptations.py').read_text().split('\nprograms =')[0]
source += '''
    plan_path=tmp/'synthetic-paired-plan.json'
    plan_path.write_text(json.dumps({'routes':[plan]}))
    report_path=tmp/'synthetic-paired-report.json'
    report_path.write_text(json.dumps({'status':'DESCRIPTIVE_PAIRED_ANALYSIS_COMPLETE',
        'new_model_GPU_or_evaluator_calls':0,'original_predictions_or_accepted_reviews_modified':False,
        'inputs':[identity(plan_path)],'program':identity(AUTHORED/'paired_shuffle_analysis.py'),
        'comparisons':comparisons}))
    review.audit(type('Arguments',(),{'plan':plan_path,'report':report_path,'output':tmp/'synthetic-review.json'})())
    synthetic_review=json.loads((tmp/'synthetic-review.json').read_bytes())
    assert synthetic_review['status']=='INDEPENDENT_COMPLETE_PAIRED_ANALYSIS_PASS'
    assert len(synthetic_review['comparisons'])==2
'''
namespace = {'__file__':str(BASE/'review_analysis_adaptations.py'), 'review':review}
exec(compile(ast.parse(source),str(BASE/'review_analysis_adaptations.py'),'exec'),namespace)


def identity(path):
    data=path.read_bytes()
    return {'path':str(path),'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}


out=BASE/'paired-review-program-analytical-test.json'
result={'status':'INDEPENDENT_PAIRED_REVIEW_PROGRAM_ANALYTICAL_CPU_PASS',
    'reviewer_task':'/root/source_audit','tested_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'program':identity(program),'test_program':identity(Path(__file__)),
    'scope':'Actual independent complete paired-review entrypoint executes on three-user manual synthetic original predictions and mechanically generated paired report; every per-user row, summary and declared case checked for both Shuffle conditions.',
    'comparisons':2,'users_per_comparison':3,'new_GPU_model_or_evaluator_calls':0,
    'synthetic_not_formal_result':True}
assert not out.exists();out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(identity(out)|{'status':result['status']}))
