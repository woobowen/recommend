#!/usr/bin/env python3
"""Exercise the web second-review gate using an isolated internal-review fixture.

The fixture is neither a real web review nor a user launch authorization.
"""
import argparse
import copy
import datetime as dt
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys
from production_control import approval_documents
from formal_train import verify_web_review
from g3_identity import source_identity, need, sha


def check_gate(fixture, release, run_commit, config_sha256):
    fixture, release = Path(fixture).resolve(), Path(release).resolve()
    need('SYNTHETIC_CPU_INTEGRATION_ONLY' in str(fixture), 'isolated synthetic review fixture required')
    need(not fixture.exists(), 'preserve previous synthetic review fixture')
    fixture.mkdir(parents=True)
    review_path = fixture / 'internal-review-negative-fixture.json'
    evidence_path = fixture / 'synthetic-evidence.json'
    evidence_path.write_text(json.dumps({'scope':'SYNTHETIC_CPU_INTEGRATION_ONLY',
        'synthetic_fixture':True,'not_a_real_web_review':True,'not_a_user_launch_authorization':True})+'\n')
    precision = release / 'lc_rec/letter/evidence/formal_20261111/precision_approval.json'
    training = release / 'lc_rec/a0_beauty/formal_train.py'
    review = {'scope':'SYNTHETIC_CPU_INTEGRATION_ONLY','synthetic_fixture':True,
        'not_a_real_web_review':True,'not_a_user_launch_authorization':True,
        'status':'PASS','RUN_COMMIT':run_commit,'config_sha256':config_sha256,
        'formal_train_sha256':sha(training),'review_kind':'INDEPENDENT_NONAUTHOR',
        'reviewer':'SYNTHETIC_INTERNAL_REVIEW_GUARD_FIXTURE',
        'evidence_files':source_identity([evidence_path])}
    review_path.write_text(json.dumps(review,indent=2)+'\n')
    plan = {'release':str(release),'run_commit':run_commit,'config_sha256':config_sha256,
        'prelaunch_review':str(review_path),'precision_approval':str(precision),
        'approval_files':{'prelaunch_review':source_identity([review_path])[0],
                          'precision_approval':source_identity([precision])[0]}}
    # These preliminary facts make the expected failure attributable to review kind.
    need(json.loads(precision.read_text()).get('formal_a_only_bf16_authorized') is True,
         'actual precision approval fixture prerequisite')
    need(review['status']=='PASS' and review['RUN_COMMIT']==plan['run_commit'] and
         review['config_sha256']==plan['config_sha256'] and review['formal_train_sha256']==sha(training),
         'synthetic internal review identity prerequisites')
    try:
        approval_documents(plan, require_pass=True)
    except ValueError as error:
        need(str(error)=='G3 web second review evidence required', 'wrong web-review guard reached: '+str(error))
        result={'status':'PASS','case':'internal_non_author_review_cannot_replace_web_second_review',
            'guard':'ValueError','message':str(error),'synthetic_fixture':True}
    else:
        raise AssertionError('internal non-author review improperly satisfied the web second-review gate')
    checks=[{key:result[key] for key in ('case','guard','message','synthetic_fixture')}]
    def reject_direct(case, path, expected):
        try:
            verify_web_review(str(path),sha(path),run_commit,config_sha256,sha(training))
        except ValueError as error:
            need(str(error)==expected,'wrong direct formal review guard reached: '+str(error))
            checks.append({'case':case,'guard':'ValueError','message':str(error),'synthetic_fixture':True})
        else:
            raise AssertionError('direct formal training review guard accepted '+case)
    reject_direct('direct_training_internal_review_is_not_web_second_review',review_path,
                  'external webpage second review required before formal training')
    invalid_evidence=copy.deepcopy(review)
    invalid_evidence['review_kind']='EXTERNAL_GPT_REVIEW'
    invalid_evidence['reviewer']='SYNTHETIC_CORRUPT_EVIDENCE_NEGATIVE_FIXTURE_NOT_A_WEB_REVIEW'
    invalid_evidence['evidence_files'][0]['sha256']='0'*64
    invalid_path=fixture/'synthetic-web-kind-corrupt-evidence.json'
    invalid_path.write_text(json.dumps(invalid_evidence,indent=2)+'\n')
    reject_direct('direct_training_web_kind_wrong_evidence_hash',invalid_path,
                  'web review evidence member identity mismatch: '+str(evidence_path))
    result.update(negative_checks=checks,
        direct_training_guard_source=source_identity([inspect.getsourcefile(verify_web_review)])[0],
        direct_training_guard_method_sha256=hashlib.sha256(inspect.getsource(verify_web_review).encode()).hexdigest(),
        scope='SYNTHETIC_CPU_INTEGRATION_ONLY',run_commit=run_commit,
        config_sha256=config_sha256,formal_train_sha256=sha(training),
        required_review_kind='EXTERNAL_GPT_REVIEW',supplied_review_kind='INDEPENDENT_NONAUTHOR',
        actual_guard_source=source_identity([inspect.getsourcefile(approval_documents)])[0],
        guard_method_sha256=hashlib.sha256(inspect.getsource(approval_documents).encode()).hexdigest(),
        fixture_files=source_identity([review_path,evidence_path,invalid_path]),
        actual_precision_approval=source_identity([precision])[0],
        formal_experiment_started=False,formal_launch_authorized=False,
        real_web_review_created=False,user_launch_authorization_created=False)
    return result


def main():
    parser=argparse.ArgumentParser(allow_abbrev=False,description=__doc__)
    parser.add_argument('--release',type=Path,required=True)
    parser.add_argument('--run-commit',required=True)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    need(not args.output.exists(),'preserve existing web-review guard result')
    args.output.mkdir(parents=True)
    result=check_gate(args.output/'SYNTHETIC_CPU_INTEGRATION_ONLY',args.release,args.run_commit,sha(args.config))
    result.update(argv=sys.argv,cwd=os.getcwd(),python=sys.executable,
                  created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),script=source_identity([__file__])[0],exit_code=0)
    (args.output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'status':'PASS','case':result['case'],'message':result['message'],'negative_checks':len(result['negative_checks']),'result':str(args.output/'result.json')}))


if __name__=='__main__':
    main()
