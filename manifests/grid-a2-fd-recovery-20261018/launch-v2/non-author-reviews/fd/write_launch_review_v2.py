"""Write v2 conditional source/plan review while preserving the c31 review."""
import hashlib
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent
prior = OUT / 'write_launch_review.py'
code = prior.read_text()
for before, after in [
    ('compiled-c31d2d3eaaafbd18', 'compiled-22286966cd544773'),
    ('deployment-plan-independent-inspection.json', 'deployment-plan-independent-inspection-v2.json'),
    ('four_GitHub_fixed_commit_raw_file_contents_and_blob_hashes_checked', 'three_v2_GitHub_fixed_commit_raw_file_contents_and_blob_hashes_checked'),
    ("path = OUT / 'launch-review.json'", "path = OUT / 'launch-review-v2.json'"),
]:
    assert before in code
    code = code.replace(before, after)
namespace = {'__file__': str(__file__), '__name__': 'independent_write_launch_review_v2'}
exec(compile(code, str(prior) + ':v2_binding', 'exec'), namespace)
namespace['main']()
path = OUT / 'launch-review-v2.json'
result = json.loads(path.read_text())
review = OUT / 'runner-adaptation-review.json'
result['independent_review']['reference_receipts'].append({'path': str(review), 'bytes': review.stat().st_size, 'sha256': hashlib.sha256(review.read_bytes()).hexdigest()})
result['independent_review']['narrow_outer_runner_16_positive_negative_readonly_cases_pass'] = True
result['source_version_change'] = {'previous_source_commit': '0d18820025e7a28675ed7112be0b1b4ff0110025', 'previous_snapshot': 'c31d2d3eaaafbd18', 'only_changed_manifest_file': 'experiments/launch_20261017/queue_runner.py', 'new_outer_runner_sha256': '6c64871028aec213646083d92b09a5b9b3a4571f73053ecc1bcb7719b1807767', 'all_other_261_manifest_files_exact': True, 'reason': 'Preserve full original absolute supervisor deadline via exact new CPU queue whitelist; fix pre-submission old-runner86400s rejection.', 'previous_c31_45fa_candidate': 'Preserved unsubmitted and NOT_ADMITTED; previous refusal is not a formal quantizer attempt.'}
result['conditions_before_formal_submission'].append('Verify deployed new outer runner6c6487 and unchanged original namespace sidecar; call its actual full read-only verify() on the final supervisor request, not just source simulation.')
path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
print(json.dumps(dict(path=str(path),final_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),bytes=path.stat().st_size,status=result['status'],formal_submission_approved=False)))
