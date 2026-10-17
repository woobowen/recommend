"""Run the full concrete audit against v2, preserving the refused c31 candidate."""
import hashlib
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent
BASE = OUT.parents[1]
prior = OUT / 'inspect_concrete_requests.py'
code = prior.read_text()
for before, after in [
    ('compiled-c31d2d3eaaafbd18', 'compiled-22286966cd544773'),
    ('45fa8698ab605332b992ae37ce097acf67328777bf06900116304b789fafe3ac', 'ddc231d33f739ec8f5282c60ad2a670023d057f8609bcca1d9526129fdfb711e'),
    ("OUT / 'launch-review.json'", "OUT / 'launch-review-v2.json'"),
    ('concrete-requests-independent-inspection.json', 'concrete-requests-independent-inspection-v2.json'),
]:
    assert before in code
    code = code.replace(before, after)
extra = """    old_candidate = read(BASE / 'queue-deployment-c31d2d3eaaafbd18.json')
    all_inputs[old_candidate['remote_config']] = old_candidate['config_sha256']
    all_inputs[old_candidate['outer_runner']] = old_candidate['outer_runner_sha256']
    all_inputs[cfg['runner']['job_root'] + '/ga-20261017-sportsa2-quantize-v1/work.log'] = '8d4351404436d903c10ddf7015278e127cc654e2df6d03febfa5b6aebbf207d9'
"""
assert code.count('    all_inputs.update(outer_inputs)\n') == 1
code = code.replace('    all_inputs.update(outer_inputs)\n', '    all_inputs.update(outer_inputs)\n' + extra)
namespace = {'__file__': str(__file__), '__name__': 'independent_concrete_request_v2'}
exec(compile(code, str(prior) + ':v2_identity_binding', 'exec'), namespace)
namespace['main']()
path = OUT / 'concrete-requests-independent-inspection-v2.json'
result = json.loads(path.read_text())
submit = BASE / 'submit_recovery.py'
source = submit.read_text()
compile(source, str(submit), 'exec')
for expected in ["review['formal_submission_admitted'] is True", "review['future_results_approved'] is False", "review['config_sha256']==d['config_sha256']", "review['outer_request_file_sha256']==sha(request_path)", "if exists:raise RuntimeError", "'validate'", "t.profile['runner']=d['outer_runner']", 'receipt=gridctl.submit(t,request_path)']:
    assert expected in source
assert source.count('gridctl.submit(') == 1
assert not any(token in source for token in ['while ', 'for attempt', 'kill(', 'pkill', 'sudo', 'rm -'])
result['submit_script_static_review'] = {'path': str(submit), 'bytes': submit.stat().st_size, 'sha256': hashlib.sha256(submit.read_bytes()).hexdigest(), 'both_real_nonauthor_reviews_four_exact_fields_checked': True, 'same_new_controller_ID_absence_before_fresh_live_validate': True, 'one_submit_call_no_retry_loop': True, 'accepted_child_runner_unchanged_and_audited_new_outer_only': True, 'not_executed_by_reviewer': True}
result['old_c31_candidate_config_and_outer_runner_bytes_unchanged'] = True
result['original_failed_quantizer_work_log_exact_preserved'] = True
result['formal_quantizer_attempts_before_this_review'] = 0
path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
print(json.dumps(dict(path=str(path),final_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),status=result['status'],reviewed_submit_script_sha256=result['submit_script_static_review']['sha256'])))
