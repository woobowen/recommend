"""Rebind the unchanged source/plan audit to the separately preserved v2 export."""
import hashlib
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent
BASE = OUT.parents[1]
prior = OUT / 'inspect_deployment_and_plan.py'
code = prior.read_text()
for before, after in [
    ('compiled-c31d2d3eaaafbd18', 'compiled-22286966cd544773'),
    ('/compiled/c31d2d3eaaafbd18', '/compiled/22286966cd544773'),
    ('0d18820025e7a28675ed7112be0b1b4ff0110025', '85b77f5e2a5b497fdc1f620649ba488fc783a245'),
    ('source-github-readback.json', 'source-github-readback-v2.json'),
    ('deployment-plan-independent-inspection.json', 'deployment-plan-independent-inspection-v2.json'),
]:
    assert before in code
    code = code.replace(before, after)
namespace = {'__file__': str(__file__), '__name__': 'independent_deployment_plan_v2'}
exec(compile(code, str(prior) + ':v2_identity_binding', 'exec'), namespace)
namespace['main']()
current = json.loads((BASE / 'source-deployment.json').read_text())
previous = json.loads((BASE / 'source-deployment-c31d2d3eaaafbd18.json').read_text())
a = {r['path']: r for r in previous['manifest']['files']}
b = {r['path']: r for r in current['manifest']['files']}
changes = [{'path': name, 'old': a.get(name), 'new': b.get(name)} for name in a.keys() | b.keys() if a.get(name) != b.get(name)]
assert len(changes) == 1 and changes[0]['path'] == 'experiments/launch_20261017/queue_runner.py'
runner_review = json.loads((OUT / 'runner-adaptation-review.json').read_text())
assert runner_review['status'] == 'OUTER_RUNNER_ENGINEERING_ADAPTATION_PASS'
assert changes[0]['new']['sha256'] == runner_review['source']['sha256']
path = OUT / 'deployment-plan-independent-inspection-v2.json'
result = json.loads(path.read_text())
result.update(changes_from_preserved_c31_source_manifest=changes,
              all_other_261_files_identical=True,
              new_outer_runner_component_review={'path': str(OUT / 'runner-adaptation-review.json'), 'sha256': hashlib.sha256((OUT / 'runner-adaptation-review.json').read_bytes()).hexdigest()},
              previous_c31_candidate_preserved_unsubmitted_not_admitted=True)
path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({'path': str(path), 'final_sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'changed_manifest_files': len(changes), 'status': result['status']}))
