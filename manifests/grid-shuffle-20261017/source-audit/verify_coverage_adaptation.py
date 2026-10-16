"""Isolated CPU regression of domain coverage validation; no model imports."""
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import time
import types

ROOT = Path('/home/addaswsw/recommend-worktrees/sid-extensions-p0')
OUT = Path(__file__).parent / 'coverage-adaptation-review.json'


def load_function(path, name, namespace):
    tree = ast.parse(path.read_bytes(), filename=str(path))
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name]
    assert len(selected) == 1
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace[name]


def identity(path):
    data = path.read_bytes()
    return {'path': str(path), 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}


common_ns = {'Path': Path, 'hashlib': hashlib}
sha256 = load_function(ROOT/'experiments/beauty_extensions/common.py', 'sha256', common_ns)
verify_asset = load_function(ROOT/'experiments/beauty_extensions/common.py', 'verify_asset', common_ns)
read_ns = {'Path': Path, 'json': json, 'Counter': Counter, 'verify_asset': verify_asset}
read_index = load_function(ROOT/'experiments/beauty_extensions/mlp_formal.py', 'read_index', read_ns)

scenario = {}
trace_saved = {}


def test_lifecycle(cfg, model, dm, output):
    if scenario.get('raise_test'):
        raise RuntimeError('synthetic lifecycle failure')


def recompute(*args, **kwargs):
    return {'coverage_exact': scenario.get('coverage', True), 'samples': scenario['samples']}


for name, values in {
    'experiments.beauty_extensions.cli': {'test_lifecycle': test_lifecycle},
    'experiments.beauty_extensions.results': {'recompute': recompute},
    'experiments.beauty_extensions.mlp_formal': {'read_index': read_index},
}.items():
    module = types.ModuleType(name)
    vars(module).update(values)
    sys.modules[name] = module


def instantiate(spec, output, checkpoint=True):
    n = spec['shuffle_n']
    model = types.SimpleNamespace(encoder_calls=2, encoder_role_counts={'loss': 1, 'generate': 1},
        shuffle_summary={'calls': 2 if n else 0}, shuffle_role_counts={'loss': int(bool(n)), 'generate': int(bool(n))},
        shuffle_events=[{'role': 'loss'}, {'role': 'generate'}] if n else [])
    return 'synthetic_cfg', model, 'synthetic_dm'


ns = {'instantiate': instantiate, 'time': time,
      'write_json': lambda path, value: trace_saved.update({str(path): value})}
evaluate = load_function(ROOT/'experiments/formal_20261015/run.py', 'evaluate', ns)
checks = []
with tempfile.TemporaryDirectory(prefix='shuffle-coverage-audit-') as tmp:
    tmp = Path(tmp)
    key = 'a'*24
    index = tmp/'index.jsonl'
    for domain, size in [('beauty', 22363), ('toys', 19412), ('sports', 35598)]:
        line = json.dumps({'sample_key': key, 'target': [0, 1, 2, 3]})+'\n'
        index.write_text(line*size)
        record = {'path': str(index), 'sha256': sha256(index)}
        counter, verified = read_index(record)
        assert len(counter) == 1 and sum(counter.values()) == size
        for n in (0, 3, 4):
            scenario.clear(); scenario['samples'] = size
            spec = {'route': 'B', 'domain': domain, 'seed': 42, 'shuffle_n': n,
                    'test_index': record, 'checkpoint': {}, 'sid': {}, 'config': {}}
            result = evaluate(spec, None, tmp)
            assert result['status'] == 'COMPLETE' and result['metrics']['samples'] == size
            assert result['coverage_index'] == verified
        checks.append({'domain': domain, 'targets': size, 'conditions': [0, 3, 4],
                       'Counter_multiplicity_not_unique_count': True})
    for description, overrides in [
        ('reject_shortened_test', {'samples': 35597}),
        ('reject_surplus_test', {'samples': 35599}),
        ('reject_nonexact_coverage', {'samples': 35598, 'coverage': False}),
    ]:
        scenario.clear(); scenario.update(overrides)
        try:
            evaluate(spec, None, tmp)
        except AssertionError:
            checks.append({'negative_check': description, 'rejected': True})
        else:
            raise AssertionError(description)
    scenario.clear(); scenario.update(samples=35598, raise_test=True)
    try:
        evaluate(spec, None, tmp)
    except RuntimeError as exc:
        assert str(exc) == 'synthetic lifecycle failure'
        trace = trace_saved[str(tmp/'intervention_trace.json')]
        assert trace['exception'] == "RuntimeError('synthetic lifecycle failure')"
        checks.append({'negative_check': 'test_failure_preserves_trace_and_propagates', 'rejected': True})
    else:
        raise AssertionError('Lifecycle failure was swallowed')
    index.write_text('')
    empty = {'path': str(index), 'sha256': sha256(index)}
    try:
        read_index(empty)
    except ValueError as exc:
        assert str(exc) == 'Empty metadata index'
        checks.append({'negative_check': 'reject_empty_verified_target_index', 'rejected': True})
    else:
        raise AssertionError('Empty index accepted')
    index.write_text(json.dumps({'sample_key': key, 'target': [0, 1, 2]})+'\n')
    bad = {'path': str(index), 'sha256': sha256(index)}
    try:
        read_index(bad)
    except ValueError:
        checks.append({'negative_check': 'reject_non_H4_target', 'rejected': True})
    else:
        raise AssertionError('Invalid target accepted')
    index.write_text(json.dumps({'sample_key': key, 'target': [0, 1, 2, 3]})+'\n')
    try:
        read_index(bad)
    except ValueError as exc:
        assert str(exc).startswith('Asset hash mismatch')
        checks.append({'negative_check': 'reject_changed_index_identity', 'rejected': True})
    else:
        raise AssertionError('Changed source accepted')

files = ['experiments/formal_20261015/run.py', 'experiments/shuffle_20261017/deploy.py',
         'experiments/shuffle_20261017/preflight.py', 'experiments/shuffle_20261017/__init__.py']
for name in files:
    ast.parse((ROOT/name).read_bytes(), filename=name)
assert '22363' not in (ROOT/files[0]).read_text()
report = {
    'status': 'INDEPENDENT_COVERAGE_ADAPTATION_CPU_PASS',
    'reviewer_task': '/root/source_audit',
    'source_files': [identity(ROOT/name) for name in files],
    'checks': checks,
    'scope': 'Execute the actual modified evaluate function AST with fake lifecycle/model/evaluator boundaries, plus actual read_index and verify_asset, to check domain target counts and failure propagation. No GPU/model dependency was imported. This does not claim real model, metric, SID semantic or runner validation.',
    'new_GPU_or_model_calls': 0,
    'program': identity(Path(__file__)),
}
assert not OUT.exists()
OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
print(json.dumps(identity(OUT) | {'status': report['status'], 'checks': len(checks)}))
