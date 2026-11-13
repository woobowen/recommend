#!/usr/bin/env python3
"""Reviewer-owned synthetic CPU tests of actual AST-extracted handoff branches.

Never imports transport, Torch or the production modules. All fabricated state
is temporary and explicitly synthetic; no formal model/result is loaded.
"""
import ast
import datetime as dt
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace


def run_node(node, namespace):
    exec(compile(ast.Module(body=[node], type_ignores=[]), '<actual-pipeline-synthetic-fixture>', 'exec'), namespace)


def main():
    scripts = Path(__file__).resolve().parents[1] / 'scripts'
    raw = (scripts / 'pipeline.py').read_bytes()
    tree = ast.parse(raw)
    main_node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    source_identity = {}
    for name in ('control.py', 'pipeline.py', 'offline_pipeline.py'):
        data = raw if name == 'pipeline.py' else (scripts / name).read_bytes()
        ast.parse(data)
        source_identity[name] = {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
    checks = []
    nested_run = next(n for n in ast.walk(main_node) if isinstance(n, ast.FunctionDef) and n.name == 'run')
    calls = []
    rc = [0]

    class Child:
        def __init__(self, command, **kwargs):
            calls.append({'command': command, **kwargs})
            self.returncode = rc[0]

        def poll(self):
            return self.returncode

    status = {'stages': []}
    namespace = {'resource': lambda: None, 'sys': sys, 'os': __import__('os'),
                 'subprocess': SimpleNamespace(Popen=Child, DEVNULL=subprocess.DEVNULL,
                                               TimeoutExpired=subprocess.TimeoutExpired),
                 'status': status, 'utc': lambda: 'SYNTHETIC', 'write': lambda *x: None,
                 'status_path': None, 'children': [], 'time': time, 'limit': 1000,
                 'started': time.monotonic()}
    run_node(nested_run, namespace)
    namespace['run']('offline_evaluation', ['/fixed-E/offline_pipeline.py', '--evaluation-commit', 'e' * 40], cpu=True)
    assert calls[-1]['command'][1] == '/fixed-E/offline_pipeline.py'
    assert calls[-1]['env']['CUDA_VISIBLE_DEVICES'] == '' and '-m' not in calls[-1]['command']
    assert 'start_new_session' not in calls[-1]
    checks.append('CPU command hides CUDA and retains accepted-runner session')
    namespace['run']('test', ['/fixed-R/evaluate.py'])
    assert calls[-1]['command'][1:3] == ['-m', 'torch.distributed.run'] and calls[-1]['env'] is None
    checks.append('GPU command path remains distinct; no process actually launched')
    rc[0] = 7
    try:
        namespace['run']('offline_evaluation', ['/fixed-E/offline_pipeline.py'], cpu=True)
    except RuntimeError:
        checks.append('CPU nonzero child exit propagates failure')
    else:
        raise AssertionError('CPU failure swallowed')
    rc[0] = 0
    before = len(calls)

    def exhausted():
        raise AssertionError('CUMULATIVE_BUDGET_EXHAUSTED')

    namespace['resource'] = exhausted
    try:
        namespace['run']('offline_evaluation', ['/fixed-E/offline_pipeline.py'], cpu=True)
    except AssertionError:
        assert len(calls) == before
        checks.append('Shared resource/budget gate runs before CPU spawn')
    else:
        raise AssertionError('CPU bypassed resource gate')

    R, E = '1' * 40, '2' * 40
    gates = [n for n in ast.walk(main_node) if isinstance(n, ast.If)
             and isinstance(n.test, ast.Compare) and ast.unparse(n.test) == "a.kind == 'formal'"]
    gate = next(n for n in gates if any(isinstance(x, ast.Name) and x.id == 'evaluation_version' for x in ast.walk(n)))
    formal = next(n for n in gates if any(isinstance(x, ast.Name) and x.id == 'prior_tests' for x in ast.walk(n)))
    resume = next(n for n in ast.walk(main_node) if isinstance(n, ast.If)
                  and ast.unparse(n.test) == 'a.resume_evaluation'
                  and any(isinstance(x, ast.Name) and x.id == 'proof' for x in ast.walk(n)))
    digest = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    with tempfile.TemporaryDirectory(prefix='a1-review-cpu-handoff-') as temporary:
        root = Path(temporary)
        source = root / 'R' / 'scripts'; source.mkdir(parents=True)
        (source / 'fixture.py').write_text('# synthetic\n')
        config = source.parent / 'config.json'; config.write_text('{"synthetic_only":true}\n')
        evaluation = root / 'releases' / E; evaluation.mkdir(parents=True)
        (evaluation / 'RUN_VERSION.json').write_text(json.dumps({'RUN_COMMIT': E}))
        gate_path = root / 'prelaunch.json'
        gate_value = {'status': 'PASS', 'RUN_COMMIT': R, 'evaluation_commit': E,
                      'config_sha256': digest(config),
                      'source_sha256': {'scripts/fixture.py': digest(source / 'fixture.py')}}
        gate_path.write_text(json.dumps(gate_value))
        a = SimpleNamespace(kind='formal', prelaunch_review=str(gate_path), evaluation_commit=E,
                            evaluation_release=str(evaluation), run_commit=R,
                            resume_evaluation=True, resume=None, job_id='synthetic-new')
        ns = {'a': a, 'Path': Path, 'json': json, 'sha': digest, 'scripts': source, 'config': config}
        run_node(gate, ns)
        checks.append('Formal gate binds distinct R and E plus config/script bytes')
        for changed in ('evaluation_commit', 'config_sha256', 'source_sha256'):
            bad = dict(gate_value, **{changed: 'wrong'})
            gate_path.write_text(json.dumps(bad))
            try:
                run_node(gate, dict(ns))
            except AssertionError:
                checks.append('Reject wrong prelaunch gate ' + changed)
            else:
                raise AssertionError('Wrong formal gate accepted')
        gate_path.write_text(json.dumps(gate_value))
        (evaluation / 'RUN_VERSION.json').write_text(json.dumps({'RUN_COMMIT': R}))
        try:
            run_node(gate, dict(ns))
        except AssertionError:
            checks.append('Reject evaluation export naming/version disagreement')
        else:
            raise AssertionError('Wrong E export accepted')
        (evaluation / 'RUN_VERSION.json').write_text(json.dumps({'RUN_COMMIT': E}))

        training = root / 'formal_seed42' / 'training'; training.mkdir(parents=True)
        adapter = training / 'best_adapter'; adapter.mkdir(); (adapter / 'adapter_model.bin').write_bytes(b'SYNTHETIC_ADAPTER_NO_MODEL')
        selected = {'run_commit': R, 'adapter_sha256': digest(adapter / 'adapter_model.bin')}
        (training / 'best_selection.json').write_text(json.dumps(selected))
        proof = {'status': 'TRAINING_COMPLETE', 'global_step': 8240, 'epoch': 4.0,
                 'run_commit': R, 'config_sha256': digest(config)}
        for rank in (0, 1):
            (training / f'rank{rank}.json').write_text(json.dumps(proof))
        old_test = root / 'synthetic-old' / 'test'; old_test.mkdir(parents=True)
        metric = {'RUN_COMMIT': R, 'adapter_model_sha256': selected['adapter_sha256'],
                  'actual_test_users': 24772, '_synthetic_only': True}
        (old_test / 'metrics.json').write_text(json.dumps(metric))
        for rank in (0, 1):
            (old_test / f'rank{rank}.json').write_text(json.dumps({'status': 'PASS', '_synthetic_only': True}))
        output = root / a.job_id; output.mkdir()
        ledger = {'attempts': [{'job_id': 'synthetic-old'}, {'job_id': a.job_id}]}
        runs = []

        def fake_run(stage, arguments, cpu=False):
            runs.append((stage, list(map(str, arguments)), cpu))
            if stage == 'offline_evaluation':
                target = Path(arguments[arguments.index('--output') + 1]); target.mkdir()
                (target / 'pipeline.json').write_text(json.dumps({'status': 'PASS', 'EVALUATION_RUN_COMMIT': E, '_synthetic_only': True}))

        state = {'stages': []}
        ns = {'a': a, 'Path': Path, 'json': json, 'sha': digest, 'scripts': source, 'config': config,
              'training': training, 'ledger': ledger, 'status': state, 'root': root,
              'output': output, 'run': fake_run, 'source': source, 'data': root / 'data', 'model': 'unused'}
        run_node(resume, ns)
        assert state['stages'][-1]['status'] == 'REUSED_VERIFIED_COMPLETE_TRAINING' and not runs
        checks.append('Evaluation-only recovery reuses verified training without train call')
        run_node(formal, ns)
        assert [r[0] for r in runs] == ['offline_evaluation']
        command = runs[-1][1]
        assert runs[-1][2] is True
        assert command[0] == str(evaluation / 'lc_rec/a1_qwen3_06b/scripts/offline_pipeline.py')
        assert command[command.index('--gpu-run-commit') + 1] == R
        assert command[command.index('--evaluation-commit') + 1] == E
        assert command[command.index('--test-dir') + 1] == str(old_test)
        assert command[command.index('--training') + 1] == str(training)
        checks.append('Completed full test reused; CPU receives fixed R/E and original test/training')
        assert state['evaluation_run_commit'] == E and state['extended_evaluation']['status'] == 'PASS'
        checks.append('Outer formal status consumes actual offline PASS identity')
        bad = dict(proof, global_step=1)
        (training / 'rank1.json').write_text(json.dumps(bad))
        try:
            run_node(resume, dict(ns))
        except AssertionError:
            checks.append('Incomplete training rejected for evaluation-only recovery')
        else:
            raise AssertionError('Incomplete recovery training accepted')
    print(json.dumps({'status': 'PASS_SYNTHETIC_CPU_HANDOFF_ONLY', 'checks': checks,
                      'directed_checks': len(checks), 'source_identity': source_identity,
                      'scope': 'Actual AST branches with mock subprocesses and temporary synthetic metadata; no SSH/GPU/formal evaluation.'}, indent=2))


if __name__ == '__main__':
    main()
