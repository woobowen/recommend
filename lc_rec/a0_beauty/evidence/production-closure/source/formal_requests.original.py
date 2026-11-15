"""Prepare fixed Beauty GPU-stage requests; does not authorize or submit jobs.

This module is an incomplete production integration: callers must verify the
training source chain, deploy a fixed export, implement per-stage validators,
and enforce the separately granted formal authorization before any submission.
No caller-supplied executable or free-form command is accepted.
"""
from pathlib import Path
import re


GPU_STAGES = ('train', 'normal', 's3', 's4')


def _absolute(value, key):
    if not isinstance(value, str) or not Path(value).is_absolute() or '..' in Path(value).parts:
        raise ValueError('absolute path required: ' + key)
    return value


def build_gpu_request(plan, stage):
    """Build, but never submit, a request for a named Beauty GPU stage.

    Inputs are explicit, previously identified files; their actual hashes are
    checked by the existing Runner at submission.  The prepare operation is
    useful only after external identity and per-stage validation are wired.
    """
    if stage not in GPU_STAGES:
        raise ValueError('unsupported GPU stage')
    if plan.get('scope') != 'FORMAL_PREPARE':
        raise PermissionError('request construction is preparation only')
    commit = plan.get('run_commit', '')
    config_sha = plan.get('config_sha256', '')
    if not re.fullmatch(r'[0-9a-f]{40}', commit) or not re.fullmatch(r'[0-9a-f]{64}', config_sha):
        raise ValueError('full code and configuration identities required')
    paths = plan['paths']
    required = ('release', 'python', 'source_root', 'data_root', 'model', 'output_root',
                'train_config', 'precision_approval', 'prelaunch_review', 'cpu_audit')
    for key in required:
        _absolute(paths[key], key)
    root = Path(paths['release']) / 'lc_rec' / 'a0_beauty'
    out = Path(paths['output_root'])
    train = out / 'training'
    adapter = train / 'best_adapter'
    command = [paths['python'], '-m', 'torch.distributed.run', '--standalone',
               '--nnodes=1', '--nproc_per_node=2']
    common = ['--source-root', paths['source_root'], '--data-root', paths['data_root'],
              '--model', paths['model']]
    if stage == 'train':
        command += [str(root / 'formal_train.py'), *common, '--output', str(train),
                    '--config', paths['train_config'], '--run-commit', commit,
                    '--precision-approval', paths['precision_approval'],
                    '--prelaunch-review', paths['prelaunch_review']]
    else:
        if stage == 'normal':
            command += [str(root / 'formal_evaluate.py')]
        else:
            command += [str(root / 'shuffle_entry.py'), '--model-key', 'a0',
                        '--shuffle-n', '3' if stage == 's3' else '4',
                        '--cpu-audit', paths['cpu_audit']]
        command += [*common, '--adapter', str(adapter), '--output', str(out / stage),
                    '--run-commit', commit]
    stage_config = plan['stages'][stage]
    job_id = stage_config['job_id']
    if not re.fullmatch(r'letter-a0-beauty-formal-[A-Za-z0-9_.-]{1,65}', job_id):
        raise ValueError('invalid Beauty formal job ID')
    timeout = stage_config['timeout_seconds']
    if type(timeout) is not int or not 0 < timeout <= 345600:
        raise ValueError('invalid job timeout')
    if not isinstance(plan.get('input_identity'), dict) or not plan['input_identity']:
        raise ValueError('data and model identities required')
    inputs = stage_config['inputs']
    if not isinstance(inputs, list) or not inputs:
        raise ValueError('fixed input manifest required')
    for entry in inputs:
        _absolute(entry['path'], 'input')
        if not re.fullmatch(r'[0-9a-f]{64}', entry.get('sha256', '')):
            raise ValueError('input SHA required')
    by_path = {entry['path']: entry['sha256'] for entry in inputs}
    if by_path.get(paths['train_config']) != config_sha:
        raise ValueError('request must bind the exact configuration file')
    runner = plan['runner_namespace']
    environment = dict(plan['environment'])
    devices = environment.get('CUDA_VISIBLE_DEVICES', '').split(',')
    if len(devices) != 2 or len(set(devices)) != 2 or any(not x.isdecimal() for x in devices):
        raise ValueError('formal two-rank GPU mapping must be supplied explicitly')
    if 'GRID_JOB_ID' in environment or 'GRID_JOB_DIR' in environment:
        raise ValueError('runner identity markers are runner-owned')
    return {'schema': 1, 'host': runner['host'], 'user': runner['user'], 'uid': runner['uid'],
            'namespace': runner['namespace'], 'deployment_sha256': runner['deployment_sha256'],
            'capture_version': commit, 'cwd': paths['release'], 'argv': command,
            'env': environment, 'timeout_seconds': timeout, 'inputs': inputs, 'job_id': job_id,
            'beauty_stage': stage, 'beauty_config_sha256': config_sha,
            'beauty_input_identity': plan['input_identity']}
