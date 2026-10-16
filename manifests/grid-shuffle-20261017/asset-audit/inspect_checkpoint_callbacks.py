"""Read CPU checkpoint selection metadata without constructing any model."""
import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import sys


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', required=True, type=Path)
    args = parser.parse_args()
    spec = json.loads(args.spec.read_text())
    assert os.environ.get('CUDA_VISIBLE_DEVICES', '') == ''
    assert sha(__file__) == spec['script']['sha256']
    snapshot = Path(spec['snapshot']['path'])
    assert sha(snapshot / 'DEPLOYMENT.json') == spec['snapshot']['deployment_sha256']
    source_before = {}
    for key in ['checkpoint', 'config', 'sid']:
        p = Path(spec[key]['path'])
        assert p.is_file() and not p.is_symlink()
        assert p.stat().st_size == spec[key]['bytes']
        source_before[key] = p.stat()
    sys.path.insert(0, str(snapshot))
    from experiments.beauty_extensions.common import bind_source, load_checkpoint
    bind_source('B')
    import torch
    from omegaconf import OmegaConf

    def convert(value):
        if isinstance(value, torch.Tensor):
            assert value.numel() == 1
            return value.item()
        if OmegaConf.is_config(value):
            return convert(OmegaConf.to_container(value, resolve=False))
        if isinstance(value, dict):
            return {str(key): convert(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [convert(item) for item in value]
        assert value is None or isinstance(value, (str, bool, int, float)), type(value)
        return value

    checkpoint = load_checkpoint(spec['checkpoint']['path'])
    assert not torch.cuda.is_initialized()
    sid = torch.load(spec['sid']['path'], map_location='cpu', weights_only=True)
    hyperparams = checkpoint['hyper_parameters']
    assert torch.equal(hyperparams['codebooks'], sid)
    callbacks = convert(checkpoint.get('callbacks', {}))
    selected = {key: val for key, val in callbacks.items() if 'ModelCheckpoint' in key}
    assert len(selected) == 1
    callback_name, selection = next(iter(selected.items()))
    assert selection['monitor'] == 'val/recall@5'
    assert Path(selection['best_model_path']).name == Path(spec['checkpoint']['path']).name
    assert int(checkpoint['global_step']) == spec['expected_selected_step']
    assert float(selection['best_model_score']) == spec['expected_best_validation_recall5']
    assert selection['best_model_path'] in selection['best_k_models']
    assert len(selection['best_k_models']) == 1
    assert all(record.device.type == 'cpu' for record in checkpoint['state_dict'].values())
    assert not torch.cuda.is_initialized()
    for key, before in source_before.items():
        after = Path(spec[key]['path']).stat()
        assert (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    output = Path(os.environ['GRID_JOB_DIR']) / 'output'
    output.mkdir(parents=True, exist_ok=True)
    write(output / 'invocation.json', {
        'argv': sys.argv, 'interpreter': sys.executable, 'cwd_after_source_bind': os.getcwd(),
        'started_at_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
        'job_id': os.environ['GRID_JOB_ID'], 'CUDA_VISIBLE_DEVICES': os.environ.get('CUDA_VISIBLE_DEVICES'),
        'spec': spec, 'spec_sha256': sha(args.spec), 'script_sha256': sha(__file__),
    })
    result = {
        'status': 'PASS', 'domain': spec['domain'], 'route': 'B',
        'checkpoint': spec['checkpoint'], 'sid': spec['sid'], 'config': spec['config'],
        'global_step': checkpoint['global_step'], 'epoch': checkpoint['epoch'],
        'callbacks': callbacks, 'selection_callback_key': callback_name,
        'csv_selection_crosscheck': {'expected_best_validation_recall5': spec['expected_best_validation_recall5'],
                                    'expected_selected_step': spec['expected_selected_step'], 'exact_match': True},
        'hyperparameters_available_names': sorted(str(key) for key in hyperparams),
        'scalar_hyperparameters': {key: convert(hyperparams[key]) for key in [
            'num_hierarchies', 'num_embeddings_per_hierarchy', 'embedding_dim',
            'top_k_for_generation', 'should_check_prefix', 'should_add_sep_token',
            'weight_tying'] if key in hyperparams},
        'embedded_sid_exact': True, 'sid_shape': list(sid.shape), 'sid_dtype': str(sid.dtype),
        'state_dict_tensor_count': len(checkpoint['state_dict']),
        'sid_embedding_state_shapes': {key: list(val.shape) for key, val in checkpoint['state_dict'].items() if 'item_sid_embedding_table' in key},
        'all_state_tensors_CPU': True, 'cuda_initialized': False,
        'model_constructions': 0, 'GPU_model_calls': 0,
        'source_file_stat_unchanged': True, 'snapshot': spec['snapshot'],
        'scope': 'CPU metadata and exact embedded SID check; no original historical Python producer claim',
    }
    write(output / 'checkpoint_callback_metadata.json', result)
    write(output / 'RESULT.json', {'status': 'PASS', 'domain': spec['domain'],
                                 'metadata_sha256': sha(output / 'checkpoint_callback_metadata.json'),
                                 'model_constructions': 0, 'GPU_model_calls': 0})
    print(json.dumps({'status': 'PASS', 'domain': spec['domain'], 'best_score': selection['best_model_score'],
                      'selected_step': checkpoint['global_step'], 'model_calls': 0, 'GPU_calls': 0}))


if __name__ == '__main__':
    main()
