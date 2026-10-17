"""Independently validate raw CPU receipts; never execute research computation."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path
import struct

ROOT = Path('/home/addaswsw/recommend-worktrees/sid-extensions-p0')
SUPPORT = Path('/home/addaswsw/recommend-support/grid-a2-fd-recovery-20261018')
RAW = Path('/home/addaswsw/recommend-support/sid-extensions-p0/jobs/new-server/ga-20261018-a2-fd-cpu-equivalence-v1/retrieved')
OUT = Path(__file__).resolve().parent
OLD = Path('/home/addaswsw/recommend-support/grid-a-launch-20261017/deliveries/a2-emfile-capture-20261017T175451Z/package/a2-failure')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def main():
    status = read(RAW / 'status.json')
    cleanup = read(RAW / 'cleanup.json')
    eq = read(RAW / 'output/equivalence.json')
    config = read(RAW / 'output/config-equivalence.json')
    assert status['status'] == 'SUCCEEDED' and status['exit_code'] == 0
    assert status['worker_may_be_live'] is False and status['cleanup']['verified']
    assert cleanup['verified'] and cleanup['last_observation']['verified_empty']
    assert not cleanup['last_observation']['owned_live']
    artifacts = []
    for row in status['artifacts']:
        path = RAW / row['path']
        assert path.stat().st_size == row['bytes'] and sha(path) == row['sha256']
        artifacts.append(row)
    log = (RAW / 'work.log').read_text()
    assert all(token not in log for token in ['Traceback', 'Too many open files', 'OSError', 'RuntimeError'])
    logged = next(json.loads(line) for line in log.splitlines() if line.startswith('{"status"'))
    assert logged == eq
    assert eq['status'] == 'CPU_EQUIVALENCE_AND_BOUNDED_IPC_PASS' and eq['gpu_requested'] == 0
    scene = read(SUPPORT / 'initial-scene.json')
    for key in ['path', 'bytes', 'sha256']:
        assert eq['input'][key] == scene['embedding'][key]
    assert eq['real_item_axis'] == dict(items=18357, unique=18357, all_dense_ids_exact=True, all_embedding_pairs_exact=True)
    assert eq['rng_states_unchanged'] and eq['actual_original_model_step_predict_step_and_writer_equal']
    assert eq['adapted_source_sha256'] == sha(ROOT / 'experiments/a2_fd_recovery_20261018/adapter.py')
    original = read(OLD / 'output/used_config.json')['data_loading']['datamodule']
    assert original == config['original']
    restored = json.loads(json.dumps(config['adapted']))
    diffs = []
    for name in ['train_dataloader_config', 'predict_dataloader_config']:
        before = restored[name]['collate_fn']['_target_']
        assert before == 'experiments.a2_fd_recovery_20261018.adapter.collate_items_contiguous'
        restored[name]['collate_fn']['_target_'] = original[name]['collate_fn']['_target_']
        diffs.append(name + '/collate_fn/_target_')
    assert restored == original
    for name in ['direct', 'dict_and_extra_features']:
        row = eq[name]
        assert row['items'] == row['unique_ids'] == 2048
        assert row['id_shape'] == [2048, 1] and row['id_dtype'] == 'torch.int32'
        assert row['embedding_shape'] == [2048, 512] and row['embedding_dtype'] == 'torch.float32'
        assert row['old_id_storages'] == 2048 and row['new_id_storages'] == 1
        assert row['old_total_storages'] - row['new_total_storages'] == 2047
        assert row['exact_values_dtype_order_all_features_pairing']
    ipc = eq['ipc']
    assert len(ipc['original']) == len(ipc['contiguous']) == 4
    for a, b in zip(ipc['original'], ipc['contiguous']):
        assert a['batch'] == b['batch'] and a['item_ids'] == b['item_ids']
        assert a['embedding_sha256'] == b['embedding_sha256']
        assert len(a['item_ids']) == 512 and a['item_storages'] == 512 and b['item_storages'] == 1
        assert a['process']['soft'] == b['process']['soft'] == 524288
    ids = [n for row in ipc['original'] for n in row['item_ids']]
    assert len(ids) == len(set(ids)) == 2048
    assert sha_bytes(struct.pack('<' + 'i' * len(ids), *ids)) == eq['direct']['ids_sha256']
    fd_files = [read(p) for p in (RAW / 'output/fd').glob('*.json')]
    native_fd = []
    for native in eq['native_loader_cpu_simulation']:
        rank = native['simulated_rank']
        assert rank in (0, 1) and native['raw_batch'] == 2048 and native['num_workers'] == 12
        assert native['prefetch_factor'] == 2 and native['persistent_workers'] and native['pin_memory']
        assert len(native['workers']) == 12 and len(native['batches']) == 3
        for batch in native['batches']:
            assert batch['items'] == 2048 and batch['item_storages'] == 1 and batch['exact_pairing']
        for worker in native['workers']:
            assert worker['ppid'] == native['batches'][0]['process']['pid']
            assert worker['soft'] == worker['hard'] == 524288 and worker['fd'] < 8192
            receipt = next(r for r in fd_files if r['process']['pid'] == worker['pid'])
            assert receipt['rank'] == rank and receipt['configured_workers'] == 12
            assert receipt['id_shape'] == [2048, 1] and receipt['id_storages'] == 1
            assert receipt['sharing_strategy'] == 'file_descriptor'
            native_fd.append(worker['fd'])
    assert eq['process_limits']['before'] == [1024, 524288]
    assert eq['process_limits']['after'] == [524288, 524288]
    assert eq['process_limits']['global_configuration_changed'] is False
    result = {
        'reviewer': '/root/fd_repair_reviewer', 'at': dt.datetime.now(dt.timezone.utc).isoformat(),
        'status': 'INDEPENDENT_CPU_RAW_RECEIPTS_PASS', 'scientific_result': 'SCIENTIFIC_RESULT_NOT_YET_AVAILABLE',
        'job': {k: status[k] for k in ['job_id', 'status', 'exit_code', 'started_at', 'ended_at']},
        'raw_artifacts_all_sha_and_bytes_checked': artifacts,
        'work_log_exact_report_and_no_traceback': True,
        'effective_config_exact_original_except': diffs,
        'raw_IPC_ids_match_and_independent_int32_bytes_hash_match': True,
        'direct_item_storages_before_after': [2048, 1],
        'direct_total_storages_before_after': [2049, 2],
        'IPC_original_FD': [r['process']['fd'] for r in ipc['original']],
        'IPC_contiguous_FD': [r['process']['fd'] for r in ipc['contiguous']],
        'native_main_FD': [[b['process']['fd'] for b in r['batches']] for r in eq['native_loader_cpu_simulation']],
        'native_worker_FD_range': [min(native_fd), max(native_fd)],
        'native_24_distinct_workers_actual_PPID_checked': True,
        'limits_before_after': [eq['process_limits']['before'], eq['process_limits']['after']],
        'scope_limit': 'Native workers/ranks are sequential CPU simulations in one main PID, not actual GPU DDP ranks. Real DDP2/rank limits/24-worker receipts and optimizer progress remain formal startup gates.',
        'model_fixture': eq['model_fixture'], 'input': eq['input'],
        'raw_receipts_path': str(RAW), 'no_new_runtime_tests_or_dependencies': True,
    }
    target = OUT / 'cpu-raw-independent-inspection.json'
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'path': str(target), 'sha256': sha(target), 'status': result['status']}))


def sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


if __name__ == '__main__':
    main()
