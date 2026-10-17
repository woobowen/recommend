"""A2 DataLoader IPC/FD engineering adaptation.

Only the item-ID container crossing DataLoader IPC changes. The preserved
collate builds every feature tensor. No loader/RNG/worker-init hook is replaced.
"""
import json
import os
from pathlib import Path
import resource
import time

import torch

_WORKER_RECORDED = set()


def process_record(pid=None):
    pid = os.getpid() if pid is None else pid
    p = Path('/proc') / str(pid)
    stat = (p / 'stat').read_text().rsplit(')', 1)[1].split()
    limits = next(s for s in (p / 'limits').read_text().splitlines() if s.startswith('Max open files')).split()
    return dict(pid=pid, ppid=int(stat[1]), start_ticks=stat[19],
                fd=len(list((p / 'fd').iterdir())), soft=int(limits[3]), hard=int(limits[4]))


def receipt_dir():
    value = os.environ.get('GRID_A2_FD_OUTPUT')
    return Path(value) if value else None


def save(name, value):
    root = receipt_dir()
    if root is None:
        return
    root.mkdir(parents=True, exist_ok=True)
    # Separate file per writer; avoid RNG-consuming temporary names.
    path = root / name
    tmp = path.with_suffix(path.suffix + '.partial')
    tmp.write_text(json.dumps(value, allow_nan=False, sort_keys=True) + '\n')
    os.replace(tmp, path)


def configure_process(output, soft_target=524288):
    before = resource.getrlimit(resource.RLIMIT_NOFILE)
    target = max(before[0], min(soft_target, before[1]))
    resource.setrlimit(resource.RLIMIT_NOFILE, (target, before[1]))
    os.environ['GRID_A2_FD_OUTPUT'] = str(Path(output) / 'fd')
    value = dict(purpose='A2 DataLoader IPC/FD engineering adaptation',
                 rank=int(os.environ.get('LOCAL_RANK', '0')), before=list(before),
                 after=list(resource.getrlimit(resource.RLIMIT_NOFILE)),
                 global_configuration_changed=False,
                 sharing_strategy=torch.multiprocessing.get_sharing_strategy(),
                 process=process_record(), time=time.time())
    save(f'rank-{value["rank"]}-limits.json', value)
    return value


def collate_items_contiguous(batch, item_id_field, feature_to_input_name):
    from src.data.loading.components.collate_functions import collate_fn_items
    result = collate_fn_items(batch, item_id_field, feature_to_input_name)
    result.item_ids = torch.stack(result.item_ids, dim=0)
    info = torch.utils.data.get_worker_info()
    key = (os.getpid(), int(os.environ.get('LOCAL_RANK', '0')))
    if info is not None and key not in _WORKER_RECORDED:
        _WORKER_RECORDED.add(key)
        save(f'rank-{key[1]}-worker-{info.id}-pid-{key[0]}.json',
             dict(rank=key[1], worker_id=info.id, configured_workers=info.num_workers,
                  process=process_record(), sharing_strategy=torch.multiprocessing.get_sharing_strategy(),
                  batch_items=len(result.item_ids), id_shape=list(result.item_ids.shape),
                  id_dtype=str(result.item_ids.dtype), id_storages=1,
                  worker_init='unchanged Lightning seed_everything(42, workers=True) path', time=time.time()))
    return result


def adapt_config(cfg):
    """Replace only collate targets; original training and inference loaders stay."""
    from omegaconf import OmegaConf
    original = OmegaConf.to_container(cfg.data_loading.datamodule, resolve=True)
    for name in ('train_dataloader_config', 'predict_dataloader_config'):
        node = cfg.data_loading.datamodule[name].collate_fn
        assert node._target_ == 'src.data.loading.components.collate_functions.collate_fn_items'
        node._target_ = 'experiments.a2_fd_recovery_20261018.adapter.collate_items_contiguous'
    revised = OmegaConf.to_container(cfg.data_loading.datamodule, resolve=True)
    restored = json.loads(json.dumps(revised))
    for name in ('train_dataloader_config', 'predict_dataloader_config'):
        restored[name]['collate_fn']['_target_'] = original[name]['collate_fn']['_target_']
    assert restored == original
    return dict(original=original, adapted=revised, only_two_collate_targets_changed=True)


def resource_callback(output, reference):
    """Record real ranks/workers; terminate this rank if scoped FD guard fails."""
    import hashlib
    import threading
    from lightning.pytorch.callbacks import Callback

    class FDGuard(Callback):
        def __init__(self):
            self.stop = threading.Event()
            self.thread = None

        def sample(self):
            root = os.getpid()
            children = {}
            for p in Path('/proc').iterdir():
                if not p.name.isdigit():
                    continue
                try:
                    if p.stat().st_uid == os.getuid():
                        s = (p/'stat').read_text().rsplit(')', 1)[1].split()
                        children[int(p.name)] = int(s[1])
                except (FileNotFoundError, ProcessLookupError, PermissionError):
                    continue
            own = {root}
            while True:
                extra = {p for p, parent in children.items() if parent in own}
                if extra <= own:
                    break
                own |= extra
            records = []
            for pid in sorted(own):
                try:
                    records.append(process_record(pid))
                except (FileNotFoundError, ProcessLookupError, PermissionError):
                    continue
            value = dict(time=time.time(), rank=int(os.environ.get('LOCAL_RANK', '0')),
                         source='actual PID/PPID descendant relation; same UID', processes=records,
                         per_process_fd_ceiling=8192, soft_fraction_ceiling=0.5)
            path = receipt_dir() / f'rank-{value["rank"]}-samples.jsonl'
            with path.open('a') as f:
                f.write(json.dumps(value, allow_nan=False) + '\n')
            danger = [p for p in records if p['fd'] >= min(8192, p['soft'] * 0.5)]
            if danger:
                save(f'rank-{value["rank"]}-FD_GUARD_STOP.json', dict(**value, dangerous= danger))
                os._exit(74)  # This job's independent runner owns precise cleanup.

        def loop(self):
            try:
                while not self.stop.is_set():
                    self.sample()
                    self.stop.wait(5)
            except BaseException as error:
                save(f'rank-{os.environ.get("LOCAL_RANK", "0")}-FD_MONITOR_FAILED.json', dict(error=repr(error)))
                os._exit(75)

        def on_fit_start(self, trainer, module):
            self.thread = threading.Thread(target=self.loop, daemon=True)
            self.thread.start()

        def on_train_batch_start(self, trainer, module, batch, batch_idx):
            value = batch[0]
            ids = value.item_ids.detach().cpu().reshape(-1)
            features = value.transformed_features['input_embedding'].detach().cpu()
            assert ids.dtype in (torch.int32, torch.int64) and len(ids) == features.shape[0]
            assert torch.equal(features, reference[ids])
            result = dict(rank=trainer.global_rank, global_step=trainer.global_step,
                          loader_batch=batch_idx, items=len(ids), item_ids=ids.tolist(),
                          embedding_sha256=hashlib.sha256(features.contiguous().numpy().tobytes()).hexdigest(),
                          exact_id_embedding_pairing=True, time=time.time())
            with (receipt_dir()/f'rank-{trainer.global_rank}-batches.jsonl').open('a') as f:
                f.write(json.dumps(result, allow_nan=False)+'\n')

        def on_predict_end(self, trainer, module):
            self.close()

        def on_exception(self, trainer, module, exception):
            self.close()

        def close(self):
            self.stop.set()
            if self.thread:
                self.thread.join(timeout=6)

    return FDGuard()
