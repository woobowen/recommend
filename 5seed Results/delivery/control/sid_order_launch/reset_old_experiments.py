"""Remove only this chat's obsolete variant outputs, preserving baseline inputs."""
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import time

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = Path('/data/jcyin/sid_order_reproduction')
OLD_GROUP = OUTPUT_ROOT / 'qwen_rkmeans_shared_sid_20261014_2255'


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def matching_processes():
    found = {}
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():
            continue
        try:
            args = proc.joinpath('cmdline').read_bytes().decode().split('\0')
            if '--output' not in args:
                continue
            output = Path(args[args.index('--output') + 1])
            if output != OLD_GROUP and output.parent != OLD_GROUP:
                continue
            if not any(arg in [str(ROOT / 'codex/sid_order_launch/launch_jobs.py'),
                               str(ROOT / 'codex/sid_order_launch/run_variant.py')] for arg in args):
                continue
            if proc.stat().st_uid != os.getuid():
                raise RuntimeError('Unexpected process owner')
            found[int(proc.name)] = args
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    return found


def main():
    assert OUTPUT_ROOT.resolve() == OUTPUT_ROOT and OUTPUT_ROOT.is_dir()
    files = []
    for baseline in (ROOT / 'experiments').iterdir():
        if baseline.name in ['beauty_rkmeans_qwen_20261010_012253',
                             'sports_rkmeans_qwen_20261010_093035',
                             'toys_rkmeans_qwen_20261010_093036']:
            quantizer = Path(json.loads((baseline / 'quantizer_train/stage_result.json').read_text())['checkpoint'])
            files += [baseline / 'item_embeddings.pt', baseline / 'semantic_ids.pt', quantizer]
    assert len(files) == 9
    before = {str(p): digest(p) for p in files}
    targets = []
    for path in OUTPUT_ROOT.iterdir():
        if path == OLD_GROUP or re.fullmatch(
            r'(?:smoke_(?:mlp|shuffle)_20261014_2253|attention_switch_check_20261015_0038(?:_v[2-5])?)(?:\.log)?',
            path.name,
        ):
            assert not path.is_symlink()
            assert path.resolve().parent == OUTPUT_ROOT
            targets.append(path)
    processes = matching_processes()
    # Freeze dispatchers before stopping children so queued old work cannot start.
    for pid, args in processes.items():
        if str(ROOT / 'codex/sid_order_launch/launch_jobs.py') in args:
            os.kill(pid, signal.SIGSTOP)
    for pid in processes:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    time.sleep(3)
    for pid in matching_processes():
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    time.sleep(1)
    assert not matching_processes(), 'Old jobs still alive; refusing to delete outputs'
    for path in targets:
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
    after = {str(p): digest(p) for p in files}
    assert before == after
    report = {'stopped_processes': processes, 'deleted_outputs': [str(p) for p in targets],
              'baseline_input_hashes': after, 'baseline_inputs_unchanged': True,
              'old_group_removed': not OLD_GROUP.exists(), 'old_processes_remaining': []}
    destination = ROOT / 'codex/sid_order_launch/旧实验清理记录.json'
    destination.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
