"""A fresh Qwen -> RQ-KMeans/RQ-VAE -> WMH TIGER experiment; no historical SIDs."""
import argparse
from contextlib import ExitStack, contextmanager
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


@contextmanager
def reserve_stage_gpus(gpu_ids, waiting):
    """Reserve only this stage's idle physical GPUs; release them at stage end."""
    lock_dir = ROOT/'stage_gpu_locks'
    lock_dir.mkdir(exist_ok=True)
    while True:
        stack = ExitStack()
        reason = None
        try:
            for gpu in sorted(gpu_ids, key=int):
                lock = stack.enter_context((lock_dir/f'gpu_{gpu}.lock').open('a'))
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    reason = f'GPU {gpu} is reserved by another stage'
                    break
            if reason is None:
                gpu_rows = subprocess.check_output(['nvidia-smi',
                    '--query-gpu=index,uuid,memory.used', '--format=csv,noheader,nounits'], text=True)
                process_rows = subprocess.check_output(['nvidia-smi',
                    '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader,nounits'], text=True)
                active = {row.split(',')[0].strip() for row in process_rows.splitlines() if ',' in row}
                available = {}
                for row in gpu_rows.splitlines():
                    index, uuid, used = (value.strip() for value in row.split(','))
                    available[index] = int(used) < 512 and uuid not in active
                missing = [gpu for gpu in gpu_ids if gpu not in available]
                if missing:
                    raise ValueError(f'Physical GPU indices do not exist: {missing}')
                busy = [gpu for gpu in gpu_ids if not available[gpu]]
                if busy:
                    reason = f'Waiting for idle physical GPUs: {busy}'
            if reason is None:
                break
        except BaseException:
            stack.close()
            raise
        stack.close()
        waiting(reason)
        time.sleep(15)
    try:
        yield
    finally:
        stack.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['beauty', 'sports', 'toys'], required=True)
    parser.add_argument('--method', choices=['rkmeans', 'rqvae'], default='rkmeans')
    parser.add_argument('--stage', choices=['all', 'embeddings', 'quantize', 'train'], default='all')
    parser.add_argument('--gpus', default='0,1', help='Free GPUs for Qwen; first GPU for quantization/TIGER')
    parser.add_argument('--embedding-batch-size', type=int, default=8)
    parser.add_argument('--run-dir', type=Path)
    args = parser.parse_args()
    gpu_ids = args.gpus.split(',')
    if any(not i.isdigit() for i in gpu_ids) or len(set(gpu_ids)) != len(gpu_ids):
        parser.error('--gpus requires distinct numeric GPU indices')
    run = (args.run_dir or ROOT/'experiments'/f'{args.dataset}_{args.method}_{datetime.now():%Y%m%d_%H%M%S}').resolve()
    if not run.is_relative_to(ROOT/'experiments'):
        parser.error('--run-dir must be inside the new project experiments directory')
    if args.stage in ('all', 'embeddings') and run.exists():
        parser.error('Choose a new --run-dir; old results will not be reused')
    if args.stage in ('quantize', 'train') and not run.exists():
        parser.error('Pass the --run-dir created by the preceding stage')
    run.mkdir(parents=True, exist_ok=True)
    data = ROOT/'original_assets/data/amazon_data'/args.dataset
    embedding = run/'item_embeddings.pt'
    sid = run/'semantic_ids.pt'
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpus, OMP_NUM_THREADS='4',
        PYTHONUNBUFFERED='1', TF_CPP_MIN_LOG_LEVEL='3', HYDRA_FULL_ERROR='1',
        TF_NUM_INTRAOP_THREADS='2', TF_NUM_INTEROP_THREADS='2')
    for key in ('PROJECT_ROOT', 'METADATA_DIR', 'LOCAL_RANK', 'RANK', 'WORLD_SIZE', 'MASTER_ADDR', 'MASTER_PORT'):
        env.pop(key, None)

    def execute(name, cmd, gpu):
        output = run/name
        output.mkdir(exist_ok=False)
        if name != 'text_vectors':
            cmd += [f'hydra.run.dir={output}', '++trainer.enable_progress_bar=false']
        (output/'command.json').write_text(json.dumps({'argv': cmd, 'gpus': gpu}, indent=2))
        print(f'{name}: {output}', flush=True)
        status = {'stage': name, 'state': 'waiting_for_gpu', 'output': str(output), 'gpus': gpu}
        last_wait_message = 0
        def waiting(reason):
            nonlocal last_wait_message
            status.update(state='waiting_for_gpu', reason=reason)
            (run/'pipeline_status.json').write_text(json.dumps(status, indent=2))
            if time.monotonic() - last_wait_message >= 60:
                print(f'{name}: {reason}', flush=True)
                last_wait_message = time.monotonic()
        (run/'pipeline_status.json').write_text(json.dumps(status, indent=2))
        with reserve_stage_gpus(gpu.split(','), waiting):
            status.update(state='running')
            status.pop('reason', None)
            (run/'pipeline_status.json').write_text(json.dumps(status, indent=2))
            with (output/'console.log').open('w') as log:
                result = subprocess.run(cmd, cwd=ROOT, env=dict(env, CUDA_VISIBLE_DEVICES=gpu),
                                        stdout=log, stderr=subprocess.STDOUT)
        status.update(state='completed' if result.returncode==0 else 'failed', exit_code=result.returncode)
        (run/'pipeline_status.json').write_text(json.dumps(status, indent=2))
        if result.returncode:
            raise RuntimeError(f'{name} failed ({result.returncode}); see {output}/console.log')
        return output

    def workers(prefix):
        return [f'++{prefix}num_workers=0', f'++{prefix}timeout=0', f'++{prefix}persistent_workers=false']

    if args.stage in ('all', 'embeddings'):
        execute('text_vectors', [sys.executable, str(ROOT/'_运行/generate_qwen_embeddings.py'),
            '--dataset', args.dataset, '--output', str(embedding), '--gpus', args.gpus,
            '--batch-size', str(args.embedding_batch_size)], args.gpus)
    import torch
    if args.stage in ('all', 'quantize'):
        info = json.loads(embedding.with_suffix('.json').read_text())
        if info['architecture'] != 'qwen2' or info['dataset'] != args.dataset:
            raise ValueError('Expected vectors generated by this Qwen experiment')
        dim = info['shape'][1]
        common = [f'data_dir={data}', f'paths.root_dir={ROOT}', 'trainer.devices=1',
            '++trainer.strategy=auto', 'trainer.precision=32-true', 'extras.print_config=false',
            f'embedding_path={embedding}', f'embedding_dim={dim}',
            'num_hierarchies=3', 'codebook_width=256']
        out = execute('quantizer_train', [sys.executable, str(ROOT/'_运行/grid_stage.py'),
            f'experiment={args.method}_train_flat', 'train=true', 'test=false', 'ckpt_path=null',
            '++callbacks.restart_job=null']+common+workers('data_loading.datamodule.train_dataloader_config.'), gpu_ids[0])
        checkpoint = json.loads((out/'stage_result.json').read_text())['checkpoint']
        if not Path(checkpoint).resolve().is_relative_to(run):
            raise ValueError('Quantizer checkpoint must belong to this run')
        out = execute('sid_generation', [sys.executable, str(ROOT/'_运行/grid_stage.py'),
            f'experiment={args.method}_inference_flat', f'ckpt_path={checkpoint}',
            'callbacks.bq_writer=null', '++callbacks.restart_job=null']+common+
            workers('data_loading.datamodule.predict_dataloader_config.'), gpu_ids[0])
        tensor = torch.load(out/'pickle/merged_predictions_tensor.pt', map_location='cpu', weights_only=False).long()
        n = info['shape'][0]
        if list(tensor.shape) != [4, n] or tensor.min()<0 or tensor.max()>=256:
            raise ValueError('Expected 3 quantized SID digits + 1 dedup digit in [0,255]')
        if len(torch.unique(tensor.T, dim=0)) != n:
            raise ValueError('Generated SIDs must distinguish every item')
        torch.save(tensor, sid)
    if args.stage in ('all', 'train'):
        if not sid.is_file() or not embedding.with_suffix('.json').is_file():
            raise FileNotFoundError('Generate this run Qwen vectors and SIDs first')
        cmd = [sys.executable, str(ROOT/'_运行/grid_stage.py'), 'experiment=tiger_train_flat',
            f'data_dir={data}', f'paths.root_dir={ROOT}', f'semantic_id_path={sid}',
            'num_hierarchies=4', 'trainer.devices=1', '++trainer.strategy=auto',
            'trainer.precision=32-true', 'train=true', 'test=true', 'ckpt_path=null',
            '++callbacks.restart_job=null', 'extras.print_config=false', 'trainer.log_every_n_steps=10']
        for split in ('train', 'val', 'test'):
            cmd += workers(f'data_loading.{split}_dataloader_config.dataloader.')
        execute('tiger_train_test', cmd, gpu_ids[0])
    print(f'Completed requested stage. Run folder: {run}', flush=True)


if __name__ == '__main__':
    main()
