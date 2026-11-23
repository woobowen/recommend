"""Prepare Qwen/RQ-KMeans/SIDs, then read-only downstream checks. Never fit TIGER."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
RUN = ROOT/'experiments/yelp_rkmeans_qwen_aligned_v1'
DATA = Path('/home/jcy/data/yelp_aligned_v1')


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as s:
        for b in iter(lambda:s.read(8*1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def write(p, value):
    p = Path(p)
    temp = p.with_suffix('.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
    temp.replace(p)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--worker', action='store_true')
    args=ap.parse_args()
    if not args.worker:
        assert not RUN.exists(), 'Fresh output required'
        RUN.mkdir(parents=True)
        with (RUN/'supervisor.log').open('w') as log:
            proc=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'--worker'],
                cwd=ROOT, stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                start_new_session=True,close_fds=True)
        write(RUN/'launch.json',dict(pid=proc.pid,detached=True,formal_training_started=False,
             argv=[sys.executable,str(Path(__file__).resolve()),'--worker']))
        print(json.dumps(dict(pid=proc.pid,run=str(RUN),formal_training_started=False)))
        return
    original_paths=[ROOT/'_运行/generate_qwen_embeddings.py',ROOT/'_运行/run_pipeline.py',
                    ROOT/'codex/plan_v1_1/run_task.py',ROOT/'codex/plan_v1_1/protocol.py',
                    ROOT/'codex/plan_v1_1/attention.py',ROOT/'codex/sid_order_launch/sid_order_shuffle_model.py',
                    ROOT/'codex/sid_order_launch/sid_order_mlp_model.py']
    originals={str(p):sha(p) for p in original_paths}
    write(RUN/'original_code_hashes.json',originals)
    spec=importlib.util.spec_from_file_location('original_pipeline',ROOT/'_运行/run_pipeline.py')
    original=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(original)
    env=dict(os.environ,OMP_NUM_THREADS='4',PYTHONUNBUFFERED='1',TF_CPP_MIN_LOG_LEVEL='3',
        HYDRA_FULL_ERROR='1',TF_NUM_INTRAOP_THREADS='2',TF_NUM_INTEROP_THREADS='2')
    for key in ('PROJECT_ROOT','METADATA_DIR','LOCAL_RANK','RANK','WORLD_SIZE','MASTER_ADDR','MASTER_PORT'):
        env.pop(key,None)
    status=dict(pid=os.getpid(),session_id=os.getsid(0),formal_training_started=False)
    def state(stage, value, **kwargs):
        status.update(stage=stage,state=value,updated_utc=datetime.now(timezone.utc).isoformat(),**kwargs)
        write(RUN/'pipeline_status.json',status)
    def execute(name,cmd,gpus,hydra=True):
        out=RUN/name
        out.mkdir()
        if hydra:
            cmd += [f'hydra.run.dir={out}','++trainer.enable_progress_bar=false']
        write(out/'command.json',dict(argv=cmd,gpus=gpus))
        state(name,'waiting_for_gpu')
        def waiting(reason):
            state(name,'waiting_for_gpu',reason=reason)
        with original.reserve_stage_gpus(gpus.split(','),waiting):
            state(name,'running')
            with (out/'console.log').open('w') as log:
                result=subprocess.run(cmd,cwd=ROOT,env=dict(env,CUDA_VISIBLE_DEVICES=gpus),
                    stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
        assert result.returncode==0, f'{name} failed; see {out}/console.log'
        state(name,'completed')
        return out
    def workers(prefix):
        return [f'++{prefix}num_workers=0',f'++{prefix}timeout=0',f'++{prefix}persistent_workers=false']
    try:
        vectors=RUN/'item_embeddings.pt'
        execute('text_vectors',[sys.executable,str(HERE/'generate_qwen_embeddings.py'),
            '--dataset','yelp_aligned_v1','--data-dir',str(DATA),'--output',str(vectors),
            '--gpus','0,1','--batch-size','8'],'0,1',hydra=False)
        import torch
        vm=json.loads(vectors.with_suffix('.json').read_text())
        assert vm['architecture']=='qwen2' and vm['shape']==[18775,3584] and sha(vectors)==vm['sha256']
        common=[f'data_dir={DATA}',f'paths.root_dir={ROOT}','trainer.devices=1','++trainer.strategy=auto',
                'trainer.precision=32-true','extras.print_config=false',f'embedding_path={vectors}',
                'embedding_dim=3584','num_hierarchies=3','codebook_width=256']
        qout=execute('quantizer_train',[sys.executable,str(ROOT/'_运行/grid_stage.py'),
            'experiment=rkmeans_train_flat','train=true','test=false','ckpt_path=null',
            '++callbacks.restart_job=null']+common+workers('data_loading.datamodule.train_dataloader_config.'),'0')
        checkpoint=Path(json.loads((qout/'stage_result.json').read_text())['checkpoint']).resolve()
        assert checkpoint.is_relative_to(RUN) and checkpoint.is_file()
        sout=execute('sid_generation',[sys.executable,str(ROOT/'_运行/grid_stage.py'),
            'experiment=rkmeans_inference_flat',f'ckpt_path={checkpoint}','callbacks.bq_writer=null',
            '++callbacks.restart_job=null']+common+workers('data_loading.datamodule.predict_dataloader_config.'),'0')
        tensor=torch.load(sout/'pickle/merged_predictions_tensor.pt',map_location='cpu',weights_only=False).long()
        assert list(tensor.shape)==[4,18775] and tensor.min()>=0 and tensor.max()<256
        assert len(torch.unique(tensor.T,dim=0))==18775
        torch.save(tensor,RUN/'semantic_ids.pt')
        write(RUN/'upstream_audit.json',dict(vector_shape=vm['shape'],vector_sha256=sha(vectors),
            quantizer_checkpoint=str(checkpoint),quantizer_sha256=sha(checkpoint),sid_sha256=sha(RUN/'semantic_ids.pt'),
            sid_shape=list(tensor.shape),unique_items=18775,unique_semantic_prefixes=len(torch.unique(tensor[:3].T,dim=0)),
            sid_digit_ranges=[dict(min=int(x.min()),max=int(x.max()),used=len(x.unique())) for x in tensor],
            dataset_manifest_sha256=sha(DATA/'file_manifest.json'),author_sid_used=False,
            shared_across_methods_and_seeds=True,quantizer_seed=42))
        for variant in ('shuffle','mlp'):
            execute(f'prepare_{variant}',[sys.executable,str(HERE/'run_task.py'),'--dataset','yelp_aligned_v1',
                '--variant',variant,'--gpu','0','--seed','42','--output',str(RUN/f'readonly_{variant}'),
                '--sample-dir',str(ROOT/'experiments/sid_order_yelp_5seed_v1/fixed_test_users'),
                '--prepare-only'],'0',hydra=False)
        assert all(sha(p)==h for p,h in originals.items()), 'Original code changed during preparation'
        state('all_preparation','completed',original_code_unchanged=True)
    except BaseException as exc:
        state(status.get('stage','initializing'),'failed',error=str(exc))
        traceback.print_exc()
        raise


if __name__=='__main__':
    main()
