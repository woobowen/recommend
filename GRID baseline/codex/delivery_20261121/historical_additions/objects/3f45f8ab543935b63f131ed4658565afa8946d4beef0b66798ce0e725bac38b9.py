from pathlib import Path
import argparse, datetime, hashlib, json, math, os, subprocess
os.environ.update(CUDA_VISIBLE_DEVICES="", TF_CPP_MIN_LOG_LEVEL="3", OMP_NUM_THREADS="2", TF_NUM_INTRAOP_THREADS="1", TF_NUM_INTEROP_THREADS="1")
import torch
from omegaconf import OmegaConf
ROOT=Path('/home/jcyin/GRID_wmh_Qwen_TIGER_20261009')
import sys
sys.path.insert(0,str(ROOT))
import src.utils
parser=argparse.ArgumentParser()
parser.add_argument('--dataset',choices=['all','sports','toys'],default='all')
selected=parser.parse_args().dataset
report={'checked_at':datetime.datetime.now().astimezone().isoformat(),'datasets':{},'runner_sha256':hashlib.sha256((ROOT/'_运行/run_pipeline.py').read_bytes()).hexdigest()}
processes=[]
for line in subprocess.check_output(['ps','-eo','pid,ppid,args'],text=True).splitlines()[1:]:
    fields=line.split(None,2)
    if len(fields)==3: processes.append((int(fields[0]),int(fields[1]),fields[2]))
for dataset,n,gpu in [('sports',18357,'1'),('toys',11924,'2')]:
    if selected!='all' and selected!=dataset: continue
    launch=json.loads(sorted((ROOT/'launches').glob(dataset+'_rkmeans_qwen_*.json'))[-1].read_text())
    run=Path(launch['run_dir'])
    vectors=torch.load(run/'item_embeddings.pt',map_location='cpu',weights_only=True)
    meta=json.loads((run/'item_embeddings.json').read_text())
    assert list(vectors.shape)==[n,3584] and torch.isfinite(vectors).all()
    assert meta['dataset']==dataset and meta['architecture']=='qwen2'
    assert hashlib.sha256((run/'item_embeddings.pt').read_bytes()).hexdigest()==meta['sha256']
    sid=torch.load(run/'semantic_ids.pt',map_location='cpu',weights_only=True)
    assert list(sid.shape)==[4,n] and int(sid.min())>=0 and int(sid.max())<256
    unique=len(torch.unique(sid.T,dim=0))
    assert unique==n
    qresult=json.loads((run/'quantizer_train/stage_result.json').read_text())
    ckpt_path=Path(qresult['checkpoint'])
    assert ckpt_path.resolve().is_relative_to(run)
    checkpoint=torch.load(ckpt_path,map_location='cpu',weights_only=False)
    initialized=checkpoint['layers_initialized']
    assert len(initialized)==3 and all(initialized)
    assert all(torch.isfinite(t).all() for t in checkpoint['state_dict'].values() if isinstance(t,torch.Tensor) and t.is_floating_point())
    cfg=OmegaConf.load(run/'tiger_train_test/.hydra/config.yaml')
    assert cfg.train is True and cfg.test is True and cfg.ckpt_path is None
    assert cfg.model._target_=='src.models.modules.semantic_id.tiger_generation_model.SemanticIDEncoderDecoder'
    assert Path(cfg.semantic_id_path).resolve()==(run/'semantic_ids.pt').resolve()
    assert str(cfg.data_dir)==str(ROOT/'original_assets/data/amazon_data'/dataset)
    assert cfg.trainer.devices==1 and cfg.trainer.accumulate_grad_batches==16
    progress=json.loads((run/'tiger_train_test/progress.json').read_text())
    assert progress['state']=='training' and progress['global_step']>0
    assert progress.get('loss_finite') and math.isfinite(progress['loss'])
    children=[pid for pid,ppid,args in processes if ppid==launch['pid'] and 'grid_stage.py' in args]
    assert len(children)==1
    env=dict(entry.split(b'=',1) for entry in Path('/proc/'+str(children[0])+'/environ').read_bytes().split(b'\0') if b'=' in entry)
    assert env[b'CUDA_VISIBLE_DEVICES'].decode()==gpu
    parent=[ppid for pid,ppid,args in processes if pid==launch['pid']][0]
    assert parent==1
    report['datasets'][dataset]={'run_dir':str(run),'pipeline_pid':launch['pid'],'training_pid':children[0],'pipeline_ppid':parent,'physical_gpu':gpu,'embedding_shape':list(vectors.shape),'embedding_sha256':meta['sha256'],'sid_shape':list(sid.shape),'sid_unique_items':unique,'sid_digit_ranges':[[int(row.min()),int(row.max())] for row in sid],'quantizer_checkpoint':str(ckpt_path),'quantizer_steps':qresult['global_step'],'layers_initialized':initialized,'train':True,'test':True,'max_steps':cfg.trainer.max_steps,'max_epochs':cfg.trainer.max_epochs,'progress':progress}
output=ROOT/('training_start_verified_sports_toys.json' if selected=='all' else 'training_start_verified_'+selected+'.json')
if output.exists():
    previous=json.loads(output.read_text())
    for dataset,row in report['datasets'].items():
        before=previous['datasets'][dataset]['progress']['global_step']
        assert row['progress']['global_step']>before
        row['previous_global_step']=before
    report['two_observations_verified']=True
else:
    report['two_observations_verified']=False
output.write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
print(json.dumps(report,indent=2,ensure_ascii=False))
