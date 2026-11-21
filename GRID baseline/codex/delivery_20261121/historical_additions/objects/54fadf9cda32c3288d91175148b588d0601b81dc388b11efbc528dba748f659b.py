import os,sys
os.environ.update(CUDA_VISIBLE_DEVICES='',TF_CPP_MIN_LOG_LEVEL='3',OMP_NUM_THREADS='2')
from pathlib import Path
from datetime import datetime
import hashlib,json,subprocess
root=Path('/home/jcyin/GRID_wmh_Qwen_TIGER_20261009')
sys.path.insert(0,str(root))
import src.utils
import torch
from omegaconf import OmegaConf
stamp=datetime.now().strftime('%Y%m%d_%H%M%S')
folder=root/'reconfigurations'/('val1600_'+stamp)
folder.mkdir(parents=True,exist_ok=False)
plan={'prepared_at':datetime.now().astimezone().isoformat(),'plan_dir':str(folder),'state':'prepared','datasets':{}}
for dataset,gpu in [('beauty','0'),('sports','1'),('toys','2')]:
    launch=json.loads(sorted((root/'launches').glob(dataset+'_rkmeans_qwen_*.json'))[-1].read_text())
    run=Path(launch['run_dir'])
    old_stage=Path(launch.get('stage_dir',str(run/'tiger_train_test')))
    if dataset=='beauty':
        candidates=list((old_stage/'checkpoints').glob('*.ckpt'))
        assert len(candidates)==1,candidates
        checkpoint=candidates[0]
        ckpt=torch.load(checkpoint,map_location='cpu',weights_only=False)
        assert ckpt['optimizer_states'] and 'loops' in ckpt
        callback=next(v for k,v in ckpt['callbacks'].items() if k.startswith('ModelCheckpoint'))
        assert Path(callback['best_model_path'])==checkpoint
        best={'path':str(checkpoint),'sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),'global_step':ckpt['global_step'],'best_recall5':float(callback['best_model_score']),'optimizer_states':len(ckpt['optimizer_states'])}
        stage=old_stage
        checkpoint_override='ckpt_path='+chr(39)+str(checkpoint)+chr(39)
    else:
        stage=run/('tiger_train_test_val1600_'+stamp)
        checkpoint_override='ckpt_path=null'
        best=None
    command=json.loads((old_stage/'command.json').read_text())['argv']
    replace={'ckpt_path=':checkpoint_override,'trainer.val_check_interval=':'trainer.val_check_interval=1600','hydra.run.dir=':'hydra.run.dir='+str(stage),'++trainer.enable_progress_bar=':'++trainer.enable_progress_bar=false'}
    command=[v for v in command if not any(v.startswith(k) for k in replace)]
    command+=list(replace.values())
    result=subprocess.run(command+['--cfg','job'],cwd=root,capture_output=True,text=True)
    assert result.returncode==0,result.stderr[-2000:]
    config=OmegaConf.create(result.stdout)
    assert config.trainer.val_check_interval==1600 and config.trainer.accumulate_grad_batches==16
    assert config.train and config.test
    assert config.ckpt_path==(str(checkpoint) if dataset=='beauty' else None)
    (folder/(dataset+'_config.yaml')).write_text(result.stdout,encoding='utf-8')
    job={'dataset':dataset,'physical_gpu':gpu,'run_dir':str(run),'stage_dir':str(stage),'argv':command,'console_filename':'console_val1600_'+stamp+'.log','source_launch':launch,'source_stage':str(old_stage),'resume_checkpoint':best,'validation_interval_batches':1600,'train':True,'test':True,'mode':'resume_best' if dataset=='beauty' else 'fresh_model'}
    job_file=folder/(dataset+'_job.json')
    job_file.write_text(json.dumps(job,indent=2),encoding='utf-8')
    plan['datasets'][dataset]={'job_file':str(job_file),**job}
    print('PREPARED',dataset,'gpu',gpu,'mode',job['mode'],'best',best,flush=True)
(folder/'plan.json').write_text(json.dumps(plan,indent=2),encoding='utf-8')
(root/'reconfigurations/latest_val1600_plan.json').write_text(json.dumps(plan,indent=2),encoding='utf-8')
print('PLAN_FILE',str(folder/'plan.json'),flush=True)
