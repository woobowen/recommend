"""Create a fresh frozen seqrec-only Beauty run; preserve Instruments and old run."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import ast,difflib,hashlib,json,os,shutil,subprocess,sys

ROOT=Path('/home/jcy/GRID_wmh_Qwen_TIGER_20261009')
OLD=ROOT/'experiments/letter_lc_rec_formal_20261121_123952'
HERE=ROOT/'codex/beauty_seqrec_batch64_20261122'
META=HERE/'new_run.json'
assert json.loads((HERE/'stop_receipt.json').read_text())['stopped']
assert not META.exists(),'New run already prepared; inspect existing task rather than duplicate'
stamp=datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S')
group=ROOT/'experiments'/('letter_beauty_seqrec_batch64_'+stamp)
runner=group/'runner';runner.mkdir(parents=True)
for f in (OLD/'runner').glob('*.py'):shutil.copy2(f,runner/f.name)
p=runner/'common.py';before=p.read_text()
oldtasks="'tasks':['seqrec','item2index','index2item'] if dataset=='Beauty' else ['seqrec']"
assert before.count(oldtasks)==1
after=before.replace(oldtasks,"'tasks':['seqrec']")
after=after.replace("'effective_batch':32 if dataset=='Beauty' else 64", "'effective_batch':64")
after=after.replace("'gradient_accumulation_steps':(32 if dataset=='Beauty' else 64)//micro", "'gradient_accumulation_steps':64//micro")
old="    if dataset=='Beauty':parts.extend(ItemFeatDataset(args,task=t,prompt_sample_num=1,sample_num=-1)\n                                    for t in ['item2index','index2item'])"
assert after.count(old)==1;after=after.replace(old,'    # User-authorized seqrec-only run; no auxiliary product-text training.')
after=after.replace("'author_reference':'Issue19 author supplement, four-card effective batches 32/64'", "'author_reference':'User 2026-11-22: Beauty seqrec only, effective batch64; other settings inherited, including temperature1.0'")
p.write_text(after)
(group/'common_changes.diff').write_text(''.join(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile='old_three_task/common.py',tofile='new_seqrec64/common.py')))
for f in runner.glob('*.py'):ast.parse(f.read_text(),filename=str(f))
shutil.copy2(HERE/'supervise.py',group/'supervise.py')
manifest={str(f.relative_to(group)):hashlib.sha256(f.read_bytes()).hexdigest() for f in [*runner.glob('*.py'),group/'supervise.py']}
meta={'group':str(group),'time_beijing':datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),'parent_run':str(OLD),
      'changes':{'tasks':['seqrec'],'effective_batch':64},'unchanged':{'temperature':1.0,'epochs':4,'train_seed':42,'shuffle_seed':2026,'shuffle_identity_allowed':True,'best_selection':'lowest eval_loss'},
      'gpus':'2,3','train_from_base':True,'reuse_old_adapter_or_optimizer':False,'source_sha256':manifest,
      'capacity_candidates':[8,4,2,1],'post_training_tests':['unshuffle','shuffle_3','shuffle_4']}
(group/'request.json').write_text(json.dumps(meta,indent=2))
env=dict(os.environ,CUDA_VISIBLE_DEVICES='2,3',USE_TF='0',USE_FLAX='0',TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',WANDB_MODE='disabled',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
python=ROOT/'environments/letter_lc_rec/bin/python'
with (group/'supervisor.log').open('ab',buffering=0) as log:
    child=subprocess.Popen([str(python),str(group/'supervise.py'),'--group',str(group)],env=env,stdout=log,stderr=log,stdin=subprocess.DEVNULL,start_new_session=True)
meta['supervisor_pid']=child.pid;META.write_text(json.dumps(meta,indent=2));(group/'launch.json').write_text(json.dumps(meta,indent=2))
print(json.dumps(meta),flush=True)
