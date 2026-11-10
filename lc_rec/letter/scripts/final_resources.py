"""Read-only closeout of task storage, resources and inherited model identity."""
import hashlib,json,os,subprocess
from pathlib import Path
root=Path('/home/bwwu/lc-rec/letter-20261110');env=Path('/data/bwwu/envs/letter-20261110')
def size(p):
 logical=allocated=0
 for d,ds,fs in os.walk(p,followlinks=False):
  ds[:]=[x for x in ds if not (Path(d)/x).is_symlink()]
  for f in fs:
   q=Path(d)/f
   if q.is_file() and not q.is_symlink():s=q.stat();logical+=s.st_size;allocated+=s.st_blocks*512
 return {'logical_bytes':logical,'allocated_bytes':allocated}
r={'new_storage':{str(p):size(p) for p in [root,env]},'memory':Path('/proc/meminfo').read_text(),'gpu':subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,driver_version,memory.total,memory.used,utilization.gpu','--format=csv'],text=True),'gpu_processes':subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv'],text=True),'filesystems':subprocess.check_output(['df','-B1','/data','/home'],text=True)}
model=Path('/data/bwwu/lc-rec/assets/llama-7b/4782ad278652c7c71b72204d462d6d01eaaf7549');p=model/'download-validation-stable-20261109.json';r['model_validation_inherited']=json.loads(p.read_text());r['model_validation_receipt_sha256']=hashlib.sha256(p.read_bytes()).hexdigest();r['current_model_files']={q.name:{'bytes':q.stat().st_size,'mtime_ns':q.stat().st_mtime_ns} for q in model.glob('*.safetensors')};r['pyvenv_cfg']= (env/'pyvenv.cfg').read_text();r['new_system_packages']=[];r['system_configuration_changes']=[]
print(json.dumps(r,indent=2))
