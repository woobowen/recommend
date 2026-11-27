"""Small read-only progress snapshot for the original job; never launches workers."""
from pathlib import Path
import json,subprocess,time
HERE=Path('/home/jcy/GRID_wmh_Qwen_TIGER_20261009/codex/title_categories_hierarchy_20261127')
def main():
 receipt=json.loads((HERE/'latest_run.json').read_text());run=Path(receipt['run'])
 out=dict(time=time.strftime('%Y-%m-%d %H:%M:%S %z'),receipt=receipt)
 for rel in ['status.json','beauty/encoding_status.json','instruments/encoding_status.json','analysis_status.json','beauty/encoding_preflight.json','instruments/encoding_preflight.json']:
  p=run/rel
  if p.exists():out[rel]=json.loads(p.read_text())
 out['gpu']=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv'],text=True)
 proc=Path('/proc')/str(receipt['pid']);out['supervisor_cmdline']=((proc/'cmdline').read_bytes().decode(errors='replace') if proc.exists() else None)
 print(json.dumps(out,indent=2))
if __name__=='__main__':main()
