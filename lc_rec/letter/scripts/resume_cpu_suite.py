#!/usr/bin/env python3
"""Detached CPU-only checkpoint inventories, original-threshold comparisons and replay checks."""
import argparse
import datetime as dt
import json
import os
from pathlib import Path
import subprocess
import sys

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--include-torchdet',action='store_true');a=p.parse_args()
 assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
 out=Path(a.output);out.mkdir(parents=True,exist_ok=True);scripts=Path(__file__).resolve().parent
 root=Path('/data/bwwu/lc-rec/resume-20261112');prefix='letter-resume-20261112-';history=Path('/data/bwwu/lc-rec/a0-normal-20261111')
 report={'started_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'cpu_only':True,'checks':[]}
 def run(name,script,args):
  cmd=[sys.executable,str(scripts/script),*map(str,args)];start=dt.datetime.now(dt.timezone.utc).isoformat()
  with (out/(name+'.log')).open('x') as f:r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
  report['checks'].append({'name':name,'argv':cmd,'started_utc':start,'ended_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'exit_code':r.returncode})
  (out/'suite_status.json').write_text(json.dumps(report,indent=2)+'\n');assert r.returncode==0,name
 def training(name):return root/(prefix+name)/'training'
 for name in ['default-cont-r2']+(['torchdet-cont-r3'] if a.include_torchdet else []):
  run('inspect-'+name,'resume_inspect.py',['--checkpoint',training(name)/'checkpoint-2','--output',out/('inspect-'+name)])
  resumed='default-resume-r2' if name=='default-cont-r2' else 'torchdet-resume-r3'
  run('state-'+name,'resume_checkpoint_state.py',['--continuous',training(name),'--resumed',training(resumed),'--inspection',out/('inspect-'+name),'--output',out/('state-'+name+'.json')])
 pairs=[('historical',history/'letter-a0-20261111-acceptance-cont-r3/training',history/'letter-a0-20261111-acceptance-resume-r3/training'),
        ('default',training('default-cont-r2'),training('default-resume-r2')),
        ('independent-resumes',training('load-r1'),training('repeat-r2')),
        ('full-trace-resumes',training('trace1-r3'),training('trace2-r3'))]
 if a.include_torchdet:pairs.append(('torchdet',training('torchdet-cont-r3'),training('torchdet-resume-r3')))
 for name,left,right in pairs:
  run('compare-'+name,'resume_cpu_verify.py',['--left',left,'--right',right,'--output',out/('compare-'+name)])
 run('loader-replay','test_formal_replay.py',['--output',out/'loader_replay.json'])
 report['ended_utc']=dt.datetime.now(dt.timezone.utc).isoformat();report['status']='CPU_CHECKS_COMPLETED_NUMERIC_GATES_SEPARATE'
 (out/'suite_status.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
if __name__=='__main__':main()
