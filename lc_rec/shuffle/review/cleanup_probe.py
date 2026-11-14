"""CPU-only bounded probe of accepted runner marked-session timeout cleanup."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
assert os.environ['GRID_JOB_ID'].startswith(('letter-a0-20261112-','letter-a1-q06-20261113-'))
out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
children=[]
for _ in range(2):
    child=subprocess.Popen([sys.executable,'-I','-S','-B','-c','import time;time.sleep(45)'],
                           stdin=subprocess.DEVNULL,start_new_session=True)
    children.append({'pid':child.pid,'stat':Path('/proc/%d/stat'%child.pid).read_text()})
(out/'children.json').write_text(json.dumps({'job_id':os.environ['GRID_JOB_ID'],
    'parent_pid':os.getpid(),'parent_session':os.getsid(0),'children':children},indent=2)+'\n')
# The independently accepted supervisor must time out at eight seconds and
# prove cleanup. Children have an additional hard 45s self-expiry as a backstop.
time.sleep(60)
