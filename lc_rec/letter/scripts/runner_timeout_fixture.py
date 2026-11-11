#!/usr/bin/env python3
"""No-GPU safety fixture: own marked worker sessions resist TERM, require KILL."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
p=argparse.ArgumentParser();p.add_argument('--child',action='store_true');a=p.parse_args()
signal.signal(signal.SIGTERM,signal.SIG_IGN)
if a.child:
    print(json.dumps({'fixture_child':os.getpid(),'session':os.getsid(0),'pgid':os.getpgrp()}),flush=True)
else:
    child=subprocess.Popen([sys.executable,__file__,'--child'],start_new_session=True,stdin=subprocess.DEVNULL)
    Path(os.environ['GRID_JOB_DIR'],'output/fixture.json').write_text(json.dumps({'parent':os.getpid(),'child':child.pid,'gpu_used':False,'purpose':'known dual-marker child session rejects TERM; runner must use verified pidfd KILL'})+'\n')
time.sleep(60)
