#!/usr/bin/env python3
import subprocess
import sys
import time
from preflight_subprocess import run
stage={};run([sys.executable,'-c','pass'],time.monotonic()+3,stage,lambda:None,.05);assert stage['exit_code']==0
stage={}
try:run([sys.executable,'-c','raise SystemExit(7)'],time.monotonic()+3,stage,lambda:None,.05)
except subprocess.CalledProcessError as e:assert e.returncode==7 and stage['exit_code']==7
else:raise AssertionError('nonzero status swallowed')
stage={}
try:run([sys.executable,'-c','import time; time.sleep(30)'],time.monotonic()+.2,stage,lambda:None,.05)
except TimeoutError:assert stage['exit_code'] is not None and 'TimeoutError' in stage['interrupted']
else:raise AssertionError('timeout not enforced')
print('PASS: subprocess success, nonzero propagation, timeout and reaping')
