#!/usr/bin/env python3
"""Invoke the saved terminal gate once when requested; never starts a model run."""
from pathlib import Path
import datetime,json,sys
sys.path.insert(0,'/home/addaswsw/recommend-support/sid-extensions-p0/scripts')
from transport import Transport
P=Path(__file__).resolve().parent
value=json.loads(Transport('new-server').ssh('CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 /data/bwwu/envs/grid-20261009-e61baf3/bin/python -B -',data=(P/'remote_terminal_audit.py').read_bytes(),timeout=60))
stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
target=P/('terminal-review-'+stamp+'.json');target.write_text(json.dumps(value,indent=2)+'\n')
print(json.dumps({'path':str(target),'status':value['status'],'final_pass':value['status']=='A2_TRAINING_NONAUTHOR_REVIEW_PASS'}))
