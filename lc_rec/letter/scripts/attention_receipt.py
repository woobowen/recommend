#!/usr/bin/env python3
"""Final CPU receipt: original checkpoint hashes, true runner charges, storage and processes."""
import datetime as dt
import json
import os
from pathlib import Path
import socket
import subprocess
from formal_train import write,sha


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    root=Path('/data/bwwu/lc-rec/attention-20261112')
    out=root/Path(os.environ['GRID_JOB_DIR']).name;out.mkdir(exist_ok=False)
    release=Path(__file__).resolve().parents[3]
    old=json.loads((release/'lc_rec/letter/evidence/resume_20261112/jobs/letter-resume-20261112-cpu-final-receipt/run/final_resources.json').read_text())
    checkpoint=[]
    for row in old['historical_checkpoint_rehashed_unchanged']:
        p=Path(row['path']);r={'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)}
        assert r==row, 'original checkpoint changed'
        checkpoint.append(r)
    jobs=[];charged=0
    for p in sorted(Path('/home/bwwu/grid-audits/sid-extensions-p0/jobs').glob('letter-attention-20261112-*')):
        req=json.loads((p/'request.json').read_text());status=json.loads((p/'status.json').read_text())
        gpu=bool(req['env'].get('CUDA_VISIBLE_DEVICES',''))
        row={'job_id':p.name,'request':req,'status':status,'gpu':gpu}
        if (p/'cleanup.json').exists():row['cleanup']=json.loads((p/'cleanup.json').read_text())
        if gpu:
            assert status.get('ended_at') and status.get('started_at'), 'GPU job not terminal'
            outer=(dt.datetime.fromisoformat(status['ended_at'])-dt.datetime.fromisoformat(status['started_at'])).total_seconds()
            pipeline=json.loads((root/p.name/'pipeline.json').read_text())
            row.update(pipeline=pipeline,outer_wall_seconds=outer,charged_device_seconds=max(outer*2,pipeline['gpu_device_seconds']))
            charged+=row['charged_device_seconds']
        jobs.append(row)
    assert sum(x['gpu'] for x in jobs)<=8 and charged<=7200
    storage={}
    for name,p in [('data',root),('exports',Path('/home/bwwu/lc-rec/letter-attention-20261112'))]:
        files=[f for f in p.rglob('*') if f.is_file() and not f.is_symlink()]
        storage[name]={'path':str(p),'files':len(files),'logical_bytes':sum(f.stat().st_size for f in files),
                       'du_bytes':int(subprocess.check_output(['du','-s','-B1',str(p)],text=True).split()[0])}
    commands={'gpu':['nvidia-smi','--query-gpu=index,uuid,name,memory.used,utilization.gpu','--format=csv'],
              'compute':['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_memory','--format=csv'],
              'disk':['df','-B1','/data/bwwu','/home/bwwu'],'ram':['free','-b'],
              'process':['ps','-u','bwwu','-o','pid,ppid,lstart,args']}
    sources=[]
    for export in sorted(Path('/home/bwwu/lc-rec/letter-attention-20261112/releases').glob('*')):
        if not (export/'RUN_VERSION.json').exists():continue
        mapping=json.loads((export/'RUN_VERSION.json').read_text())
        for row in mapping['files']:
            assert sha(export/row['path'])==row['sha256'], 'fixed export changed'
        sources.append({'RUN_COMMIT':mapping['RUN_COMMIT'],'path':str(export),'verified_files':len(mapping['files']),'mapping_sha256':sha(export/'RUN_VERSION.json')})
    result={'sources':sources,'utc':dt.datetime.now(dt.timezone.utc).isoformat(),'host':socket.gethostname(),'uid':os.getuid(),
            'original_checkpoint':checkpoint,'jobs':jobs,'charged_gpu_device_seconds':charged,
            'charged_gpu_device_minutes':charged/60,'storage':storage,
            'ledger':json.loads((root/'ledger.json').read_text()),
            'resources':{k:subprocess.check_output(v,text=True) for k,v in commands.items()},
            'new_installations':[],'A0_FORMAL_NORMAL':'NOT_STARTED'}
    write(out/'final_resources.json',result)
    print(json.dumps({'out':str(out),'gpu_device_minutes':charged/60,'storage':storage}))


if __name__=='__main__':main()
