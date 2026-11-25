
import datetime,os,json,shutil,subprocess
from pathlib import Path
j=Path('/home/bwwu/grid-audits/sid-extensions-p0/jobs/grid-letter-rerun-20261125-formal-r0')
s=json.loads((j/'status.json').read_text())
out={'time':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':s,'records':{name:json.loads((j/(name+'.json')).read_text()) for name in ['request','accepted','attempt']}}
out['processes']={}
for k in ('worker','supervisor'):
    pid=s[k]['pid']
    proc=Path('/proc')/str(pid)
    argv=proc.joinpath('cmdline').read_bytes().strip(b'\0').decode().split('\0')
    assert argv==s[k]['argv'], (k,argv)
    l=proc.joinpath('stat').read_text().split(') ',1)[1].split()
    assert l[19]==s[k]['start_ticks']
    assert l[4]=='0'
    item={'pid':pid,'ppid':int(l[1]),'pgid':int(l[2]),'sid':int(l[3]),'tty_nr':int(l[4]),'start_ticks':l[19],'argv':argv,'cwd':os.readlink(proc/'cwd'),'fds':{str(i):os.readlink(proc/'fd'/str(i)) for i in range(3)}}
    if k=='worker':
        env=dict(elem.split(b'=',1) for elem in (proc/'environ').read_bytes().split(b'\0') if b'=' in elem)
        item['cuda_visible_devices']=env[b'CUDA_VISIBLE_DEVICES'].decode()
    out['processes'][k]=item
out['gpu']=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu','--format=csv'],text=True)
out['gpu_apps']=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv'],text=True)
out['space']={str(f):shutil.disk_usage(f).free for f in [j,Path('/data/bwwu/grid-letter-rerun-20261125')]}
for f in (j/'work.log',j/'output/run/upstream/quantizer_train.log'):
    if f.exists():
        with f.open('rb') as stream:
            size=os.fstat(stream.fileno()).st_size
            stream.seek(max(0,size-8192))
            out[f.name]={'size':size,'mtime_ns':f.stat().st_mtime_ns,'tail':stream.read(min(size,8192)).decode('utf-8','replace')}
print(json.dumps(out))
