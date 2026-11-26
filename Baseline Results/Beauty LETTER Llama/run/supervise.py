"""Bounded capacity then one fresh training and three-condition test pipeline."""
from pathlib import Path
import argparse,fcntl,json,os,subprocess,sys,time,traceback

def save(root,stage,**kw):
    v={'time_unix':time.time(),'stage':stage,**kw};temp=root/'status.tmp';temp.write_text(json.dumps(v,indent=2));temp.replace(root/'status.json')
    with (root/'events.jsonl').open('a') as f:f.write(json.dumps(v)+'\n')
    print(json.dumps(v),flush=True)

def main():
    a=argparse.ArgumentParser();a.add_argument('--group',type=Path,required=True);args=a.parse_args();root=args.group
    sys.path.insert(0,str(root/'runner'));from common import PREP
    locks=[]
    try:
        assert os.environ['CUDA_VISIBLE_DEVICES']=='2,3'
        for gpu in ['2','3']:
            f=(PREP/f'gpu{gpu}.lock').open('a');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(f)
        selected=None
        for micro in [8,4,2,1]:
            out=root/'Beauty'/f'capacity_micro{micro}';out.mkdir(parents=True,exist_ok=False)
            argv=[sys.executable,str(root/'runner/train.py'),'--dataset','Beauty','--out',str(out),'--micro',str(micro),'--capacity']
            with (out/'stdout_stderr.log').open('ab',buffering=0) as log:
                worker=subprocess.Popen(argv,stdout=log,stderr=log,stdin=subprocess.DEVNULL)
                save(root,'capacity_running',micro=micro,worker_pid=worker.pid,argv=argv,effective_batch=64)
                rc=worker.wait(timeout=1200)
            path=out/'capacity_result.json'
            if rc==0 and path.exists() and json.loads(path.read_text()).get('passed'):
                selected=(micro,out);break
            failure=out/'progress.json';state=json.loads(failure.read_text()) if failure.exists() else {}
            save(root,'capacity_failed',micro=micro,exit_code=rc,error=state.get('error'))
            if not any(t in str(state.get('error','')) for t in ['OutOfMemoryError','CUDA out of memory']):raise RuntimeError('Non-memory capacity failure; do not alter experiment to bypass')
        assert selected,'No capacity layout passed'
        micro,out=selected
        # The tested pipeline owns the same exact GPU locks after this supervisor releases them.
        for f in locks:fcntl.flock(f,fcntl.LOCK_UN);f.close()
        locks=[]
        argv=[sys.executable,str(root/'runner/pipeline.py'),'--dataset','Beauty','--group',str(root),'--capacity',str(out),'--micro',str(micro)]
        with (root/'Beauty/pipeline.log').open('ab',buffering=0) as log:
            worker=subprocess.Popen(argv,stdout=log,stderr=log,stdin=subprocess.DEVNULL)
            save(root,'formal_pipeline_running',micro=micro,gradient_accumulation_steps=64//micro,effective_batch=64,worker_pid=worker.pid,argv=argv,automatic_tests=['unshuffle','shuffle_3','shuffle_4'])
            rc=worker.wait()
        assert rc==0,f'Pipeline exit {rc}'
        assert json.loads((root/'Beauty/testing/evaluation_result.json').read_text())['complete']
        save(root,'complete',complete=True)
    except Exception as e:
        save(root,'failed',complete=False,error=repr(e),traceback=traceback.format_exc());raise

if __name__=='__main__':main()
