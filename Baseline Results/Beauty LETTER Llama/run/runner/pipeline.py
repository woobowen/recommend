"""Finite train -> best checkpoint -> three full-test modes pipeline."""
from pathlib import Path
import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from common import PREP,event,write


def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);p.add_argument('--group',type=Path,required=True)
    p.add_argument('--capacity',type=Path,required=True);p.add_argument('--micro',type=int,required=True)
    args=p.parse_args();out=args.group/args.dataset;here=Path(__file__).resolve().parent
    locks=[]
    try:
        for i in os.environ['CUDA_VISIBLE_DEVICES'].split(','):
            f=(PREP/f'gpu{i}.lock').open('a');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(f)
        event(out,'waiting_for_capacity',capacity=str(args.capacity),micro=args.micro)
        deadline=time.monotonic()+1800
        while not (args.capacity/'capacity_result.json').exists():
            state=args.capacity/'progress.json'
            if state.exists() and json.loads(state.read_text())['stage']=='failed':raise RuntimeError('Capacity/integration check failed')
            if time.monotonic()>deadline:raise TimeoutError('Capacity wait exceeded 30 minutes')
            time.sleep(5)
        capacity=json.loads((args.capacity/'capacity_result.json').read_text())
        assert capacity['passed'] and capacity['micro']==args.micro
        train_out=out/'training';train_out.mkdir(exist_ok=True)
        if not (train_out/'training_result.json').exists():
            argv=[sys.executable,str(here/'train.py'),'--dataset',args.dataset,'--out',str(train_out),'--micro',str(args.micro)]
            resume=sorted((train_out/'checkpoints').glob('checkpoint-*/COMPLETE.json'),key=lambda p:int(p.parent.name.split('-')[-1]))
            if resume:argv.extend(['--resume',str(resume[-1].parent)])
            with (out/'training.log').open('ab',buffering=0) as log:
                worker=subprocess.Popen(argv,stdout=log,stderr=log)
                event(out,'training_running',pid=worker.pid,micro=args.micro,
                    global_gpus=os.environ['CUDA_VISIBLE_DEVICES'],automatic_test=True)
                rc=worker.wait()
            if rc:raise RuntimeError(f'Training exited {rc}; testing not started')
        trained=json.loads((train_out/'training_result.json').read_text());assert trained['complete']
        test_out=out/'testing'
        if not (test_out/'evaluation_result.json').exists():
            with (out/'testing.log').open('ab',buffering=0) as log:
                worker=subprocess.Popen([sys.executable,str(here/'test_recommendation.py'),'--dataset',args.dataset,
                    '--training',str(train_out),'--out',str(test_out)],stdout=log,stderr=log)
                event(out,'testing_running',pid=worker.pid,best_checkpoint=trained['best_checkpoint'],
                    variants=['unshuffle','shuffle_3','shuffle_4'])
                rc=worker.wait()
            if rc:raise RuntimeError(f'Testing exited {rc}; completed parts are retained')
        assert json.loads((test_out/'evaluation_result.json').read_text())['complete']
        event(out,'complete',complete=True,dataset=args.dataset)
        both=[args.group/ds/'testing/evaluation_result.json' for ds in ['Beauty','Instruments']]
        if all(p.exists() for p in both):
            cols=['Dataset','Mode','Recall@5','Recall@10','NDCG@5','NDCG@10','Change vs unshuffle (R5/R10/N5/N10)']
            rows=[cols]
            for file in both:
                result=json.loads(file.read_text());baseline=result['results']['unshuffle']['metrics']
                for mode,r in result['results'].items():
                    keys=['recall@5','recall@10','ndcg@5','ndcg@10']
                    delta=['NA' if baseline[k]==0 else f"{(r['metrics'][k]/baseline[k]-1)*100:+.2f}%" for k in keys]
                    rows.append([result['dataset'],mode]+[f"{r['metrics'][k]:.6f}" for k in keys]+[' / '.join(delta)])
            widths=[max(len(r[i]) for r in rows) for i in range(len(cols))]
            def line(a,b,c):return a+b.join('─'*(w+2) for w in widths)+c
            formatted=['│'+'│'.join(' '+v.ljust(widths[i])+' ' for i,v in enumerate(row))+'│' for row in rows]
            text='```text\n'+line('┌','┬','┐')+'\n'+('\n'+line('├','┼','┤')+'\n').join(formatted)+'\n'+line('└','┴','┘')+'\n```\n'
            tmp=args.group/'metrics.md.tmp';tmp.write_text(text,encoding='utf-8');tmp.replace(args.group/'metrics.md')
    except Exception as exc:
        import traceback
        event(out,'pipeline_failed',complete=False,error=repr(exc),traceback=traceback.format_exc());raise


if __name__=='__main__':main()
