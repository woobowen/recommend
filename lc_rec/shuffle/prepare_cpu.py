"""Run authorized CPU-only full inputs and A1 nonauthor point audit."""
import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import time


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',required=True)
    a=p.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    root=Path(__file__).resolve().parent
    config=json.loads((root/'config.json').read_text())
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    calls=[]
    def run(name, argv):
        tick=time.monotonic()
        with (out/(name+'.log')).open('x') as f:
            r=subprocess.run(argv,stdout=f,stderr=subprocess.STDOUT,timeout=900)
        calls.append({'name':name,'argv':argv,'exit_code':r.returncode,'wall_seconds':time.monotonic()-tick})
        (out/'calls.json').write_text(json.dumps(calls,indent=2)+'\n')
        assert r.returncode == 0, name
    for key,b in config['baselines'].items():
        run(key+'_inputs',[b['python'],str(root/'cpu_inputs.py'),'--model-key',key,
            '--adapter',b['adapter'],'--data-root',b['data-root'],'--source-root',b['source-root'],
            '--normal-test-dir',b['normal_test_dir'],'--output',str(out/key)])
    # Both tokenizer scans use exactly the same row/item keyed permutations.
    left=json.loads((out/'a0/audit.json').read_text());right=json.loads((out/'a1/audit.json').read_text())
    assert left['history_lengths']==right['history_lengths']
    assert left['permutation_distributions']==right['permutation_distributions']
    for l,r in zip((out/'a0/permutations.jsonl').open(),(out/'a1/permutations.jsonl').open()):
        l,r=json.loads(l),json.loads(r)
        assert l['row_idx']==r['row_idx'] and l['user_id']==r['user_id']
        for n in ['3','4']:assert l[n]['permutations']==r[n]['permutations']
    b=config['baselines']['a1'];data=Path(b['data-root'])/'Instruments'
    normal=Path(b['normal_test_dir']);primary=normal.parent/'offline_evaluation/primary'
    run('a1_independent_points',[b['python'],str(root.parent/'a1_qwen3_06b/review/final_20261114/independent_points.py'),
        *[str(data/('Instruments.'+kind+'.json')) for kind in ['index','inter','item']],
        str(normal/'predictions.rank0.jsonl'),str(normal/'predictions.rank1.jsonl'),
        str(Path(b['adapter'])/'tokenizer.json'),str(primary/'per_user.csv'),
        str(primary/'metrics.json'),str(primary/'groups.json'),str(out/'a1_independent_points.json')])
    summary={'status':'PASS_CPU_PREPARATION','calls':calls,'GPU_used':False,
             'same_user_permutations_across_models':True,
             'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary))


if __name__=='__main__':main()
