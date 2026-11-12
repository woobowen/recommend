#!/usr/bin/env python3
"""Independent CPU JSON fingerprint/event joins; no model or GPU import."""
import argparse
from pathlib import Path
import json

def differences(a,b,path=''):
    if type(a)!=type(b):return [path+':type']
    if isinstance(a,dict):
        out=[path+'/'+str(k)+':missing' for k in a.keys()^b.keys()]
        for k in sorted(a.keys()&b.keys()):out+=differences(a[k],b[k],path+'/'+str(k))
        return out
    if isinstance(a,list):
        out=[path+':length'] if len(a)!=len(b) else []
        for i,(x,y) in enumerate(zip(a,b)):out+=differences(x,y,path+'/'+str(i))
        return out
    return [] if a==b else [path]

def read(p):return json.loads(Path(p).read_text())
def effective_bnb(x):
    return {n:{'CB':r['state'].get('CB') or r['weight_attrs']['CB'],
               'SCB':r['state'].get('SCB') or r['weight_attrs']['SCB'],
               'threshold':r['state'].get('threshold'),
               'has_fp16_weights':r['state'].get('has_fp16_weights'),
               'active_adapter':r['active_adapter'],'disable_adapters':r['disable_adapters']}
            for n,r in x['bnb'].items()}

def pair(left,right):
    out={'left':str(left),'right':str(right),'ranks':[]}
    for rank in [0,1]:
        l=read(Path(left)/f'before_first_forward.rank{rank}.json');r=read(Path(right)/f'before_first_forward.rank{rank}.json')
        row={'rank':rank,'pre_forward':{k:differences(l[k],r[k]) for k in ['model','optimizer','scheduler','rng','parameters','gradients']},
             'effective_bnb':differences(effective_bnb(l),effective_bnb(r)),
             'runtime':{'left':l['runtime'],'right':r['runtime']},'updates':{},'forward':[],'parameter_gradient':[]}
        for s in [2,3]:
            for label in ['S4','S5']:
                x=read(Path(left)/f'{label}-step{s}.rank{rank}.json');y=read(Path(right)/f'{label}-step{s}.rank{rank}.json')
                row['updates'][f'{label}-step{s}']=differences(x,y)
        events=[]
        for root in [left,right]:events.append([json.loads(v) for v in (Path(root)/f'events.rank{rank}.jsonl').read_text().splitlines()])
        for event,field in [('diagnosis_forward','forward'),('diagnosis_parameter_gradient','parameter_gradient')]:
            x=[v for v in events[0] if v['event']==event];y=[v for v in events[1] if v['event']==event]
            assert len(x)==len(y)
            for a,b in zip(x,y):
                key='module' if field=='forward' else 'parameter';value='output' if field=='forward' else 'gradient'
                assert (a['step'],a['micro'],a[key],a.get('grad_enabled'))==(b['step'],b['micro'],b[key],b.get('grad_enabled'))
                row[field].append({'step':a['step'],'micro':a['micro'],key:a[key],'grad_enabled':a.get('grad_enabled'),'exact':a[value]==b[value],
                  **({'left_norm':a['norm'],'right_norm':b['norm']} if field!='forward' else {})})
        out['ranks'].append(row)
    return out

def main():
    p=argparse.ArgumentParser();p.add_argument('--left');p.add_argument('--right',required=True);p.add_argument('--s1');p.add_argument('--output',required=True);a=p.parse_args()
    if a.s1:
        root=Path(a.s1);model=read(root/'mp_rank_00_model_states.pt.json')['module'];out={'s1':str(root),'loaded':a.right,'ranks':[]}
        for rank in [0,1]:
            loaded=read(Path(a.right)/f'S2-train-begin.rank{rank}.json')
            opt=read(root/f'bf16_zero_pp_rank_{rank}_mp_rank_00_optim_states.pt.json')['optimizer_state_dict']
            out['ranks'].append({'rank':rank,'module_differences':differences(model,loaded['model']),
                                'optimizer_differences':differences(opt,loaded['optimizer'])})
    else:out=pair(a.left,a.right)
    Path(a.output).write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({k:v for k,v in out.items() if k!='ranks'}))
    for r in out['ranks']:
        print(json.dumps({k:v for k,v in r.items() if k not in ['forward','parameter_gradient','runtime','updates']}))

if __name__=='__main__':main()
