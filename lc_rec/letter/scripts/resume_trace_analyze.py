#!/usr/bin/env python3
"""Align full first-micro probes by semantic key; no CUDA or tensor imports."""
import argparse
import collections
import json
from pathlib import Path

def main():
 p=argparse.ArgumentParser();p.add_argument('--left',required=True);p.add_argument('--right',required=True);p.add_argument('--output',required=True);a=p.parse_args();report={'left':a.left,'right':a.right,'ranks':[]}
 for rank in [0,1]:
  def read(root):return [json.loads(s) for s in (Path(root)/f'events.rank{rank}.jsonl').read_text().splitlines()]
  left,right=read(a.left),read(a.right);r={'rank':rank,'events':{}}
  for kind in ['trace_forward','trace_gradient_output','trace_parameter_gradient']:
   def index(events):
    counts=collections.Counter();out={}
    for e in events:
     if e['event']!=kind:continue
     base=(e.get('module',e.get('parameter')),e.get('grad_enabled'));counts[base]+=1;key=(*base,counts[base]);out[key]={k:v for k,v in e.items() if k in ['input','output','gradient','norm']}
    return out
   l,rr=index(left),index(right);diff=[{'key':k,'left':v,'right':rr.get(k)} for k,v in l.items() if v!=rr.get(k)]
   r['events'][kind]={'left_count':len(l),'right_count':len(rr),'missing_keys':[k for k in l if k not in rr],'extra_keys':[k for k in rr if k not in l],'exact':sum(v==rr.get(k) for k,v in l.items()),'differences_in_left_execution_order':diff}
  report['ranks'].append(r)
 out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,indent=2)+'\n')
 print(json.dumps([{'rank':r['rank'],'events':{k:{'count':v['left_count'],'exact':v['exact'],'first_difference':v['differences_in_left_execution_order'][:1]} for k,v in r['events'].items()}} for r in report['ranks']],indent=2))
if __name__=='__main__':main()
