#!/usr/bin/env python3
"""Cross-check published keyed/fingerprint exports and actual LETTER split references."""
import argparse,csv,hashlib,json,pickle
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--supplement',type=Path,required=True);p.add_argument('--p5',type=Path,required=True);p.add_argument('--letter',type=Path,required=True);p.add_argument('--pkl',type=Path,required=True);p.add_argument('--bridge',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def require(c,m):
 if not c: raise ValueError(m)
require(sha(a.pkl)=='dcbfb61369cde2627cea50cc42bb80be753e2d9854b85f1825460174486bd590','PKL hash')
require(sha(a.letter)=='8809584b8b78fa6561771c88911ed65bd716fc9926fca1e581e6a1ab578ccfd5','LETTER inter hash')
require(sha(a.p5)=='226cce9c3105299ca0db9615d7d3fb32b3175e90da43100ae352599f0f0107b8','P5 seq hash')
rows=pickle.loads(a.pkl.read_bytes()); require(len(rows)==12101 and len({r['item_id'] for r in rows})==12101,'keys')
with (a.supplement/'beauty_g3_keyed_predictions.csv').open(newline='') as f: keyed=list(csv.DictReader(f))
require(len(keyed)==len(rows),'keyed count')
by_id={r['item_id']:r['cluster_ids'] for r in rows}
require(all(int(r['item_id'])==i and [int(r[f'sid_{j}']) for j in (1,2,3)]==by_id[i] for i,r in enumerate(keyed)),'keyed export')
seq={}
for line in a.p5.read_text().splitlines():
 x=list(map(int,line.split()));u=x[0]-1;require(u not in seq,'duplicate user');seq[u]=[i-1 for i in x[1:]]
with (a.supplement/'p5_grid_sequence_fingerprints.csv').open(newline='') as f: fps=list(csv.DictReader(f))
require(len(fps)==len(seq) and len({r['grid_user_id'] for r in fps})==len(seq),'fingerprints axis')
for r in fps:
 u=int(r['grid_user_id']);hist=seq[u][-21:]; digest=hashlib.sha256(','.join(map(str,hist)).encode()).hexdigest()
 require(int(r['p5_user_id'])==u+1 and int(r['sequence_length'])==len(hist) and r['zero_based_sequence_sha256']==digest,'fingerprint')
inter=json.loads(a.letter.read_text());require(set(inter)==set(map(str,seq)) and all(v==seq[int(k)] for k,v in inter.items()),'LETTER sequence')
with a.bridge.open(newline='') as f: bridge=list(csv.DictReader(f))
ids=[int(r['letter_item_id']) for r in bridge];require(len(ids)==12101 and set(ids)==set(range(12101)),'bridge bijection')
splits={'train':set(),'valid':set(),'test':set()};counts={'train':0,'valid':0,'test':0}
for values in inter.values():
 # Actual LETTER SeqRecDataset uses all prefixes with >=1 history item.
 for i in range(1,len(values)-2):
  splits['train'].update(values[max(0,i-20):i+1]);counts['train']+=1
 splits['valid'].update(values[max(0,len(values)-22):len(values)-1]);counts['valid']+=1
 splits['test'].update(values[max(0,len(values)-21):]);counts['test']+=1
result={'keyed_records':len(keyed),'testing_fingerprints':len(fps),'letter_split_rows':counts,'letter_split_distinct_items':{k:len(v) for k,v in splits.items()},'letter_split_unmapped':{k:len(v-set(ids)) for k,v in splits.items()},'bridge_sha256':sha(a.bridge),'script_sha256':sha(Path(__file__))}
require(not any(result['letter_split_unmapped'].values()),'unmapped split refs');a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
