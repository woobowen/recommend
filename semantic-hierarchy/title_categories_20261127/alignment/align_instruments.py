"""Use original author ID mappings, bound by full metadata and interaction validation.

No fuzzy title matching, SID-derived matching, model training, or original edits.
"""
import argparse,ast,collections,gzip,hashlib,html,json,re,shutil
from pathlib import Path

WORKSPACE=Path(__file__).resolve().parents[2]
AUTHOR=WORKSPACE/'codex/instruments_author_preprocessing_20261115'
RAW=WORKSPACE/'codex/instrument_semantic_hierarchy_20261115/local_transfer/raw'
QINPUT=WORKSPACE/'codex/github_delivery_audit_20261121/metadata/hierarchy/control/instrument_semantic_hierarchy_20261115/inputs/instrument'
OFFICIAL=WORKSPACE/'codex/letter_lc_rec_smoke_20261121/vendor/LETTER/data/Instruments'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf8'))
def jsonl(p):return [json.loads(s) for s in p.read_text(encoding='utf8').splitlines()]
def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
def lines(p,rows):
 with p.open('w',encoding='utf8',newline='\n') as f:
  for r in rows:f.write(json.dumps(r,ensure_ascii=False)+'\n')
def field_key(v):return json.dumps(v,sort_keys=True,ensure_ascii=False,separators=(',',':'))
def load_author_metadata():
 scope=dict(collections=collections,gzip=gzip,json=json,html=html,re=re,tqdm=lambda x,**kw:x)
 for path,names in [(AUTHOR/'LC-Rec/data_process/utils.py',{'clean_text'}),(AUTHOR/'LC-Rec/data_process/amazon18_data_process.py',{'load_meta_items'})]:
  tree=ast.parse(path.read_text(encoding='utf8'))
  sub=ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names],type_ignores=[])
  exec(compile(sub,str(path),'exec'),scope)
 return scope['load_meta_items'](str(RAW/'meta_Musical_Instruments.json.gz'))

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=Path(__file__).resolve().parent/'aligned')
 args=parser.parse_args();out=args.output.resolve();assert not out.exists(),'Choose a new output directory; originals are protected'
 meta=load_author_metadata();raw=[]
 with gzip.open(RAW/'Musical_Instruments_5.json.gz','rt',encoding='utf8') as f:
  for i,s in enumerate(f):
   r=json.loads(s);raw.append((r['reviewerID'],r['asin'],int(r['unixReviewTime']),i))
 # Author retains earliest user-item occurrence. Stable row order only breaks timestamp ties.
 earliest={}
 for u,a,t,i in raw:
  key=(u,a)
  if key not in earliest or (t,i)<earliest[key]:earliest[key]=(t,i)
 edges=[(u,a,*ti) for (u,a),ti in earliest.items() if a in meta]
 dedup_count=len(edges);rounds=[]
 while True:
  uc=collections.Counter(u for u,a,t,i in edges);ic=collections.Counter(a for u,a,t,i in edges)
  new=[e for e in edges if uc[e[0]]>=5 and ic[e[1]]>=5]
  if len(new)==len(edges):break
  rounds.append(dict(before=len(edges),after=len(new)))
  edges=new
 raw_users=collections.defaultdict(list);raw_items=collections.defaultdict(set)
 for u,a,t,i in edges:raw_users[u].append((t,i,a));raw_items[a].add(u)
 assert (len(raw_users),len(raw_items),len(edges))==(24772,9922,206153)
 released_items=read(OFFICIAL/'Instruments.item.json');released_users=read(OFFICIAL/'Instruments.inter.json');released_sids=read(OFFICIAL/'Instruments.index.json')
 assert set(released_items)==set(released_sids)=={str(i) for i in range(9922)}
 assert len(released_users)==24772 and sum(map(len,released_users.values()))==206153
 assert all(len(s)==len(set(s)) for s in released_users.values())
 item_keys_raw={a:field_key(meta[a]) for a in raw_items}
 item_keys_letter={i:field_key(r) for i,r in released_items.items()}
 assert collections.Counter(item_keys_raw.values())==collections.Counter(item_keys_letter.values())
 # Author mapping is authoritative; exact full released data identity binds it to LETTER.
 original=Path(__file__).resolve().parent/'author_mapping'
 for name in ['Instruments.item.json','Instruments.inter.json']:
  assert read(original/name)==read(OFFICIAL/name),name
 def inverse_mapping(path):
  values={}
  for line in path.read_text().splitlines():
   raw_id,integer_id=line.split('\t');assert integer_id not in values;values[integer_id]=raw_id
  assert len(set(values.values()))==len(values)
  return values
 item_map=inverse_mapping(original/'Instruments.item2id')
 user_map=inverse_mapping(original/'Instruments.user2id')
 assert set(item_map)==set(released_items) and set(user_map)==set(released_users)
 assert set(item_map.values())==set(raw_items) and set(user_map.values())==set(raw_users)
 assert all(meta[a]==released_items[i] for i,a in item_map.items())
 history=[dict(stage='official_item2id_and_user2id',items=len(item_map),users=len(user_map),
  source='LC-Rec official dataset download, identical item/inter JSON to LETTER release')]
 # Complete graph equality is independent of integer ordering and within-day timestamp ties.
 matched_edges={(user_map[u],item_map[str(a)]) for u,s in released_users.items() for a in s}
 assert matched_edges=={(u,a) for u,a,t,i in edges}
 chronology_errors=[];strict_sequence_equal=ties=0
 for u,s in released_users.items():
  rs=raw_users[user_map[u]];times={a:t for t,i,a in rs};seq=[item_map[str(a)] for a in s]
  ts=[times[a] for a in seq]
  if ts!=sorted(ts):chronology_errors.append(u)
  strict_sequence_equal+=seq==[a for t,i,a in sorted(rs)]
  ties+=len(ts)-len(set(ts))
 assert not chronology_errors,'Interaction graph matches but chronological order is incompatible'
 qmapping=jsonl(QINPUT/'item_mapping.jsonl');qsids=jsonl(QINPUT/'item_sids.jsonl')
 assert sha(QINPUT/'item_mapping.jsonl')=='262c3bc796ece5826f2e171f585c1664750e9d5b3ef336b5d13899f9715246e8'
 assert [r['item_id'] for r in qmapping]==list(range(10611)) and [r['item_id'] for r in qsids]==list(range(10611))
 aq={r['asin']:r['item_id'] for r in qmapping};assert len(aq)==10611 and set(item_map.values())<=aq.keys()
 mapping=[dict(item_id=i,letter_item_id=i,asin=item_map[str(i)],qwen_original_item_id=aq[item_map[str(i)]],
  identity_basis='original author item2id; full official metadata/interaction equality and ASIN-to-Qwen source mapping') for i in range(9922)]
 qnew={str(r['item_id']):qsids[r['qwen_original_item_id']]['sid'] for r in mapping}
 assert all(len(s)==4 for s in qnew.values()) and len({tuple(s) for s in qnew.values()})==9922
 removed=[r for r in qmapping if r['asin'] not in set(item_map.values())];assert len(removed)==689
 out.mkdir(parents=True)
 lines(out/'item_mapping.jsonl',mapping);lines(out/'removed_qwen_items.jsonl',removed)
 lines(out/'qwen_item_sids.jsonl',[dict(item_id=i,sid=qnew[str(i)]) for i in range(9922)])
 write(out/'Instruments.qwen.index.json',qnew)
 for name in ['Instruments.index.json','Instruments.inter.json','Instruments.item.json']:shutil.copy2(OFFICIAL/name,out/name)
 lines(out/'items_title_categories.jsonl',[dict(item_id=i,asin=item_map[str(i)],title=released_items[str(i)]['title'],categories=released_items[str(i)]['categories']) for i in range(9922)])
 lines(out/'user_identity.jsonl',[dict(letter_user_id=int(u),reviewerID=user_map[u]) for u in sorted(user_map,key=int)])
 report=dict(complete=True,old_qwen_items=10611,aligned_items=9922,removed_items=689,letter_axis_order=True,
  mapping_one_to_one=True,exact_four_field_multiset_equal=True,identity_refinement=history,
  full_interaction_edges_checked=len(edges),full_graph_exact=True,users_bijective=len(user_map),chronological_order_valid=True,
  source_row_tie_order_exact_users=strict_sequence_equal,timestamp_tie_extra_occurrences=ties,
  sequence_note='Official LETTER sequence preserved byte-for-byte; within-timestamp tie ordering may differ from local raw-row order. Full edges and nondecreasing timestamps verified.',
  qwen_sid_values_unchanged=True,letter_sid_bytes_unchanged=True,learned_quantizers_unchanged=True,no_model_inference_or_training=True,
  text_source='Identical official LETTER title/categories on common LETTER item axis for future paired comparison; historical embeddings not silently reused across texts.',
  nonempty_categories=sum(bool(r.get('categories','').strip()) for r in released_items.values()),
  source_counts=dict(raw_users=len({r[0] for r in raw}),raw_items=len({r[1] for r in raw}),raw_reviews=len(raw),dedup_user_items=len(earliest),dedup_with_metadata=dedup_count,core_rounds=rounds),
  sources={str(p):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in [RAW/'Musical_Instruments_5.json.gz',RAW/'meta_Musical_Instruments.json.gz',QINPUT/'item_mapping.jsonl',QINPUT/'item_sids.jsonl',OFFICIAL/'Instruments.item.json',OFFICIAL/'Instruments.inter.json',OFFICIAL/'Instruments.index.json',original/'Instruments.item2id',original/'Instruments.user2id',original/'Instruments.item.json',original/'Instruments.inter.json',Path(__file__)]},
  outputs={p.name:dict(bytes=p.stat().st_size,sha256=sha(p)) for p in out.iterdir() if p.is_file()})
 write(out/'alignment_verification.json',report)
 print(json.dumps({k:v for k,v in report.items() if k not in ['sources','outputs','identity_refinement']},ensure_ascii=False),flush=True)
if __name__=='__main__':main()
