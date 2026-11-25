"""Create new recommendation splits on the existing Qwen Instruments item axis."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',TF_CPP_MIN_LOG_LEVEL='3',TF_NUM_INTRAOP_THREADS='1',TF_NUM_INTEROP_THREADS='1')
import argparse,collections,hashlib,json
from pathlib import Path

ROOT=Path('/home/jcy/GRID_wmh_Qwen_TIGER_20261009')
OLD=ROOT/'codex/instrument_semantic_hierarchy_20261115'
UP=OLD/'runs/20261115_104723/upstream'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf8')

def main():
 p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);args=p.parse_args();run=args.run
 import torch,tensorflow as tf
 tf.config.set_visible_devices([],'GPU')
 mapping=OLD/'inputs/instrument/item_mapping.jsonl'
 axis=[json.loads(s) for s in mapping.read_text().splitlines()]
 assert [r['item_id'] for r in axis]==list(range(10611))
 amap={r['asin']:r['item_id'] for r in axis};assert len(amap)==10611
 sid=UP/'semantic_ids.pt';assert sha(sid)=='0f12825d1de428e8cba9d9827537be229446f0f076effbf6bd695efa42c83015'
 tensor=torch.load(sid,map_location='cpu',weights_only=False)
 assert list(tensor.shape)==[4,10611] and len(torch.unique(tensor.T,dim=0))==10611
 source=Path('/home/jcy/data/amazon2018_instruments/interactions_5core_aligned.jsonl')
 groups=collections.defaultdict(list);rows=0
 for i,line in enumerate(source.read_text().splitlines()):
  r=json.loads(line);assert amap[r['asin']]==r['item_id']
  groups[r['reviewerID']].append((r['unixReviewTime'],i,r['item_id']));rows+=1
 assert len(groups)==27530 and rows==231312
 sequences={i:[v[2] for v in sorted(groups[u])] for i,u in enumerate(sorted(groups))}
 assert min(map(len,sequences.values()))>=3
 data=run/'data';data.mkdir(exist_ok=False)
 write(run/'user_mapping.json',[dict(user_id=i,reviewerID=u) for i,u in enumerate(sorted(groups))])
 with (run/'sequences.jsonl').open('w') as f:
  for u,s in sequences.items():f.write(json.dumps(dict(user_id=u,items=s))+'\n')
 opts=tf.data.Options();opts.threading.private_threadpool_size=1
 reports={}
 for split in ['training','evaluation','testing']:
  folder=data/split;folder.mkdir();file=folder/'part-00000.tfrecord.gz'
  expected={u:(s[:-2] if split=='training' else s[:-1][-21:] if split=='evaluation' else s[-21:]) for u,s in sequences.items()}
  # A one-item training sequence has no next-item example; retain these users in evaluation/testing.
  expected={u:s for u,s in expected.items() if len(s)>=2}
  feat=lambda a:tf.train.Feature(int64_list=tf.train.Int64List(value=a))
  with tf.io.TFRecordWriter(str(file),options='GZIP') as w:
   for u,s in expected.items():
    fields={'user_id':feat([u]),'sequence_data':feat(s)}
    w.write(tf.train.Example(features=tf.train.Features(feature=fields)).SerializeToString())
  seen=set()
  for raw in tf.data.TFRecordDataset(str(file),compression_type='GZIP').with_options(opts):
   fields=tf.train.Example.FromString(raw.numpy()).features.feature
   u=fields['user_id'].int64_list.value[0];s=list(fields['sequence_data'].int64_list.value)
   assert u not in seen and s==expected[u];seen.add(u)
  assert seen==set(expected)
  reports[split]=dict(users=len(seen),excluded_short=len(sequences)-len(seen),sha256=sha(file),all_records_readback_exact=True)
 report=dict(complete=True,items=10611,users=len(sequences),interactions=rows,sid_path=str(sid),sid_sha256=sha(sid),
  mapping_path=str(mapping),mapping_sha256=sha(mapping),interactions_path=str(source),interactions_sha256=sha(source),
  duplicate_user_item_extra=sum(len(s)-len(set(s)) for s in sequences.values()),
  ordering='unixReviewTime ascending; source row index breaks ties',
  filtering='No additional core filtering or interaction deduplication. Only 5 users with one-item training histories excluded from training; all users evaluated.',
  splits=reports,split_policy='train=S[:-2]; validation=S[:-1][-21:]; test=S[-21:]; evaluation has last 20 history items plus target, matching Beauty baseline',
  source_sha256=sha(__file__))
 write(run/'data_audit.json',report);print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':main()
