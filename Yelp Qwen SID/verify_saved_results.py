"""Verify saved test records and regenerate metrics; no checkpoint inference."""
from pathlib import Path
import csv,hashlib,json,os
def main():
 here=Path(__file__).resolve().parent
 if os.name=='nt':here=Path('\\\\?\\'+str(here))
 index=json.loads((here/'run_index.json').read_text(encoding='utf-8'));rows=[]
 for r in index:
  run=here/'runs'/r['published_run'];stage=json.loads((run/'stage_result.json').read_text(encoding='utf-8'));assert stage['completed']
  manifest=json.loads((run/'source_manifest.json').read_text(encoding='utf-8'))['files']
  for p,h in manifest.items():assert hashlib.sha256((here/'source_snapshots'/r['source_signature']/p).read_bytes()).hexdigest()==h,p
  expected={'unshuffle','shuffle_3','shuffle_4'} if 'unshuffle' in stage['test_metrics'] else {'unshuffled','shuffle_3','shuffle_4'}
  assert set(stage['test_metrics'])==expected,'Unexpected evaluation condition set'
  baseline=stage['test_metrics']['unshuffle' if 'unshuffle' in expected else 'unshuffled']['metrics'][0]
  for mode,v in stage['test_metrics'].items():
   assert json.loads((run/f'test_{mode}.json').read_text(encoding='utf-8'))==v
   assert v['full_test_set'] and v['checkpoint_sha256']==stage['checkpoint_sha256']
   metric=v['metrics'][0]
   rows.append({'dataset':r['dataset'],'condition':mode,'seed':stage['train_seed'],**{k:metric['test/'+k] for k in ['recall@5','recall@10','ndcg@5','ndcg@10']},**{k+'_change_percent':100*(metric['test/'+k]/baseline['test/'+k]-1) for k in ['recall@5','recall@10','ndcg@5','ndcg@10']}})
 print(json.dumps({'saved_tests':len(rows),'models':len(index),'rows':rows,'model_inference':False},indent=2))
if __name__=='__main__':main()
