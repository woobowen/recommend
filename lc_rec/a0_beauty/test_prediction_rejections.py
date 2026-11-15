"""Small real-Beauty-axis synthetic prediction rejection checks; no model inference."""
import argparse,copy,json,math,tempfile
from pathlib import Path
import formal_evaluate as ev

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--data-root',default='/home/addaswsw/recommend-support/a0-beauty-preliminary/author_data');args=parser.parse_args()
 repo=Path(__file__).resolve().parents[2]; evaluator=ev.load_evaluator(repo/'lc_rec/letter/vendor/LETTER/LETTER-LC-Rec')
 _,catalog,allusers,_=ev.catalog_and_users(args.data_root);users=allusers[:3]
 def make(i,rank):
  u=users[i];cs=[{'generation_order':j,'raw_decoded':'Response: '+u['target'],'sequence_score':float(20-j),'sid':u['target']} for j in range(20)]
  return {**u,'rank':rank,'prompt_id':0,'candidates':cs,'sorted_generation_order':list(range(20)),'relevance':[1]*20}
 good=[[make(0,0),make(2,0)],[make(1,1)]];results={}
 with tempfile.TemporaryDirectory() as d:
  paths=[Path(d)/'rank0.jsonl',Path(d)/'rank1.jsonl']
  def execute(data):
   for path,rows in zip(paths,data):path.write_text(''.join(json.dumps(x)+'\n' for x in rows))
   return ev.aggregate_predictions(paths,users,catalog,evaluator,require_rank_done=False)
  assert execute(good)['actual_test_users']==3
  def negative(name,mutate):
   rows=copy.deepcopy(good);mutate(rows)
   try:execute(rows)
   except (AssertionError,ValueError,KeyError,TypeError):results[name]='REJECTED'
   else:raise AssertionError(name+' accepted')
  negative('missing_user',lambda x:x[1].clear())
  negative('duplicate_user',lambda x:x[1].append(copy.deepcopy(x[1][0])))
  negative('row_out_of_range',lambda x:x[1][0].update(row_idx=3))
  negative('rank_mismatch',lambda x:x[1][0].update(rank=0))
  negative('wrong_user',lambda x:x[1][0].update(user_id='WRONG'))
  negative('wrong_target_item',lambda x:x[1][0].update(target_item_id='WRONG'))
  negative('wrong_target_sid',lambda x:x[1][0].update(target='WRONG'))
  negative('fewer_beams',lambda x:x[1][0]['candidates'].pop())
  negative('nonfinite_score',lambda x:x[1][0]['candidates'][0].update(sequence_score=math.nan))
  negative('candidate_score_order',lambda x:x[1][0]['candidates'][0].update(sequence_score=-99.))
  negative('wrong_saved_order',lambda x:x[1][0]['sorted_generation_order'].reverse())
  negative('wrong_relevance',lambda x:x[1][0]['relevance'].__setitem__(0,0))
 print(json.dumps({'status':'PASS_SYNTHETIC_SMALL','rejections':results,'scope':'three actual Beauty user identities; no full-population nor formal runner execution'}))
if __name__=='__main__':main()
