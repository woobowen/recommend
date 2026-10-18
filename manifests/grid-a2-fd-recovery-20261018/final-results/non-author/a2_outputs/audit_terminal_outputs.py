"""A2 output-only nonauthor audit, run only after a terminal collection.

Reads saved predictions/NPZ/PNG and receipts. Does not run a model or evaluator.
Dependency training completeness is checked but complete training review belongs
to its separate reviewer. No final output PASS is written for missing artifacts.
"""
from pathlib import Path
from collections import Counter
import argparse,gzip,itertools,json,math,statistics,sys
import numpy as np
from PIL import Image
import yaml
from numeric_core import ident,load,read_rows,read_sid_tensor,sha,ranking,pair_metrics,read_bundle,independent_paths
from verify_preparation import CAPTURE,EXPECTED_COHORT,EXPECTED_SID

ROOT=Path(__file__).resolve().parent
PREP=ROOT/'preparation-v2/raw/jobs'
RAW=ROOT/'terminal_outputs/raw/jobs'
META=PREP/'ga-20261018-sportsa2-metadata-fd-recovery-v1/output'

def job(role):return RAW/f'ga-20261018-sportsa2-{role}-fd-recovery-v1'
def write(p,d):
    text=json.dumps(d,indent=2,allow_nan=False)+'\n'
    if p.exists():assert p.read_text()==text,p
    else:p.write_text(text)
def terminal(p):
    s=load(p/'status.json');r=load(p/'request.json')
    assert s['status']=='SUCCEEDED' and s['exit_code']==0 and s['cleanup']['verified']
    assert not s['worker_may_be_live'] and s['cleanup']['last_observation']['verified_empty']
    assert r['capture_version']==CAPTURE
    return {k:s[k] for k in ['job_id','status','exit_code','started_at','ended_at','worker_may_be_live']}|{'cleanup_verified':True}

def check_config(out,n,best,best_step):
    spec=load(out/'invocation.json')['effective_spec'];loaded=load(out/'loading.json');strict=loaded['strict_load']
    assert spec['capture_version']==CAPTURE and spec['sid']['sha256']==EXPECTED_SID
    assert spec['checkpoint']==best and spec['seed']==42 and spec['generation_k']==10 and spec['shuffle_n']==n
    assert sha(META/'fixed_attention_samples.json')==spec['attention_cohort']['sha256']==EXPECTED_COHORT
    assert strict['missing_keys']==strict['unexpected_keys']==[] and strict['global_step']==best_step
    assert strict['all_parameter_keys_shapes_values_equal'] and strict['embedded_sid_exact'] and strict['state_tensor_count']==178
    assert loaded['shared_sid_table'] is False and loaded['parameter_count']==13259136
    assert loaded['model_class']=='experiments.beauty_extensions.author.models.AuthorShuffleModel'
    cfg=yaml.safe_load((out/'effective_config.yaml').read_text())
    assert cfg['model']['top_k_for_generation']==10 and cfg['model'].get('should_check_prefix',False) is False
    assert cfg['model']['shuffle_embedding_on_test']==bool(n) and cfg['model']['hierarchies_to_shuffle']==(n or 3)
    assert not any(k.startswith('limit_') for k in cfg['trainer']) and cfg['trainer']['precision']=='32-true'
    loader=cfg['data_loading']['test_dataloader_config']['dataloader']
    assert loader['batch_size_per_device']==8 and loader['num_workers']==8 and loader['should_shuffle_rows'] is False
    assert loader['drop_last'] is False and loader['sequence_length']==120
    remote_output=str(Path(spec['job_root'])/out.parent.name/'output')
    normalized=json.loads(json.dumps(cfg).replace(remote_output,'<JOB_OUTPUT>'))
    for key in ['shuffle_embedding_on_test','hierarchies_to_shuffle']:normalized['model'].pop(key,None)
    return {'checkpoint':spec['checkpoint'],'sid':spec['sid'],'config':spec['config'],'strict_load':strict,
            'effective_config':ident(out/'effective_config.yaml')},normalized

def evaluate(n,best,best_step,legal):
    p=job(f'eval-{n}');out=p/'output';status=terminal(p);result=load(out/'RESULT.json')
    assert result['status']=='COMPLETE' and result['formal'] and result['test_complete'] and result['shuffle_n']==n
    config,normalized=check_config(out,n,best,best_step)
    rows=read_rows(out/'test/predictions.jsonl');oldper=read_rows(out/'test/per_sample_metrics.jsonl')
    assert len(rows)==len(oldper)==35598 and sha(out/'test/predictions.jsonl')==result['metrics']['prediction_sha256']
    index=read_rows(META/'test_target_index.jsonl')
    assert Counter((r['sample_key'],tuple(r['target'])) for r in rows)==Counter((r['sample_key'],tuple(r['target'])) for r in index)
    assert len({r['sample_key'] for r in rows})==35598
    values={k:[] for k in ['recall@5','ndcg@5','recall@10','ndcg@10']};users={};invalid=duplicates=tied=0;max_per=0.0
    for row,old in zip(rows,oldper):
        r=ranking(row);assert r['target'] in legal and row['sample_key']==old['sample_key']
        invalid+=sum(x not in legal for x in r['ordered']);duplicates+=10-len(set(r['ordered']));tied+=r['score_tied']
        for k,v in r['metrics'].items():values[k].append(v);max_per=max(max_per,abs(v-old['metrics'][k]))
        users[row['sample_key']]=r
    assert duplicates==0 and invalid==result['metrics']['invalid_candidate_count'] and max_per<1e-7
    actual={k:math.fsum(v)/len(v) for k,v in values.items()};original=result['metrics']['metrics']
    errors={k:abs(actual[k]-original[k]) for k in actual};assert max(errors.values())<1e-7
    trace=load(out/'intervention_trace.json')
    assert trace['seed']==42 and trace['exception'] is None and trace['encoder_calls']==8900
    assert trace['encoder_role_counts']=={'loss':4450,'generate':4450} and trace['shuffle_call_count']==(8900 if n else 0)
    assert bool(trace['permutation_events'])==bool(n)
    for e in trace['permutation_events']:
        perm=np.array(e['permutation']);assert e['protocol']=='original_rand_argsort_per_encoder_call'
        assert perm.shape[-1]==n and np.array_equal(np.sort(perm,-1),np.broadcast_to(np.arange(n),perm.shape))
    life=load(out/'test/lifecycle_cleanup.json');assert life['exception'] is None and not life['is_testing_after']
    assert all(life[k] for k in ['encoder_mask_cleared','replay_cleared','evaluator_restored','author_replay_queue_cleared'])
    batches=life['real_batch_devices'];assert [r['batch_index'] for r in batches]==list(range(4450))
    assert all(r['input_device']=='cuda:0' and r['input_dtype']=='torch.int64' and r['parameter_devices']==['cuda:0']
               and r['parameter_dtypes']==['torch.float32'] and r['is_testing'] for r in batches)
    return {'status':'A2_FULL_EVALUATION_SAVED_OUTPUT_PASS','job_id':p.name,'terminal':status,'identity':config,
        'samples':35598,'coverage_exact':True,'metrics_original':original,'metrics_independent_float64':actual,
        'maximum_metric_error':max(errors.values()),'maximum_per_user_error':max_per,'invalid_candidates':invalid,
        'duplicate_candidates':duplicates,'rows_with_irrelevant_score_ties':tied,'prediction':ident(out/'test/predictions.jsonl'),
        'intervention_calls':trace['shuffle_call_count'],'candidate_treatment':'Unmodified original candidates, no filtering/completion/rescoring/reordering'},users,normalized

def paired(n,normal,shuffled):
    assert normal.keys()==shuffled.keys();counts=Counter();transitions={k:Counter() for k in [5,10]};rankpairs=Counter();overlap=[]
    out=ROOT/f'a2-normal-shuffle{n}-paired.jsonl.gz';assert not out.exists()
    with gzip.open(out,'wt') as f:
        for key,x in normal.items():
            y=shuffled[key];assert x['target']==y['target']
            c={'top1_changed':x['ordered'][0]!=y['ordered'][0],'top10_order_changed':x['ordered']!=y['ordered'],
               'top10_set_changed':set(x['ordered'])!=set(y['ordered'])}
            counts.update({k:int(v) for k,v in c.items()});common=len(set(x['ordered'])&set(y['ordered']));overlap.append(common)
            for k in [5,10]:transitions[k][f'{int(0<x["rank"]<=k)}->{int(0<y["rank"]<=k)}']+=1
            rankpairs[(x['rank'],y['rank'])]+=1
            f.write(json.dumps({'sample_key':key,'target':x['target'],'normal_rank':x['rank'],'shuffle_rank':y['rank'],
                               'top10_common':common,**c})+'\n')
    return {'samples':35598,**dict(counts),'mean_top10_overlap':statistics.mean(overlap),
       'hit_migration':{str(k):dict(v) for k,v in transitions.items()},
       'rank_transition_counts':[{'normal':x,'shuffle':y,'count':c} for (x,y),c in sorted(rankpairs.items())],
       'rank_missing_value':0,'per_user_output':ident(out)}

def attention(n,best,best_step,sid,stats_reference):
    p=job(f'attention-{n}');out=p/'output';status=terminal(p);result=load(out/'RESULT.json')
    assert result['formal'] and result['status']=='COMPLETE' and result['samples']==32 and result['shuffle_n']==n
    config,normalized=check_config(out,n,best,best_step);cohort=load(META/'fixed_attention_samples.json')
    assert result['cohort']['sha256']==EXPECTED_COHORT and len(result['batches'])==4
    reference=next(r for r in stats_reference['jobs'] if r['job_id']==p.name)
    refrows={(r['sample_key'],r['path'],tuple(r['query_pair'])):r for r in reference['per_sample_query_pairs']}
    rows=[];keys=[];npzs=[];pngs=0;max_stat=max_head_sum=max_masked=0.;nonzero_parents=0
    for batch,entry in enumerate(result['batches']):
        q=out/f'batch-{batch:02d}';record=q/'bundle/record.json';assert sha(record)==entry['bundle_sha256']
        b=read_bundle(record);npzs.append(ident(record.with_name('arrays.npz')))
        assert not b['metadata']['engineering_only'] and b['metadata']['shuffle_n']==n
        assert b['metadata']['spec']['sid']['sha256']==EXPECTED_SID and b['metadata']['spec']['checkpoint']==best
        expected_keys=cohort['sample_keys'][batch*8:batch*8+8]
        assert b['capture']['sample_keys']==entry['sample_keys']==expected_keys and b['capture']['selected_indices']==list(range(8))
        keys.extend(expected_keys)
        assert b['capture']['source']=='actual generation; no prefix/encoder replay' and b['capture']['encoder_calls']==1
        inputs=np.array(cohort['input_ids'][batch*8:batch*8+8]);inputmask=np.array(cohort['attention_mask'][batch*8:batch*8+8])
        assert np.array_equal(b['input_sids'],inputs) and np.array_equal(b['input_mask'],inputmask)
        mask=b['encoder_mask'].astype(bool);assert mask.shape==(8,150)
        assert np.array_equal(mask.reshape(8,30,5)[:,:,:4].reshape(8,120),inputmask)
        assert np.array_equal(mask.reshape(8,30,5)[:,:,4],inputmask.reshape(8,30,4)[:,:,-1])
        perms=b['metadata']['source_permutations']
        if n:
            perms=np.array(perms);assert perms.shape==(8,30,4)
            assert np.array_equal(np.sort(perms[:,:,:n],-1),np.broadcast_to(np.arange(n),(8,30,n)))
            if n==3:assert (perms[:,:,3]==3).all()
            assert np.any(perms!=np.arange(4))
        else:assert perms is None
        for sample,mapping in enumerate(b['position_map']):
            assert len(mapping)==150;items=cohort['samples'][batch*8+sample]['visible_history_item_ids']
            for pos,m in enumerate(mapping):
                item,code=divmod(pos,5);visible=bool(inputmask[sample,item*4])
                assert m['encoder_position']==pos and m['history_slot']==item and m['item_id']==items[item]
                assert bool(m['visible'])==bool(mask[sample,pos])
                if code==4:assert m['role']=='sep';continue
                src=int(perms[sample,item,code]) if n and visible else code
                assert m['role']=='sid_lookup_vector' and m['lookup_code_position']==src
                assert m['raw_sid_value']==int(inputs[sample,item*4+src])
                expected_lookup=int(inputs[sample,item*4+src])+256*src if visible else 0
                assert m['lookup_table_index']==expected_lookup
                if visible:assert int(inputs[sample,item*4+src])==sid[src,items[item]]
        rebuilt,err=independent_paths(b['steps']);max_head_sum=max(max_head_sum,err)
        for key,value in rebuilt.items():assert np.array_equal(value,b['paths'][key]),key
        assert np.array_equal(b['final_generated_ids'],b['steps'][-1]['generated_ids'])
        assert np.array_equal(b['final_scores'],b['steps'][-1]['scores'])
        nonzero_parents+=int((rebuilt['final_parent_indices']!=0).any(-1).sum())
        for h,step in enumerate(b['steps']):
            max_masked=max(max_masked,float(np.max((step['cross_heads']*(~mask[:,None,None,:])).sum(-1))))
            if h:assert np.array_equal(step['local_first_prefix'],b['steps'][h-1]['generated_ids'][:,0])
            else:assert step['query_identity']=='BOS'
            assert step['last_query_index']==step['query_length_before_last_selection']-1
        observed=load(q/'off_on_checks.json')
        assert observed['sid_exact'] and observed['scores']['exact'] and observed['scores']['max_absolute']==0
        assert all(observed[k] for k in ['cpu_rng_same','cuda_rng_same','parameter_sha256_unchanged','model_overrides_restored','stack_hooks_restored'])
        assert load(q/'RESULT.json')['exception_cleanup']
        plot=load(q/'figures/plot_receipt.json');assert plot['source_record_sha256']==sha(record) and len(plot['products'])==8
        for prod in plot['products']:
            image=q/'figures'/prod['file'];assert sha(image)==prod['sha256'] and image.stat().st_size==prod['bytes']
            s=prod['sample_record_index'];maximum=float(max(rebuilt['local_first_cross'][s].max(),rebuilt['final_top1_cross'][s].max()))
            assert prod['scale_min']==0 and prod['scale_max']==maximum
            with Image.open(image) as im:im.load();assert list(im.size)==[prod['width'],prod['height']]==[2880,1280]
            pngs+=1
        for path in ['final_top1_cross','local_first_cross']:
            matrix=rebuilt[path].astype(np.float64)
            for sample,key in enumerate(expected_keys):
                visible=matrix[sample][:,mask[sample]]
                for i,j in itertools.combinations(range(4),2):
                    r={'sample_key':key,'path':path,'query_pair':[i+1,j+1],**pair_metrics(visible[i],visible[j])}
                    old=refrows[(key,path,(i+1,j+1))]
                    for metric in ['cosine','pearson','total_variation','jensen_shannon_nats']:
                        if r[metric] is None:assert old[metric] is None
                        else:max_stat=max(max_stat,abs(r[metric]-old[metric]))
                    rows.append(r)
    assert keys==cohort['sample_keys'] and len(rows)==384 and pngs==32
    assert max_stat<1e-12 and max_head_sum<2e-6 and max_masked==0
    summary={}
    for path in ['final_top1_cross','local_first_cross']:
        summary[path]={}
        for metric in ['cosine','pearson','total_variation','jensen_shannon_nats']:
            vals=[r[metric] for r in rows if r['path']==path and r[metric] is not None]
            desc={'n':len(vals),'min':min(vals),'median':statistics.median(vals),'mean':math.fsum(vals)/len(vals),'max':max(vals)}
            for k,v in desc.items():assert abs(v-reference['summary'][path][metric][k])<1e-12
            summary[path][metric]=desc
    return {'status':'A2_FIXED32_ATTENTION_AND_CPU_STATISTICS_SAVED_OUTPUT_PASS','job_id':p.name,'terminal':status,
        'identity':config,'fixed32':ident(META/'fixed_attention_samples.json'),'cohort_split':'evaluation','users':32,'batches':4,
        'raw_heads':6,'autoregressive_queries':4,'NPZ_files':npzs,'PNG_count':32,'both_paths_reconstructed_exactly':True,
        'top1_paths_with_nonzero_parent':nonzero_parents,'maximum_head_row_sum_error':max_head_sum,'maximum_masked_mass':max_masked,
        'statistics':summary,'maximum_independent_statistic_error':max_stat,'all_query_pair_rows':rows,
        'PNG_scope':'All files hash/size/dimensions/raw-scale/source checked. No pixel-perfect independent rerender claimed.'},normalized

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--after-terminal-confirmed',action='store_true',required=True);parser.parse_args()
    if not (ROOT/'terminal_outputs/collection-receipt.json').exists():
        print(json.dumps({'status':'PENDING_RESULT_ARTIFACTS','scientific_acceptance':False}));return 2
    train=load(job('train')/'output/RESULT.json');train_status=terminal(job('train'))
    assert train['training_complete'] and not train.get('incomplete',False) and train['sid']['sha256']==EXPECTED_SID
    best=train['best_checkpoint'];best_step=train['best_checkpoint_gate']['best_global_step']
    sid=read_sid_tensor(PREP/'ga-20261018-sportsa2-quantize-fd-recovery-v1/output/semantic_ids.pt');legal=set(map(tuple,sid.T))
    stats=load(job('stats')/'output/RESULT.json');stats_terminal=terminal(job('stats'))
    assert stats['status']=='DESCRIPTIVE_STATISTICS_COMPLETE' and stats['sid']['sha256']==EXPECTED_SID and stats['capture_version']==CAPTURE
    evaluations={};users={};attentions={};configs=[]
    for n in [0,3,4]:
        evaluations[str(n)],users[n],cfg=evaluate(n,best,best_step,legal);configs.append(cfg)
        attentions[str(n)],_=attention(n,best,best_step,sid,stats)
    assert configs[0]==configs[1]==configs[2]
    pairs={str(n):paired(n,users[0],users[n]) for n in [3,4]}
    report={'status':'A2_EVALUATION_ATTENTION_STATISTICS_NONAUTHOR_REVIEW_PASS','reviewer':'/root/toys_result_audit',
        'scope':'A2 saved full prediction, fixed32 attention and CPU statistics only; separate training/quantization acceptance required for overall final PASS',
        'scientific_capture':CAPTURE,'new_SID_sha256':EXPECTED_SID,'best_checkpoint':best,'best_step':best_step,
        'training_dependency_status':train_status,'stats_terminal':stats_terminal,'evaluations':evaluations,'paired_predictions':pairs,
        'attention':attentions,'cohort_split':'evaluation','formal_metrics_split':'testing','GPU_calls':0,'model_loads':0,
        'new_evaluator_calls':0,'server_mutations':0,'overall_final_internal_review_pass':False,
        'limits':['Historical producer and distinct scratch-training conditions cannot be conflated with original author Sports weights.',
                  'No strict permutation-invariance or target-prefix nonuse proof; one seed and fixed descriptive cohort.',
                  'The audit did not reload a large generator checkpoint; actual per-job strict-load receipts are checked and training reviewer owns direct checkpoint acceptance.'],
        'audit_program':ident(Path(__file__)),'numeric_core':ident(ROOT/'numeric_core.py')}
    write(ROOT/'A2_OUTPUTS_NONAUTHOR_REVIEW.json',report)
    print(json.dumps({'status':report['status'],'prediction_rows':106794,'NPZ':12,'PNG':96,'query_pair_rows':1152,
                      'report':str(ROOT/'A2_OUTPUTS_NONAUTHOR_REVIEW.json'),'overall_final_internal_review_pass':False}));return 0
if __name__=='__main__':sys.exit(main())
