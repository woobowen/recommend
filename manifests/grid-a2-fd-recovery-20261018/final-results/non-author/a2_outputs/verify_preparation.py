"""Audit real new SID / metadata / fixed32 bindings before result existence."""
from pathlib import Path
from collections import Counter
import ast,json,math
import numpy as np
from numeric_core import ident,load,read_rows,read_sid_tensor,sha,ranking,pair_metrics
ROOT=Path(__file__).resolve().parent
PREP=ROOT/'preparation-v2/raw'
JOBS=PREP/'jobs'
SOURCE=PREP/'scientific_source'
EXPECTED_SID='06cfa6ba0f64d6c68e0e94c4111edd51d086796e8cdb474ac3fcd424a4a2adf0'
EXPECTED_COHORT='0a2bbfbd62ab5bbd82b3032330e1f0f411af015d674fccc6b5f23c0ddacafe01'
CAPTURE='85b77f5e2a5b497fdc1f620649ba488fc783a245/22286966cd544773f21b96e4185d56c66afd7183fdc4e8888716d8ea1d77acb2'
def main():
    q=JOBS/'ga-20261018-sportsa2-quantize-fd-recovery-v1/output';metadata=JOBS/'ga-20261018-sportsa2-metadata-fd-recovery-v1/output'
    sidpath=q/'semantic_ids.pt';assert sha(sidpath)==EXPECTED_SID
    sid=read_sid_tensor(sidpath);legal=set(map(tuple,sid.T));assert len(legal)==18357 and ((sid>=0)&(sid<256)).all()
    quant=load(q/'RESULT.json');assert quant['sid']['sha256']==EXPECTED_SID and quant['capture_version']==CAPTURE
    result=load(metadata/'RESULT.json');assert result['sid']['sha256']==EXPECTED_SID and result['capture_version']==CAPTURE
    test=read_rows(metadata/'test_target_index.jsonl');evaluation=read_rows(metadata/'evaluation_target_index.jsonl')
    coverage=load(metadata/'test_coverage.json')
    assert sha(metadata/'test_target_index.jsonl')==coverage['target_index_sha256']
    assert coverage['split']=='testing' and coverage['targets']==35598 and coverage['original_label_targets_verified']
    assert coverage['complete_target_hidden_groups_verified'] and not coverage['model_execution']
    for rows in [test,evaluation]:
        assert len(rows)==35598 and len({r['sample_key'] for r in rows})==35598
        assert all(tuple(r['target']) in legal and len(r['target'])==4 for r in rows)
    cohortpath=metadata/'fixed_attention_samples.json';cohort=load(cohortpath)
    assert sha(cohortpath)==EXPECTED_COHORT and cohort['sid']['sha256']==EXPECTED_SID
    assert cohort['split']=='evaluation' and not cohort['model_outputs_inspected'] and cohort['selected_users']==32
    assert cohort['sample_keys']==[r['sample_key'] for r in evaluation[:32]]
    assert cohort['targets']==[r['target'] for r in evaluation[:32]]
    inputs=np.array(cohort['input_ids'],dtype=np.int64);mask=np.array(cohort['attention_mask'],dtype=np.int64)
    assert inputs.shape==mask.shape==(32,120) and np.isin(mask,[0,1]).all()
    grouped=mask.reshape(32,30,4);assert (grouped==grouped[:,:,:1]).all()
    positions=0
    for row,sample in enumerate(cohort['samples']):
        assert sample['sample_key']==cohort['sample_keys'][row] and sample['target']==cohort['targets'][row]
        assert sample['target_hidden'] and sample['input_dtype']==sample['target_dtype']=='torch.int64'
        for item,item_id in enumerate(sample['visible_history_item_ids']):
            visible=bool(mask[row,item*4])
            if visible:
                assert 0<=item_id<18357 and np.array_equal(inputs[row,item*4:item*4+4],sid[:,item_id])
            else:assert item_id==-1
            positions+=4
    # Actual training input remains separate from still-future formal outputs.
    train=JOBS/'ga-20261018-sportsa2-train-fd-recovery-v1/output'
    invocation=load(train/'invocation.json');spec=invocation['effective_spec'];used=load(train/'used_config.json');structure=load(train/'structure.json')
    assert spec['sid']['sha256']==EXPECTED_SID and spec['research_identity']=='Sports-A2-T5RKMeans-DualTable-NewTraining'
    assert invocation['code_identity']==CAPTURE and used['model']['should_check_prefix'] is False
    assert used['model']['top_k_for_generation']==10 and used['model']['shuffle_embedding_on_test'] is False
    assert used['trainer']['precision']=='32-true' and used['trainer']['accumulate_grad_batches']==16
    assert used['eval']['evaluator']['_target_']=='src.components.eval_metrics.SIDRetrievalEvaluator'
    assert used['eval']['evaluator']['top_k_list']==[5,10]
    assert structure['shared_sid_table'] is False and structure['parameters_total']==structure['parameters_trainable']==13259136
    assert structure['encoder_sid_table_shape']==structure['decoder_sid_table_shape']==[1024,128]
    deployment=load(SOURCE/'DEPLOYMENT.json');source_map={r['path']:r for r in deployment['files']}
    source_files=[]
    for p in sorted(SOURCE.rglob('*')):
        if p.is_file() and p.name!='DEPLOYMENT.json':
            rel=str(p.relative_to(SOURCE));assert sha(p)==source_map[rel]['sha256'];source_files.append(ident(p)|{'relative':rel})
    assert len(source_files)==20
    # Hand-computable CPU probes exercise rank hit/miss/discount and probability metrics.
    ids=[[i,0,0,0] for i in range(10)];scores=list(range(10,0,-1))
    fixture={'sids':ids,'scores':scores,'target':ids[5]};r=ranking(fixture)
    assert r['rank']==6 and r['metrics']['recall@5']==0 and r['metrics']['recall@10']==1
    assert abs(r['metrics']['ndcg@10']-1/math.log2(7))<1e-15
    fixture['target']=[999,999,999,999];assert ranking(fixture)['rank']==0
    x=np.array([1.,0.]);y=np.array([0.,1.]);m=pair_metrics(x,y)
    assert m['cosine']==0 and m['total_variation']==1 and abs(m['jensen_shannon_nats']-math.log(2))<1e-15
    for script in ROOT.glob('*.py'):ast.parse(script.read_text(),filename=str(script))
    out={'status':'A2_OUTPUT_REVIEW_PREPARATION_VERIFIED','not_final_scientific_result_acceptance':True,
        'reviewer':'/root/toys_result_audit','scientific_capture':CAPTURE,'SID':ident(sidpath)|{'shape':list(sid.shape),'unique_full_SIDs':len(legal)},
        'metadata_job':'ga-20261018-sportsa2-metadata-fd-recovery-v1','testing_targets':35598,'evaluation_targets':35598,
        'fixed32':ident(cohortpath)|{'split':'evaluation','selected_before_model_outputs':True,'exact_first32_from_evaluation_index':True,
             'all_visible_item_SID_pairings_verified':True,'positions_checked':positions},
        'actual_static_train_config':ident(train/'used_config.json'),'source_files':source_files,
        'pending':['generator training terminal/best checkpoint identity (owned by main/training reviewer)',
            'Normal/Shuffle3/Shuffle4 complete 35598-user predictions','three fixed32 actual generation attention sets',
            'CPU descriptive statistics final output','independent final output acceptance after terminal collection'],
        'CPU_math_probes_pass':True,'new_model_or_GPU_calls':0,'large_checkpoint_downloads':0,'server_mutations':0,
        'collector_note':'Initial local preparation listing hit the 18 MB guard before any transfer because gpu-gate recursively included engineering checkpoints. Failed 0-byte capture retained in preparation/. Second capture used explicit small gpu-gate files; no checkpoint downloaded.'}
    destination=ROOT/'preparation-verification.json';assert not destination.exists();destination.write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({'status':out['status'],'testing':35598,'fixed32_split':'evaluation','cohort_sha256':EXPECTED_COHORT,'source_files':20,'final_output_review':'PENDING'}))
if __name__=='__main__':main()
