"""Independently reconstruct all decoder query identities from raw beam arrays."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

REPO=Path('/home/addaswsw/recommend-worktrees/sid-extensions-p0')
sys.path.insert(0,str(REPO))
from experiments.beauty_extensions.attention.records import load_bundle


def identity(path):
    data=Path(path).read_bytes()
    return {'path':str(path),'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}


def read(path):
    return json.loads(Path(path).read_bytes())


def check(record):
    assert identity(record['path'])==record, record['path']


def audit(path):
    report=read(path)
    assert report['status']=='DERIVED_DECODER_QUERY_TOKEN_IDENTITY_COMPLETE'
    assert report['T5Stack_inputs_embeds_direct_capture_claimed'] is False
    assert report['raw_NPZ_or_PNG_modified'] is False and report['new_model_or_GPU_calls']==0
    for name in ('program','invocation','original_attention_acceptance','source_SID_dictionary'):
        check(report[name])
    for record in report['byte_bound_source_mapping_inputs']+report['unchanged_raw_bundle_inputs']:
        check(record)
    attention=read(report['original_attention_acceptance']['path'])
    invocation_path=Path(report['invocation']['path']);invocation=read(invocation_path)
    root=invocation_path.parent.parent
    spec=invocation['spec'];deployment=invocation['code_identity']
    assert (report['domain'],report['route'],report['shuffle_n'])==(spec['domain'],spec['route'],spec['shuffle_n'])
    assert attention['status']=='INDEPENDENT_FORMAL_ATTENTION_RESULT_PASS'
    assert attention['job_id']==report['job_id']
    assert attention['invocation']==report['invocation']
    check(attention['accepted_evaluation_review'])
    evaluation=read(attention['accepted_evaluation_review']['path'])
    assert evaluation['status']=='FORMAL_EVALUATION_ACCEPTANCE_PASS'
    assert evaluation['legal_vocabulary']['sha256']==report['source_SID_dictionary']['sha256']
    used=read(root/'output/used_config.json')
    dictionary=read(report['source_SID_dictionary']['path'])
    assert dictionary['source']==spec['sid']
    width=1+max(code for row in dictionary['legal_raw_sids'] for code in row)
    assert width==used['model']['decoder']['embed_tokens']['num_embeddings']
    assert report['raw_SID_maximum']==width-1
    assert report['source_commit']==deployment['source_commit'] and report['snapshot_sha256']==deployment['snapshot_sha256']
    source_map={record['path']:record for record in deployment['files']}
    for record in report['byte_bound_source_mapping_inputs']:
        relative=str(Path(record['path']).relative_to(REPO))
        assert all(record[key]==source_map[relative][key] for key in ('bytes','sha256'))
    actual=report['sample_query_identities'];offset=0;queries_checked=0
    record_paths=sorted((root/'output').glob('batch-*/bundle/record.json'))
    assert len(record_paths)==4 and len(report['unchanged_raw_bundle_inputs'])==8
    for record_path in record_paths:
        bundle=load_bundle(record_path)
        steps=bundle['steps'];assert len(steps)==4
        full_parents=[]
        for h,step in enumerate(steps):
            assert step['hierarchy']==h
            if h==0:
                parents=np.zeros(step['generated_ids'].shape[:2],dtype=np.int64)
            else:
                previous=steps[h-1]['generated_ids']
                candidates=step['generated_ids'][...,:h]
                matching=(candidates[:,:,None,:]==previous[:,None,:,:]).all(-1)
                assert (matching.sum(-1)==1).all()
                parents=matching.argmax(-1)
            assert np.array_equal(parents,step['parent_indices'])
            full_parents.append(parents)
        batch_size=len(bundle['capture']['sample_keys'])
        assert batch_size==8
        selected=np.zeros(batch_size,dtype=np.int64);ancestry=np.empty((batch_size,4),dtype=np.int64)
        tokens=np.empty((batch_size,4),dtype=np.int64)
        for h in (3,2,1,0):
            tokens[:,h]=steps[h]['generated_ids'][np.arange(batch_size),selected,-1]
            selected=full_parents[h][np.arange(batch_size),selected]
            ancestry[:,h]=selected
        assert np.array_equal(tokens,bundle['final_generated_ids'][:,0])
        assert np.array_equal(ancestry,bundle['paths']['final_parent_indices'])
        for sample,key in enumerate(bundle['capture']['sample_keys']):
            row=actual[offset];offset+=1
            assert row['sample_key']==key and row['source_record']==str(record_path)
            assert row['final_Top1_raw_SID']==tokens[sample].tolist()
            assert set(row['paths'])=={'final_top1','local_first'}
            for name in ('final_top1','local_first'):
                assert len(row['paths'][name])==4
                for h,observed in enumerate(row['paths'][name]):
                    parent=int(ancestry[sample,h]) if name=='final_top1' else 0
                    prefix=[] if h==0 else steps[h-1]['generated_ids'][sample,parent].tolist()
                    token=None if h==0 else prefix[-1]
                    expected={'target_generated_code_1based':h+1,'preselection_parent_beam_index':parent,
                        'raw_parent_SID_prefix':prefix,
                        'query_kind':'learned decoder.bos_token vector; no SID integer' if h==0 else 'last raw SID token in saved selected-parent prefix',
                        'raw_query_SID_integer':token,'query_source_code_position_0based':None if h==0 else h-1,
                        'derived_embedding_lookup_index':None if h==0 else token+(h-1)*width,
                        'num_embeddings_per_hierarchy':width,
                        'actual_observed_cross_attention_query_length':steps[h]['query_length_before_last_selection'],
                        'actual_observed_cross_attention_last_query_index':steps[h]['last_query_index'],
                        'identity_basis':'saved beam prefixes + independently checked parent ancestry + byte-bound decoder source mapping; not direct T5Stack inputs_embeds capture'}
                    assert observed==expected,(key,name,h)
                    assert steps[h]['query_length_before_last_selection']==(2 if h==1 else 1)
                    assert steps[h]['last_query_index']==(1 if h==1 else 0)
                    if name=='final_top1':assert prefix==tokens[sample,:h].tolist()
                    elif h:assert prefix==steps[h]['local_first_prefix'][sample].tolist()
                    queries_checked+=1
    assert offset==len(actual)==report['samples']==32
    assert queries_checked==report['total_query_identity_records']==256
    assert report['paths_per_sample']==2 and report['queries_per_path']==4
    return {'job_id':report['job_id'],'domain':spec['domain'],'route':spec['route'],'shuffle_n':spec['shuffle_n'],
        'report':identity(path),'complete_users':32,'complete_query_identity_records':256,
        'all_raw_beam_parents_independently_reconstructed':True,'both_path_every_query_identity_exact':True,
        'BOS_has_no_fabricated_integer_ID':True,'actual_query_lengths_and_last_indices_checked':True,
        'embedding_table_width':width,'derived_vs_direct_observer_scope_preserved':True}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reports',required=True,nargs='+',type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args();assert not args.output.exists()
    reviewed=[audit(path) for path in args.reports]
    assert len({item['job_id'] for item in reviewed})==len(reviewed)
    receipt={'status':'INDEPENDENT_DECODER_QUERY_IDENTITIES_PASS','reviewer_task':'/root/source_audit',
        'reviewed_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'program':identity(Path(__file__)),
        'conditions':reviewed,'condition_count':len(reviewed),'total_query_identity_records':256*len(reviewed),
        'method':'Independently reconstruct every native beam parent and reverse final Top1 ancestry from raw generated SID arrays, then derive every raw selected-parent prefix/query and offset index; compare all saved CPU-derived identity records.',
        'scientific_scope':'These are source-derived query identities from actual saved raw beam arrays, with observed cross-attention query lengths. T5Stack input_ids/inputs_embeds were not separately captured.',
        'new_model_GPU_or_evaluator_calls':0,'original_NPZ_PNG_or_source_writes':0}
    args.output.write_text(json.dumps(receipt,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps(identity(args.output)|{'status':receipt['status'],'conditions':len(reviewed)}))
