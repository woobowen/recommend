"""Detached official-data -> Qwen SID + independent LLM2Vec -> statistics pipeline."""
import os
os.environ.update(OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',
                  TF_CPP_MIN_LOG_LEVEL='3',TF_NUM_INTRAOP_THREADS='1',TF_NUM_INTEROP_THREADS='1',
                  CUDA_VISIBLE_DEVICES='')
import argparse
from contextlib import ExitStack
import fcntl
import importlib.util
from pathlib import Path
import subprocess
import sys
import time
import traceback
from runtime_common import ROOT,HERE,sha,read_json,read_jsonl,write_json,write_jsonl,status

DATA=Path('/home/jcy/data/amazon2018_instruments')
GRID=ROOT/'environments/grid/bin/python'


def execute(run,name,cmd,env=None):
    folder=run/name;folder.mkdir(exist_ok=True)
    write_json(folder/'command.json',{'argv':list(map(str,cmd)),'visible_gpus':(env or os.environ).get('CUDA_VISIBLE_DEVICES','')})
    with (folder/'console.log').open('ab') as log:
        result=subprocess.run(list(map(str,cmd)),cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
    assert result.returncode==0,f'{name} exit={result.returncode}; see {folder}/console.log'


def prepare_configuration(run):
    inputs=HERE/'inputs/instrument'
    prep=read_json(HERE/'preparation/input_preparation_verified.json');n=prep['retained_items']
    assert n>=3072
    enc={'seed':20261113,'max_length':256,'doc_max_length':224,'batch_size':2,
         'output_dimension':4096,'instruction':'Represent this product for semantic similarity:',
         'dtype':'float16','pooling':'mean','l2_normalize':True,'expected_items':{'instrument':n},
         'model_download_manifest_sha256':sha(ROOT/'codex/llm2vec_sid_hierarchy_20261113/preparation/model_download_verified.json'),
         'inputs':{'instrument':[]}}
    for variant,filename in [('title','items_title.jsonl'),('title_brand','items_title_brand.jsonl')]:
        path=inputs/filename
        enc['inputs']['instrument'].append({'name':variant,'path':str(path),'sha256':sha(path)})
    write_json(run/'encoding_config.json',enc)
    # This encoding config deliberately contains no SID path or SID-dependent filtering.
    qwen={'items':n,'input_path':str(inputs/'items_qwen.jsonl'),'input_sha256':sha(inputs/'items_qwen.jsonl'),
          'model':'Qwen2.5-7B-Instruct','batch_size':8,'max_length':128,'seed':42,
          'pooling':'WMH MeanAggregation','padding':'right','quantizer_steps':30,
          'num_semantic_codes':3,'codebook_width':256,'normalize_residuals':True}
    write_json(run/'qwen_config.json',qwen)
    write_json(run/'input_audit.json',prep)
    write_json(run/'download_manifest.json',read_json(DATA/'download_manifest.json'))
    (run/'instrument').mkdir(exist_ok=True)
    return n


def start_encoder(run,filename,gpus,extra):
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpus)
    log=(run/(filename.replace('.py','')+'.log')).open('ab')
    command=[sys.executable,'-u',str(run/'source'/filename),'--run',str(run),*extra]
    process=subprocess.Popen(command,cwd=run/'source',env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
    log.close()
    return process


def quantize(run,n):
    module_spec=importlib.util.spec_from_file_location('original_qwen_pipeline',ROOT/'_运行/run_pipeline.py')
    module=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(module)
    common=[f'data_dir={DATA}',f'paths.root_dir={ROOT}','trainer.devices=1','++trainer.strategy=auto',
            'trainer.precision=32-true','extras.print_config=false',f'embedding_path={run}/upstream/item_embeddings.pt',
            'embedding_dim=3584','num_hierarchies=3','codebook_width=256','seed=42']
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',PYTHONUNBUFFERED='1',HYDRA_FULL_ERROR='1')
    for key in ['PROJECT_ROOT','METADATA_DIR','LOCAL_RANK','RANK','WORLD_SIZE','MASTER_ADDR','MASTER_PORT']:
        env.pop(key,None)
    def workers(prefix):
        return [f'++{prefix}num_workers=0',f'++{prefix}timeout=0',f'++{prefix}persistent_workers=false']
    for inference in [False,True]:
        name='sid_generation' if inference else 'quantizer_train'
        status(run/'experiment_status.json',name,physical_gpu=0)
        args=['experiment=rkmeans_inference_flat',f'ckpt_path={checkpoint}','callbacks.bq_writer=null'] if inference else ['experiment=rkmeans_train_flat','train=true','test=false','ckpt_path=null']
        prefix='predict_dataloader_config.' if inference else 'train_dataloader_config.'
        command=[GRID,ROOT/'_运行/grid_stage.py',*args,'++callbacks.restart_job=null',*common,
                 *workers('data_loading.datamodule.'+prefix),f'hydra.run.dir={run/name}','++trainer.enable_progress_bar=false']
        with module.reserve_stage_gpus(['0'],lambda reason:status(run/'experiment_status.json','waiting_for_quantizer_gpu',reason=reason)):
            execute(run,name,command,env)
        if not inference:
            result=read_json(run/name/'stage_result.json');checkpoint=Path(result['checkpoint']).resolve()
            assert checkpoint.is_relative_to(run) and checkpoint.exists() and result['global_step']==30
    import torch
    tensor=torch.load(run/'sid_generation/pickle/merged_predictions_tensor.pt',map_location='cpu',weights_only=False).long()
    assert list(tensor.shape)==[4,n] and tensor.min()>=0 and tensor.max()<256
    assert len(torch.unique(tensor.T,dim=0))==n
    torch.save(tensor,run/'upstream/semantic_ids.pt')
    sid_path=HERE/'inputs/instrument/item_sids.jsonl'
    write_jsonl(sid_path,[{'item_id':i,'sid':sid} for i,sid in enumerate(tensor.T.tolist())])
    audit={'complete':True,'items':n,'sid_shape':list(tensor.shape),'unique_full_sids':n,
           'unique_three_code_prefixes':len(torch.unique(tensor[:3].T,dim=0)),
           'sid_sha256':sha(run/'upstream/semantic_ids.pt'),'qwen_embedding_sha256':sha(run/'upstream/item_embeddings.pt'),
           'quantizer_checkpoint':str(checkpoint),'quantizer_checkpoint_sha256':sha(checkpoint),
           'quantizer_steps':30,'quantizer_seed':42,'width':256,'normalize_residuals':True,
           'model_unchanged':True,'downstream_recommendation_training':False,
           'digit_ranges':[{'min':int(x.min()),'max':int(x.max()),'used':len(x.unique())} for x in tensor]}
    write_json(run/'upstream/upstream_audit.json',audit)
    write_json(run/'analysis_config.json',{'seed':20261113,'epsilon':1e-6,'minimum_group_size':2,
        'threshold_sensitivity_runs':False,'primary_transition':'1_to_2','third_level_requires_at_least_items':2,
        'variants':['title','title_brand'],'permutations':999,'bootstrap_repetitions':2000,
        'path_condition':'S1<S2<=S3','datasets':{'instrument':{'items':n,'sid_path':str(sid_path),'sid_sha256':sha(sid_path)}},
        'primary_test_family':'one new instrument/title/delta12 test; no retroactive pooling with prior datasets',
        'no_holm_needed_for_single_primary_test':True})


def verify(run):
    import numpy as np
    results={}
    rows=read_jsonl(HERE/'inputs/instrument/item_sids.jsonl')
    for variant in ['title','title_brand']:
        folder=run/'instrument'/variant;out=folder/'evaluation';summary=read_json(out/'summary.json')
        assert sha(out/'summary.json')==read_json(out/'evaluation_complete.json')['summary_sha256']
        z=np.load(folder/'embeddings_normalized.npy',mmap_mode='r')
        assert np.isfinite(z).all() and max(abs(np.linalg.norm(z.astype(np.float64),axis=1)-1))<1e-5
        nodes=read_jsonl(out/'prefix_nodes.jsonl');edges=read_jsonl(out/'parent_child_edges.jsonl')
        paths=read_jsonl(out/'complete_paths.jsonl');gaps=read_jsonl(out/'branch_gaps.jsonl')
        def close(a,b):assert abs(a-b)<1e-10,(a,b)
        checked=0
        for depth in [1,2,3]:
            eligible=[a for a in nodes if a['depth']==depth and a['items']>=2]
            close(np.mean([a['mean_pairwise_cosine'] for a in eligible]),summary['node_macro_cosines'][str(depth)])
            close(sum(a['items'] for a in eligible)/len(rows),summary['coverage'][str(depth)]['coverage'])
            for node in sorted(eligible,key=lambda a:-a['items'])[:3]:
                ids=[i for i,row in enumerate(rows) if row['sid'][:depth]==node['prefix']]
                block=z[ids].astype(np.float64);gram=block@block.T
                close(float(gram[np.triu_indices(len(ids),1)].mean()),node['mean_pairwise_cosine']);checked+=1
        for tag in ['12','23']:
            transition=tag[0]+'_to_'+tag[1]
            eligible=[a for a in edges if a['transition']==transition and a['eligible_proper_refinement']]
            close(np.mean([a['delta'] for a in eligible]),summary['observed']['delta_'+tag])
            close(np.mean([a['delta']>1e-6 for a in eligible]),summary['observed']['positive_'+tag])
            close(np.mean([a['gap'] for a in gaps if a['transition']==transition]),summary['observed']['gap_'+tag])
        eligible=[a for a in paths if a['proper_both_edges']]
        close(np.mean([a['monotonic_as_image'] for a in eligible]),summary['observed']['path_monotonic'])
        assert len(read_jsonl(out/'permutation_statistics.jsonl'))==1998
        results[variant]={'passed':True,'explicit_pairwise_nodes_checked':checked,'all_aggregates_and_coverages_match':True,
                          'normalized_finite_vectors':True,'summary_hash_valid':True,'permutation_records':1998}
    write_json(run/'final_verification.json',{'passed':True,'results':results})


def worker(run):
    source=run/'source';launch=read_json(run/'launch.json')
    for name,digest in launch['source_sha256'].items():assert sha(source/name)==digest,'Frozen source changed'
    for path,digest in launch['original_core_sha256'].items():assert sha(path)==digest,'Original protocol code changed'
    try:
        status(run/'experiment_status.json','data_validation_and_preparation')
        execute(run,'prepare_data',[sys.executable,'-u',source/'download_prepare.py'])
        n=prepare_configuration(run)
        execute(run,'write_item_records',[GRID,'-u',source/'write_item_records.py','--inputs',HERE/'inputs/instrument','--data',DATA])
        execute(run,'metric_unit_tests',[sys.executable,source/'test_metrics.py'])
        status(run/'experiment_status.json','parallel_qwen_llm2vec_encoding',items=n,gpu_allocation={'qwen':[0,1],'llm2vec':[2,3]})
        qwen=start_encoder(run,'qwen_worker.py','0,1',[])
        llama=start_encoder(run,'encoder_worker.py','2,3',['--dataset','instrument'])
        write_json(run/'processes.json',{'coordinator_pid':os.getpid(),'qwen_pid':qwen.pid,'llm2vec_pid':llama.pid})
        while qwen.poll() is None:
            assert llama.poll() in [None,0],'LLM2Vec failed; inspect encoder_worker.log'
            time.sleep(5)
        assert qwen.returncode==0,'Qwen encoding failed; inspect qwen_worker.log'
        quantize(run,n)
        status(run/'experiment_status.json','waiting_for_llm2vec_before_evaluation',qwen_and_sid_complete=True)
        while llama.poll() is None:time.sleep(5)
        assert llama.returncode==0,'LLM2Vec encoding failed; inspect encoder_worker.log'
        status(run/'experiment_status.json','cpu_evaluation')
        execute(run,'evaluation',[sys.executable,'-u',source/'evaluate_worker.py','--run',run,'--dataset','instrument'])
        status(run/'experiment_status.json','final_verification')
        verify(run)
        execute(run,'report',[sys.executable,'-u',source/'build_report.py','--run',run])
        for path,digest in launch['original_core_sha256'].items():assert sha(path)==digest,'Original code unexpectedly changed'
        status(run/'experiment_status.json','complete',report=str(run/'Instrument_语义层级实验结果.md'),
               completed_unix=time.time(),original_core_code_unchanged=True,all_gpus_released=True)
    except BaseException as exc:
        status(run/'experiment_status.json','failed',error_type=type(exc).__name__,error=str(exc))
        traceback.print_exc();raise


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--launch',action='store_true');parser.add_argument('--run',type=Path)
    args=parser.parse_args()
    if not args.launch:
        assert args.run;worker(args.run);return
    required=['Musical_Instruments_5.json.gz','meta_Musical_Instruments.json.gz']
    for name in required:assert (DATA/'raw'/name).exists(),'Transfer raw files first'
    previous=HERE/'latest_run.json'
    if previous.exists():
        old=read_json(previous);pid=old['coordinator_pid']
        if Path(f'/proc/{pid}').exists():raise RuntimeError('Existing coordinator still alive; inspect it instead of duplicating')
    with ExitStack() as stack:
        for gpu in range(4):
            lock=stack.enter_context((ROOT/'stage_gpu_locks'/f'gpu_{gpu}.lock').open('a'));fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used','--format=csv,noheader,nounits'],text=True)
        busy=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True)
        for row in rows.splitlines():
            index,uuid,used=[a.strip() for a in row.split(',')]
            assert int(used)<512 and uuid not in busy,f'GPU {index} is occupied'
        run=HERE/'runs'/time.strftime('%Y%m%d_%H%M%S');run.mkdir(parents=True,exist_ok=False)
        source=run/'source';source.mkdir()
        import shutil
        hashes={}
        for path in sorted(HERE.glob('*.py')):
            shutil.copy2(path,source/path.name);hashes[path.name]=sha(source/path.name)
        core=[ROOT/'_运行/grid_stage.py',ROOT/'_运行/run_pipeline.py',ROOT/'configs/experiment/rkmeans_train_flat.yaml',
              ROOT/'configs/experiment/rkmeans_inference_flat.yaml',ROOT/'src/modules/clustering/residual_quantization.py',
              ROOT/'src/models/modules/clustering/mini_batch_kmeans.py']
        write_json(run/'launch.json',{'user_authorized':True,'started_unix':time.time(),'run':str(run),
             'gpu_allocation':{'qwen':[0,1],'llm2vec':[2,3],'quantizer':[0],'statistics':'CPU'},
             'source_sha256':hashes,'original_core_sha256':{str(p):sha(p) for p in core},
             'threshold_sensitivity_runs':False,'primary_transition':'1_to_2','only_one_new_primary_test':True})
    with (run/'coordinator.log').open('ab') as log:
        process=subprocess.Popen([sys.executable,'-u',str(source/'run_instrument.py'),'--run',str(run)],
              cwd=source,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    receipt={'run':str(run),'coordinator_pid':process.pid,'started_unix':time.time()}
    write_json(HERE/'latest_run.json',receipt);print(receipt,flush=True)


if __name__=='__main__':main()
