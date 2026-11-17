"""Freeze the authorized protocol, reserve idle pairs, and detach one run."""
import argparse
from contextlib import ExitStack
import fcntl
from pathlib import Path
import shutil
import subprocess
import sys
import time
from runtime_common import ROOT,HERE,read_json,sha,write_json


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--execute',action='store_true');args=ap.parse_args()
    assert args.execute, 'Pass --execute only for an authorized launch'
    assert read_json(HERE/'preparation/model_download_verified.json')['complete']
    assert read_json(HERE/'preparation/smoke_check_verified.json')['complete']
    subprocess.run([sys.executable,str(HERE/'test_metrics.py')],check=True,cwd=HERE)
    allocation={'beauty':[0,1],'toys':[2,3]}
    with ExitStack() as stack:
        for index in range(4):
            lock=stack.enter_context((ROOT/'stage_gpu_locks'/f'gpu_{index}.lock').open('a'))
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free','--format=csv,noheader,nounits'],text=True)
        occupied=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True)
        for row in rows.splitlines():
            index,uuid,free=[v.strip() for v in row.split(',')]
            if int(index) in range(4):assert int(free)>=10240 and uuid not in occupied,f'GPU {index} occupied'
        stamp=time.strftime('%Y%m%d_%H%M%S');run=HERE/'runs'/stamp;run.mkdir(parents=True,exist_ok=False)
        source=run/'source';source.mkdir()
        names=['runtime_common.py','encoder_worker.py','semantic_metrics.py','evaluate_worker.py','run_experiment.py','test_metrics.py']
        hashes={}
        for name in names:shutil.copy2(HERE/name,source/name);hashes[name]=sha(source/name)
        expected={'beauty':12101,'toys':11924}
        encoding={'seed':20261113,'max_length':256,'doc_max_length':224,'batch_size':2,
                  'output_dimension':4096,'instruction':'Represent this product for semantic similarity:',
                  'dtype':'float16','pooling':'mean','l2_normalize':True,'expected_items':expected,'inputs':{},
                  'model_download_manifest_sha256':sha(HERE/'preparation/model_download_verified.json')}
        analysis={'seed':20261113,'epsilon':1e-6,'minimum_group_size':2,'threshold_sensitivity_runs':False,
                  'primary_transition':'1_to_2','third_level_requires_at_least_items':2,
                  'variants':['title','title_brand'],'permutations':999,'bootstrap_repetitions':2000,
                  'path_condition':'S1<S2<=S3','datasets':{}}
        for dataset in expected:
            encoding['inputs'][dataset]=[]
            for variant,filename in [('title','items_title.jsonl'),('title_brand','items_title_brand.jsonl')]:
                path=HERE/'inputs'/dataset/filename
                encoding['inputs'][dataset].append({'name':variant,'path':str(path),'sha256':sha(path)})
            path=HERE/'inputs'/dataset/'item_sids.jsonl'
            analysis['datasets'][dataset]={'items':expected[dataset],'sid_path':str(path),'sid_sha256':sha(path)}
            (run/dataset).mkdir()
        write_json(run/'encoding_config.json',encoding);write_json(run/'analysis_config.json',analysis)
        launch={'started_unix':time.time(),'gpu_allocation':allocation,'source_sha256':hashes,
                'user_authorized_full_experiment':True,'primary_evidence':'first_to_second',
                'no_threshold_sensitivity':True,'run':str(run)}
        write_json(run/'launch.json',launch)
    # Workers enforce the same nonblocking locks for their entire encoding stage.
    with (run/'coordinator.log').open('ab') as log:
        process=subprocess.Popen([sys.executable,'-u',str(source/'run_experiment.py'),'--run',str(run)],
                                 cwd=source,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    write_json(HERE/'latest_run.json',{'run':str(run),'coordinator_pid':process.pid,'started_unix':time.time()})
    print({'run':str(run),'coordinator_pid':process.pid,'gpu_allocation':allocation},flush=True)


if __name__=='__main__':main()
