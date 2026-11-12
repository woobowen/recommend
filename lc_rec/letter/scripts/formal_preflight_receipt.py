"""CPU-only receipt for terminal final preflight, measured budget and inputs."""
import argparse
import datetime as dt
import json
import os
from pathlib import Path
import statistics
import subprocess
from formal_train import write,sha,verify_checkpoint
from formal_ledger import reconcile


def main():
    p=argparse.ArgumentParser();p.add_argument('--job',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and 'DS_ACCELERATOR' not in os.environ
    import torch
    assert not torch.cuda.is_initialized()
    base=Path('/data/bwwu/lc-rec/a0-normal-20261111');run=base/a.job;training=run/'training'
    job=Path('/home/bwwu/grid-audits/sid-extensions-p0/jobs')/a.job
    status=json.loads((job/'status.json').read_text())
    assert status['status']=='SUCCEEDED' and status['exit_code']==0 and status['cleanup']['verified']
    ledger=reconcile(base/'formal_det_preflight_20261112_ledger.json')
    charged=sum(x['gpu_device_seconds'] for x in ledger['attempts']);assert charged<=3600
    reports=[json.loads((training/('rank%d.json'%r)).read_text()) for r in [0,1]]
    commit=reports[0]['run_commit'];config_sha=reports[0]['config_sha256']
    events=[[json.loads(x) for x in (training/('events.rank%d.jsonl'%r)).read_text().splitlines()] for r in [0,1]]
    measured=[]
    for r,(report,rows) in enumerate(zip(reports,events)):
        assert report['status']=='PROBE_COMPLETE' and report['global_step']==12
        assert report['run_commit']==commit and report['config_sha256']==config_sha
        assert '--diagnosis' not in report['argv'] and report['formal_attention']['tensor_observers'] is False
        calls=report['formal_attention']['calls'];assert calls['native_varlen']==calls['extension_varlen']==1536
        assert any(x['event']=='formal_fa_backward_dispatch' and x['deterministic'] is True for x in rows)
        assert any(x['event']=='bnb_hot_scb_alias_loaded' and x['hot_modules']==224 for x in rows)
        assert any(x['event']=='strict_ds_restored' and x['global_step']==12 for x in rows)
        assert report['optimizer_membership_exact'] and report['training_arguments']['save_total_limit']==2
        steps={x['step']:x for x in rows if x['event']=='optimizer_step'}
        durations=[steps[i]['elapsed_seconds']-steps[i-1]['elapsed_seconds'] for i in range(4,13)]
        measured.append({'rank':r,'steady_update_seconds':durations,'median_seconds':statistics.median(durations),
                         'max_seconds':max(durations),'train_peak_allocated_bytes':report['peak_allocated_bytes'],
                         'train_peak_reserved_bytes':report['peak_reserved_bytes'],
                         'checkpoint_seconds':[x['elapsed_seconds_checkpoint'] for x in rows if x['event']=='checkpoint_complete'],
                         'validation':[x['metrics'] for x in rows if x['event']=='preflight_validation']})
    checkpoints={x.name:verify_checkpoint(x,commit,config_sha) for x in training.glob('checkpoint-*')}
    assert set(checkpoints)=={'checkpoint-2','checkpoint-12'}
    expected=torch.load(training/'checkpoint-12/adapter_model.bin',map_location='cpu',weights_only=True)
    exported=torch.load(training/'preflight_adapter/adapter_model.bin',map_location='cpu',weights_only=True)
    assert set(expected)==set(exported) and len(expected)==450
    assert all(torch.equal(expected[k],exported[k]) for k in expected)
    del expected,exported
    tests=[json.loads((run/'test_preflight'/('rank%d.json'%r)).read_text()) for r in [0,1]]
    for v in tests:
        assert v['status']=='PREFLIGHT_INTERFACE_PASS' and v['completed_users']==8
        assert v['strict_adapter_reload']['keys']==450
        assert v['generation']['num_beams']==v['generation']['num_return_sequences']==20
    train_hours=8240*max(x['median_seconds'] for x in measured)/3600
    valid_hours=49544*4*max(v['eval_runtime'] for x in measured for v in x['validation'])/32/3600
    test_hours=12386*max(v['elapsed_generation_seconds'] for v in tests)/8/3600
    checkpoint_hours=4*max(v for x in measured for v in x['checkpoint_seconds'])/3600
    reload_init_hours=.25
    conditional=(train_hours+valid_hours+test_hours+checkpoint_hours+reload_init_hours)*1.25
    assert conditional<96
    result={'status':'PASS','scope':'terminal bounded preflight; not four epochs or full test',
            'preflight_run_commit':commit,'config_sha256':config_sha,'job_id':a.job,
            'actual_runner_status':status,'ledger':ledger,'charged_gpu_device_minutes':charged/60,
            'measured':measured,'test_interfaces':[{k:v[k] for k in ['status','completed_users','elapsed_generation_seconds','gpu_peak_allocated_bytes','gpu_peak_reserved_bytes','model_load_and_strict_verification_seconds','strict_adapter_reload']} for v in tests],
            'selected_checkpoint_adapter_export_450_tensors_exact':True,
            'checkpoints':checkpoints,'conditional_budget_hours':{'train':train_hours,'validation_small_probe_extrapolation':valid_hours,
              'beam20_small_probe_extrapolation':test_hours,'four_checkpoint_writes':checkpoint_hours,
              'reload_initialization_allowance':reload_init_hours,'margin_fraction':.25,'total':conditional},
            'budget_limits':['Few training steps and sixteen test users do not calibrate long-run variability',
                             'Validation extrapolation includes small-batch startup/rank timing differences',
                             'Use max-rank time, never sum ranks as duplicate wall time; 96h is a hard cap, not a target'],
            'utc':dt.datetime.now(dt.timezone.utc).isoformat(),
            'resources':{k:subprocess.check_output(v,text=True) for k,v in {'gpu':['nvidia-smi','--query-gpu=index,uuid,memory.used,memory.total,utilization.gpu','--format=csv'],'ram':['free','-b'],'disk':['df','-B1','/data/bwwu','/home/bwwu']}.items()}}
    assert not torch.cuda.is_initialized();write(a.output,result)
    print(json.dumps({'status':'PASS','output':a.output,'gpu_device_minutes':charged/60,'conditional_total_hours':conditional}))


if __name__=='__main__':main()
