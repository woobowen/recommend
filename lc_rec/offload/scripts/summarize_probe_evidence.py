#!/usr/bin/env python3
"""Read terminal raw evidence; distinguish real model updates, pressure, and formal gates."""
import argparse
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text()) if path.is_file() else None


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evidence',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();root=a.evidence
    status=read(root/'status.json');chain=read(root/'output/chain-status.json')
    assert status and status['status'] not in ('RUNNING','ACCEPTED')
    result={'kind':'TERMINAL_RAW_EVIDENCE_DERIVATION','job_id':status['job_id'],
            'runner_status':status['status'],'runner_exit_code':status.get('exit_code'),
            'RUN_COMMIT':chain['RUN_COMMIT'] if chain else None,'chain_status':chain,
            'candidates':[],'formal_training_runs':0,'formal_six_task_updates':0,
            'save_resume_export_generation':'NOT TESTED','strict_one_to_one':'BLOCKED by documented data, initialization, loss grouping and save-selection gates'}
    for name in ['optimizer-only','optimizer-parameter']:
        base=root/'output'/name
        if not base.exists():continue
        reports=[read(q) for q in sorted((base/'probe').glob('probe_rank*.json'))]
        resources=read(base/'resource-summary.json')
        rawsamples=base/'resource-samples.jsonl'
        samples=[json.loads(s) for s in rawsamples.read_text().splitlines()] if rawsamples.exists() else []
        candidate={'candidate':name,'resource_summary':resources,'ranks':[],'sampled_gpu_max_mib':{}}
        for sample in samples:
            for line in sample.get('nvidia_smi','').splitlines():
                fields=[x.strip() for x in line.split(',')]
                if len(fields)>=2:
                    try:candidate['sampled_gpu_max_mib'][fields[0]]=max(candidate['sampled_gpu_max_mib'].get(fields[0],0),int(fields[1]))
                    except ValueError:pass
        for d in reports:
            groups=d.get('trainer_optimizer_groups_before_cpuadam_mapping',[])
            names=[n for g in groups for n in g.get('names',[])]
            updates=d.get('optimizer_updates',[])
            losses=d.get('losses',[])
            batch=d.get('actual_micro_batches',[])
            timed=[]
            for index in range(len(losses)//64):
                rows=losses[index*64:(index+1)*64]
                timed.append({'update_index':index+1,'local_training_step_wall_seconds_including_audit':sum(r.get('training_step_wall_seconds',0) for r in rows),
                              'forward_wall_seconds':sum(r.get('forward_wall_seconds_including_pending_dependencies',0) for r in rows)})
            candidate['ranks'].append({'rank':d['rank'],'raw_status':d['status'],'phase':d['phase'],
              'errors':d.get('errors',[]),'model':d.get('model'),'tokenizer_new_tokens':d.get('tokenizer',{}).get('new_tokens'),
              'optimizer_parameter_names_count':len(names),'optimizer_parameter_names_unique':len(set(names)),
              'all_recorded_parameter_objects_in_optimizer':bool(d.get('model')) and len(set(names))==d['model']['parameter_objects'] and len(names)==len(set(names)),
              'optimizer_initial_state':d.get('initial_optimizer_state'),
              'gradient_accumulation_buffer':d.get('gradient_accumulation_buffer'),
              'base_weight_samples_all_exact':bool(d.get('base_weight_sample_verification')) and all(x['exact_sample_match'] for x in d['base_weight_sample_verification']),
              'local_forwards':len(losses),'all_losses_finite':bool(losses) and all(r['finite'] for r in losses),
              'actual_micro_records':len(batch),'shifted_supervised_counts':sorted(set(sum(v!=-100 for v in r['labels'][1:]) for r in batch)),
              'actual_nonpadding_token_range':[min((r['nonpadding_tokens'] for r in batch),default=0),max((r['nonpadding_tokens'] for r in batch),default=0)],
              'max_actual_history_items':max((r['history_items'] for r in batch),default=0),
              'optimizer_updates':[{'index':x['update_index'],'global_steps_before':x['engine_global_steps_before'],
                 'global_steps_after':x.get('engine_global_steps_after'),'lr_before':x['lr_before'],
                 'all_rank_gradients_finite':x['all_rank_gradients_finite'],
                 'gradient_scan':x.get('gradient_finite_scan'),'sample_witnesses':x.get('global_sample_witnesses'),
                 'global_grad_norm':x.get('actual_global_grad_norm'),'overflow':x.get('overflow_flag'),
                 'engine_step_wall_seconds':x.get('engine_step_wall_seconds_including_cpu_optimizer_and_partition_copies'),
                 'audit_boundary_wall_seconds':x.get('elapsed_seconds_in_audited_boundary')} for x in updates],
              'update_group_timings':timed,'gpu_peak_after_real_updates':d.get('gpu_peak_after_real_updates'),
              'gpu_allocator_peak':d.get('gpu_allocator_peak'),'process_resources':d.get('process_resources'),
              'synthetic_pressure':d.get('synthetic_pressure'),'elapsed_wall_seconds':d.get('elapsed_wall_seconds'),
              'actual_config':d.get('actual_deepspeed_config'),'trainer_result':d.get('trainer_result')})
        candidate['complete_real_update_chain']=len(reports)==2 and all(
             len(d.get('optimizer_updates',[]))==2 and d['optimizer_updates'][-1].get('engine_global_steps_after')==2
             and all(all(x.values()) for x in d['optimizer_updates'][-1].get('global_sample_witnesses',{}).values())
             and bool(d['optimizer_updates'][-1].get('global_sample_witnesses'))
             and d['optimizer_updates'][-1]['all_rank_gradients_finite'] for d in reports)
        candidate['effective_batch128_real_path_pass']=candidate['complete_real_update_chain'] and all(len(d.get('losses',[]))==128 for d in reports)
        tau=[r['update_group_timings'][1]['local_training_step_wall_seconds_including_audit'] for r in candidate['ranks'] if len(r['update_group_timings'])>=2]
        if len(tau)==2:
            seconds=max(tau)
            candidate['conditional_time_substitution']={'same_seqrec_path_seconds_per_global128_update':seconds,
                'one_seed_6912_updates_hours':seconds*6912/3600,'five_seeds_34560_updates_hours':seconds*34560/3600,
                'scope':'Conditional substitution of second real SeqRec update only; includes audit work; not measured six-task throughput or a promised bound',
                'excluded':'model load, full validation, checkpoint I/O (possibly repeated), best reload, export, seventeen-template beam test'}
        result['candidates'].append(candidate)
    ledger=read(root/'gpu-budget.json');result['gpu_budget_ledger']=ledger
    if ledger:
        result['conservative_gpu_wall_seconds']=ledger['prior_stage_a_seconds']+sum(r.get('elapsed_seconds',0) for r in ledger['runs'])
    result['real_model_update_status']='PASS' if any(c['complete_real_update_chain'] for c in result['candidates']) else 'BLOCKED'
    result['batch128_observed_seqrec_status']='PASS' if any(c['effective_batch128_real_path_pass'] for c in result['candidates']) else 'NOT TESTED'
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({'path':str(a.output),'real_model_update_status':result['real_model_update_status'],'batch128_observed_seqrec_status':result['batch128_observed_seqrec_status']}))

if __name__=='__main__':main()
