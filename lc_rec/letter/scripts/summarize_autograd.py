#!/usr/bin/env python3
"""Summarize already-exported continuation evidence; no model, data audit or GPU."""
import hashlib
import json
from pathlib import Path
import statistics
import subprocess


def main():
    root=Path(__file__).resolve().parents[1]
    repo=root.parents[1]
    evidence=root/'evidence/autograd_20261110'
    run='8e6399c2a1269dfde58d705eef25a499400dc59b'
    def read(path):
        return json.loads(path.read_text())
    jobs=evidence/'jobs'
    train_job='letter-20261110-autograd-gpu-r0'
    inference_job='letter-20261110-autograd-inference-r0'
    training=[read(jobs/train_job/'output/probe'/f'rank{i}.json') for i in (0,1)]
    inference=[read(jobs/inference_job/'output/inference'/f'inference_rank{i}.json') for i in (0,1)]
    assert all(r['status']=='PASS' and len(r['updates'])==9 for r in training)
    assert all(r['status']=='RELOAD_BEAM20_TECHNICAL_PASS' for r in inference)
    budget=read(evidence/'budget_estimate.json')
    ledger=read(evidence/'gpu_budget_ledger.json')
    resources=[read(jobs/j/'output/resources/resource-summary.json') for j in (train_job,inference_job)]
    ledger_used=sum(r['charged_gpu_device_seconds'] for r in ledger['runs'])+ledger['prior_stage_a_seconds']
    current_used=sum(r['charged_gpu_device_seconds'] for r in ledger['runs'] if r['job_id'] in (train_job,inference_job))
    review_names=['cpu_result_review.json','gpu_runtime_review.json','inference_budget_runtime_review.json']
    reviews={name:read(root/'review/autograd_20261110'/name) for name in review_names}
    sources={}
    for rel in ['scripts/compat.py','scripts/gpu_probe.py','scripts/inference_probe.py','scripts/resource_guard.py',
                'scripts/estimate_budget.py','config/probe_instruments.json','config/budget.json','config/guard_autograd.json']:
        path='lc_rec/letter/'+rel
        old=subprocess.check_output(['git','show',run+':'+path],cwd=repo)
        assert old==(root/rel).read_bytes(), 'GPU/estimate source changed after fixed run: '+path
        sources[path]=hashlib.sha256(old).hexdigest()
    job_summaries=[]
    for job in ['letter-20261110-autograd-cpu-r0',train_job,inference_job]:
        q=read(jobs/job/'controller-query.json')
        assert q['status']=='SUCCEEDED' and q['exit_code']==0 and q['cleanup_verified']
        assert not q['worker_alive'] and not q['workgroup_live']
        job_summaries.append({'job_id':job,'status':q['status'],'exit_code':q['exit_code'],'cleanup_verified':True})
    report={'schema':'letter_autograd_continuation_summary_v1',
        'overall':'GPU_TECHNICAL_PASS_BOUNDED_CHAIN',
        'full_seed_96h_acceptance':budget['time_limit_assessment'],
        'source':{'RUN_COMMIT':run,'CPU_RUN_COMMIT':'059f9feb55fde067187afbf3b80f285a00a63ec3',
            'prior_delivery':'61854bb1df0f26a71e68d95ab8eb564f7c762f76',
            'author_LETTER_commit':'8d0154e28de37dbb6e24871c508ad8ddb1921cda',
            'model_revision':'4782ad278652c7c71b72204d462d6d01eaaf7549',
            'server_run_release':'/home/bwwu/lc-rec/letter-20261110/releases/'+run,
            'DELIVERY_COMMIT_resolution':'Exact full SHA is bound by the final reply and ZIP final_receipts/publication.json; not self-embedded in its own commit.',
            'runtime_and_estimator_files_unchanged_from_RUN_COMMIT':sources},
        'versions':training[0]['versions'],
        'trainable_identity':training[0]['trainable_identity'],
        'training':{'completed_global64_updates_per_rank':[len(r['updates']) for r in training],
            'complete_microsteps_per_rank':[sum(len(u['microsteps']) for u in r['updates']) for r in training],
            'nonzero_lr_updates_per_rank':[sum(u['nonzero_learning_rate'] for u in r['updates']) for r in training],
            'finite_gradient_tensors_per_rank':[len(r['gradient_evidence']) for r in training],
            'changed_A_tensors_per_rank':[r['parameter_changes']['changed_lora_A_tensors'] for r in training],
            'changed_B_tensors_per_rank':[r['parameter_changes']['changed_lora_B_tensors'] for r in training],
            'active_SID_max_abs_changes':[{n:v for n,v in r['parameter_changes']['max_abs_by_parameter'].items() if 'modules_to_save.default' in n} for r in training],
            'frozen_originals_all_bytes_unchanged':all(x['all_bytes_identical'] for r in training for x in r['frozen_original_checks'].values()),
            'cross_rank_parameters_equal':all(r['cross_rank_parameter_synchronization']['all_equal'] for r in training),
            'steady_global_samples_per_second':budget['measurements']['steady_mixed_global_samples_per_second'],
            'steady_effective_tokens_per_second':budget['measurements']['steady_mixed_effective_tokens_per_second'],
            'steady_global_seconds_with_data':budget['measurements']['steady_mixed_update_seconds_with_data'],
            'all_update_seconds':[{ 'phase':u['phase'],'global64_seconds':u['max_rank_seconds']} for u in training[0]['updates']],
            'temperature_loss':[r['temperature_loss'] for r in training],
            'forward_backward_optimizer_boundary':'PASS all36microsteps/rank, Adam step1..9, exact optimizer membership',
            'nan_inf':'No nonfinite loss/gradient/trainable parameter observed',
            'sampling_limit':training[0]['measurement_sampling_limit']},
        'validation':budget['measurements']['validation'],
        'adapter':{'save_keys':len(training[0]['adapter_save']['keys']),
            'strict_direct_trainable_save':training[0]['adapter_save']['exact_direct_trainable_key_and_value_match'],
            'file':training[0]['adapter_save']['files']['adapter_model.bin'],
            'server_directory':training[0]['adapter_save']['path'],
            'write_seconds':training[0]['adapter_save']['write_seconds_rank0'],
            'save_and_verification_seconds':training[0]['adapter_save']['seconds_rank0'],
            'reload_keys_per_rank':[r['adapter_exact_key_and_value_reload']['keys'] for r in inference],
            'reload_seconds_per_rank':[r['adapter_reload_seconds'] for r in inference],
            'base_reload_seconds_per_rank':[r['base_load_seconds'] for r in inference],
            'actual_bin_read_and_all_values_verified':True,
            'safetensors_adapter':'NOT_TESTED; pinned PEFT0.5 path explicitly required .bin and rejected ambiguous formats'},
        'beam20':{'protocol':inference[0]['generation_definition'],
            'timed_distinct_users':8,'warmup_user_presentations':2,
            'all20_beams_per_user_callback_and_EOS_verified':True,
            'timings':budget['measurements']['generation'],
            'attention_and_dtype':'Author inference LoRA branch: full BF16 base, vanilla HF attention; training uses int8/FastChat',
            'recommendation_metrics':'NOT_RUN'},
        'resources':{'training_allocated_peak_gib_per_rank':[r['gpu_entire_job_peak_allocated_gib'] for r in training],
            'training_reserved_peak_gib_per_rank':[r['gpu_entire_job_peak_reserved_gib'] for r in training],
            'inference_allocated_peak_gib_per_rank':[r['cuda_peak_allocated_bytes']/2**30 for r in inference],
            'inference_reserved_peak_gib_per_rank':[r['cuda_peak_reserved_bytes']/2**30 for r in inference],
            'training_rank_ru_maxrss_gib':[r['cpu_process_peak_rss_gib'] for r in training],
            'inference_rank_ru_maxrss_gib':[r['cpu_process_peak_rss_bytes']/2**30 for r in inference],
            'minimum_sampled_host_available_gib':min(r['peaks']['min_mem_available_bytes'] for r in resources)/2**30,
            'sampled_task_tree_rss_max_gib_per_job':[r['peaks']['cpu_rss_sum_peak_bytes']/2**30 for r in resources],
            'cpu_peak_limit':'5-second task-tree sampling can miss peaks; rank ru_maxrss is per-process history and is not summed as a measured concurrent peak.',
            'sampled_swap_increase_and_task_swap_bytes':[r['peaks']['max_swap_used_increase_bytes']+r['peaks']['task_vm_swap_peak_bytes'] for r in resources],
            'guard_tracked_storage_peak_bytes':max(r['peaks']['new_disk_allocated_peak_bytes'] for r in resources)},
        'budget':{'center_conditional_hours':budget['center_estimate']['total_hours'],
            'conservative_conditional_hours':budget['conservative_estimate']['total_hours'],
            'center_remaining_to_96_hours':budget['center_estimate']['remaining_to_96_hours'],
            'conservative_remaining_to_96_hours':budget['conservative_estimate']['remaining_to_96_hours'],
            'conditional_measured_components_support_96h_feasibility':budget['conditional_measured_components_support_96h_feasibility'],
            'minimum_training_samples_s_with_measured_other_costs':budget['minimum_train_samples_per_second_with_measured_other_costs_and_center_margin'],
            'formal_full_seed_guarantee':False,'unmeasured_formal_overheads':budget['center_estimate']['unquantified_formal_checkpoint_overheads']},
        'gpu_budget':{'historical_gpu_device_minutes':(ledger_used-current_used)/60,
            'current_gpu_device_minutes':current_used/60,'cumulative_gpu_device_minutes':ledger_used/60,
            'limit_gpu_device_minutes':120,'remaining_gpu_device_minutes':(7200-ledger_used)/60},
        'corrections':{'historical_used':3,'this_task_used':1,'this_task_initial_round':0,'this_task_limit':3,
            'round1_scope':'Correct delivery-draft sampler wording; no GPU source/config change or new computation'},
        'reviews':{name:{'reviewer':r['reviewer'],'status':r['status']} for name,r in reviews.items()},
        'jobs':job_summaries,'active_task_jobs':[],
        'formal_training_and_metrics_run':False,'Qwen_or_Shuffle_run':False,
        'new_packages_toolchains_or_environment_configuration':[],
        'formal_adaptation_approval':'BF16 A-only and PEFT non-inplace adaptation are approved only for this bounded technical probe; formal baseline adoption remains unapproved.',
        'GPT_UPLOAD_REQUIRED':'YES'}
    (evidence/'summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    rounds=read(evidence/'correction_rounds.json')
    rounds['round0'].update(gpu_job=train_job,inference_job=inference_job,RUN_COMMIT=run,
        actual_complete_updates=9,gpu_training='PASS',adapter_reload_and_beam20='PASS')
    rounds['current_correction_rounds_used']=1
    rounds['round1']={'component':'Delivery draft sampler description',
        'root_cause':'Draft described DistributedSampler arithmetic as actual runtime sampling; captured Trainer loader is Accelerate DataLoaderShard/BatchSamplerShard with inner RandomSampler.',
        'minimal_change':'Correct only unified review-entry prose. Runtime source, raw records, estimator and all numeric costs unchanged.',
        'verification':'Re-read actual rank formal_train_dataloader_inspection; actual length8240/rank and scheduler8240/four epochs remain supported. Non-author entry review is separately retained.',
        'new_GPU_execution':False,'additional_optimizer_updates':0}
    rounds['terminal_outcome']={'core_gpu_technical':'PASS','full_seed_96h':budget['time_limit_assessment'],
        'active_jobs':[],'current_correction_rounds_used':1,'no_additional_GPU_work_authorized_or_started':True}
    (evidence/'correction_rounds.json').write_text(json.dumps(rounds,indent=2)+'\n')
    print(json.dumps({'overall':report['overall'],'budget':report['budget'],'GPU_minutes':report['gpu_budget']},indent=2))


if __name__=='__main__':
    main()
