#!/usr/bin/env python3
"""Read-only static Beauty prelaunch audit; never submits or executes a GPU job."""
import hashlib,json,sys
from pathlib import Path
root=Path(__file__).resolve().parent
cfg=json.loads((root/'config/formal_beauty_seed42.json').read_text())
train=(root/'formal_train.py').read_text();evaluate=(root/'formal_evaluate.py').read_text();metrics=(root/'offline_metrics.py').read_text();shuffle=(root/'shuffle_entry.py').read_text();old=(root.parent/'shuffle/evaluate_condition.py').read_text();paired=(root/'paired_beauty.py').read_text()
sys.path.insert(0,str(root));from epoch_schedule import schedule
checks={}
checks['train_recipe_config']=(cfg['dataset']=='Beauty' and cfg['epochs']==4 and cfg['world_size']==2 and cfg['per_device_batch_size']==8 and cfg['gradient_accumulation_steps']==4 and cfg['global_batch']==64 and cfg['model_revision']=='4782ad278652c7c71b72204d462d6d01eaaf7549')
checks['formal_vs_preflight_dataset_boundaries']=all(x in train for x in ("assert (len(train), len(valid)) == (131413, 44726)","if a.boundary_preflight:\n        from torch.utils.data import Subset",'max_steps=10 if a.boundary_preflight else 8214','sync_with_dataloader\']=False'))
sched=schedule(8214,4,4);checks['epoch_arithmetic']=[x['updates'] for x in sched]==[2053,4107,6160,8214] and [x['pending_microbatches'] for x in sched]==[2,0,2,0]
checks['formal_train_guard']='assert a.prelaunch_review' in train and "gate['status']=='PASS'" in train
checks['formal_shard_sizes']='11182 if rank == 0 else 11181' in evaluate and 'validate_assignments(assignments, len(data))' in evaluate
checks['preflight_manifest_disallowed_formally']='bool(args.preflight_users_per_rank) != bool(args.preflight_manifest)' in evaluate
checks['beam20']='BEAMS = 20' in evaluate and 'num_return_sequences=BEAMS' in evaluate and 'filter_items' in cfg and not cfg['filter_items']
checks['metrics_beauty_axis']='N = 22363' in metrics and "cfg['sid_coverage_denominator'] == 12088" in metrics and 'np.random.PCG64' in metrics and 'range(0,5000,8)' in metrics
checks['paired_beauty_adaptation']='24772' not in paired and '22363' in paired and '7430079e2b917dfa5e4bb8ead2702fb6c622fb48ffb684dd2889780f9480af7f' in paired
checks['old_shuffle_controller_unsafe_for_beauty']="completed_users']==12386" in old and "'Instruments'" in old
# The historical condition/evaluation orchestration is Instruments-specific;
# no approved Beauty series runner can be inferred from standalone entries.
report={'status':'PARTIAL','checks':checks,'epoch_schedule':sched,'blockers':['No verified Beauty formal train→Normal→Shuffle3→Shuffle4 controller with authorization and resource gates','Historical Shuffle evaluation controller hardcodes Instruments user counts and assets','Beauty paired adapter is only synthetically tested; full-population pipeline and independent recomputation are not validated'],'scope':'static source inspection and arithmetic only; no formal job submitted','files_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (root/'formal_train.py',root/'formal_evaluate.py',root/'offline_metrics.py',root/'paired_beauty.py')}}
assert all(checks.values()),checks
if len(sys.argv)>1:
 with Path(sys.argv[1]).open('x') as f:json.dump(report,f,indent=2);f.write('\n')
print(json.dumps(report,indent=2))
