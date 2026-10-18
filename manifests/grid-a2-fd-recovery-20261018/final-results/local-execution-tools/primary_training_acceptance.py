"""Primary full-event acceptance of completed A2 training; no model execution."""
from pathlib import Path
import datetime as dt
import hashlib
import json
import math

BASE=Path(__file__).resolve().parent
RAW=BASE/'terminal-captures/20261018T042205665551Z/raw/ga-20261018-sportsa2-train-fd-recovery-v1'

def read(path):return json.loads(path.read_text())
def rows(path):return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

def main():
    status=read(RAW/'status.json');cleanup=read(RAW/'cleanup.json')
    assert status['status']=='SUCCEEDED' and status['exit_code']==0 and cleanup['verified']
    output=RAW/'output';result=read(output/'training_result.json')
    updates=rows(output/'optimizer_updates.jsonl');validations=rows(output/'validations.jsonl')
    assert len(updates)==7900 and len(validations)==79
    windows=0;losses=[];gradient_samples={}
    for number,record in enumerate(updates,1):
        assert record['optimizer_update']==number and record['cumulative_microbatches']==16*number
        assert len(record['expanded_windows'])==len(record['returned_training_losses'])==16
        windows+=sum(record['expanded_windows'])
        assert windows==record['cumulative_expanded_windows']
        losses.extend(record['returned_training_losses'])
        optimizer=record['optimizer']
        assert optimizer['step_before']==number-1 and optimizer['all_available_gradients_finite']
        assert optimizer['optimizer_covers_every_trainable_parameter']
        for name,value in optimizer['watched_gradient_norms'].items():
            assert math.isfinite(value) and value>=0
            gradient_samples.setdefault(name,[]).append(value)
    assert all(math.isfinite(value) for value in losses)
    assert all(any(value>0 for value in values) for values in gradient_samples.values())
    assert windows==result['expanded_windows']==39426025
    for number,record in enumerate(validations,1):
        assert record['validation_number']==number and record['global_step']==100*number
        assert record['targets']==35598 and record['batches']==4450 and record['exact_key_target_multiset']
        assert record['expected_index']['sha256']=='87822a4b8b174bd6e4e7f49ca4bc3c413736e4c2630696898c9e488b5cdbac12'
        assert all(math.isfinite(value) for value in record['metrics'].values())
    best=max(validations,key=lambda record:record['metrics']['val/recall@5'])
    assert best['global_step']==6900 and result['global_step']==7900
    assert result['termination']=='patience10_complete_validation_early_stop'
    assert all(record['metrics']['val/recall@5']<=best['metrics']['val/recall@5'] for record in validations[-10:])
    assert result['early_stopping']==dict(wait_count=10,patience=10)
    assert result['fresh_initialization'] and result['full_parameter_training'] and result['independent_sid_tables']
    assert result['training_complete'] and not result['test_complete']
    gate=result['best_checkpoint_gate']
    assert gate['all_keys_shapes_values_exact'] and gate['embedded_sid_exact'] and gate['independent_sid_tables']
    assert not gate['missing_keys'] and not gate['unexpected_keys']
    assert result['sid']['sha256']=='06cfa6ba0f64d6c68e0e94c4111edd51d086796e8cdb474ac3fcd424a4a2adf0'
    used=read(output/'used_config.json');actual=read(output/'actual_train_loader.json')
    assert used['trainer']['precision']=='32-true' and used['trainer']['accumulate_grad_batches']==16
    assert used['trainer']['val_check_interval']==1600 and used['trainer']['max_steps']==320000
    assert used['trainer']['max_epochs']==10 and actual['raw_microbatch']==16
    assert actual['fresh_optimizer_state_empty'] and actual['single_gpu_world_size']==1
    assert used['optim']['optimizer']['_target_']=='torch.optim.Adam'
    assert used['optim']['optimizer']['lr']==0.001 and used['optim']['optimizer']['weight_decay']==0.0001
    assert used['optim']['scheduler'] is None
    assert used['model']['top_k_for_generation']==10 and not used['model']['should_check_prefix']
    wall=(dt.datetime.fromisoformat(status['ended_at'])-dt.datetime.fromisoformat(status['started_at'])).total_seconds()
    assert wall<=43200
    checked=0
    for entry in status['artifacts']:
        path=RAW/entry['path']
        if path.exists():
            assert path.stat().st_size==entry['bytes'] and hashlib.sha256(path.read_bytes()).hexdigest()==entry['sha256']
            checked+=1
    report=dict(status='PRIMARY_A2_TRAINING_EVENT_ACCEPTANCE_PASS',checked_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
      author='/root',optimizer_updates=len(updates),raw_microbatches=16*len(updates),expanded_windows=windows,
      complete_validations=len(validations),each_validation_targets=35598,all_target_multisets_exact=True,
      best_step=6900,final_step=7900,best_validation_recall5=best['metrics']['val/recall@5'],
      last_ten_nonimproving_steps=[record['global_step'] for record in validations[-10:]],
      returned_loss_range=[min(losses),max(losses)],
      gradient_ranges={name:[min(values),max(values)] for name,values in gradient_samples.items()},
      original_run_strict_checkpoint_gate=gate,request_gpu_seconds=wall,training_limit_seconds=43200,
      artifact_members_rehashed=checked,sid=result['sid'],best_checkpoint=result['best_checkpoint'],
      final_checkpoint=result['final_recovery']['checkpoint'],
      checkpoint_fresh_CPU_load='Covered by real independent nonauthor CPU audit; this primary pass only reads original events and strict-load receipts.',
      final_full_task_review_pass=False)
    destination=BASE/'primary-result-acceptance/a2-training.json'
    assert not destination.exists()
    destination.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(path=str(destination),status=report['status'],best_step=6900,final_step=7900,
                         validations=len(validations),gpu_seconds=wall),ensure_ascii=False,indent=2))

if __name__=='__main__':main()
