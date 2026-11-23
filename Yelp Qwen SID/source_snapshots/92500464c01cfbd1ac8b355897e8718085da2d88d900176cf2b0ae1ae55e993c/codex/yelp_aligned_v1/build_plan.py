"""Store the Yelp plan as executable job arguments, not a Word document."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
UPSTREAM=ROOT/'experiments/yelp_rkmeans_qwen_aligned_v1'
OUTPUT=ROOT/'experiments/sid_order_yelp_5seed_v1'
DATA=Path('/home/jcy/data/yelp_aligned_v1')


def main():
    tasks=[]
    waves=[ [('shuffle',42),('mlp',42),('shuffle',43),('mlp',43)],
            [('mlp',44),('shuffle',44),('mlp',45),('shuffle',45)],
            [('shuffle',46),('mlp',46),None,None] ]
    lanes={str(gpu):[] for gpu in range(4)}
    for wave,jobs in enumerate(waves,1):
        for gpu,job in enumerate(jobs):
            if job is None:
                continue
            variant,seed=job
            task_id=f'Y{len(tasks)+1:02d}'
            out=OUTPUT/variant/f'seed_{seed}'
            cmd=[sys.executable,'-u',str(HERE/'run_task.py'),'--dataset','yelp_aligned_v1',
                 '--variant',variant,'--gpu',str(gpu),'--seed',str(seed),'--output',str(out),
                 '--sample-dir',str(OUTPUT/'fixed_test_users')]
            task=dict(id=task_id,wave=wave,gpu=gpu,variant=variant,seed=seed,
                      status='not_started',completed=False,output=str(out),argv=cmd)
            tasks.append(task)
            lanes[str(gpu)].append(task_id)
    required=[UPSTREAM/'item_embeddings.pt',UPSTREAM/'semantic_ids.pt',UPSTREAM/'upstream_audit.json',
              UPSTREAM/'readonly_shuffle/preparation_verified.json',UPSTREAM/'readonly_mlp/preparation_verified.json']
    ready=all(p.is_file() for p in required)
    if ready:
        for variant in ('shuffle','mlp'):
            audit=json.loads((UPSTREAM/f'readonly_{variant}/preparation_verified.json').read_text())
            ready=ready and audit['passed'] and not audit['fit_called'] and audit['optimizer_steps']==0
    plan=dict(plan_id='yelp_aligned_v1_5seed',updated_utc=datetime.now(timezone.utc).isoformat(),
        user_request='Prepare aligned Yelp and plan the same 5-seed shuffle/MLP experiments; no Word document; do not launch formal experiments yet.',
        state='ready_not_started' if ready else 'preparing_inputs',formal_training_started=False,
        readiness_audit=str(UPSTREAM/'readiness_verified.json'),
        auto_launch=False,downstream_training_authorized=False,dataset='yelp_aligned_v1',
        data_dir=str(DATA),dataset_audit=str(DATA/'processing_audit.json'),
        upstream_dir=str(UPSTREAM),fixed_upstream_seed=42,upstream_frozen_across_all_10_tasks=True,
        upstream_method=dict(text_encoder='Qwen2.5-7B-Instruct',pooling='WMH MeanAggregation',max_text_tokens=128,
            right_padding=True,weight_dtype='fp16',stored_dtype='fp32',dimension=3584,
            quantizer='RQ-KMeans',semantic_digits=3,dedup_digits=1,codebook_width=256),
        seeds=[42,43,44,45,46],train_tasks=10,full_tests=20,attention_groups=20,attention_users_per_group=100,
        protocol=dict(reference='codex/plan_v1_1/run_task.py, protocol.py and attention.py',
            embedding_tables='one shared encoder/decoder table, fresh initialization per task',
            train_shuffle=False,validation_shuffle=False,
            checkpoint='max val/recall@5; each shuffle task tests the exact same best checkpoint in all 3 modes',
            shuffle_test_modes=['unshuffled','shuffle_3','shuffle_4'],mlp_test_modes=['mlp'],
            shuffle_operation='Only item-internal history SID embedding order; targets/item chronology unchanged',
            mlp='Ordered SID concatenation 512 -> 256 -> 128; ReLU; dropout 0.1; no SEP',
            full_test_enabled=True,full_test_attention_export=False,
            attention_export=True,attention_users=100,attention_sampling_seed=2026,
            attention_sampling='Before any outcomes, sorted history-length/user-ID thirds, sample 34/33/33; exact shared users across methods/seeds/modes',
            attention_layers='all 4 encoder self, 4 decoder self and 4 cross; all 6 heads',
            attention_path='Recompute coherent final Top1 prefix; save raw tensors, masks, permutations, labels and prediction references',
            eval_seed=2026,permutation='SHA256(eval_seed,dataset,user_id,item_position,mode)',
            precision='32-true',devices_per_task=1,strategy='auto',deterministic=True,
            train_source_batch=16,train_expanded_batch_cap=128,gradient_accumulation=16,
            val_test_batch=8,workers=0,timeout=0,persistent_workers=False,
            sequence_sid_slots=120,max_steps=320000,max_epochs=10,
            val_check_interval_train_batches=1600,early_stopping_validation_checks=10,
            optimizer='Adam',learning_rate=0.001,weight_decay=0.0001,scheduler=None,
            d_model=128,encoder_layers=4,decoder_layers=4,heads=6,d_ff=1024,d_kv=64,dropout=0.15,beam_size=10,
            fit_checkpoint=None,fresh_output_required=True,automatic_resume=False,automatic_retry=False),
        tasks=tasks,gpu_lanes=lanes,
        scheduling='Per-GPU asynchronous sequential queue; wave columns express lane order, not a global barrier. Each task includes fit, all full tests and attention before the next lane task. Failure stops that lane. Use stage_gpu_locks before launching.',
        aggregation=dict(conditions=['unshuffled','shuffle_3','shuffle_4','mlp'],
            metrics=['recall@5','recall@10','ndcg@5','ndcg@10'],
            seeds_required=5,report='mean and sample standard deviation (ddof=1)',
            comparisons='Within-seed shuffle_3/4 minus unshuffled; paired MLP minus unshuffled. Never substitute another seed or choose checkpoint by test metrics.'),
        processing_summary=json.loads((DATA/'processing_audit.json').read_text()),
        prepared_inputs={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in required if p.is_file()})
    if (UPSTREAM/'original_code_hashes.json').exists():
        plan['original_code_sha256']=json.loads((UPSTREAM/'original_code_hashes.json').read_text())
    (HERE/'experiment_plan.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(state=plan['state'],tasks=len(tasks),formal_training_started=False),indent=2))


if __name__=='__main__':
    main()
