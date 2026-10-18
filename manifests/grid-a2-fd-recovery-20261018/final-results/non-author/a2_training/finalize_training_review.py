#!/usr/bin/env python3
"""Assemble independent terminal evidence without running a model again."""
from pathlib import Path
import datetime,hashlib,json
P=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
primary_path=P/'terminal-review-20261018T042225Z.json';final_path=P/'final-checkpoint-strict-review-20261018T042344Z.json'
r=read(primary_path);f=read(final_path);phase=read(P/'stage_review.json');raw=P/'terminal-originals'
assert r['status']=='A2_TRAINING_NONAUTHOR_REVIEW_PASS' and f['status']=='A2_FINAL_RECOVERY_CHECKPOINT_STRICT_LOAD_PASS'
assert sha(P/'stage_review.json')=='4cfc012774690e30a0aa93c1beab9216bd4b6128cde1b8f10dbe2e470ac22f44'
assert (r['final_step'],r['best_step'],r['full_validations'],r['actual_microbatches'],r['expanded_windows'])==(7900,6900,79,126400,39426025)
assert r['best']['checkpoint']['sha256']=='1287c3f60404726dbda165d5242ba17d40ecaf6520ded69d6e3165170586513d'
assert r['final_checkpoint']['sha256']==f['checkpoint']['sha256']=='74770b7d1166fa16df1a931b6ffa221bbf9e6f9b3eb84fa2a9d2ad8055b55676'
assert f['global_step']==7900 and f['state_tensors']==178 and f['keys_shapes_values_exact'] and f['embedded_sid_exact'] and f['independent_tables']
receipts=read(P/'terminal-copy-receipts.json')
for v in receipts:
 p=Path(v['local']);assert p.stat().st_size==v['bytes'] and sha(p)==v['sha256']
validation=[json.loads(x) for x in (raw/'output/validations.jsonl').read_text().splitlines()]
assert len(validation)==79 and [x['global_step'] for x in validation[-10:]]==list(range(7000,7901,100))
assert max(validation,key=lambda x:x['metrics']['val/recall@5'])['global_step']==6900
assert all(x['metrics']['val/recall@5']<=r['best_validation_recall5'] for x in validation[-10:])
status=read(raw/'status.json');assert status['status']=='SUCCEEDED' and status['exit_code']==0 and status['cleanup']['verified']
out={**r,'final_checkpoint_strict_load':f,'first_full_validation':phase['first_full_validation'],'A1_effective_protocol_comparison':phase['effective_config_A1_comparison'],'all_effective_config_differences':phase['all_effective_config_differences'],'last_ten_validation_steps':[x['global_step'] for x in validation[-10:]],'last_ten_validation_recall5':[x['metrics']['val/recall@5'] for x in validation[-10:]],'started_at':status['started_at'],'ended_at':status['ended_at'],'unused_training_budget_seconds':43200-r['wall_GPU_request_seconds'],'terminal_raw_files_verified':len(receipts),'terminal_raw_bytes':sum(x['bytes'] for x in receipts),'large_checkpoint_downloads':0,'CPU_model_instantiations_total_for_terminal_review':2,'scientific_test_results_approved':False,'audit_evidence':[{'path':str(p),'sha256':sha(p)} for p in [primary_path,final_path,P/'stage_review.json',P/'terminal-copy-receipts.json']],'limitations':['Training PASS is separate from formal testing and attention; those still require their own result audit.','From-scratch initialization is backed by frozen source, actual config/empty optimizer and real updates; no full random initial parameter dump exists.','Past full-validation key-target equality is verified against the original complete-validation events and their frozen index identity; no GPU validation rerun.','A2 is a new T5-RKMeans-SID dual-table training method reproduction, not recovered author historical Sports recommender weights.','Timing includes requested GPU wall under shared resource conditions and is not a fair standalone speed comparison.']}
(P/'final_training_review.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
text=f'''# Sports A2 生成器训练终态：真实非作者审核

`A2_TRAINING_NONAUTHOR_REVIEW_PASS`

审核者：`/root/sports_a1_result_audit`。本代理未编写或运行本次生成器实现，直接读取原执行记录并在服务器CPU严格加载原best/final checkpoint；没有GPU重评、训练或大checkpoint下载，没有修改任何原件。此PASS只覆盖A2生成器训练，正式三条件测试与attention结果须独立验收。

## 真实训练身份与终态

- job：`ga-20261018-sportsa2-train-fd-recovery-v1`。
- 源码：`85b77f5e2a5b497fdc1f620649ba488fc783a245`，snapshot `22286966cd544773f21b96e4185d56c66afd7183fdc4e8888716d8ea1d77acb2`。
- 原runner started：`{status['started_at']}`；ended：`{status['ended_at']}`（北京时间2026-10-18 12:19:28.582003结束）。
- 原状态SUCCEEDED、exit0、cleanup verified；另按原boot_id/PID/start_ticks核查supervisor/worker/known members，没有仍属该训练作业原身份的活进程。
- 真正终止原因：`patience10_complete_validation_early_stop`。final step7900；best step6900；不是12小时预算停止，不是存在某个checkpoint就提前认定完成。

## 新SID、从零初始化与A1协议

训练SID来自本次成功的新量化job：`/home/bwwu/grid-audits/sid-extensions-p0/jobs/ga-20261018-sportsa2-quantize-fd-recovery-v1/output/semantic_ids.pt`，588597字节，SHA `06cfa6ba0f64d6c68e0e94c4111edd51d086796e8cdb474ac3fcd424a4a2adf0`。训练effective spec、原结果、checkpoint embedded SID与实际重哈希全部一致。

原训练used_config与已审核A1逐字段比较，恰有7个允许的SID/输出路径差异；其他实际模型、数据加载/文件分配、worker、训练、选模与evaluator配置均相同。保留seed42、32-true、Adam lr0.001/weight_decay0.0001、无scheduler、raw16/cap512/accumulation16、每1600 loader batches完整验证、max_steps320000/max_epochs10、val/recall@5最大选模、patience10、K10/prefix关闭。

固定源码fresh instantiate、fit ckpt_path=None，实际spec无checkpoint/resume；原Adam起始state为空。13259136参数全部可训练。实际best/final模型严格载入后均确认独立encoder/decoder SID表各[1024,128]、storage不同且数值不同；无A1/Sports B推荐生成器权重载入。原实验没有完整随机初始参数dump，本审核不捏造该资产。

## 全训练和完整验证

直接流式审阅全7900 optimizer update记录，无缺步/重号，累计126400 raw microbatches、39426025 expanded windows，每次恰16 accumulation且每microbatch window数在原cap512内。原运行每次确认optimizer覆盖全部trainable参数、所有可用梯度有限；两SID表、BOS和被监视的encoder/decoder transformer梯度全程非零有限。原returned_training_losses范围{r['loss_range']}，没有把这些返回值额外解释成未缩放loss。

完整验证79次：step100–7900每100 optimizer updates一次；每次35598 targets、4450 batches，完整key-target multiset=true，并绑定新SID条件的原evaluation index SHA `87822a4b8b174bd6e4e7f49ca4bc3c413736e4c2630696898c9e488b5cdbac12`。没有重跑GPU验证；独立检查全部事件序列、目标数量、索引身份和记录的finite metrics。

最高验证Recall@5为{r['best_validation_recall5']:.17g}，在step6900。step7000、7100、7200、7300、7400、7500、7600、7700、7800、7900连续10次未改善；独立重算wait_count=10。final checkpoint原EarlyStopping state中wait_count=10、patience=10，ModelCheckpoint state指向同一best和同一score。未用testing指标选模。

## 原checkpoint直接CPU strict load

| 资产 | 字节 | SHA-256 |
|---|---:|---|
| best `output/checkpoints/checkpoint_epoch=000_step=006900.ckpt` | 161660326 | `1287c3f60404726dbda165d5242ba17d40ecaf6520ded69d6e3165170586513d` |
| final `output/recovery/training-final/full-state.ckpt` | 161653798 | `74770b7d1166fa16df1a931b6ffa221bbf9e6f9b3eb84fa2a9d2ad8055b55676` |

二者均留在上述原job根下。本审核先按原终态manifest和结果声明独立重哈希载荷；best与final随后分别执行真正`model.load_state_dict(...,strict=True)`，各178个state tensors的keys/shapes/values精确一致，missing/unexpected均空、全部finite、embedded SID exact、独立双表成立。best/final的128份Adam参数state step分别全为6900/7900，lr/weight_decay保持原值，callbacks完整。

本次有界CPU审核共instantiate两个CPU模型（完整终态审查best一次、额外final-only严格加载一次），未执行forward，CUDA隐藏，没有重读best进行额外实验。终态主回执`terminal-review-20261018T042225Z.json`与final-only回执`final-checkpoint-strict-review-20261018T042344Z.json`共同覆盖上述结论。

## 预算、原始证据和边界

原训练GPU请求wall为{r['wall_GPU_request_seconds']:.6f}秒（{r['wall_GPU_request_seconds']/3600:.9f}小时），低于43200秒；未用额度{43200-r['wall_GPU_request_seconds']:.6f}秒，不授权新增实验。峰值allocated {r['gpu_peak_allocated_bytes']}字节、reserved {r['gpu_peak_reserved_bytes']}字节，均在单4090物理容量内。共享环境下的wall不作为公平加速证据。

定向取回20份必要轻量原件，共7739531字节，包含全7900条optimizer日志、79条validation日志、原训练结果/配置/strict-load记录/原终态状态等；所有大小和SHA已再次核验，未export整job或下载checkpoint。路径`terminal-originals/`，回执`terminal-copy-receipts.json`。

合并机器验收为`final_training_review.json`，完整保存来源回执SHA和逐项结果。初期`stage_review.json`仍保持原SHA `4cfc012774690e30a0aa93c1beab9216bd4b6128cde1b8f10dbe2e470ac22f44`；既有A1、scheduler和旧阶段审核未改。

A2属于T5-RKMeans新SID条件下的新双表方法复现，不是找回学长原Sports权重。本训练PASS不能替代量化、testing、Shuffle、attention或跨域因果结论的独立审核；没有用与A1或Beauty指标接近与否判正确。无依赖安装、环境或Git/实验改动。

GPT_UPLOAD_REQUIRED: YES（本目录原始日志与机器回执由主控并入最终单一审核ZIP。）
'''
(P/'FINAL_TRAINING_REVIEW.md').write_text(text)
print(json.dumps({'status':out['status'],'review':str(P/'final_training_review.json'),'review_sha256':sha(P/'final_training_review.json'),'report':str(P/'FINAL_TRAINING_REVIEW.md'),'report_sha256':sha(P/'FINAL_TRAINING_REVIEW.md'),'raw_files':len(receipts),'raw_bytes':out['terminal_raw_bytes'],'stage_receipt_preserved':True}))
