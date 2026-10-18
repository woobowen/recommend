# Sports A2 生成器训练终态：真实非作者审核

`A2_TRAINING_NONAUTHOR_REVIEW_PASS`

审核者：`/root/sports_a1_result_audit`。本代理未编写或运行本次生成器实现，直接读取原执行记录并在服务器CPU严格加载原best/final checkpoint；没有GPU重评、训练或大checkpoint下载，没有修改任何原件。此PASS只覆盖A2生成器训练，正式三条件测试与attention结果须独立验收。

## 真实训练身份与终态

- job：`ga-20261018-sportsa2-train-fd-recovery-v1`。
- 源码：`85b77f5e2a5b497fdc1f620649ba488fc783a245`，snapshot `22286966cd544773f21b96e4185d56c66afd7183fdc4e8888716d8ea1d77acb2`。
- 原runner started：`2026-10-17T19:10:58.920653+00:00`；ended：`2026-10-18T04:19:28.582003+00:00`（北京时间2026-10-18 12:19:28.582003结束）。
- 原状态SUCCEEDED、exit0、cleanup verified；另按原boot_id/PID/start_ticks核查supervisor/worker/known members，没有仍属该训练作业原身份的活进程。
- 真正终止原因：`patience10_complete_validation_early_stop`。final step7900；best step6900；不是12小时预算停止，不是存在某个checkpoint就提前认定完成。

## 新SID、从零初始化与A1协议

训练SID来自本次成功的新量化job：`/home/bwwu/grid-audits/sid-extensions-p0/jobs/ga-20261018-sportsa2-quantize-fd-recovery-v1/output/semantic_ids.pt`，588597字节，SHA `06cfa6ba0f64d6c68e0e94c4111edd51d086796e8cdb474ac3fcd424a4a2adf0`。训练effective spec、原结果、checkpoint embedded SID与实际重哈希全部一致。

原训练used_config与已审核A1逐字段比较，恰有7个允许的SID/输出路径差异；其他实际模型、数据加载/文件分配、worker、训练、选模与evaluator配置均相同。保留seed42、32-true、Adam lr0.001/weight_decay0.0001、无scheduler、raw16/cap512/accumulation16、每1600 loader batches完整验证、max_steps320000/max_epochs10、val/recall@5最大选模、patience10、K10/prefix关闭。

固定源码fresh instantiate、fit ckpt_path=None，实际spec无checkpoint/resume；原Adam起始state为空。13259136参数全部可训练。实际best/final模型严格载入后均确认独立encoder/decoder SID表各[1024,128]、storage不同且数值不同；无A1/Sports B推荐生成器权重载入。原实验没有完整随机初始参数dump，本审核不捏造该资产。

## 全训练和完整验证

直接流式审阅全7900 optimizer update记录，无缺步/重号，累计126400 raw microbatches、39426025 expanded windows，每次恰16 accumulation且每microbatch window数在原cap512内。原运行每次确认optimizer覆盖全部trainable参数、所有可用梯度有限；两SID表、BOS和被监视的encoder/decoder transformer梯度全程非零有限。原returned_training_losses范围[0.26116400957107544, 1.4493671655654907]，没有把这些返回值额外解释成未缩放loss。

完整验证79次：step100–7900每100 optimizer updates一次；每次35598 targets、4450 batches，完整key-target multiset=true，并绑定新SID条件的原evaluation index SHA `87822a4b8b174bd6e4e7f49ca4bc3c413736e4c2630696898c9e488b5cdbac12`。没有重跑GPU验证；独立检查全部事件序列、目标数量、索引身份和记录的finite metrics。

最高验证Recall@5为0.031434349715709686，在step6900。step7000、7100、7200、7300、7400、7500、7600、7700、7800、7900连续10次未改善；独立重算wait_count=10。final checkpoint原EarlyStopping state中wait_count=10、patience=10，ModelCheckpoint state指向同一best和同一score。未用testing指标选模。

## 原checkpoint直接CPU strict load

| 资产 | 字节 | SHA-256 |
|---|---:|---|
| best `output/checkpoints/checkpoint_epoch=000_step=006900.ckpt` | 161660326 | `1287c3f60404726dbda165d5242ba17d40ecaf6520ded69d6e3165170586513d` |
| final `output/recovery/training-final/full-state.ckpt` | 161653798 | `74770b7d1166fa16df1a931b6ffa221bbf9e6f9b3eb84fa2a9d2ad8055b55676` |

二者均留在上述原job根下。本审核先按原终态manifest和结果声明独立重哈希载荷；best与final随后分别执行真正`model.load_state_dict(...,strict=True)`，各178个state tensors的keys/shapes/values精确一致，missing/unexpected均空、全部finite、embedded SID exact、独立双表成立。best/final的128份Adam参数state step分别全为6900/7900，lr/weight_decay保持原值，callbacks完整。

本次有界CPU审核共instantiate两个CPU模型（完整终态审查best一次、额外final-only严格加载一次），未执行forward，CUDA隐藏，没有重读best进行额外实验。终态主回执`terminal-review-20261018T042225Z.json`与final-only回执`final-checkpoint-strict-review-20261018T042344Z.json`共同覆盖上述结论。

## 预算、原始证据和边界

原训练GPU请求wall为32909.661350秒（9.141572597小时），低于43200秒；未用额度10290.338650秒，不授权新增实验。峰值allocated 10177158656字节、reserved 20919091200字节，均在单4090物理容量内。共享环境下的wall不作为公平加速证据。

定向取回20份必要轻量原件，共7739531字节，包含全7900条optimizer日志、79条validation日志、原训练结果/配置/strict-load记录/原终态状态等；所有大小和SHA已再次核验，未export整job或下载checkpoint。路径`terminal-originals/`，回执`terminal-copy-receipts.json`。

合并机器验收为`final_training_review.json`，完整保存来源回执SHA和逐项结果。初期`stage_review.json`仍保持原SHA `4cfc012774690e30a0aa93c1beab9216bd4b6128cde1b8f10dbe2e470ac22f44`；既有A1、scheduler和旧阶段审核未改。

A2属于T5-RKMeans新SID条件下的新双表方法复现，不是找回学长原Sports权重。本训练PASS不能替代量化、testing、Shuffle、attention或跨域因果结论的独立审核；没有用与A1或Beauty指标接近与否判正确。无依赖安装、环境或Git/实验改动。

GPT_UPLOAD_REQUIRED: YES（本目录原始日志与机器回执由主控并入最终单一审核ZIP。）
