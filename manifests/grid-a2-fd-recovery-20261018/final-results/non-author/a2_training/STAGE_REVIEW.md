# Sports A2 新双表生成器：非作者初期核验

`A2_TRAINING_STAGE_EVIDENCE_CHECKED_RUNNING`

**`final_training_pass: false`**。审核者`/root/sports_a1_result_audit`。本报告只确认已经直接取得的初期证据，不提前批准训练终态、best选模或未来科研结果。

实际取证时间为北京时间2026-10-18 03:19:54，作业`ga-20261018-sportsa2-train-fd-recovery-v1`仍RUNNING。执行源码固定`85b77f5e2a5b497fdc1f620649ba488fc783a245`、snapshot `22286966cd544773f21b96e4185d56c66afd7183fdc4e8888716d8ea1d77acb2`；没有把当前Git HEAD `acb2f395c62af074a9649e99af17a99aa60dd83e`冒充运行源码。

## 直接确认的事实

- 实际新SID来自本次成功量化job，588597字节，SHA-256 `06cfa6ba0f64d6c68e0e94c4111edd51d086796e8cdb474ac3fcd424a4a2adf0`；本审核直接重哈希该小资产，与新量化RESULT、训练effective spec及training_protocol记录逐项对应。
- 训练`used_config.json`与原已审核A1逐字段比较，恰有7处变化：模型和train/val/test三个SID loader的SID路径，以及trainer/logger/checkpoint输出路径。其他实际模型、数据加载、训练、选模、evaluator配置全部相同。
- seed42、32-true、Adam lr0.001/weight_decay0.0001、无scheduler、raw16/cap512/accumulation16、每1600 loader batches验证、max_steps320000/max_epochs10、val/recall@5最大选模、patience10、K10/prefix关闭均保留。
- 原实际train loader记录与A1完全相同，包含原文件分配顺序、worker数、raw batch和空Adam初态。新spec没有checkpoint或resume字段；冻结源码重新instantiate `_fresh_model`，`trainer.fit(...,ckpt_path=None)`。结构记录与A1完全相同：13259136参数全部可训练、独立encoder/decoder SID表各[1024,128]、H4/K10、有SEP、无user table。该初期证明基于固定源码、实际配置和原运行记录；没有声称存在未保存的完整随机初始参数dump。
- 已逐条核128个真实optimizer update（从1开始、step_before从0开始），2048 microbatches，651951 expanded windows；每次严格16 accumulation，所有被监视的encoder SID、decoder SID、BOS、encoder/decoder transformer梯度非零且有限，每次原运行检查optimizer覆盖全部trainable参数且所有可用梯度有限。`returned_training_losses`范围[0.8760535717010498,1.4493671655654907]，不是对原始未缩放loss的额外解释。
- 第一次full validation真实完成于北京时间03:19:01：step100，35598 targets、4450 batches、exact key-target multiset=true，expected evaluation index SHA `87822a4b8b174bd6e4e7f49ca4bc3c413736e4c2630696898c9e488b5cdbac12`，与实际spec一致。val Recall@5为0.0026967807207256556；所有验证metrics有限。没有把第一轮指标当作最终结果或选模终点。
- 新训练的12小时预算从原runner started_at（北京时间03:10:58.920653）开始计算，内部保留180秒，internal_stop为北京时间15:07:58.920653。该安全停止不是科研完成条件，也没有重新获得另一份12小时额度。

## 待终态后必须完成

完整optimizer/validation事件流、真正训练终止原因、验证best及patience、best/final checkpoint完整载荷和CPU strict=True、SID嵌入、双表storage、128份Adam状态和callbacks、所有产物terminal manifest哈希、exit/cleanup/原PID身份和12小时预算，均未在本阶段提前判PASS。

已保存有界终态审核程序：

```bash
python3 -B /home/addaswsw/recommend-support/grid-a2-fd-recovery-20261018/reviews/a2_training/run_terminal_audit.py
```

程序先读终态门：RUNNING时不import模型、不加载checkpoint；失败/INCOMPLETE保留真实结论。只有原job SUCCEEDED/exit0/cleanup verified后，才一次性流式读完整日志并在隐藏CUDA的2-thread CPU进程里instantiate一个模型、strict load原best、读取原final optimizer/callback。没有forward、没有训练、没有GPU调用、没有大checkpoint下载，结果保存在本审核目录；脚本不会提交、停止、恢复或写入原job。

本阶段只定向读取必要原件和最多256条optimizer/validation前缀，两次取证均明确是非原子观察，不频繁轮询。未来持续监视由主控负责；本审核将在主控通知真实终态后继续。终态程序当前仅做了语法检查，尚未在运行中的模型上执行最终checkpoint审核。

机器回执：`stage_review.json`；原始只读快照位于`observation-*`；脚本`check_stage.py`保留A1全字段差异与逐事件检查。没有新增依赖、环境或Git改动；既有A1/scheduler原件未改。

GPT_UPLOAD_REQUIRED: YES（由主控并入最终单一审核ZIP。）
