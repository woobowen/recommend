# LETTER 官方 SID + LETTER-LC-Rec：Beauty / Instruments

本轮从第一代 LLaMA-7B 基础权重重新训练；不把六步短测 adapter 当成正式模型初始化。官方 SID、交互和文本信息均使用提交 `8d0154e28de37dbb6e24871c508ad8ddb1921cda` 的配套文件，不重训上游、不删 SID 冲突、不改变用户与商品映射。

## 正式配置

| 项目 | Instruments | Beauty |
|---|---|---|
| GPU | 0、1：一个模型切分两卡 | 2、3：一个模型切分两卡 |
| 作者补充任务 | seqrec | seqrec,item2index,index2item |
| 数据量 | 24772用户、9922商品 | 22363用户、12101商品 |
| 训练样本 | 131837 | 155615（131413+12101+12101） |
| 每用户验证提示词 | 固定抽取2个，共49544样本 | 固定抽取2个，共44726样本 |
| epoch | 4 | 4 |
| 物理 micro-batch | 8 | 1 |
| 梯度累积 | 8 | 32 |
| 有效 batch | 64 | 32 |
| 训练温度 | 0.8 | 1.0 |
| 学习率 | 2e-4 | 2e-4 |
| LoRA | r8/alpha16/dropout0.05，Q/K/V/O及MLP | 同左 |
| 其他训练模块 | embedding、lm_head 活跃副本 | 同左 |
| 损失 | response-only CE，logits除以温度 | 同左 |
| 优化器、调度 | AdamW、weight decay0.01、cosine、warmup0.01 | 同左 |
| 最大历史 / token | 20件 / 2048 | 同左 |
| 训练/测试精度 | Int8主干+FP16计算 / 未量化FP16 | 同左 |
| 随机种子 | 训练42，shuffle2026 | 同左 |

按作者补充配方的四卡布局，有效 batch 分别为64、32。本轮单进程两卡切分，用梯度累积保持有效 batch；FP16替代2080 Ti不支持的BF16，采用原生SDPA，不调用FastChat FlashAttention或DeepSpeed。实际使用作者模型、数据集、提示词、损失和评价函数。物理 micro-batch 改变会影响混合任务中按 token 平均的微批损失权重，不能声称数值轨迹与原四卡环境逐位一致。

## 长输入与显存检查

正式 Trainer 启动前检查真实最长历史样本，以及 Beauty 两种文本辅助任务的2048-token长样本。Beauty物理batch2在长文本上OOM，改为batch1，并把累积改为32，不缩短2048上限。

作者原始collator对少数超长文本提示词右截断后会截掉整个答案。本轮仅修正没有任何监督标签的行：保留提示词开头与末尾的Response标记，缩短中间文本，为原回答留出空间；普通行原样调用作者collator。训练记录`repaired_long_prompt_rows`。不修改官方数据文件。

## 训练、保存与自动测试

每epoch对全体验证用户的两条固定抽样提示词评估，按最低eval_loss选最佳。每100个优化器更新额外保存恢复点，保存本身不触发额外验证，不改变选模标准；保留最佳与最近checkpoint。保存optimizer、scheduler、GradScaler、Python/NumPy/Torch和两张卡的CUDA RNG状态，`COMPLETE.json`标记完整checkpoint。

两个独立后台pipeline持有对应GPU锁。训练退出成功且`training_result.json`确认完整后，自动新进程加载同一个最佳adapter，依次测试unshuffle、shuffle_3、shuffle_4。若训练失败则不进入测试，并记录错误，不自动掩盖异常重启。

测试覆盖全部用户，单用户一个真实下一商品目标，无负采样；固定Prompt0、beam20、batch1、四个SID token加EOS。按作者公开代码的逐位允许token集合解码；不额外过滤或补齐不在目录的组合。指标按完整SID匹配，冲突商品不可区分。主指标Recall@5/10、NDCG@5/10。

shuffle严格复用GRID `plan_v1_1`：seed2026、dataset小写、实际user_id、可见历史商品位置，以同一SHA256字符串取模选择全部6/24种排列，允许原序。仅改变历史商品内部SID token顺序；商品时间顺序、提示词、mask、目标和生成顺序保持原样。仅在测试时启用。

测试每100用户原子保存一份预测JSON，20用户更新进度；中断恢复时核对checkpoint/数据/代码/协议manifest，按连续用户位置读回，避免重复或遗漏。六组全量测试完成后在实验组目录生成只包含指标的`metrics.md`，附相对unshuffle的百分比。

## 入口与证据

服务器根目录：`/home/jcy/GRID_wmh_Qwen_TIGER_20261009`。

任务目录：`codex/letter_lc_rec_formal_20261121`。`latest_run.json`指向本轮实验组，`last_status.json`为最近查询快照。实验组`runner/`为冻结脚本和SHA256，`launch_manifest.json`记录GPU与进程；各数据集有capacity预检、training、testing和pipeline日志。

查询使用独立环境：

```bash
/home/jcy/GRID_wmh_Qwen_TIGER_20261009/environments/letter_lc_rec/bin/python /home/jcy/GRID_wmh_Qwen_TIGER_20261009/codex/letter_lc_rec_formal_20261121/inspect_progress.py
```

本轮正式队列的存在和早期健康检查不代表训练已经完成；全量测试指标必须等待`testing/evaluation_result.json`的complete标记。
