# GRID G3 Beauty Shuffle V2

第三组检验固定 G3 Beauty LLaMA-7B 推荐器对历史商品内部 SID 顺序的敏感性。第一、二组已经通过网页二审。第三组以 G3 自己的 Normal 为对照，正式对比为四种 Shuffle V2 分别减 Normal，以及 S3b−S3a。

## 资产与干预

固定上游采用 GRID 的三位量化码与原第四位去重位，沿已审商品桥接转换为 LETTER-LC-Rec Beauty index。测试集 22,363 人、商品与唯一 SID 各 12,101、碰撞组 0。词表 32,786，含 786 个新增 token。下游沿用原 SeqRec Prompt0、最近至多 20 件历史和原 LLaMA-7B revision 4782ad278652c7c71b72204d462d6d01eaaf7549。G3 原四 epoch、8,214 次更新，根据验证 loss 选中 checkpoint-6160。冻结 Normal 使用同一 G3 权重，单 rank 22,363 条、447,260 个原始候选。 [原 G3 生成入口](https://github.com/woobowen/recommend/blob/21d495b0be5b606853ec7cec68a3eaa1b58acbd7/lc_rec/a0_beauty/formal_evaluate.py) · [原 CPU 收尾](https://github.com/woobowen/recommend/blob/7368d301d0d928f77d50551c3b53a5b30c23d63f/lc_rec/a0_beauty/grid_sid_preparation/g3_conversion/production_preparation/cpu_closure/results/README.md)

| 模式 | 活动槽，从 0 开始 | 固定槽 | 字典序排列数 |
| --- | --- | --- | ---: |
| S2 | 2, 3 | 0, 1 | 2 |
| S3a | 0, 1, 2 | 3 | 6 |
| S3b | 1, 2, 3 | 0 | 6 |
| S4 | 0, 1, 2, 3 | 无 | 24 |

协议 GRID-LETTER-SHUFFLE-V2，shuffle seed 42，allow_identity。对每次可见历史商品出现，计算紧凑 JSON 键 [protocol,42,m,row_idx,visible_item_idx,counter] 的完整 SHA-256，按无符号大端整数做拒绝采样，余数索引字典序全排列。三位模式共享相对排列，按 new[A[j]] = old[A[perm[j]]] 应用。row_idx 为全量冻结轴行号，32 人预跑沿用原编号与样本。自然恒等直接保留。

原 G3 tokenizer 与 collator 构建 Prompt0 后，在原生 generate 前，对 seq[:-1][-20:] 中每件历史商品的四个完整 SID token ID 做组内置换。固定槽、组外 token、组间顺序、目标 SID、长度和 mask 延续原生成接口。移动后的 token 使用当前原生位置与空初始 KV cache，RoPE 配对和因果上下文随之变化。原逐码位 callback、canonical 目标、EOS 2、pad 0、稳定分数排序与候选保留策略沿用。单 GPU 单 rank，batch 1，beam/return 20，最多 10 新 token，确定性 beam，early_stopping=True，use_cache=True，filter_items=False，无人工 padding。 [冻结 V2 core](https://github.com/woobowen/recommend/blob/6212bb302ff4f8db87232ca1e7993f411961eac9/lc_rec/shuffle_v2/core.py) · [V1 排除恒等的历史说明](../../README.md#历史-v1)

## 代码与审核

- [a0_beauty_g3.json](../../../shuffle_v2/a0_beauty_g3.json) 和 [g3_assets.py](../../../shuffle_v2/g3_assets.py) 固定原实现、数据、Normal 和模型。原训练入口版本是 21d495b0be5b606853ec7cec68a3eaa1b58acbd7。
- [g3_replay.py](../../../shuffle_v2/g3_replay.py) 使用冻结 tokenizer/collator 重建完整 22,363 人输入并逐行对照 Normal，逐出现与独立 reference 比较。旧 V1 OFF 路径用作输入结构验证。
- [g3_generate.py](../../../shuffle_v2/g3_generate.py) 派生原 G3 入口，经 generate_one → g3_transform → core.transform。验证原 FORMAL adapter 的训练来源，同时标明新 V2 运行 commit 和 PREFLIGHT 输出 scope。进模型前绑定 CPU manifest 的行、模式、输入及排列。装载核对 450 项 tensor、448 项 LoRA、288,571,392 个元素。基座、embedding、lm_head 为 BF16，LoRA 为 FP32。实际 active embedding 与 decoder prefill 核查全部 20 个 beam、input IDs、mask、position IDs 和空初始 cache，逐行检查参数身份、storage 和变更计数。
- [g3_evaluate.py](../../../shuffle_v2/g3_evaluate.py) 复用冻结评价器和 CCE，对 16 个指标独立复算。SID Hit@1/5/10/20、NDCG 和 MRR@5/10/20，ItemHit 和 ItemNDCG@5/10/20。目录外和重复候选保留原位置，按原 CCE 碰撞展开和均匀 credit 计算。五项比较在配对用户轴上共享 PCG64 seed 20261113 的 5,000 次有放回抽样，95% 线性百分位区间。32 人预跑统计用于工程核验。
- [g3_control.py](../../../shuffle_v2/g3_control.py) 固定独立 job、单 GPU 串行、预算、CUDA UUID 和服务器回执。派生 runner 仅修改既有已审 runner 的作业前缀，保留 uid、PID、进程创建时间、pidfd 和清收验证。
- [g3_formal.py](../../../shuffle_v2/g3_formal.py) 构造本地正式计划。每个正式模式拟覆盖 22,363 用户和 447,260 个候选，按 S2→S3a→S3b→S4，每个 GPU 作业成功并清收后进入 CPU 核验及评价，最后运行五项配对。正式单项最多 8 小时，整组最多 36 GPU-device-hours。父作业及各子作业待网页二审和用户另行授权。断线优先查原 ID。 [实际准备、回执与未提交计划](preflight/README.md)

## 离线重算与权重获取

审核 ZIP 包含冻结 Normal 原始候选、真实四模式预跑候选及边界、全量 CPU manifest、数据和 tokenizer。按照包内 README 运行 [g3_offline_review.sh](../../../shuffle_v2/g3_offline_review.sh)，可重算全量 CPU 输入、冻结 Normal、四模式边界以及评价配对。离线重放从冻结 tokenizer/collator 保存的输入开始。服务器实际运行另有当前完整 tokenization 重建证据。

固定 checkpoint 复评使用服务器 /data/bwwu/lc-rec/g3-formal-20261118/21d495b0be5b/training/best_adapter 与配置所列 LLaMA-7B revision。权重需要该服务器读取权限，按固定配置、模型身份和 best_selection 回执核对。本轮沿用原 G3 upstream，无重新选模、模型训练、文本编码、SID 重生成或 Normal/OFF GPU 推理。
