# Toys A 双候选真实非作者结果审核

结论：`INDEPENDENT_TOYS_SAVED_RESULT_REVIEW_PASS`。审核者 `/root/toys_result_audit` 为实际子代理，未编写原实验源码、未运行原实验。此结论仅覆盖 Toys A-5300 / A-5000 的本轮已保存结果，不代表 Sports A2 或全任务最终 PASS。

本轮直接读取两候选六组全部预测（116,472 行）、24 组数值 NPZ、对应 JSON、192 PNG、固定 cohort、全量覆盖索引、strict-load / invocation / 有效配置 / 终态与 cleanup 记录；以独立 NumPy/Python 实现复算四指标、逐用户配对、beam 父链及 2,304 行 query-pair 描述统计。没有调用 GPU 或重新加载生成器。

## 身份与实施

- 科研源码：`b44399926c98369191c2ff5c2425377c8c07d3c8`。服务器不可变快照：`/home/bwwu/grid-audits/sid-extensions-p0/grid-a-launch-20261017/snapshots/b6203414ee58a400`，完整 snapshot SHA 为 `b6203414ee58a40034d8a5e9c311ff509d897002804c2cb81f4c83bf419d2d41`。13 个直接审阅源文件从该服务器快照定向读取，逐字节匹配部署 manifest。
- 两个 checkpoint 分别为 step5300 / step5000；本轮只读重验载荷 SHA，未重新加载模型。每个实际 eval / attention 的 strict_load 记录均为 178 state tensors、missing/unexpected keys 均空、shape/value 全相等、embedded SID exact，双表不共享且参数数目 13,259,136。
- 共同 SID SHA：`a487a4bdadef63f3a87386ddce52ec03f4771bedbaf554f63272c45fe373e7d1`，形状 `[4,11924]`。CPU 定向读取全 SID，用于独立核对候选合法性、item 轴与 lookup offset。
- 原 `encoder_forward_pass` 确认先加层级 offset、再 lookup、随后对每个历史 occurrence 的完整向量在前三/四位内随机重排，最后插入 SEP。不改整数 SID、目标 SID 或物品时间轴。原 helper 被原样调用；bounded trace 不额外消耗全局 RNG。
- 每条件 2,427 个实际 test batch（raw microbatch 8，8 workers）均为 CUDA 输入 int64 / 模型 FP32，Normal 的 shuffle_call_count=0，Shuffle3/4=4,854（loss / generate 各 2,427）。固定种子42，K=10，prefix check沿保全源码默认 False。各模型三条件有效配置除预定 shuffle flags / 输出路径外完全一致。

## 正式原指标

| 模型 | 条件 | Recall@5 | NDCG@5 | Recall@10 | NDCG@10 | 非法候选 | 重复候选 |
|---|---|---:|---:|---:|---:|---:|---:|
| toys5300 | Normal | 0.040078300982714 | 0.027255861088634 | 0.061302285641432 | 0.034059926867485 | 175 | 0 |
| toys5300 | Shuffle3 | 0.040232844650745 | 0.026425955817103 | 0.060271997004747 | 0.032848287373781 | 194 | 0 |
| toys5300 | Shuffle4 | 0.040129818022251 | 0.026403736323118 | 0.059859879314899 | 0.032738946378231 | 221 | 0 |
| toys5000 | Normal | 0.040644962340593 | 0.026936978101730 | 0.059859879314899 | 0.033158622682095 | 157 | 0 |
| toys5000 | Shuffle3 | 0.039820730686188 | 0.026538845151663 | 0.059396248310804 | 0.032830830663443 | 140 | 0 |
| toys5000 | Shuffle4 | 0.038841951638460 | 0.026054665446281 | 0.058056872338057 | 0.032222375273705 | 160 | 0 |

全部六条件各覆盖 19,412 唯一用户，(sample_key,target) 全量 multiset 与原 testing 索引精确相等，无重复或遗漏。原始预测 SHA 与正式结果一致。独立 double 精度复算对原 FP32 指标最大绝对差 `1.94e-09`，逐用户指标误差小于 1e-7；所有候选分数均无并列。非法候选原样保留，没有过滤、补齐、改分或重排后再评价。

## 严格配对推荐变化

| 模型 | 条件对 Normal | Top1变化 | Top10顺序变化 | Top10集合变化 | Hit@5 0→1 / 1→0 | Hit@10 0→1 / 1→0 |
|---|---|---:|---:|---:|---:|---:|
| toys5300 | Shuffle3 | 5609 (28.894%) | 19289 | 17427 | 88 / 85 | 102 / 122 |
| toys5300 | Shuffle4 | 6163 (31.748%) | 19361 | 17918 | 101 / 100 | 106 / 134 |
| toys5000 | Shuffle3 | 4994 (25.726%) | 19221 | 16825 | 63 / 79 | 75 / 84 |
| toys5000 | Shuffle4 | 5521 (28.441%) | 19314 | 17433 | 64 / 99 | 75 / 110 |

四组各 19,412 个逐用户配对及完整命中排名迁移已保存为 `*-paired.jsonl.gz`，rank=0 表示 Top10 未命中。聚合 Recall/NDCG 的小变化同时伴随大量个体推荐变化；不能据此称严格置换不变。两 checkpoint 全部保留，不按测试指标择优删除。

## fixed32 注意力和统计

两候选共用同一份 SHA 为 `3a82afccb0d944d537039ae3da51c971c1b6abba2a869052d3197b2e6bbb2ab6` 的 fixed32 evaluation cohort。选择规则是模型输出检查前按实际文件/行顺序取前32个不同用户；三条件完全一致。每条件4批×8人、4个autoregressive query、最后层6个head、150个encoder位置，保存两条路径及head mean。

独立从每步 generated prefix 重建全部 parent_indices，从最终 child0逆向追踪真实Top1完整祖先链；逐项等于已保存父链、Top1 SID和raw attention。local-first路径另按每步parent0独立提取。全 cohort 原输入、mask、历史item轴、SID值、原始层级offset及每次实际shuffle映射全核对。

24个 NPZ / 192 PNG 全核对哈希、数组shape/dtype、来源绑定、raw共同色阶和解码尺寸；对两个固定首样本做人工视觉核对。没有宣称对192张图逐像素独立重新渲染。24批observer关闭/开启的生成 SID / score完全相等、CPU/CUDA RNG相同、参数未变，hook/方法恢复和异常清理均通过。

| 模型 | 条件 | 路径 | mean Cosine | mean Pearson | mean TV | mean JS(nats) |
|---|---|---|---:|---:|---:|---:|
| toys5300 | Normal | final_top1_cross | 0.922560468 | 0.843515832 | 0.208174069 | 0.061342915 |
| toys5300 | Normal | local_first_cross | 0.923674937 | 0.844845803 | 0.208432087 | 0.060495926 |
| toys5300 | Shuffle3 | final_top1_cross | 0.916571580 | 0.835002114 | 0.215473242 | 0.063988954 |
| toys5300 | Shuffle3 | local_first_cross | 0.918029999 | 0.839249607 | 0.217333417 | 0.063669626 |
| toys5300 | Shuffle4 | final_top1_cross | 0.922275906 | 0.843831613 | 0.206119224 | 0.060272369 |
| toys5300 | Shuffle4 | local_first_cross | 0.922612940 | 0.844628808 | 0.209711390 | 0.060629565 |
| toys5000 | Normal | final_top1_cross | 0.835023734 | 0.477951396 | 0.258415310 | 0.085673783 |
| toys5000 | Normal | local_first_cross | 0.837000007 | 0.491397833 | 0.252595881 | 0.083725650 |
| toys5000 | Shuffle3 | final_top1_cross | 0.827439441 | 0.453370045 | 0.266083590 | 0.090103745 |
| toys5000 | Shuffle3 | local_first_cross | 0.828886516 | 0.456017675 | 0.262266093 | 0.087341609 |
| toys5000 | Shuffle4 | final_top1_cross | 0.839915247 | 0.466016631 | 0.257264211 | 0.083375832 |
| toys5000 | Shuffle4 | local_first_cross | 0.842042001 | 0.468111266 | 0.254427217 | 0.080777582 |

每条件每路径保留32人×6个query pair=192行；全量独立复算对producer最大差5.56e-16。masked mass全为0，单head raw row-sum最大误差2.3842e-7。Cosine/Pearson使用可见原值，TV/JS只在计算概率分布时除以可见行和，包含可见SEP，不改矩阵和图。跨条件没有未经对齐直接比较列。完整min/median/max、逐人逐pair记录在JSON中，不只保留均值。

## 科研结论、终态及边界

实施正确、约定范围运行完成、原始结果完整这三项在本审核范围内通过。现象是聚合指标变化相对小，而个体推荐与不同query的注意力可明显不同。没有由attention相似性推出decoder不使用目标prefix，也没有严格不变性证明。

六eval、六attention、两stats的保存状态均SUCCEEDED/exit0、cleanup verified/owned group empty；这是原runner终态原件，本审核没有用exit0替代内容验收。当前服务器整体GPU/进程保护及剩余额度由主控负责。

Toys上游历史producer仍不能唯一确定：Qwen/3584配置与保全512维producer之间的历史绑定缺口未补造。两候选结果仅按其实际checkpoint/SID/源码绑定解释。本次未重审未变化的Beauty/B档案，没有模型训练、generation或GPU复评，没有安装新依赖或修改配置。

## 可审查原件入口

- `toys-independent-review.json`：全量结构化验收、原指标、独立指标、配对、全部attention统计及身份。
- `toys-supplemental-review.json`：实际生命周期、observer对照、来源和明确审核范围。
- `raw/REMOTE_MANIFEST.json`：568个定向收集成员的远端路径/大小/SHA；`collection-receipt.json`给压缩传输校验。
- `scientific-source-server-bindings.json`和`scientific_source/`：实际旧快照的13份源码与两checkpoint/原config只读SHA校验。
- `legal_sid_cpu_export.json`：小型完整SID数值、来源和形状；未包含大模型。
- `audit_toys.py`、`finish_review.py`和六个`*-independent-statistics.json`：可独立复跑的CPU验收与原始复算。

GPT_UPLOAD_REQUIRED: YES（由主控合并这些原始证据到唯一最终审核 ZIP，不要求用户另上传本子任务散件）。
