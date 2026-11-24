# Instruments A0 允许恒等排列的 Shuffle V2

`wbw-lc-rec-shuffle-v2` 将历史 LETTER V1 的干预扩展为包含恒等项的组内排列，研究同一固定模型及完整测试用户轴上不同码位扰动的结果。GitHub activity 在 2026-11-20T16:15:35Z 记录 `woobowen` 创建该研究分支，研究交付 `692cd94d6acd17292c2bdef7fabc6bb50ec9ba5d` 含启动和七次初始观察。整理前远端版本 `50507089981cb0b9d36e8977bd1d94cd9085f031` 属于之前的规范发布。分支继承 G3 Beauty 结束时的 V1 历史。Instruments A0 和原 LETTER Beauty 的正式系列均已完成。[仓库活动](https://github.com/woobowen/recommend/activity)

本轮 Shuffle V2 四组正式实验的结果均已通过网页审核。各组入口保留各自的运行记录与历史审核版本。

## 第一组正式结果

第一组研究及原结果交付已通过网页二审，状态为 `FIRST_GROUP_COMPLETE_WEB_REVIEW_PASS`。本轮说明和新审核包待网页差异复核。 [整组结果、五项配对、原作业与复现入口](../shuffle/results/v2-a0-instruments/final-20261121/README.md) · [本轮收尾记录](../shuffle/results/v2-a0-instruments/final-20261121/closeout/README.md)

## 第二组 LETTER Beauty

原 LETTER Beauty 的四项正式生成、四项 CPU 评价和五项配对已通过网页 GPT 二审，状态为 `SECOND_GROUP_COMPLETE_WEB_REVIEW_PASS`。每个条件覆盖 22,363 名用户。 [正式结果](../shuffle/results/v2-a0-beauty-letter/final-r0/README.md) · [第二组总入口](../shuffle/results/v2-a0-beauty-letter/README.md) · [正式启动观察历史](../shuffle/results/v2-a0-beauty-letter/launch-observation-r0/README.md)

## 第三组 GRID G3 Beauty

[G3 Beauty Shuffle V2](../shuffle/results/v2-a0-beauty-g3/README.md) 复用 G3 自己的单 rank 冻结 Normal、tokenizer 和最佳 adapter。该入口保存本轮准备、实际审核证据及未提交的正式计划。

## 第四组 Instruments A1 Qwen3-0.6B-Base

第四组整组完成，两向非作者终审及网页 GPT 最终二审通过，状态为 `FOURTH_GROUP_COMPLETE_WEB_REVIEW_PASS`。 [正式结果](../shuffle/results/v2-a1-instruments/final-r0/README.md) · [方案与预跑](../shuffle/results/v2-a1-instruments/README.md) · [正式启动观察](../shuffle/results/v2-a1-instruments/formal-launch-observation-r0/README.md)

## 历史只读观测

截至 `2026-11-21T07:22:39.422045+00:00`，服务器保存的父 job `letter-a0-20261112-v2-20261120-group-series-r0` 仍为 `RUNNING`，正式运行源 `e2f54ebd2d7709821de1071693f91644f5ed9a30`。S2、S3a、S3b 各完成 24,772 人，三组 GPU 生成和对应的 CPU 评价均退出 0 并完成清收；S4 仍运行，rank0 实际记录 5,725／24,772 人。四条件全量评价和配对未见终态。此处是带时间的观测，文末历史观测及原运行提交保留各自时点。[只读观察依据](../shuffle/results/v2-a0-instruments/repository-delivery-20261120/FINAL_CLOSEOUT_OBSERVATION.json)

## 实际生效方法

`core.transform` 先沿用冻结 V1 OFF 的组、词表、padding 与 mask 验证，随后独立置换每次 history item occurrence 的完整 token IDs。键为 `[协议, 42, 活动槽数量, 全局用户 row_idx, 可见物品出现下标, 拒绝计数]`，使用规范紧凑 JSON 的 UTF-8 SHA-256，拒绝超出均匀取模界的 digest。Shuffle3a 和 Shuffle3b 采用同一相对排列，映射到不同槽位。全输入和每组的 `nonidentity` 与 `tokens_changed` 有各自证据。S2 活动槽为 2、3，S3a 为 0、1、2，S3b 为 1、2、3，S4 为 0、1、2、3，槽位均从 0 起数。理论恒等抽取概率依次为 1/2、1/6、1/6、1/24。

扰动发生在原 tokenizer/collator 完成后与 `generate` 之前。保留 item 时间顺序、组边界、组外 token、mask、合法码位约束和 canonical target。扰动后的相对 token ID 顺序决定模型实际看到的绝对位置、RoPE 配对及因果上下文。Beam20/return20、Prompt0、10 新 token、确定性 beam 搜索、原候选约束、原稳定分数排序、EOS2/pad0；原 `filter_items=False` 允许目录外完整组合，保留重复并按冻结 CCE 定义处理 item collision。所有指标、五个配对及共享 PCG64 seed 20261113、5,000 次有放回用户 Bootstrap 在完整终态中计算。冻结 Normal 仍由原双 rank 的预测按 row_idx 匹配，四组新扰动均单 rank。

## 模块和输入定位

| 模块 | 实际职责 |
|---|---|
| [core.py](core.py) 和 [reference.py](reference.py) | 前者生产排列与输入证据，后者独立实现抽样和 active-slots 映射，用于逐组验证。 |
| [runtime.py](runtime.py)、[entry.py](entry.py)、[beauty.py](beauty.py) | 约束输入形状、原生 prefill 和词嵌入，校验冻结源入口，导出派生生成文件与 diff；Beauty 适配已通过真实 GPU 预跑、四模式正式运行和全量边界核验。 |
| [replay.py](replay.py)、[single_shard.py](single_shard.py) | 读取冻结 LETTER data、tokenizer、Normal 原预测全轴，确定行索引和 raw/shuffled 流；单 shard 与冻结双 shard 在固定 sampler 的行顺序关系由 CPU 审核验证。 |
| [generate.py](generate.py)、[evaluate.py](evaluate.py)、[statistics.py](statistics.py) | 生成使用冻结权重、词表及运行契约，评价独立核对预测覆盖、目标、目录与单/双 rank 入口，最后对完整矩阵做配对。 |
| [control.py](control.py)、[series.py](series.py)、[series_control.py](series_control.py) | 固定 request、预算及依赖，服务器按 S2 → S3a → S3b → S4 推进。每条件先经过完整生成和 CPU 评价门槛，再继续下一个。 |
| [collect.py](collect.py)、[package.py](package.py) 与测试模块 | 收集原 job、stage、资源和派生实际文件，按完成状态打包本实验所需材料，CPU fixtures 与单元测试检查置换、边界、评价和统计接口。 |

配置 [a0_instruments.json](a0_instruments.json) 固定实际 Base 模型 `huggyllama/llama-7b@4782ad278652c7c71b72204d462d6d01eaaf7549`、最佳 A0 adapter（step8240、SHA-256 `f4651c5c6e6ed841cd3384c0c8ebfab30880e94597e515ffa21916b876001ed5`）、数据 index/inter/item 摘要、918 SID token 词表语义摘要，以及冻结双分片 Normal。正式研究 source `e2f54ebd2d7709821de1071693f91644f5ed9a30`，串行控制 `4f5758ffb578f3a8d3d06a54262f59d3f63db370`，服务器导出由[部署回执](../shuffle/results/v2-a0-instruments/launch-observation-20261120/e2f54ebd2d7709821de1071693f91644f5ed9a30.deployment.json)映射。固定解释器为 `/data/bwwu/envs/letter-20261110/bin/python`，Python 3.10.22、Torch 2.6.0、Transformers 4.31.0、PEFT 0.5.0、Accelerate 0.21.0，正式生成使用 BF16 Base 和严格载入的 LoRA 及 adapter 内 embedding/head。[实际首行](../shuffle/results/v2-a0-instruments/launch-observation-20261120/formal-shuffle2-first-row.json)和[原串行计划](../shuffle/results/v2-a0-instruments/launch-observation-20261120/series-plan.json)可追溯最终生效条件。

## 三阶段证据和复现范围

- 前置。完整 24,772 用户、每模式 172,929 组历史在原 tokenizer 上 CPU replay，模式数与恒等计数可查 [CPU 结果](../shuffle/results/v2-a0-instruments/cpu-replay-r1/audit.json)。两份冻结 Normal 原预测各 12,386 用户，已有 CPU 重评的 24,772×16 NPZ 与冻结 Normal 同字节。四模式各 32 用户的独立 GPU 预跑与 prefill 边界审查保留在 [启动记录](../shuffle/results/v2-a0-instruments/launch-observation-20261120/README.md)。2026-11-20 仓库整理时再次执行 `python3 -B -m unittest lc_rec.shuffle.test_core lc_rec.shuffle_v2.test_protocol`，7 项通过。[实际复核输出](../shuffle/results/v2-a0-instruments/repository-delivery-20261120/verification.txt)
- 历史过程快照。原父作业 `letter-a0-20261112-v2-20261120-group-series-r0` 已脱离 SSH。首次 30 分钟记录后，2026-11-20 通过原 SSH alias 读取 [过程快照](../shuffle/results/v2-a0-instruments/repository-delivery-20261120/observation.json)，时间为 `2026-11-20T20:34:09Z`，S2 同一 GPU worker `3367671` 已记录 21,200/24,772，状态为 RUNNING，输出文件观测大小 345,033,673 bytes。快照含原父子 status、rank 回执、有限日志尾、首条完整预测。文件读取发生在该报告的起止时刻之间，运行中输出可继续变化。
- 后置。2026-11-20 的早期过程快照当时尚无单条件终态。后续正式终态与系列配对见上方整组结果入口。后续只用原 ID 查询，位置为 `/home/bwwu/grid-audits/sid-extensions-p0/jobs/<job_id>/`。保留现场 controller 与其阶段原件。编排使用 [runner 仓库源码](../a1_qwen3_06b/scripts/runner.py)，与 [服务器部署原件](../shuffle/results/v2-a0-instruments/repository-delivery-20261120/runner-a1-96h-1fefc09a15c1cb73.py) SHA-256 一致。父作业的计划、CPU、GPU 与收尾状态参照 [启动记录](../shuffle/results/v2-a0-instruments/launch-observation-20261120/README.md)。

保存输出复查可从上方整组结果进入。四条件的全量预测索引、评价、配对和离线复算材料已经完成核验。固定 checkpoint 复评需原 LETTER 资产、冻结 base 与 adapter、上述导出、配套 evaluator 和 interpreter，资产取得需要相应授权。重跑本实验从同一 checkpoint 产生四种干预测试，不包含重训；本轮整理未启动重复运行。上游数据和 SID 采用固定 LETTER 发布资产，未执行 Qwen embedding 或量化重建。


2026-11-21T06:12:43Z 的[原作业回查](../shuffle/results/v2-a0-instruments/repository-delivery-20261120/observation-20261121.json)记录 S2 与 S3a 的生成和评价子作业 exit 0，父作业和 S3b 生成仍在运行。该次按实际状态归档，完整系列指标及配对结论仍以原作业终态为准。
