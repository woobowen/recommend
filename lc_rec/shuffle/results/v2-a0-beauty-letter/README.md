# LETTER Beauty Shuffle V2 第二组

**正式前准备已通过，等待网页 GPT 二审及独立正式启动授权。** 真实预跑作业 `letter-beauty-v2-gpu-preflight-r1` 以退出码 0 结束并核验清收。服务器 jobs、requests 和输出目录的只读核查均显示，本轮正式父子请求提交数和正式 GPU 新增量为 0。 [预跑终态](prep/preflight-r1-status.json) · [正式路径核查](prep/formal-inventory.json)

## 条件与接续点

冻结原 LETTER Beauty 的 22,363 名用户、12,101 件商品、12,088 个 SID、11 个碰撞组和总词表 32,965。沿用原 LLaMA-7B、step 6160 best adapter 和两个不等长的 Normal 分片。实际 GPU adapter 严格核对的 450 项张量中，448 项 LoRA 为 FP32，embedding 和 lm_head 为 BF16，基座为 BF16。原 Beauty 运行来源为 `6e1b211d9c37f358c43b1540b40b17f6703150d2`，V2 预跑快照为 `1182dfd914c20bd3dc11786587eae6879fba6b9d`，真实预跑 CPU 评价快照为 `95cc10dd2411cd9567f89e4a5952fc8ed9131385`。 [方法和复现](../../../shuffle_v2/beauty_reproduction.md) · [上一轮 PARTIAL_BLOCKED 历史](https://github.com/woobowen/recommend/blob/25b3c341e4fe6114d8bf899e76f3d36021601d7a/lc_rec/shuffle/results/v2-a0-beauty-letter/README.md)

原 `cpu-replay-r3` 的每模式 22,363 行和 159,966 个历史组，以及旧 Normal 原双片 447,260 候选和 357,808 个逐用户指标值，复用原快照与已验结果。此次核对 replay 输入清单、原 Normal 复评和正式评价源码身份。V2 保留 seed 42、四模式中包括恒等的字典序排列，S3a 和 S3b 共享同一相对排列。Prompt0、最近 20 个历史、batch 1、beam/return 20、最大 10 个新 token 和原逐码位回调沿用冻结合同。 [继承证据与原路径](prep/inherited-evidence.json)

## 恢复及真实预跑

准备工作接续此前三轮，修正记录从第 4 轮起。原 GPU r0 异常直接归因为缺失 `CUDA_HOME`。对照原冻结请求还查实 `LD_LIBRARY_PATH` 缺失。恢复两值后，GPU 请求、主包装器和真实 rank 的有效环境一致。新 Beauty runner 用同一个 Beauty 前缀约束发现和发信号，保留 UID、启动时间、boot、双任务标记、PGID/session 和 pidfd。隔离的实际生命周期测试覆盖普通退出、跨 session 残留 TERM、超时 TERM→KILL 和同时运行的对照进程。 [生命周期](prep/lifecycle.json) · [执行非作者核验](prep/nonauthor-execution.md) · [真实回执读回](prep/runner-readback.json)

| 条件 | 用户数 | 原始候选 | 自然整条恒等 | 严格装载秒 | 样本生成秒 |
| --- | ---: | ---: | --- | ---: | ---: |
| S2 | 32 | 640 | 行 20014，分数最大差 0.0 | 116.457 | 25.253 |
| S3a | 32 | 640 | 0 | 116.499 | 22.067 |
| S3b | 32 | 640 | 0 | 116.770 | 22.035 |
| S4 | 32 | 640 | 0 | 116.574 | 25.266 |

每个模式通过实际 20 beam 的首次 prefill、mask、原生位置、空初始 cache、移动后 embedding 查表及 canonical 生成约束检查。S2 的自然恒等样本与归档 Normal 的输入、候选、顺序及原精度分数一致，覆盖限于该用户。共计 128 个模式–用户样本和 2,560 条候选。 [预跑原汇总](prep/preflight-summary.json) · [研究非作者核验](prep/nonauthor-research.json)

实际 CPU 接口读取四个单片预跑输出和冻结 Normal 的同一 32 人，检查原 16 项及 CCE。样本级五项比较运行了 5,000 次共享 PCG64 抽样。独立实现复核了全部 80 项工程输出，区间端点最大绝对差 `1.39e-17`。正式入口仍要求 22,363 人，正式指标和配对结果待实际计算。 [评价工程回执](prep/preflight-evaluation.json) · [统计非作者核验](prep/nonauthor-statistics.json)

## 未提交的正式计划

草案包含同一张卡依次 S2→S3a→S3b→S4，每项生成成功后 CPU 评价，再运行五项 16 指标配对。草案在 WSL 构建，服务器复查无正式父子请求。未来提示词须明确授权启动，复查卡可用性、代码和资产身份、预算及运行依赖后才可走正式提交路径。 [预跑门槛](prep/prelaunch-gate.json) · [正式计划](prep/formal-plan.json)

按实测 32 人生成 wall time 外推，加测得的装载和其他开销，四模式分别约 4.94、4.32、4.32、4.94 小时，整组约 18.52 GPU 小时。增加 25% 敏感性系数后最慢条件约 6.18 小时、整组约 23.15 小时，符合参考的 8 小时单次和 36 GPU 小时累计上限。固定样本的平均输入略长，长期 IO、缓存及资源竞争仍会影响外推。正式 raw JSONL 投影约 1.97 GB；计划核查时输出盘约有 2.70 TB 可用。正式评价每项和最终配对分别预留 600 秒、2,100 秒 CPU 时限，审核时间单列。累计预跑实际 648.991733 device-seconds；审阅导出前累计 CPU worker wall time 538.158041 秒，包含上一轮并保留失败记录。 [估时及容量](prep/assessment.json)
