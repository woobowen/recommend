# LETTER Beauty Shuffle V2 正式启动观察 r0

**SECOND_GROUP_FORMAL_RUNNING_30MIN_OBSERVATION_PASS**。服务器父作业 letter-beauty-v2-group-series-r0 已独立接管全组。观察结束时 S2 继续推理，其余三模式及正式配对待顺序执行。原研究快照 1182dfd914c20bd3dc11786587eae6879fba6b9d，已审本地提交入口 a2a684625ff61f79967ec4152de98526b52936f2。审定草案与实际父计划中的九项子请求完全一致。变化为启动时 GPU 选择快照。 [用户授权摘录](AUTHORIZATION.md) · [实际计划](launch/plan.json) · [审定草案](launch/reviewed-draft.json) · [提交回执](launch/submit.json) · [服务器源码清单](launch/RUN_VERSION.json)

## 研究条件和实证

沿用原 LETTER Beauty 的 22,363 人、12,101 件商品、12,088 个完整 SID 和 11 个碰撞组。新 SID token 965，词表 32,965。原 LLaMA-7B revision 4782ad2… 加载原 step 6160 adapter，448 项 LoRA 实载 FP32，embedding、lm_head 和基座实载 BF16。V2 使用 seed 42、允许恒等，四模式活动位为 (2,3)、(0,1,2)、(1,2,3)、(0,1,2,3)。S3a／S3b 共享相对排列。Prompt0 和 collator 后，只在最近 20 个可见历史内按商品移动完整 token ID。单卡单 rank、batch 1、beam／return 20、最多 10 新 token、原逐码位约束、EOS／cache 和原分数精度均由正式 guard 核实。固定模型测试时推理沿用原 Normal 双片。四次正式评价各覆盖 22,363 用户、447,260 候选、16 指标及原 CCE，最后在同一用户轴完成四个条件减 Normal 和 S3b−S3a 的 5,000 次共享 PCG64 配对抽样。 [冻结资产身份](prior/assets.json) · [完整单卡轴](prior/single-shard.json) · [正式样本和实际 rank](samples/formal-first-six.json)

真实 S2 rank 采样时的 PID 为 3399867，GPU UUID 是 GPU-59cac3bd-3645-668b-79c8-839e0d878893，RTX 4090 物理卡 1 映射逻辑 GPU 0。新连接验证了原父和 S2 子进程身份、作业标记和独立运行。S2 第一条是 row_idx 8755，历史 15 组、20 个完整 token ID 改变。前六条的输入哈希、固定槽、组外 prompt、mask、20 beam 首次 prefill、原生位置和空初始缓存均正确，120 个原精度候选分数有限，token 码位、EOS 和排序通过。 [样本核验](samples/formal-first-six-verified.json) · [非作者样本核验](reviews/first-samples-nonauthor.json) · [非作者启动核验](reviews/launch-nonauthor.json)

## 七次观察

窗口长度 1804.959 秒。T0 为真实 rank 已启动并处于模型加载时的新连接观察。后续目标点以其实际 UTC 时刻为基准。监视每次从上次已确认的换行结束处累加完整 JSONL 记录，当次文件末尾的半行不计。候选数按每条正式写入已有守卫规定的 20 条计算。日志检查限于有来源字节范围的尾部，rank 的原 checkpoint 每 25 个用户更新。七点均为同一 S2 正式子作业和同一 GPU；确认生成阶段持续增长，未发现 OOM、异常退出、重复正式 GPU worker 或第二张卡被本任务占用。准备阶段原人轴已审，正式推荐指标待全量生成后评价。

| 点位 | America/New_York | 模式及阶段 | 完成用户 | 增量 | 文件字节数（增量） | 显存 MiB／利用率 |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| T0 | 12:58:47 | S2 模型加载 | 0/22,363 | 0 | 0（+0） | 10,452／16% |
| T+5 | 13:03:56 | S2 正式生成 | 308/22,363 | 308 | 4,995,015（+4,995,015） | 21,852／87% |
| T+10 | 13:08:52 | S2 正式生成 | 762/22,363 | 454 | 12,566,933（+7,571,918） | 21,852／45% |
| T+15 | 13:13:52 | S2 正式生成 | 1,237/22,363 | 475 | 20,503,652（+7,936,719） | 21,852／18% |
| T+20 | 13:18:53 | S2 正式生成 | 1,723/22,363 | 486 | 28,384,278（+7,880,626） | 21,852／42% |
| T+25 | 13:23:53 | S2 正式生成 | 2,194/22,363 | 471 | 36,164,613（+7,780,335） | 21,852／30% |
| T+30 | 13:28:51 | S2 正式生成 | 2,665/22,363 | 471 | 43,989,610（+7,824,997） | 21,852／14% |

[原始点和统计](observations/) · [机器汇总](observations/summary.json) · [非作者窗口检查](reviews/window-nonauthor.json)

原正式 worker 连续运行，研究和服务器代码保持经审快照。观察工具准备时的一次本地命令因 NUL 字符编码被工具拒绝，改用原样引用后定向复核通过，发生在 T+5 前。没有正式计算修复或重启。

## 预算、估时与恢复

T+30 记录 2,665 人和 53,300 条按合同计算的候选。交付前单独在 13:29:03 复核 2,683 人，输出盘余量 2.700 TB。按 T+5 至 T+30 的实测 94.59 人/分外推，T+30 时 S2 预计剩余 3.47 小时，含后续三模式预跑外推的全组预计剩余 17.05 GPU 小时。后续 CPU 评价各预留 600 秒，配对预留 2,100 秒。GPU 外推排除 CPU、审核和资源排队；缓存、I/O 和负载波动会影响实际时长。

交付检查时，累计正式 GPU 1839.754 device-seconds，剩余 127760.246 device-seconds，包含 S2 装载；预跑累计 648.992 device-seconds；CPU worker 累计 571.625 秒。四个 GPU 子请求的合计上限预留为 115,080 秒。预算口径计算正式子 worker 时间，与父监督时间不重复计量。 [交付时刻原始记录](launch/delivery-observation.json) · [先前估时](launch/preflight-assessment.json)

服务器将在 S2 达到完整、退出和评价门槛后推进 S3a、S3b、S4，并在各模式通过评价后产出最终配对。子任务失败会停止依赖链并保存已生成产物和原回执。本阶段到期停止定时监视，可以关闭本地 Codex、SSH、WSL 或 VPN，服务器原监督进程按请求继续。需要找回时使用下面的只读查询。后续 Codex 分析、恢复、通知及发布以新的在线指令为准。

```bash
/mnt/c/Windows/System32/OpenSSH/ssh.exe -T -o BatchMode=yes -o StrictHostKeyChecking=yes new-server '/usr/bin/python3 -I -S -B /home/bwwu/grid-audits/sid-extensions-p0/scripts/runner-beauty-cbb963fab68c34cc.py query /home/bwwu/grid-audits/sid-extensions-p0/jobs/letter-beauty-v2-group-series-r0'
```

父控制器 summary 保存在 /data/bwwu/lc-rec/letter-shuffle-v2-beauty/group-series-r0/summary.json，S2 正式输出位于同根的 gpu-formal-shuffle2-r0/test/。 [服务器回执及路径](launch/capture-manifest.json)
