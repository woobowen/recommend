# 第四组 Instruments A1 Shuffle V2 正式启动与观察

**FOURTH_GROUP_FORMAL_RUNNING_30MIN_OBSERVATION_PASS**

服务器 `new-server`（`zwlab-01`）已接受父任务 `a1-instruments-v2-formal-parent-r0`。T+30 时 `a1-instruments-v2-formal-gpu-shuffle2-r0` 保持 `RUNNING`，已写入 4,758／24,772 条完整用户记录。七点实际跨度 1820.541 秒。新的独立 SSH 在 2026-11-23T12:45:01.279071-05:00 确认服务器接管和原 rank 继续写入后，本轮主动观察结束。完整 S2 指标、S3a／S3b／S4 及 paired 待原父任务推进。

## 已审计划与实际请求

本次用户授权覆盖四个正式 Shuffle 条件、对应 CPU 评价及最终五项 paired。九阶段依次为 S2 GPU → S2 CPU → S3a GPU → S3a CPU → S3b GPU → S3b CPU → S4 GPU → S4 CPU → paired。后继阶段要求前一阶段退出成功、产物校验通过并经 runner 清收。

- 固定计划 SHA-256 `f37fbe7ac295480a181daf2fed098810bd5b2d3c9499e19a7eec67d160603c7b`，授权 SHA-256 `d4c86662ccfbd46fe59bb03d1945ab8862783b2ecd3b955f4d100b6eadc36a94`。已接受的父请求和 S2 请求按 canonical JSON 哈希核对。 [已审计划](../formal-plan.json) · [本轮请求证据](observation-summary.json)
- GPU 运行源 `c487f7e7176bc223445d3deecfab87e0bcc30d1a`，CPU 评价／控制源 `3cc1110097c4764ebfd93bb38ee2913c8afb6958`。现场对两个 export 的每个固定文件重新核对 SHA-256。 [GPU 部署](../c487f7e7176bc223445d3deecfab87e0bcc30d1a.deployment.json) · [CPU 部署](../3cc1110097c4764ebfd93bb38ee2913c8afb6958.deployment.json)
- 父任务提交于 `2026-11-23T17:14:34.317352+00:00`，接受于 `2026-11-23T17:14:35.052461+00:00`，worker 启动于 `2026-11-23T17:14:35.144763+00:00`。S2 接受于 `2026-11-23T17:14:36.248383+00:00`，worker 启动于 `2026-11-23T17:14:36.409576+00:00`。时间均为 UTC。
- 父 supervisor／worker PID 3437877／3437878，S2 supervisor／worker PID 3437914／3437915。模型 PID 3437932，`WORLD_SIZE=1, RANK=0, LOCAL_RANK=0`，物理 GPU 1，逻辑 CUDA 0，UUID `GPU-59cac3bd-3645-668b-79c8-839e0d878893`。

## 固定方案

基座为 Qwen/Qwen3-0.6B-Base revision `da87bfb608c14b7cf20ba1ce41287e8de496c0cd`，未量化 BF16、native SDPA，原 best adapter checkpoint-8240（394 个保存键、392 个 BF16 LoRA 键、tied embedding/head）。冻结 LETTER Instruments 用户轴共 24,772 人，商品 9,922、不同四位 SID 9,897、碰撞组 23，沿用原 tokenizer 的 918 个新增 token 与 Prompt0。

四条件使用 `GRID-LETTER-SHUFFLE-V2`、`allow_identity`、seed 42。S2 活动槽 2／3，S3a 为 0／1／2，S3b 为 1／2／3，S4 为 0／1／2／3。S3a 与 S3b 共用相对排列。按全局 row_idx 与可见历史出现位置，用紧凑 JSON、SHA-256 和拒绝采样改变 `seq[:-1][-20:]` 的组内完整 token ID。固定槽、组外 Prompt、目标、输入长度与 mask 保持。生成采用原生位置、空 cache、20 beam／20 candidates、batch 1、max_new_tokens 10、early_stopping、deterministic、`use_cache=True`、原逐码位 Allowed、EOS／PAD 151643、稳定分数顺序、`filter_items=False`。

冻结 Normal 双分片用于比较，原先的 32 人 GPU 预跑和本次 S2 全量计划的预测目录分别保存。正式 CPU 按原 16 项 SID 和 CCE Item 指标评价。五项 paired 使用 5,000 次共享用户有放回 bootstrap、PCG64 seed 20261113 和线性 95% 百分位区间。 [前置结果与方案](../README.md)

## 七次观测

| 点位 | 纽约（EST） | UTC | 阶段 | 完整用户／总数 | 新增 | 用户／秒 | 预测字节（增量） | GPU 总／本进程 MiB | 瞬时利用率 | 日志最近检查点 |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| T+0 | 12:14:38.706 | 17:14:38.706 | S2 初始化 | 0／24,772 | 0 | — | 0（+0） | 1／0 | 0% | 0 |
| T+5 | 12:19:59.168 | 17:19:59.168 | S2 生成 | 857／24,772 | 857 | 2.674 | 13,810,344（+13,810,344） | 2,986／2,976 | 24% | 850 |
| T+10 | 12:25:01.891 | 17:25:01.891 | S2 生成 | 1,532／24,772 | 675 | 2.230 | 24,668,619（+10,858,275） | 2,986／2,976 | 12% | 1,525 |
| T+15 | 12:30:00.045 | 17:30:00.045 | S2 生成 | 2,358／24,772 | 826 | 2.770 | 38,138,951（+13,470,332） | 2,986／2,976 | 12% | 2,350 |
| T+20 | 12:34:56.540 | 17:34:56.540 | S2 生成 | 3,163／24,772 | 805 | 2.715 | 51,184,132（+13,045,181） | 2,986／2,976 | 12% | 3,150 |
| T+25 | 12:39:57.031 | 17:39:57.031 | S2 生成 | 3,947／24,772 | 784 | 2.609 | 63,848,214（+12,664,082） | 2,986／2,976 | 26% | 3,925 |
| T+30 | 12:44:59.246 | 17:44:59.246 | S2 生成 | 4,758／24,772 | 811 | 2.684 | 77,001,700（+13,153,486） | 2,986／2,976 | 16% | 4,750 |

采样按实际时刻计算吞吐。JSONL 计数仅包含换行已写完且按冻结用户轴成功解析的记录。rank 日志每 25 用户保存同步检查点。T0 模型处于初始化，零预测属于该现场状态。

## 模型边界与两方向审核

- 非作者 A 直接复核预测 JSONL 第一至第六行、CPU manifest、冻结 tokenizer 和 Normal 对照。全局 row_idx 依次为 5928, 12379, 12005, 11239, 23303, 14124，共 120 个有限 FP32 候选。绝对位置、mask 和 20 beam 输入摘要与正式 prefill 回执一致。S2 活动槽按协议重建，通过固定槽、目标、组外守恒、原 Allowed、EOS、稳定排序核验。审核 A 通过另一次 SSH 复读全部六个非空预测区间，核对相同的完整行数、字节边界与 SHA-256。
- 第 3 行 row_idx 12005 是自然恒等输入，20 个候选、原始顺序、token、EOS、稳定排序和分数均与冻结 Normal rank0 第 2 行精确相等。正式加载检查 BF16 活跃 embedding／head 同一对象与 storage、392 个 LoRA 键和全部 394 个 BF16 保存键。两个重叠用户在冻结预跑中另有真实 20 beam embedding 及 28 层空 cache 原始证据。正式逐行 guard 保存 prefill 摘要、空 cache 长度并核对真实 cache_position，参数版本连续性总检查待 S2 终态。
- 非作者 B 直接校验固定计划、实际请求、父子阶段状态和 GPU 归属，从服务器独立重读各观察区间的原预测与原日志。覆盖 4,758 个完整预测记录，核对计数、字节边界、日志进度、资源、后继 CPU／paired 接口、冻结 Normal 身份和 SSH 接管。结论范围为正式启动及观察窗口。

## 资源、异常和后续

- T+30 正式 GPU 占用上界 1822.837 device-s，整组 36 h 上限余 35.49 h。S2 尝试 8 h 上限余 7.49 h。四项计划各预留 8 h，合计 32 h。四个 CPU 评价阶段各限 1,800 秒，paired 限 4,500 秒，父监督限 136,800 秒。预跑 GPU 独立使用 167.244 device-s，准备 CPU 上界 144.011 worker-s。本阶段 CPU 评价启动数为 0。
- 测得最近速率 2.684 用户／秒；以 T+5 到 T+30 的加权速率外推 S2 余 2.14 h。后续三模式合计参考 8.56 h。整组剩余参考 10.07 h 至 12.53 h，合并当前稳态区间、预跑历史长度重加权和 CPU 估算。
- 观察窗口内 GPU UUID 与模型 PID 持续一致，父子运行中，未见新增 attempt、重复 rank、第二张卡上的本任务上下文、显存 OOM、I/O 或输入／mask／callback／prefill 断言。启动日志含 torch_dtype 弃用提示和 NCCL 设备发现 advisory，现场实际映射和持续预测均已核对。
- 提交前两次现场核查调整涉及 Python 3.10 哈希实现与未使用的 Torch 扩展缓存目录检查。材料采集时按正式解释器的模块搜索路径定位复用的 Torch。正式提交和 S2 均为原 r0，发生零次正式重试，研究源码和请求模板保持固定版本。
- T+30 后重新建立 SSH，两个 supervisor 都由 PID 1 接管、独立 SID，文件描述符接服务器日志及 /dev/null。GPU rank 仍为 PID 3437932。实际工作目录、输出及请求对照见 [结构化启动记录](observation-summary.json)。

查询原作业使用既有 Windows SSH 和同一 runner，在此分支工作树执行

```bash
python3 -B -m lc_rec.shuffle_v2.a1_control query formal-parent-r0
python3 -B -m lc_rec.shuffle_v2.a1_control query formal-gpu-shuffle2-r0
```

结果根 `/data/bwwu/lc-rec/a1-instruments-v2/`，作业根 `/home/bwwu/grid-audits/sid-extensions-p0/jobs/`。服务器会继续执行同一序列。全量终态证据、正式评价及最终发布按用户后续授权处理。
