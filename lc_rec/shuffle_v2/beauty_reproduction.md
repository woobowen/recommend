# LETTER Beauty Shuffle V2

本轮用冻结的原 LETTER Beauty 数据和 LLaMA-7B 测试 SID 顺序干预，用户轴为原交互 JSON 的 22,363 行。每条可见历史由 `seq[:-1][-20:]` 得到。先构造 Prompt0 并调用原 `TestCollator`，再按历史商品出现位置移动四位 SID 内的完整 token ID。模型按移动后的位置使用原生 RoPE 和因果上下文。目标、固定位、组外 token、mask 和原逐码位生成约束沿用冻结实现。

| 条件 | 活动码位（从 1 开始） | 全排列数 | 单组恒等概率 |
| --- | --- | ---: | ---: |
| S2 | 3、4 | 2 | 1/2 |
| S3a | 1、2、3 | 6 | 1/6 |
| S3b | 2、3、4 | 6 | 1/6 |
| S4 | 1、2、3、4 | 24 | 1/24 |

抽样键为紧凑 UTF-8 JSON `[protocol,42,m,row_idx,visible_item_idx,counter]`。完整 SHA-256 经无符号大端解释与拒绝采样映射到字典序排列，S3a 与 S3b 共享三槽相对排列。恒等出现时按实际抽样执行。源索引方向是 `new[A[j]] = old[A[perm[j]]]`。

## 原版与单卡派生

原 Beauty 运行提交为 `6e1b211d9c37f358c43b1540b40b17f6703150d2`。历史 Normal 由其两个不等长分片生成。原选模按验证 loss 选择 step 6160，对应 adapter SHA-256 `dd9f3d2983af7ce85ace99534bd23df9362d9cd91ac39d0b356b219cb790b1c1`，基座 revision 为 `4782ad278652c7c71b72204d462d6d01eaaf7549`。基座、embedding 和 lm_head 以 BF16 装载，冻结 PEFT 0.5.0 实际以 FP32 装载 LoRA 张量。身份、生效路径和历史来源见配置。 [原正式入口](https://github.com/woobowen/recommend/blob/6e1b211d9c37f358c43b1540b40b17f6703150d2/lc_rec/a0_beauty/formal_evaluate.py) · [资产与预算配置](a0_beauty_letter.json)

`beauty.py` 对冻结 Beauty 脚本及原 V1 接入做哈希匹配后，导出派生源码与 diff。原 `uneven_sampler` 在同一 seed 0、epoch 0 排列上取单 rank 全轴，其偶、奇位置分别对应历史 11,182 和 11,181 行。生成用 batch 1、beam/return 各 20、最多 10 个新 token、early stopping 和原 EOS/pad/cache。运行时严格核对每个输入与 CPU 清单、真实模型边界处的 20 个 beam、原生 position IDs 和空初始 cache。GPU 预跑另核对逐 token embedding 查表结果。 [生成派生](beauty.py) · [模型守卫](runtime.py)

`beauty_evaluate.py` 将冻结 16 项指标的读取合同改成 Normal 两片、新条件一片，保留其稳定排序、原精度分数、重复候选及 CCE 实现。无效完整 SID 消耗一个位置，碰撞组按 item 展开，在首次目标 SID 的组内均匀分配 credit。四种干预分别与同一 Normal 配对，另比较 S3b − S3a。五项比较使用共享 PCG64 用户 Bootstrap，5,000 次有放回抽样，seed 20261113，线性百分位 95% 区间未经多重比较调整。 [评价](beauty_evaluate.py) · [配对统计](statistics.py) · [独立复算](independent_beauty_review.py)

## 执行与复算

固定 checkpoint 重评使用配置声明的服务器资产、依赖环境与受审导出。使用已有 runner 依次执行 S2、S3a、S3b、S4，每次生成完整退出并通过单分片全量评价之后进入下一条件。首次提交与恢复按已持久化请求和原 job ID 核对；独立工作证据见 `beauty_control` 和 `beauty_observe`。正式 GPU 单次尝试最多 8 小时，累计 36 GPU-device-hours，四条件预跑累计 60 GPU-device-minutes。CPU 限额按 runner 作业 wall time 单列。 [提交和恢复](beauty_control.py) · [现场观察](beauty_observe.py)

在解压审核包后运行根目录的 `offline_review.sh`，给出独立审核 JSON 的输出路径。脚本用包内冻结原始候选、原精度分数、用户及目标轴、SID 目录、逐用户矩阵和统计数组复算，依赖 Python 和 NumPy。该入口与 checkpoint 重评是两个复现起点。

结论只覆盖此固定权重、冻结 Beauty 用户轴及 seed 42 的置换实例。

## 正式前恢复与操作接口

Beauty 和 Instruments 共用 V2 排列核心、独立参考、模型边界与 PCG64 配对逻辑。Beauty 的原 Normal 来自 11,182／11,181 两个不等长分片。Beauty 由冻结 `uneven_sampler` 还原全轴，并用冻结的 Beauty 16 项评价和 CCE 实现。两组各自绑定数据、adapter、GPU 请求、预算和结果。

上一轮的导入异常直接证明 `CUDA_HOME` 缺失；对照冻结请求还查实 `LD_LIBRARY_PATH` 同时缺失。Beauty 的公共 GPU request 构造器恢复原值，生成包装器对比 request 与实际环境，rank 在 LLaMA 真实导入和严格装载通过后记录同一环境。继续使用原 Python、PyTorch、Transformers、PEFT、CUDA 12.4.131 和原 dtype。工作进程继续使用已验证的 `TORCHINDUCTOR_COMPILE_THREADS=1`。

旧 runner 的清收范围未覆盖 Beauty 前缀。此次使用单独部署的 Beauty runner，观察和发信号的路径均采用同一个 Beauty 前缀，并和原身份保护绑定。实际生命周期证据覆盖正常退出、跨 session 残留、超时 TERM→KILL、同时运行的隔离对照。旧 runner 及其历史回执独立保留。 [Beauty runner](beauty_runner.py) · [CPU 隔离验证](beauty_lifecycle.py)

`beauty_preflight_eval` 仅使用冻结顺序的 32 人样本，逐条核对真实 GPU 候选、输入边界、原 CCE 及归档 Normal 子集。它以同一统计程序生成 5×16 小样本工程检查，并保持正式入口的 22,363 人覆盖检查。整组正式档案的 Offline Normal 复算仍从原双分片读取全量数据。 [预跑评价入口](beauty_preflight_eval.py) · [独立 Normal 复算](independent_beauty_review.py)

新的预跑请求累计上限为 180 GPU-device-minutes，单次上限为 60 分钟。准备 CPU 累计上限 8 小时，单次上限 1 小时。账本接续 `letter-beauty-v2-` 原身份，失败和装载耗时均记录。两张卡均占用时，以 60 秒间隔、最多两小时等待。同一作业持续使用一张物理卡映射为逻辑 GPU 0。

`beauty_control plan` 只构建并保存本地正式计划，不上传 review/request，也不提交父子任务。网页二审和独立启动授权到位后，由 WSL 上述研究分支使用指定快照并再次核对资源，才可执行下列入口。参数值来自最终计划及审核回执。

```bash
python3 -B -m lc_rec.shuffle_v2.beauty_control series \
  --commit <GPU_RUN_COMMIT> --review <LOCAL_PRELAUNCH_GATE_JSON> \
  --replay cpu-replay-r3 --normal cpu-eval-normal-r0 \
  --preflight gpu-preflight-r1 --attempt r0 --confirm-formal-launch
```

运行恢复以原 job ID、不可变请求和 cleanup 核对为基础。正式父控制器按四模式串行生成并评价，每个子作业完整成功后才推进，最后运行 80 项配对结果。预跑包内 `offline_review.sh` 会验证冻结 Normal 全量和 32 人预跑工程指标。包内 `source/support/` 保存实际 WSL transport 与 gridctl 源码。在线操作仍依赖既有 `new-server` 连接、项目 support profile 和冷冻资产路径及访问权。离线评价仅需 Python 和 NumPy，使用包内的数据及 Python 源码。 [正式控制器](series.py)
