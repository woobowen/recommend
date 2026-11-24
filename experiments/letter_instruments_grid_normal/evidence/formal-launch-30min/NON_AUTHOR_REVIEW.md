# 正式启动与 30 分钟观察的非作者审核

审核者 `/root/stage_reviewer`。2026-11-24 在本地直接读取固定 Git 对象、逐项核对现场原始 JSON，并用独立只读 Python 断言比对采样、模型身份、指标、预算和发布草稿。主控执行了正式提交、采集和证据整理。审核期间没有连接服务器，也未运行训练或预跑。

**通过启动与 30 分钟观察阶段审核。** 一项正式作业持续运行。观察窗口内三次验证各完成 3,097 批，并按 `val/recall@5` 产生三个依次替换的 best。截止时已记录 step 400／microbatch 6,400，正在进行第四次验证。另一次交付前只读查询显示同一 worker 仍在该阶段。

## 载荷、准入与接管

- 直接读取 `098677ab130eae4dfb14bae5f2c56922817d8183` 的模板，与实际本地请求逐字节相同。服务器请求为同一 JSON 的 runner 规范序列化，SHA-256（不含文件末尾换行）为 `6ab8df5a8f2c02a61d0c18488f0f7a07ec0e7e9dde98d013c71aae9dbedb042f`。该值与原始接受、claim、七点 runner query 和交付查询一致。独立从 `390df9f508b7dc11eaa57b79a5a91fb5a7f99385` 取出并重算 104 个请求绑定运行文件的 SHA-256，74 个 Python 文件 AST 解析通过。九项资产来自准入机器记录中的实际读回哈希，其语义配对沿用已审证据，并交叉核对了本轮启动覆盖。部署哈希与请求相符，固定 runner 文件 SHA-256 为 `398768a732bbfebde4308486a0cedd924595157fda6cbff6a88f0bfaa512beff`，与接受回执一致。[实际请求](request.json) · [服务器请求](server-request.json) · [现场准入](admission.json) · [提交回执](submission.json)
- 从原始重连读回和 T+30 文件前缀独立解码实际配置和 overrides，分别与已审固定 Git 对象逐字节相同。验证了 `formal`、seed 42、`ckpt_path=null`、step 0 和空 Adam state、单卡 FP32、batch 16 和累积 16、原 collator、Adam 与无 scheduler、1600 microbatch 验证、patience 10 和完整测试数据路径。模型回执证实 13,128,064 可训练参数、四个 256×128 预测头、BOS/SEP、共享 SID embedding 与官方 SID 匹配。启动回执三份数据各 24,772 行，9,922 件商品。启动 JSON 的 `preflight_users` 来自通用资产核验清单，实际测试加载器指向完整 `/data/testing`，正式请求登记 `test_users=24772`。[有效配置](captured/output/run/effective_config.yaml) · [启动回执](captured/output/run/startup.json) · [模型身份](captured/output/run/model.json)
- 准入时原 job 和请求均不存在，也没有匹配的正式 worker。接受后七点、启动重连、交付查询均只见一个已排队请求、一个正式接受记录和一个本任务 worker。进程 PID、boot ID、start ticks 连续，supervisor 3467516 的 PPID 为 1，worker 3467518 的 PPID 为 3467516，两者各自的 SID 和 PGID 等于自身 PID，TTY 为 0，stdin 为 `/dev/null`，stdout 和 stderr 指向服务器各自日志。实际 CUDA 计算进程在重连和 T+5 到 T+30 均只在 UUID `GPU-59cac3bd-3645-668b-79c8-839e0d878893`，对应物理 1／唯一可见逻辑 0。T0 仍在初始化，采集尚未见本任务 CUDA 进程。服务器本地读取固定快照、资产和解释器的事实与接管方式一致。[T0](snapshots/T00.json) · [SSH 重连](snapshots/start-reconnect.json) · [交付查询](delivery-status.json)

## 七点观察与指标

本地独立计算从 `/proc` start ticks 推导的 worker T0 为 13:36:30.954090 EST／18:36:30.954090 UTC。七个目标相距 300 秒，UTC 与 America/New_York 转换、实际时间和偏差均与现场记录相符。依次偏差为 +0.454、+1.528、+1.377、+1.845、+2.274、+4.058、+1.330 秒。事件取值为采集区间内最后已写记录。T+10 的 step 130 事件比快照开始晚约 0.090 秒，发生于同一次采集结束前。事件 `global_step` 与 CSV 给出的 step 保留两套原始编号，三次验证事件在 100／200／300，对应 CSV 99／199／299。[七点表](timeline.md) · [七点现场记录](snapshots/)

| 验证事件 step | 结束 UTC | 批数 | 用时秒 | CSV Recall@5 |
|---:|---|---:|---:|---:|
| 100 | 18:45:29.136850 | 3,097 | 318.300242 | 0.06297432631254196 |
| 200 | 18:54:05.748219 | 3,097 | 313.641331 | 0.07532697916030884 |
| 300 | 19:02:37.963853 | 3,097 | 313.739104 | 0.07916195690631866 |

检查了 T+30 完整前缀中的 41 项 `train_batch` 事件和 43 行 CSV，其中 3 行验证。更新按记录点增加，microbatch 与累积设置一致，`shuffle_calls=0`。所有已记录 train／val 数值有限。T+5、T+15、T+25 和 T+30 的验证中状态均有 start 和首批事件支持，前三次还可从后续已完成事件核对。工作日志保存记录与 T+30 留存的 159,007,470 B `checkpoint_epoch=000_step=000300.ckpt` 元数据吻合。只复核 checkpoint 文件身份和保存记录，按固定入口，fit 结束后由正式 worker 严格重载并校验 best。T+30 最后记录的原生 `train/loss_step` 为 11.680148124694824（CSV step 399）。`candidates.jsonl`、`stage_result.json` 和离线结果尚未产生，严格重载、全量 Normal 与离线评价待验证。[原始事件](captured/output/run/events.jsonl) · [原始指标 CSV](captured/output/run/csv/version_0/metrics.csv) · [保存日志](captured/work.log)

## 预算、修正与发布对应

- 七点单卡设备分配墙钟代理值由每个服务端采集起始时间减 T0。T+30 已用 1,801.330365、余 84,598.669635 GPU-device-seconds。14:06:47.114749 EST／19:06:47.114749 UTC 交付查询同一任务运行中，已用 1,816.160659、余 84,583.839341 GPU-device-seconds。观察期间服务器、WSL 暂存和 Windows Downloads 均留足 2 GiB。准备阶段 273.744618 GPU-device-seconds 和 CPU worker 69.758122 秒保留原账，本轮仅一项正式尝试，无正式失败消耗或新 CPU runner。估时用本轮 4 个训练窗口和 3 次完整验证计算出中位每 100 更新 204.365612 秒、验证 313.739104 秒，训练终点由原验证、早停及框架上限决定。Runner 86400 秒到限启动有界清收。[预算](budget.json) · [交付查询](delivery-status.json)
- 执行记录列出启动前第一轮准入和本地 CLI 工程修正。已核 `batch_size_per_device` 取值及准入复核通过、提交请求维持原版与作业唯一性；现场未见 OOM、非有限记录、重复运行或方法变化。审核命令初次误用了 YAML 键并把准入记录的 `size` 纳入模板比较，修正只读检查后复核通过，未修改任务或证据。[阶段报告](README.md) · [准入脚本](precheck.py)
- 对实际原始七点与发布草稿逐项核了运行、采样、文件和回执字段，全部相等。GPU 输出中的无关外部 PID 按既有整理方式隐去，本任务 PID 保留。T+30 的 8 项文件前缀均与草稿解码文件逐字节一致。该审核在发布及打包前完成。交付时以 GitHub 固定提交树和关键 blob 回读、ZIP 内成员 SHA-256、CRC、解压后复算核对发布和包。主动定时采集到 T+30 结束，之后现场记录仅有一次只读交付查询。[字节边界与捕获清单](captured/capture-manifest.json) · [阶段校验器](verify_phase.py)
