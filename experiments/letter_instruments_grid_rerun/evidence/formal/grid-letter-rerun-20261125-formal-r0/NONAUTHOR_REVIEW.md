# GRID／LETTER Instruments 正式作业非作者复核

审核者为 `/root/independent_review`。复核直接读取固定 Git 对象、原始服务器捕获、正式阶段回执、运行数据和源码。本审核在 WSL 本地读取、校验并复算。审核者没有编写或改动正式流水线，也没有提交作业、写远端运行目录或操作 GPU。

## 请求和独立启动

- 用同一个 canonical JSON 序列化规则独立计算正式请求 SHA-256 `bf0226675093e04ff777a76077cdd8173b627f977e6153a175b969faafae4eb1`。它贯穿原模板、服务器 request、accepted、attempt、submit 和独立连接 query。提交前原 ID 查询返回文件不存在，随后目录和请求列表复核证实只有历史 preview ID。首次提交回执为 `ACCEPTED`，独立连接查到真实 `RUNNING`。
- 逐项把请求的 208 个快照实体与本地快照匹配，207 个源码及依赖文件与 Git 提交 `b08c63940f5351dce63dc4cb8a0f9c6f1d8ac86e` 的原始文件内容相同，第 208 个是 `VERSION`。核对部署 runner 文件和部署清单 SHA-256、10 个共同输入 scope 条目、实际命令 `pipeline.py execute --mode formal`、原单设备 UUID、86,400 秒时限及空 `reuse_job`。
- 正式 `started_at` 为 2026-11-25 13:36:38.616311 EST。观察起点为 13:37:12.354664 EST。独立连接读出 `/proc` 中 `supervisor` PID 3526583、PPID 1，`worker` PID 3526585 以该 supervisor 为父。二者各有独立 PGID 和 SID、无控制终端、stdin 为 `/dev/null`。worker 的 argv、cwd、设备环境和日志 FD 对应固定请求。状态查询中的受管 GPU 计算 PID 3526781 实际出现在物理 1 号 4090 `GPU-59cac3bd-3645-668b-79c8-839e0d878893`，正式量化日志已写入。另一次标为 `T00` 的补充采样发生于 13:38:56.964647 EST，当时正式 SID 阶段已进入，监视窗口以首次独立确认计时。

## 正式量化、SID 准入和数据绑定

- 读取三份成功阶段回执并逐文件 SHA-256 对比闭合产物。`quantizer` 131.894758 秒、`sid` 31.076956 秒、`bound` 47.273670 秒，退出均为 0，`reused_from` 均为空。逐字段比较量化和 SID 推理的实际 resolved YAML 与已受审正式配置，替换实际输出路径后只剩运行 ID 时钟差异。检查真实 SHA 贯穿正式 checkpoint、量化 gate、SID gate 和最终数据 binding。
- 从全量 `fit_batches.jsonl` 复算 36 个原生批次、30 次 Lightning optimizer update、73,728 次含重复商品行访问以及全量 9,922 商品覆盖。每层 1 个缓冲批、1 个改动 active codebook 的初始化批及 10 个计步更新。初始化标志在该层第一次计步更新时第一次变为 true，三个层最终都已初始化。30 次计步均改动当前层中心，未改动其他层中心，观测损失和中心有限。三份中心形状均为 `[256,3584]`。
- 从正式 checkpoint 的稳定读取和字节无损提取核对完整三层 FP32 中心。对 `[9922,3584]` 的正式共同 embedding 验证全部行哈希、有限性和商品轴。独立使用 float64 的欧氏距离最小值及逐层归一化残差算法复算 29,766 个中心指派，三个层全部与实际原生预测一致。提取省去不影响此项复算的 Lightning 配置和循环对象，原始 checkpoint 在服务器保留，SHA-256 为 `864abd1e2028b6144f9300782c3f9bcea522ee106560dba21881cad84c21a6a0`。该检查复算正式 checkpoint 的指派，不重跑量化拟合。
- 独立解码真实 TFRecord 的每个上游帧，检查 CRC32C、9,922 个原始数轴 ID 和每个固定编码文本的 SHA-256。逐项对比分配张量、正式四码张量、第四位去重序数、行级 ASIN lookup 和绑定目录。完整 SID 9,922 个全部唯一，8,625 个前三码组，784 个发生前三码碰撞，涉及 2,081 个商品，第四位最大 16，所有码均为 0 到 255 的整数。正式 SID SHA-256 为 `e1cc4d2b0fddb7a48d9a26f5e99d338965590657ddaa18ec90fa3814806e948e`。
- 对照已固定的全部有序交互、用户轴和正式 SID，独立验证 `training`、`evaluation`、`testing` 三个 TFRecord 的帧 CRC32C、每个用户的完整序列、行顺序和目标边界。每个 split 含 24,772 人，保存项分别为 156,609、173,699、197,701，验证和测试可见历史分别为 148,927 及 172,929 项。证据中的无损提取和重算入口见 `review_formal.py` 和对应机器收据。

## Normal 训练观察

- T+10（13:47:16.190845 EST）实际进程进入 `downstream.py fit --mode formal`。训练 CSV 完整行覆盖 `step` 9 至 99，最后 `train/loss_step=14.45938491821289`，当时完整验证回执和 best 尚未产生。GPU 1 观察到受管进程，显存 4,006 MiB，利用率 7%。
- T+20（13:57:14.913739 EST）同一 worker 和受管 GPU 进程持续运行，训练 CSV 最后 `step=199`，`train/loss_step=11.876701354980469`。第一轮验证 callback `global_step=100`、3,097 批，Recall@5 为 `0.060108184814453125`，与 CSV `step=99` 的评估行数值相同。该 CSV 的 `val/loss=13.089644432067871`、`val/recall@10=0.0720975324511528`、`val/ndcg@5=0.03632291406393051`、`val/ndcg@10=0.04017835855484009`。日志和实际目录均给出 step 100 top-1 checkpoint。GPU 1 显存 4,008 MiB，利用率 7%。
- T+30（14:07:15.995579 EST）与独立启动确认相隔 1,803.640915 秒。相对 T+20，训练 CSV 新增 10 条有限训练步记录，最后 `step=299`，`train/loss_step=9.923819541931152`。已完成两次验证，均为原生 3,097 批，按已核验的验证轴和 batch 8 覆盖 24,772 人。第二轮 callback `global_step=200` 与 CSV `step=199` 的原值复核一致，`val/loss=11.213273048400879`、`val/recall@5=0.05869530141353607`、`val/recall@10=0.07149200886487961`、`val/ndcg@5=0.04362117499113083`、`val/ndcg@10=0.047757215797901154`。仅完成的两轮验证可以计数，实时第三轮验证批次尚无直接证据。GPU 1 显存 4,008 MiB，利用率 9%，计算进程与同一受管组对应。
- 复核 T+30 的 17 个捕获文件 SHA-256 和长度，检查真实 UTC 读出边界和各文件元数据，读出期间各记录文件的前后 stat 相同，捕获仍是顺序逐文件记录。完整 `work.log` 无未处理 traceback 或 OOM，三阶段均有成功回执，后续真实执行进入 Normal 拟合。普通 DataLoader worker 数警告保留在原日志。实际模型记录为 13,128,064 个 FP32 可训练参数、四层 encoder 和 decoder、共享 SID embedding、新 SID 匹配、确定性与 `medium` matmul 生效。实际 downstream YAML 与正式配置在执行路径和运行 ID 归一后逐字段相同。训练 CSV 的索引、完整验证 callback 步数及 checkpoint 步数分别按原出处记录。
- 目录和稳定元数据提取确认 T+30 只有一个 top-1 checkpoint，原文件 SHA-256 为 `798010e851fc8588339d59ab59bda61562858c6f655a9e20a9d56fc1f9070502`。callback 选择 `val/recall@5`，第一轮 `global_step=100` 的 `0.060108184814453125` 高于第二轮，因此保持 step 100。checkpoint 内 127 项 Adam 参数状态的 step 均为 100，学习率 0.001、weight decay 0.0001。独立对照 T+30 时文件身份和两轮验证原值。
- 首次只读 checkpoint 取证遇到 Python 循环导入，错误来自提取进程导入固定源码的顺序。按正式入口的导入顺序提前装载 `src.utils.launcher_utils` 后，重新读取相同 inode、mtime 和大小的文件成功取得状态。核对前后文件身份、最初的 stderr、采集脚本和复核结果。此修正限于采集工具，正式作业继续原 attempt，仍绑定相同源码及配置。

**审核结论为 PASS，覆盖实际启动及约定观察窗口。** 真实进程、日志和数值证明正在持续优化，完整验证及观察时 top-1 选择互相印证。正式训练尚未完结，最终模型选择、三个测试条件、离线复算与最终研究结论等待原作业结果。验证 `non-author-cutoff.json` 只使用固定代码和实际捕获，`non-author-closed-reassignment.json` 另保留全量量化指派复算。
