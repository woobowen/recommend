# 正式启动前证据

本轮在 new-server 完成隔离预跑，正式 request 只保存在本地和本目录。其绑定的执行快照为 b08c63940f5351dce63dc4cb8a0f9c6f1d8ac86e，服务器副本在 /data/bwwu/grid-letter-rerun-20261125/source-b08c63940f5351dce63dc4cb8a0f9c6f1d8ac86e。共同输入在 /data/bwwu/grid-letter-rerun-20261125/inputs。GitHub 的运行证据提交是该执行提交的后续提交。冻结执行用文件保持同一字节。

## 数据和运行

- [全量输入 manifest](identity/assets_manifest.json) 记录全 9,922 件商品按 ASIN 一一对应，LETTER 独有 0 件，旧 GRID 独有 689 件。全 24,772 条完整有序用户序列经双向商品映射后精确相同。206,153 条原始交互，重复 user–item 为 0，用户和商品最少各 5 次。
- [实际 TFRecord 验证](preview/bound/binding.json) 逐行读回三种划分，每种 24,772 用户。训练保存 156,609 项，验证 148,927 历史与 24,772 目标，测试 172,929 历史与 24,772 目标。映射、原始归档文件与原向量在单 ZIP。
- [实际预跑 request](preview/request.json) 与 [终态](preview/status.json) 属于 grid-letter-rerun-20261125-preview-r0。物理 GPU 1 为 RTX 4090，逻辑 CUDA 0。程序正式入场前该卡占用 1 MiB，另一张卡已有 20,638 MiB 占用。runner 工作进程运行 223.77 秒，退出 0，独立查询确认无残留工作组。
- [上游拟合](preview/upstream/fit_gate.json) 的六次 Lightning step 分布为每层两次有实际中心变化的更新。另有每层两批 buffer／初始化，全流程实际读取 12 批，每批 2,048，含重复 24,576 项并覆盖所有商品。量化、SID 阶段分别约 77.76、35.23 秒。临时 SID 为 [4,9922]，9,922 个唯一完整 SID，3,161 个三位前缀，1,611 个碰撞组，第四位最大 43。9,922 件推理采用 77 个 128 项批和一个 66 项批。 [SID 关卡](preview/upstream/sid_gate.json)
- [下游训练](preview/training/training.json) 从空 Adam 状态启动，48 microbatch，扩展 104 到 128，累积 16，3 次更新。三个更新的有限梯度证据、参数前后哈希和 Adam 计数均保存。只做一次、3 批的 Normal 验证，选中 step 3。训练阶段耗时 20.46 秒，阶段内记录的 fit 验证前 7.89 秒、验证 1.16 秒。PyTorch 峰值分配 2,775,259,648 B，保留 3,275,751,424 B。 [best 绑定](preview/training/best.json)
- [三条件原始结果](preview/evaluation/) 使用同一临时 best，三个条件均 strict load，分别测试 20 用户，批大小为 8、8、4，各有 200 个原精度候选。短历史、超出 20 截断、同值裸码和尾 batch 由固定规则覆盖。Normal 与全恒等适配的原始候选、分数位模式、native 排名和指标完全一致。预跑没有命中目标，Recall、NDCG、统一 SID 和 CCE 指标为 0，相对差零分母保存为 null。 [离线汇总](preview/evaluation/summary.json)
- [V2 全量 CPU 回放](preview/bound/full_cpu_replay.json) 覆盖真实临时 SID 下的验证和测试历史、逆序分块及独立 factoradic reference。测试 S3a 在 172,929 个位置抽到 28,935 次恒等，S4 抽到 7,147 次。每种预跑模式有 185 个可见历史组，S3a 的非恒等抽样、实际 token 改变均为 147，S4 均为 182。两种 Shuffle 各有 1 个用户候选 ID 数组改变、20 个用户分数位模式改变。这些数值只用于核查预跑机制。
- [准入负例](preview/admission_preview_r0.json) 使用实际预跑副本。交换商品轴的两列后拒绝加载，预跑也被正式关卡拒绝。额外 CPU 负例覆盖第四位 256、错轴、强制拒绝重抽、全 6／24 个合法排列、历史不完整组。
- [独立非作者审核](../review/NONAUTHOR_REVIEW.md) 直接比对原向量、重新计算每位码对质心的最近邻、读回原始 TFRecord，独立复算全部预跑候选和指标。其旁列原始核验输出。初次预跑通过，GPU 计算未发生修正重试。

## 正式预算和提交前边界

正式配置和本地模板见 [formal](formal/)。定量资源参考保存在 [budget.json](budget.json)。

按预跑原量化步骤实测外推，正式上游约 181 秒，其估计固定采样／初始化成本和本机资源互斥。下游 48 个 microbatch 的短预算测得约 6.08 microbatch/s；历史 C 原训练约 7.79 microbatch/s，完整验证平均约 330 秒。相同评估周期可按约 9–10 分钟每 100 次更新估算，加上 CPU 资产和阶段启动开销。实际停步由验证轨迹决定。C 历史在 6,200 步结束，仅用于说明该历史 run 约 9 小时 19 分钟，不为新实验指定停点。

每个正式测试条件 3,097 批、24,772 用户与 247,720 候选。C 实测一次全量 Normal 约 319 秒，乘以三约 16 分钟是参照值。新 V2 带有历史、lookup 与候选写出验证；预跑只有三批，启动开销使直接线性外推偏大。正式测试以原始逐阶段日志的实测为准。GPU 1 预跑前空闲，正式时由 runner 再查设备 UUID、单卡可用显存和空间。

正式 trainer 上限 320,000 次更新和 10 个 trainer epoch。既有 runner 的单作业墙钟上限独立为 24 小时。达到 runner 上限时按实际终态处理。接续只复用已完整验收的阶段，未完整训练不会自动另起一份随机训练。正式上游、SID、下游和正式评测均未提交或创建。用户另行授权后，可从同一 P0 gridctl.py 提交正式模板。若预跑用卡届时有占用，用同一源码的 pipeline.py request 以当前空闲设备 UUID 重新生成未提交模板，再提交同一正式作业身份。
