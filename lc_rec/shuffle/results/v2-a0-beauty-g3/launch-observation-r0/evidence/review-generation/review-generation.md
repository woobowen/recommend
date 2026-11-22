# 正式启动观察非作者审核｜生成与模型边界

审核者为 Codex 独立代理 `/root/review_generation`。独立脚本直接读取服务器的首批正式记录，与冻结 CPU manifest、Beauty 商品索引和测试交互交叉核对。主控提交、生成与观察脚本由其他执行者负责。

## 正式首批原始行

在 2026-11-22T15:33:39.566724+00:00 通过新的只读 SSH 读取正式 S2 文件从字节 0 开始的六条完整行。终点字节 163,878，SHA-256 为 `68dd2bae9b3f1acb7810a6be25b0f4093d4689c350c1f21f2fa0e9601fd9db1f`。与主控首次采集逐字节一致，预测文件 inode 为 78513579。原始文件位置保存在直接读回执中。

按全轴 sampler 检查真实行号依次为 8755, 21472, 6726, 17283, 12152, 4046。冻结商品目标、历史出现位置、完整 SID 分组及 mask 与实际输入一致。通过独立 SHA-256 计算重建 S2 置换，每行与固定槽、组外 token、CPU 完整 manifest 行摘要及实际 decoder 输入吻合。全量任务记录显示 FORMAL scope、22,363 用户范围、单卡 logical 0，冻结 G3 best adapter checkpoint-6160 身份和 32,786 词表吻合。

每行 20 个 beam 的实际 embedding／decoder prefill 回执通过。核对原生 position IDs、空初始 cache、BF16 embedding 形状及等值回执，基座和 LM head 的 BF16、LoRA 的 FP32 严格装载证据，以及固定 beam／EOS／pad/callback。120 个保存的有限候选分数全都能精确还原为 FP32，排序稳定，首六条范围内目录外 SID 为零。该条诊断仅覆盖六条。

## 七点运行区间

从 2026-11-22T10:30:03.373015-05:00 至 2026-11-22T11:00:09.460118-05:00，跨度 1806.087 秒，实测时点相对名义时刻的最大偏差 38.872 秒。T0 是实际 rank 已启动后的模型加载阶段。T05 至 T30 的预测文件连续增长，七点用户数为 0 / 238 / 741 / 1202 / 1672 / 2124 / 2587。由 T05 至 T30 得到 1.5799 用户/秒。逐点情况用完整落盘行和独立 rank 进度交叉检查。

正式 rank PID 3427779，单卡 UUID `GPU-59cac3bd-3645-668b-79c8-839e0d878893`，物理卡 1 对应逻辑卡 0。预测写出后各点从 NVIDIA 仅观察到本条件这一 CUDA rank。每点单独复算正式 GPU、预跑 GPU、CPU worker 和父监督墙钟账本。首轮加载日志中保留了 NCCL barrier 设备映射提示，随后实际单卡进程身份和输出持续推进均符合预期。

T0 后，本地观察器细化进程分类，按 torchrun 父 PID 排除 fork 出的 CPU DataLoader helper。T0 保留。父控制器、S2 launcher 与 rank 的 PID、启动 ticks、boot ID 和请求哈希在七点一致。
事后另开一次只读 SSH，仅重新计算七点已经采到的原始日志及预测字节范围的 SHA-256 和完整行计数。T30 原始读取边界为 64,074,085 字节，对应 2,587 条完整预测。该复核没有增加运行状态观测。

## 复核定位

`observer-first-six-review.json` 与 `direct-prefix-readback.json` 保存首批和直接来源对照。用 `verify_formal_rows.py` 重算绑定、置换、模板与模型边界。用 `verify_seven_points.py` 复算七点快照。用 `verify_live_ranges.py` 核对当时只读捕获的原始区间，执行回执为 `live-byte-range-review.json`。

生成与模型边界的正式启动阶段审核通过。完整 S2 和其他三个条件的评价、最终结果与终审待服务器完成。
