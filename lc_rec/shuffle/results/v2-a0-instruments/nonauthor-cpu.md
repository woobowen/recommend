# Shuffle V2 CPU 准备阶段的非作者复核

复核范围为冻结历史、V2 源码、CPU 输入与 Normal 评价。真实 GPU 预跑和 V2 新预测需另做非作者复核。

## `/root/history_review`

2026-11-20 16:17–16:24 UTC。直接读取固定 Git 对象、CPU 导出及回执，执行独立重算，未编辑研究源码。

- 在 Instruments A0/A1、原 LETTER Beauty、G3 Beauty 的冻结 RUN 中核验同一 V1 core，SHA-256 `1faf505bde5a7bc68fb9cf7ff45e0bc0c8d3ca2a1cf2bd7dc35618849e71ea2f`。重新派生 Instruments 八份历史 rank 源码，与固定交付逐字节一致。
- 使用冻结数据、原 Prompt0、与真实 adapter 文件哈希匹配的 tokenizer 文件，独立重建 24,772 个输入。原输入和 V1 S3/S4 摘要均吻合。独立构造 V2 字典序排列，并以 `hexdigest` 和显式目标槽映射复算全部路径。四种完整 V2 token 流 SHA-256 均与 `a245824835a3f7457d577bf28320530b6dddf779` 的实跑 `cpu-replay-r1/audit.json` 精确一致。组频数、位置移动与恒等用户计数一致。
- 从冻结源码编译四条派生入口，逐个对比 `model.generate` 的 AST。Instruments A1 保留前缀回调约束，Beauty 包装器阻断旧的 Normal/OFF 预跑开关。Beauty 与 A1 的复核只涉及所读源码及编译。
- 独立解码并评价历史验收中 32 个用户的 640 条 Normal 候选。SID 与 CCE 的 512 个逐用户值和本轮重评矩阵的最大绝对差为 0。

## `/root/runner_review`

2026-11-20 16:24–16:26 UTC。直接核对 Git 对象、已下载回执与 NumPy 数据，模拟控制器异常，不连接服务器启动作业。

- Normal 重评实际代码提交 `4275efcb6482cecb9bd5ccebb5ab717f27c2c884`。24 个部署文件与 Git 对象及请求对应。作业退出码 0，清理回执确认无 worker。两个原始分片 SHA、大小、rank 及路径与历史身份一致。24,772×16 的逐用户矩阵与固定 NPZ 逐字节一致，16 项均值精确一致。
- 用 `bincount` 加权独立重算 80,000 个历史 bootstrap 均值与 32 个区间端点，最大差分别为 `5.83e-16` 和 `3.47e-16`。独立小型数值用例含重复、碰撞与无效 SID，验证原 CCE 展开及均匀 credit；五项合成配对统计共用固定索引，重算最大误差为 `5.55e-16`。
- 发现各阶段共用 attempt 编号会妨碍保留成功片段，以及请求上传前断线会卡在查询。针对修正版 SHA-256 `87fc1a8d7fff8f87f5112ac903930f4c791e60a8b8797990e90ac1ec3e843526` 模拟混合成功尝试、重复成功、未清理、提交前后断线和异常探测。混合成功可正确接续，歧义与异常保守停止。后续补入同 ID 重送前的提交身份和剩余预算检查。

以上证据覆盖的是 CPU 准备阶段。两名复核者未观察 A0 真实 GPU prefill，未审阅本轮任一模式的正式预测或实测配对结果。
