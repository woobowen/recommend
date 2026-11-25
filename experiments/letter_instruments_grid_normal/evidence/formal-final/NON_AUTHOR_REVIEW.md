# 正式终态非作者审核

审核者 `/root/stage_reviewer`。2026-11-25 EST 直接读取固定 Git 源码、原 runner 终态导出、现场 T+30 原始记录、候选与逐用户评价、原始 checkpoint 和 CPU 严格重载回执。使用本机标准库独立核对所有事件、所有候选、checkpoint 张量字节和预算。审核者没有参与正式提交、训练、测试或产物生成，也未连接训练服务器。

**终态与评价复核通过。** 原正式 r0 由 `val/recall@5` 早停，完成best 严格重载、一次全量 Normal 和离线评价。独立 CPU 核对通过。结论覆盖本次单 seed 的作业链路和结果，跨种子波动及其他设置的因果效果未检验。

## 作业、配置和清收

- 终态 `SUCCEEDED`、exit code 0，任务 `grid-letter-instruments-normal-20261124-formal-r0`。请求 canonical SHA-256 `6ab8df5a8f2c02a61d0c18488f0f7a07ec0e7e9dde98d013c71aae9dbedb042f`，supervisor 3467516、worker 3467518、boot ID、start ticks、接受和 claim 与阶段原记录逐项一致。T+30 原始事件和日志字节为终态文件的前缀，四份有效配置／启动／模型材料与阶段记录相符。受审运行提交为 `390df9f508b7dc11eaa57b79a5a91fb5a7f99385`。 [固定正式入口](https://github.com/woobowen/recommend/blob/390df9f508b7dc11eaa57b79a5a91fb5a7f99385/experiments/letter_instruments_grid_normal/run_normal.py) · [原 30 分钟阶段](https://github.com/woobowen/recommend/tree/219fb199d74adc8ed19f727cc6f9aa21ba9ce404/experiments/letter_instruments_grid_normal/evidence/formal-launch-30min)
- 独立复算原终态归档 139,666,813 B 的 SHA-256 `f1c2a977635aa4425a56d45ff293a296e6223d620d70566cffbbd6edd7da8abe`，并逐项复核 export manifest 的 21 份文件及终态列出的 14 份输出。清收原因 `NORMAL_EXIT_CHECK`。初末扫描和后来 query 同为完整、工作组为空，无确认或待确认的成员、无 zombie 和扫描错误。后来 query 的两个进程均已退出。 [终态复核](independent-terminal.json)

## 训练、选模和完整 Normal

- 实际 `formal` 单卡训练以 seed 42、step 0、空 Adam state 开始，固定有效配置保持 `ckpt_path=null`。6200 次 optimizer update 对应 99,200 microbatch。核对 621 个训练事件、620 个 CSV 步级 loss、62 个完整验证及 CSV 编号偏移，全部已记录训练与验证数值有限。每次验证 3,097 批，共 192,014 验证批。最优验证在 step 5200，Recall@5 为 0.10257548838853836；后续 10 次均低于它，工作日志明示触发 patience 10，结束于 step 6200。框架验证事件提供该轨迹，测试输出不参与选模。 [终态复核](independent-terminal.json)
- 最终只保留 step 5200 的完整 checkpoint，159,007,534 B，SHA-256 `1bf78fe86100474887f1f5851b9e478cd772a7a45bff5997520946ed4da87a03`。独立检查了 checkpoint 内部 518 成员 CRC。从原存储逐键重算 177 个张量的模型摘要 `e258aa95b5383e3c6feef765829f53187d763308312ca8e03c5b7b8ea274fdbe`，并核对 127 个 Adam 状态的 step 均为 5200、无 scheduler、两组 callback 最佳指标一致、保存时 early-stop wait 为 0 且 patience 为 10。模型摘要与正式 `reload_enabled`、CPU fresh strict 回执完全一致，二者均无 missing／unexpected keys。CPU fresh 另验证 seed 42 初始参数摘要、官方 SID tensor、Adam 参数及实际 best 状态。 [CPU 严格重载](checkpoint-cpu.json)
- 62 次验证全部先于唯一 `test_start` 和 `test_end`。原日志一次加载 best、设置 eval seed 2026，运行中 `shuffle_calls=0`。尾批保留 4 人，另有 3,096 批各 8 人。原生四项指标的分母各为 24,772，终态 CSV 与模型直接写出的原生指标 逐值相等。 [终态复核](independent-terminal.json)

## 独立候选复算

- 按原请求逐项核九份资产 SHA-256。从三份官方 JSON 建立完整 item、用户和四码 SID 对应。逐行核 24,772 个原始用户的历史截断、目标 item／SID、输入 token 与 mask，解析 247,720 个 float32 分数及 native top-k 实际索引。独立实现稳定排序、SID 首次命中及重复 gain、CCE 首目标碰撞组展开、目录外项占一槽等规则。对照 371,580 个逐用户指标，最大差值为 0，聚合结果复核一致。复算原生四项与 GPU 原生日志的最大绝对差为 `2.6503518546494576e-09`。本轮包含 122 个碰撞目标用户、9 个目录外候选。观测到的候选没有重复 SID、分数并列或 native 与稳定排序差异。另用同分的重复目标、cutoff 和 CCE 部分覆盖样例复核相应规则。 [独立全量复算](independent-candidates.json)

## 资源和复核边界

- 以原观察的内核启动时间到 runner 终态时间为单卡设备分配墙钟代理，含初始化及清收。终点 2026-11-24 22:55:37.831428 EST／2026-11-25 03:55:37.831428 UTC，使用 33,546.877338 GPU-device-seconds，余 52,853.122662 秒。runner 持久记录的开始时间晚约 0.003 秒，按其时间计算 33,546.874403 秒。此处只有一项正式尝试，准备阶段 GPU 账仍单列。审核者的候选复算和 checkpoint 容器核验均为本地 CPU，分别约 2.28 和 0.20 秒。 [终态复核](independent-terminal.json)
- 主控的 CPU fresh checkpoint 复核初次使用 `weights_only=True` 被 checkpoint 中的 `functools.partial` 拒绝。原失败保留。对经 runner SHA 验证且由本任务产出的权重，改用固定正式入口采用的完整反序列化后，CPU strict 成功。该修正仅发生在终态独立校验路径。工作日志包含已有 dataloader worker 数和 hparams 序列化告警。事件、CSV、候选和退出状态均支持实际作业完整结束。 [阶段报告](README.md) · [CPU 严格重载](checkpoint-cpu.json)
- 此次独立审阅在终态 GitHub 发布和最终 ZIP 组装之前完成。按固定远端提交树与关键文件字节、实际 Windows ZIP CRC／SHA-256、解压逐文件复算核对交付对应。
