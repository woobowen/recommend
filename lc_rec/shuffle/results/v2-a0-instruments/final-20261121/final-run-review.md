# Shuffle V2 第一组正式运行终审

审核者 `/root/final_run_review`，验证完成时间 **2026-11-21 06:24:17 EST**（11:24:17 UTC）。本审核未参与正式生成、评价实现、结果整理或训练。直接读取服务器冻结源码、原始作业终态、预测及前置资料，另行实现只读复核。无新增作业、模型调用、计算条件变化或远端文件写入。

**PASS，范围为执行身份、全量输入与候选完整性、终态和预算。** 16 项数值与配对统计的独立复算由另一项终审覆盖。最终仓库和完整 ZIP 的发布一致性需用相应交付回执判断。

## 执行与收尾

父作业 `letter-a0-20261112-v2-20261120-group-series-r0` 于 **2026-11-21 05:43:14.898670 EST** 成功退出。直接读取 `request.json`、`accepted.json`、`status.json`、`cleanup.json`，并对同一组 18 个已有 ID 使用已冻结 runner 的 `query`，比对 job 身份、请求哈希、真实清收和当前位置无遗留 worker。9 个串行正式步骤的原请求与服务器计划一致，实际按 S2 GPU → S2 CPU → S3a GPU → S3a CPU → S3b GPU → S3b CPU → S4 GPU → S4 CPU → 配对执行。四个正式 GPU 和四个 CPU 评价均为 r0、退出 0、已清收。配对 `cpu-summary-r0` 退出 0、已清收，服务器计算汇总标记 `PASS_V2_PAIRED`，对应 24,772 人与相同用户轴。

| 条件 | 正式 GPU 用时（秒） | CPU 评价用时（秒） | GPU 结束 EST |
|---|---:|---:|---|
| S2 | 15,767.625 | 44.115 | 2026-11-20 16:11:31 |
| S3a | 15,917.539 | 43.207 | 2026-11-20 20:38:00 |
| S3b | 16,854.311 | 43.573 | 2026-11-21 01:20:05 |
| S4 | 15,512.751 | 43.266 | 2026-11-21 05:39:47 |

正式 GPU 累计 **64,052.227417 秒（17.7923 device-hours）**，低于原 36 小时累计门槛，每次低于 8 小时。四个 rank 原报告及 request 环境均指向同一物理 RTX 4090 `GPU-59cac3bd-3645-668b-79c8-839e0d878893`，`WORLD_SIZE=1`、`rank=local_rank=0`、batch=1，且每个正式输出仅有一个 rank。预跑为 **633.755594 秒（10.5626 device-minutes）**。原 CPU 作业累计 **685.477960 秒**，其中配对 **114.060352 秒**。这项额外只读复核用时 41.791 秒，另记，不混入原作业账。唯一非 `SUCCEEDED` 的原接受尝试是 `cpu-replay-r0`，其主程序退出 0 后，runner 以 SIGTERM 清理仍带本 job 标记的 PID 3359417，状态 `WORKERS_TERMINATED` 并证实已清收。后续 `cpu-replay-r1` 成功，正式评价和复核均绑定 r1 输入。未见正式 GPU 重试、CPU 结果重算、OOM 或超时。

## 独立核对范围与方法

- 从两份归档 Normal 原始分片检查 24,772 个全局编号与输入，逐行复算原输入哈希，并将两 shard 顺序交织对照正式单 rank assignment。分别按 SHA-256 比对两份原文件与冻结资产清单。使用冻结 `Instruments.inter.json`、`Instruments.index.json` 与 `Instruments.item.json` 核对交互顺序、SID–item 关系、最后一个目标、可见历史范围 `seq[:-1][-20:]` 和连续四个注册 token 的边界。数据源文件哈希均比对冻结清单。对照的 9,922 件物品形成 9,897 个不同 SID，其中 23 个 SID 关联多件物品。
- 使用独立 SHA-256 实现按 `GRID-LETTER-SHUFFLE-V2`、seed 42、活动宽度、全局用户行号、可见历史出现次序和拒绝计数重算完整置换。对每个正式原始行逐 token 核对实际输入，检验固定槽、商品边界、组外输入、长度和 mask。将重算结果与 CPU replay 的置换 ID、输入哈希、保存的 decoder 首次 prefill 哈希及原生 position ID 哈希逐行比较，检查 20 beams 和空初始 cache。每个条件全量 24,772 行、172,929 个历史组，置换分布、各位置移动计数与 r1 回执一致。S3a/S3b 每个历史位置的相对排列相同。自然恒等正常保留。
- 直接读取四个 `gpu-formal-<mode>-r0/test/predictions.rank0.jsonl` 的完整字节流并比对各 CPU 评价的原文件 SHA-256。连同两份 Normal 归档，检查 2,477,200 个原候选的生成顺序、固定 SID 码位、EOS、分数有限性、稳定排序、目标和 relevance。候选 SID 独立从保存的四个生成 token ID 解码，与保存 SID 相符。保留无效 SID、原始重复和碰撞计数。此项验证候选依据与评价入口之间的身份和结构，未另外重算 16 项结果及配对区间。
- 对照正式代码 `e2f54ebd2d7709821de1071693f91644f5ed9a30` 的 `entry.py`、`runtime.py`、`core.py` 和 `evaluate.py`，确认修改发生在编码后、原生 `generate` 前。实际先复用冻结 collator，按历史商品四码位就地排列完整 token，再检查模型的实际入参和 prefill。四个正式代用入口均与从冻结 Git Blob 重新派生的字节 SHA-256 `04a664a27a9b34293eb1203b90a437d826a39329ad7c66b9362748955cef3f29` 相符。每个正式 rank 报告均核验冻结 adapter 身份、beam=return=20、max_new_tokens=10 和 `filter_items=False`。
- 服务端 `RUN_VERSION.json` 中正式 RUN 的 30 个文件、串行控制 `4f5758ffb578f3a8d3d06a54262f59d3f63db370` 的 33 个文件，由审核者直接重算服务器 SHA-256，并逐一同固定提交的 Git Blob 字节匹配。有效 runner 位于 `lc_rec/a1_qwen3_06b/scripts/runner.py`，固定 blob 及服务器文件 `/home/bwwu/grid-audits/sid-extensions-p0/scripts/runner-a1-96h-1fefc09a15c1cb73.py` 均为 `1fefc09a15c1cb73878417f9eb50d0fe9e8fa1b2f60cc70b27f7cc4ed9cb0947`。本地 transport 和 gridctl 从已记录的 `/home/addaswsw/recommend-support/sid-extensions-p0/scripts/` 导入，当前实际字节 SHA-256 分别为 `eced9c0f87833aef91382fc037dd1b909e2d226fb0158147033921ad29f81d9a` 和 `b6fa7f86472cba5e0c00398ccc0925940dd1524027dd827aac3e82bfc1105c13`。

| 条件 | 用户 | 候选 | 历史组 | 恒等历史组 | 完整输入自然未变用户 | 不在 catalog 的 SID 候选 | 重复候选 | 碰撞 SID 候选 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| S2 | 24,772 | 495,440 | 172,929 | 86,483 | 714 | 481 | 0 | 4,421 |
| S3a | 24,772 | 495,440 | 172,929 | 28,935 | 6 | 226 | 0 | 3,166 |
| S3b | 24,772 | 495,440 | 172,929 | 28,935 | 6 | 176 | 0 | 4,267 |
| S4 | 24,772 | 495,440 | 172,929 | 7,147 | 0 | 93 | 0 | 2,879 |

机器证据 `final-run-independent-check.json` 记录每项实际检查、原始预测哈希、作业预算及部署映射，SHA-256 `849fa147559f1d99336e3411c43422b0f7e0318dd98163084cee41c729450360`。相邻 `final-run-independent-check.py` 为实际执行的独立核对脚本，指向服务端冻结原件。审核复用了已通过的预跑模型身份及正式保存的 prefill 边界检查，不重新执行模型或重读完整权重。运行结果的判断限于这套固定 A0、用户轴、生成协议及 Shuffle 实例。
