# LETTER Instruments → GRID 原生 Normal 正式结果

**正式作业自然早停并正常退出，完整结果核验通过，待网页 GPT 最终二审。** 原 job `grid-letter-instruments-normal-20261124-formal-r0`，服务器 `new-server` 的实际主机为 `zwlab-01`。runner 终态 `SUCCEEDED`、退出码 0、`NORMAL_EXIT_CHECK`，工作组清空且未经清收信号。查询为 2026-11-25 07:36:43.789 EST／12:36:43.789 UTC。工作从 2026-11-24 13:36:30.957 至 22:55:37.831 EST，UTC 跨至次日 03:55:37.831。 [现场查询](query.json) · [正式 status](run/status.json) · [清收原件](run/cleanup.json)

运行版本 `390df9f508b7dc11eaa57b79a5a91fb5a7f99385`，正式启动及 30 分钟已审发布 `219fb199d74adc8ed19f727cc6f9aa21ba9ce404`。实际请求签名 `6ab8df5a8f2c02a61d0c18488f0f7a07ec0e7e9dde98d013c71aae9dbedb042f`，绑定 104 项运行文件与 9 项正式资产。配置、overrides 与已审 dry-run 同字节。完整逐轴检查核实三份 TFRecord 各 24,772 人及原划分，SID tensor 与 9,922 件商品的四位 SID 逐值一致。 [正式请求](run/request.json) · [现场有效配置](run/output/run/effective_config.yaml) · [运行入口](https://github.com/woobowen/recommend/blob/390df9f508b7dc11eaa57b79a5a91fb5a7f99385/experiments/letter_instruments_grid_normal/run_normal.py)

## 训练与选模

单张 RTX 4090，物理 1、逻辑 0，UUID `GPU-59cac3bd-3645-668b-79c8-839e0d878893`。原生 GRID 共 13,128,064 个可训练参数，四个自回归预测头、共享 SID embedding、BOS/SEP。seed 42 的随机初始化，step 0、空 Adam state、`ckpt_path=null`。单卡 FP32、用户 batch 16、原 causal duplicate collator、累积 16，Adam 学习率 0.001、weight decay 0.0001、无 scheduler。每 100 次更新完整验证，保留一个最大验证 Recall@5 的 checkpoint。原框架上限 320,000 steps、10 trainer epochs，runner 上限 86,400 秒。 [启动与环境](run/output/run/startup.json) · [模型身份](run/output/run/model.json)

完成 **6,200 次更新、99,200 microbatch 和 62 次全量验证**，每次 3,097 批，共 192,014 个验证批次。首轮 Recall@5 为 0.06297432631254196，step 5200 达到 **0.10257548838853836**，最后为 0.10063781589269638。之后十次验证低于 best，原日志明确触发 patience 10 早停。共 21 次刷新 best，只有最终 best 留在原目录。620 条原生 `train/loss_step` 均为有限值，前 1,000 次更新对应记录的中位数为 10.99233，最后 1,000 次为 7.94998。最后一次记录为 8.38540（CSV step 6199）。 [全量训练和验证事件](run/output/run/events.jsonl) · [原始 CSV](run/output/run/csv/version_0/metrics.csv) · [验证轨迹](validation-trajectory.csv) · [早停和选模日志](run/work.log)

Best 为 `checkpoints/checkpoint_epoch=000_step=005200.ckpt`，**159,007,534 字节**，SHA-256 `1bf78fe86100474887f1f5851b9e478cd772a7a45bff5997520946ed4da87a03`。正式测试实际 strict load 177 个 state keys，missing/unexpected 均为空，model state SHA-256 `e258aa95b5383e3c6feef765829f53187d763308312ca8e03c5b7b8ea274fdbe`。CPU fresh instance 严格回读得到相同状态哈希，检查了实际权重的所有浮点张量、127 份 Adam 状态和全套 callback 数据。非作者另从 checkpoint 的存储原字节独立复算。 [正式 strict 回执](run/output/run/reload_enabled.json) · [CPU checkpoint 回读](checkpoint-cpu.json) · [独立容器回读](independent-checkpoint-container.json)

## 正式 Normal 与 CPU 复算

eval seed 2026，严格重载后走原生四步自回归 Beam10，保留概率乘积、cache 和关闭的前缀过滤。测试一次，3,096 个 8 人批和最后一个 4 人批，`shuffle_calls=0`。候选覆盖原顺序 24,772 人，每人 10 个四位候选，共 247,720 个。CPU 对照所有用户、历史物品、输入 SID、mask、目标 item/SID、候选 FP32 位模式和四组实际 top-k 索引。全量逐用户评价逐行相等，独立复算核对了 371,580 个指标值。 [正式阶段结果](run/output/run/stage_result.json) · [产物取得位置与原始哈希](artifact-location.json) · [内部复算](verification.json) · [独立逐用户复算](independent-candidates.json)

| 评价 | 指标 | 正式结果 |
|---|---|---:|
| GRID 原生 | Recall@5 | 0.08307766914367676 |
| GRID 原生 | Recall@10 | 0.10047634690999985 |
| GRID 原生 | NDCG@5 | 0.07116406410932541 |
| GRID 原生 | NDCG@10 | 0.07676350325345993 |
| GRID 原生 | 测试 loss | 8.439092636108398 |
| 统一 SID | Hit@1 | 0.058977878249636684 |
| 统一 SID | Hit@5 | 0.08307766833521718 |
| 统一 SID | Hit@10 | 0.10047634425964799 |
| 统一 SID | NDCG@5 | 0.07116406301915479 |
| 统一 SID | NDCG@10 | 0.07676350151747216 |
| 统一 SID | MRR@5 | 0.06726142418860004 |
| 统一 SID | MRR@10 | 0.0695557131835995 |
| 统一 Item／CCE | ItemHit@5 | 0.08275472307443889 |
| 统一 Item／CCE | ItemHit@10 | 0.10023413531406426 |
| 统一 Item／CCE | ItemNDCG@5 | 0.07050755805906864 |
| 统一 Item／CCE | ItemNDCG@10 | 0.07613222132209727 |

原生四项 CPU 最大绝对差 `2.6503518546494576e-09`，已审容差 `1.430511474609375e-06`。统一排序按 FP32 分数稳定降序，原生按实际记录的 top-k。正式候选中 0 行并列、0 重复候选、0 行两种排序不同。9 个目录外 SID 预测保留原位置，按大小 1 计算 CCE 展开。本轮目录包含 23 个碰撞组、48 件碰撞商品，122 人的目标 SID 碰撞，76 人的碰撞目标 SID 被命中。正式候选中有 2,393 个落在已有碰撞组的预测，涉及 2,061 人。原生 Recall 的相关数分母按生成列表计算，本轮受零重复条件影响而与 SID Hit 的每用户值相同，汇总仍各用原口径。 [原生测试输出](run/output/run/native_metrics.json) · [离线评价原件](run/output/run/offline_evaluation.json)

## 历史与资源

已核实历史也是同一 Instruments 资产、同一用户轴及 GRID Beam10。相较该历史，本轮原生 Recall@5／10 分别下降约 0.190／0.230 个百分点，NDCG@5／10 分别下降 0.144／0.162 个百分点，测试 loss 下降约 0.02257。历史 best step 5400、终步 6400，历史 best 验证 Recall@5 为 0.10241401940584183。数据打乱和环境差异限制归因，两边只各运行一个 seed。历史完整候选缺失，统一 SID／CCE 的历史差值未计算。 [历史比较和原始依据](HISTORY.md)

fit 33,209.655 秒，其中验证 20,477.069 秒，fit 其余时间 12,732.586 秒。测试 319.355 秒，含候选写出约 1.300 秒，随后离线评价 4.424 秒。runner 分配墙钟 33,546.874 秒，即 9 小时 19 分 6.874 秒，单卡预算余 52,853.126 GPU-device-seconds。PyTorch CUDA 峰值已分配 2,780,016,640 B、保留 4,288,675,840 B，约为 2.589 和 3.994 GiB。准备阶段 273.744618 GPU-device-seconds 独立记账。本轮的 CPU 审核另计，GPU 使用为原单次 r0。 [实际阶段计时](run/output/run/stage_result.json) · [终态核验](independent-terminal.json)

## 材料与复算

正式输出位于 `/home/bwwu/grid-audits/sid-extensions-p0/jobs/grid-letter-instruments-normal-20261124-formal-r0/output/run`。服务器终态导出的原始清单及传输 SHA-256 已验证。GitHub 的全量候选与逐用户结果采用无损 gzip。单个用户 ZIP 含两份原始 JSONL、实际 best、用于复算的官方文件／TFRecords／SID、完整绑定源码和核验入口。权重也保留在原服务器路径，具有 `new-server` 项目账号权限可从原 runner 终态导出。 [终态导出清单](run/export-manifest.json) · [传输回执](fetch.json) · [完整材料位置](artifact-location.json)

主控 CPU 核验用冻结 evaluator，非作者另行实现逐用户公式并对比原件。工程修正共两轮。第一次 CPU 回读的 restricted deserializer 拒绝 Lightning 保存的 `functools.partial`。绑定原作业 checkpoint SHA-256 后复用原先已审的完整反序列化方式，CPU fresh strict 成功。第二轮修正目录外候选附加审计中的单分数 hex 切片，保留正确的 8 个十六进制字符并重跑 CPU 核验。原实验文件、实际候选和指标未编辑。无新增依赖或环境配置。 [CPU 复算入口](verify_formal.py) · [独立审核](NON_AUTHOR_REVIEW.md)
