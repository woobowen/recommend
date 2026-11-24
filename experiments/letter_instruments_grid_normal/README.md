# LETTER Instruments → GRID 原生 Normal

**正式前准备为 PARTIAL / BLOCKED。** CPU 全轴与模型核验完成，GPU 预跑在首个 microbatch 的新增审计回调处失败。正式未提交。证据和修正范围见[验收状态](evidence/STATUS.md)。

本次准备采用 LETTER 发布的 Musical Instruments 四位 SID 与对应交互。正式执行拟在单张 RTX 4090 上以 seed 42 从随机初始化训练一次，再从本轮验证选出的最佳 checkpoint 测试一次 Normal。测试随机种子为 2026。正式提交需网页二审及用户明确启动授权。

## 固定依据

- 学弟历史运行的实际源码、配置、选模与资产对应均固定到发布 (74911e3ce668e265a8c3e7bc9a3929ae9247f11a)。[来源目录](https://github.com/woobowen/recommend/tree/74911e3ce668e265a8c3e7bc9a3929ae9247f11a/LETTER%20SID%20%2B%20TIGER)
- [历史有效配置](historical/config.yaml)、[Hydra overrides](historical/overrides.json)、[源码文件哈希](historical/source_manifest.json)、[划分审计](historical/mapping_audit.json)与[选模记录](historical/stage_result.json)用于对照。历史 best step 5400，结束 step 6400。
- [LETTER 固定上游](https://github.com/HonghuiBao2000/LETTER/tree/8d0154e28de37dbb6e24871c508ad8ddb1921cda/data/Instruments)提供 index、inter、item。本轮数据入口在服务器 `/data/bwwu/grid-letter-instruments-normal-20261124/assets`。复核时三份原 JSON 与派生资产在审核 ZIP 的 `assets/`，也可从固定上游取得 JSON 并用准备脚本从空输出目录重建。
- 统一 SID 评价复用 [A1 冻结评价源码](https://github.com/woobowen/recommend/blob/851425db7de3c70c6c4925740ecc20d38bae3686/lc_rec/a1_qwen3_06b/scripts/offline_metrics.py)的稳定排序、Hit、NDCG、MRR 定义及 CCE 对照公式。CCE Item 调用 [冻结 CCE 实现](https://github.com/woobowen/recommend/blob/1706a9eb145cf58b42ffa4e80aed58e962a2b864/lc_rec/letter/review/final_audit_20261113/external/official_cce.py)中的 `compute_cce_metrics`。两份来源原文见 [frozen_eval](frozen_eval/)。

## 数据与调用链

`data_assets.py prepare` 调用 [原转换脚本](native/codex/letter_official_downstream_20261116/prepare_data.py)，验证 24,772 人、9,922 件商品、206,153 条交互。四个码位转换成 `[4,9922]` 的 int64 tensor。每码位保留 256 槽，模型映射到共享表的四段。9,897 个完整 SID，23 个碰撞组、48 件商品，122 名测试用户的目标落在碰撞组中。

每个划分覆盖同一用户数字轴。训练保存 `S[:-2]`，验证保存 `S[:-1][-21:]`，测试保存 `S[-21:]`。独立读回全量 TFRecord 并逐行比对用户、商品序列和 SID 存储。三份分片和 SID tensor 的内容哈希与历史一致。

[启动脚本](run_normal.py)从原 Hydra YAML 与覆盖列表装配，先按已发布的 Instruments 有效配置逐项比较。迁移路径、记录用 run ID 并移除一项未被 datamodule 引用的占位字段。训练和评估通过 [pipeline_launcher](native/src/utils/launcher_utils.py)、[datamodule](native/src/data/loading/datamodules/sequence_datamodule.py)、[TFRecordIterator](native/src/data/loading/components/iterators.py)、[UnboundedSequenceIterable](native/src/data/loading/components/dataloading.py)与[原 collator](native/src/data/loading/components/collate_functions.py)。训练源 batch 16，原 causal duplicate collator 枚举连续历史与目标窗口，超过 128 时有放回采样索引并按成员判断保留，单 microbatch 扩展数因重复抽样而可能低于 128。最长窗口裁到 120 token，包含最多 29 件历史加目标。验证和测试保留至多 20 历史加目标。原标签模块取最后四个非 padding 码作为标签，目标位置置为 `-1`，输入 mask 屏蔽它们。

## 训练、生成与评价

[实际模型](native/codex/sid_order_launch/sid_order_shuffle_model.py)使用 T5 风格编码器、解码器各 4 层，模型宽 128、6 heads、key/value 宽 64、FFN 宽 1024，原两层隐藏 MLP、dropout 0.15。13,128,064 参数，SID embedding 共享，保留可学习 BOS/SEP 与四个 256 类预测头。拼入 SEP 后编码器张量最多 150 个位置。训练使用四码位交叉熵之和。初始 `ckpt_path=null`，launcher 在实例化前 seed 42，fit 前再设 seed 42。Adam `lr=0.001`、`weight_decay=0.0001`、无 scheduler。单卡 FP32 `32-true`、`compile=false`、`matmul_precision=medium`、原确定性开关及 `CUBLAS_WORKSPACE_CONFIG=:4096:8`。TFRecord 训练行 shuffle 保留原 buffer=128 与无显式 TensorFlow seed，因此跨环境不保证训练次序或权重逐位复现。

每 16 microbatch 更新一次，实际更新规模随扩展 batch 变化。每 1600 个训练 microbatch 执行验证，相当于本配置下 100 次更新。验证 batch 8，以 `val/recall@5` 最大值选一个完整 checkpoint，同一指标连续 10 次验证无改进则早停。正式配置上限是 320,000 次更新和 10 个 trainer epoch。训练流持续重开 TFRecord，trainer 的 epoch 不代表一次全量数据遍历。测试结果位于训练和选模之后。

[ProtocolMixin.eval_step](native/codex/plan_v1_1/protocol.py)沿原路径先算 teacher-forcing loss，再用独立 encoder 历史和新 decoder cache 进行四步 beam 10。原生 beam 保留 softmax 后逐码位概率乘积、cache 重排及前缀过滤关闭。模型三阶段均使用 Normal。原生 evaluator 直接处理同一组 ids 和 scores。两个写出钩子见 [观测 diff](native_observation.diff) 与 [候选保存](candidates.py)，保存原生成位置、四码、score 的 little-endian float32 位模式、四项指标实际使用的 top-k 索引、用户、目标商品/SID、历史 item 序列、编码前输入与 mask。正式预期 3097 batches、24,772 人与 247,720 条候选，最后 4 人保留。

[离线入口](evaluate.py)复算原生 Recall@5/10 和 NDCG@5/10，并从同一原始候选按稳定分数降序计算统一 SID Hit@1/5/10、NDCG/MRR@5/10 和 CCE ItemHit/ItemNDCG@5/10。重复目标时原生 Recall 的分母按生成候选中的相关数计算，SID Hit 只关心首次命中。统一 SID NDCG 累计重复目标 gain，CCE 从首个目标 SID 展开碰撞位置，对组内商品给均匀 credit。目录外 SID 保留位置且展开大小为一。原生排序可能与统一排序在并列时不同。历史 Beam20 系统有不同的搜索预算。预跑分数用于检查工程链路。

## 审核入口

- `evidence/` 含本轮结果、独立审核、正式 dry-run、资源估计、版本和 runner 终态。完整审核 ZIP 另附全量数据、预跑 checkpoint、请求与日志。
- 在审核 ZIP 根目录运行 `python3 verify_delivery.py <解包根目录>` 可复算离线数据、候选和评价。服务器 GPU 预跑和正式启动使用已部署的 P0 runner。正式计划文件仅供后续授权后提交，需先核对应代码资产与 GPU 占用。
