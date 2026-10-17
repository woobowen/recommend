# Sports A1 真实非作者结果审核

SPORTS_A1_NONAUTHOR_REVIEW_PASS

审核代理：`/root/sports_a1_result_audit`。该代理未编写或运行原 Sports A1 研究实现，也未参与本轮 A2 修复。审阅并独立执行的脚本、原始取证副本和逐项 JSON 验收均保存在本目录；本报告不是对整个 A2 专项任务的最终验收。

## 结论与范围

Sports A1 的原训练、Normal/Shuffle3/Shuffle4 全量测试、同 cohort fixed32 注意力和 CPU 描述统计均通过本次独立核验。没有重新训练、没有 GPU 重评，也没有写入旧 job、旧模型或冻结源码。独立执行了 CPU checkpoint strict load、全部原预测指标与配对复算、全部 NPZ beam 祖链与统计复算、原 TFRecord 的有界 32 用户 cohort 检查。

指标下降是保留的真实结果；它不影响本次实现和完整性验收。该范围不能证明严格置换不变，也不能据注意力相似度推断 Decoder 不使用目标前缀。

## 版本与原始来源

- 科研源码：`b44399926c98369191c2ff5c2425377c8c07d3c8`。
- 科研快照：`/home/bwwu/grid-audits/sid-extensions-p0/grid-a-launch-20261017/snapshots/b6203414ee58a400`，完整 snapshot hash `b6203414ee58a40034d8a5e9c311ff509d897002804c2cb81f4c83bf419d2d41`。
- 共享接续控制器：`6dafbc7215b58a2dbb47905a687c930caecf7b44`；没有把它当作研究模型源码。
- 本轮最初读取本地 Git HEAD：`9ab3d0ea5d32db00e278f74c599a1943885453db`，分支 `wbw`。
- 对 11 份关键冻结科研源码直接重哈希，与服务器 DEPLOYMENT manifest 和 `git show b443999…:<path>` 字节逐项一致。包括训练、评价、Shuffle 模型、原 evaluator、attention capture/records/plot 和统计。
- 原始作业路径：`/home/bwwu/grid-audits/sid-extensions-p0/jobs/ga-20261017-sportsa1-{metadata,gpu-gate,train,eval-0,eval-3,eval-4,attention-0,attention-3,attention-4,stats}-v1/`。
- 已定向复制 334 份原产物、共 114735999 字节；取证入口 `copy_receipts.json`。模型载荷留服务器，没有大资产搬运或全服务器扫描。

## 训练与 checkpoint

原始 `optimizer_updates.jsonl` 逐条确认 9600 次真实 Adam 更新，153600 raw microbatches，47892458 个 causal expanded windows；每次严格 accumulation16，所有记录 loss 有限，每次被监视的 encoder/decoder SID 表梯度非零且有限。optimizer 原 callback 明确检查所有 trainable parameters 均受覆盖。

原 `validations.jsonl` 共 96 次 full validation，每次 35598 targets、4450 batches，step100 到 step9600 每100 optimizer updates 一次；这对应原批准的1600 loader batches。原完整 key-target multiset 检查均为真，索引 hash 一致。最高验证 Recall@5 = 0.03368166834115982，发生在 step8600，随后 step8700–9600 连续10次未改善。原 final checkpoint 的 EarlyStopping wait_count=10、patience=10，实际正常早停，没有借预算停止冒充完整训练。

训练入口固定 `seed42`、`32-true`、Adam `lr0.001/weight_decay0.0001`、无 scheduler、raw16/accumulation16、原 source cap512、val cadence1600、K10、prefix check关闭。冻结 `_fresh_model` 实例化新模型，`trainer.fit(...,ckpt_path=None)`；原实际训练记录表明 Adam state 起始为空，没有从 A1 以外推荐权重初始化。13259136 参数全部可训练，encoder/decoder SID表各为[1024,128]、不同storage且数值不同。

本审核额外在服务器隐藏 CUDA 的 CPU 进程中直接加载原 best/final checkpoint：best step8600 的178个state tensors通过 strict=True，missing/unexpected均空，所有keys/shapes/values精确相同且finite，embedded SID完全相同。best/final 所有128份 Adam parameter state 的step分别全为8600/9600。没有模型forward或GPU调用。

| 资产 | 字节 | SHA-256 |
|---|---:|---|
| Qwen SID `[4,18357] int64` | 588597 | `89b2c2bbb94eda26ef0e147404fbff29e6ec8246fbdc754c5d885f313f2de293` |
| best `checkpoint_epoch=000_step=008600.ckpt` | 161660262 | `0f2ce9ab6155eb9864363c32972d9f9ab775291e58f392b56e1db9901f3cb6f1` |
| final `recovery/training-final/full-state.ckpt` | 161653734 | `e67d76fdde33c228f82fde8dfa39579b698c7bfa0b793ddcc2dfe75f9a0fbc5d` |

SID 原路径：`/data/bwwu/sid-extensions-assets/563dc50b1c7e/experiments/sports_rkmeans_qwen_20261010_093035/semantic_ids.pt`。best/final 位于上述 train job 的 output 下。SID全18357物品各完整四码唯一，所有码在[0,255]。这是本次新训练配套资产，不能据它反推学长历史 Sports 权重来源。

## 正式测试与独立配对

以下均为原 evaluator 产出的正式原值，不以本审核float64复算值替代。每条件35598个独立用户、target键值multiset与原 testing index精确一致，均严格加载同一best和SID。所有原候选原样保留，非法项没有过滤、补齐或重排，重复项均为0。

| 条件 | Recall@5 | NDCG@5 | Recall@10 | NDCG@10 | 非法候选 |
|---|---:|---:|---:|---:|---:|
| Normal | 0.025816056877374649 | 0.01716848649084568 | 0.037586379796266556 | 0.020962946116924286 | 20 |
| Shuffle3 | 0.019748300313949585 | 0.013000275008380413 | 0.02949603833258152 | 0.016144171357154846 | 10 |
| Shuffle4 | 0.019860666245222092 | 0.012999451719224453 | 0.029299398884177208 | 0.016071824356913567 | 14 |

独立脚本不调用原结果统计函数，逐行按四码exact match与原score顺序复算；与原FP32指标最大绝对差小于1.47e-9。每用户原per-sample metric也逐项相符。Normal与Shuffle4各有1行score相等，均未涉及正确target，未产生相关rank歧义。

原 Normal trace 8900 次 encoder调用中无Shuffle；Shuffle3/4各8900次Shuffle，loss与generate各4450、每次分别采样。所有保存的原随机排列为原rand→argsort语义。源码核实先层级offset、再lookup、再物品occurrence内完整向量重排；第四码在Shuffle3保留，目标/历史物品时间轴/decoder顺序未变。三份有效配置的独立recursive diff仅出现Shuffle开关、Shuffle4的n=4和各job输出目录。

| 与自身Normal配对 | Top1改变 | Top10顺序改变 | Top10集合改变 | Hit@5丢失/获得 | Hit@10丢失/获得 | Top10重叠均值 |
|---|---:|---:|---:|---:|---:|---:|
| Shuffle3 | 21075 | 35547 | 35030 | 373/157 | 522/234 | 4.809230856 |
| Shuffle4 | 21024 | 35587 | 35152 | 366/154 | 515/220 | 4.889825271 |

完整逐用户rank/Hit迁移在 `paired_normal_shuffle3.jsonl`、`paired_normal_shuffle4.jsonl`，未按测试分数挑模型或样本。rank缺失明确定义为11，both-hit rank差另外单列在JSON中。

## fixed32注意力与独立统计

特别说明：原fixed32明确来自 **evaluation/validation split**，正式全量测试来自 **testing split**。相同user key不代表二者历史/target相同。没有把注意力Top1强行绑定为正式testing预测；三种条件注意力之间使用完全相同的已固定evaluation cohort。

cohort hash：`3935dc0c085ad6134d6b5da50c387ee59e7b990e7d21226afcc3655ad211f565`。按实际evaluation文件分配和行顺序预先选择首32 distinct users，保存时 `model_outputs_inspected=false`。本审核另从原`evaluation/partition_26.tfrecord.gz`直接只读解析前32条原Example，独立重建最后target/hidden完整四码/120维历史与mask，32条均与cohort精确一致；没有完整数据重扫描。

每条件4×8用户，4 NPZ＋4 JSON records＋32 PNG＋4 plot receipts，三条件合计12 NPZ、96 PNG。实际raw heads [8,4,6,150]，head mean [8,4,150]。每条件独立核对1280个beam child前缀→parent映射、全部final Top1完整反向祖链、local-first每step第0 parent、head mean、4800个encoder位置的item/SID/offset/mask及真实排列身份；全部精确一致。首步为BOS，之后query使用真实选中prefix最后token。

所有原PNG均解码、核对SHA/大小/图尺寸2880×1280、对应record与原未缩放矩阵scale；每条件首张还实际看图核对。原plot源码直接imshow两条raw head-mean路径，统一paired scale，保留全部150列，没有逐行拉伸。没有声称跨matplotlib版本bit-exact重绘。masked mass全部0，raw head行和最大误差≤2.39e-7。

独立CPU统计覆盖全部32×6query pairs×2paths=384条/条件，cosine/Pearson用visible raw weights，TV/JS仅在visible行转换概率后计算，包含可见SEP；未改变保存矩阵。与原每条及汇总统计最大差<6.7e-16。

| 条件/路径 | mean Cosine | mean Pearson | mean TV | mean JS (nats) |
|---|---:|---:|---:|---:|
| Normal / final Top1 | 0.842806477 | 0.277576508 | 0.173100347 | 0.050504576 |
| Normal / local first | 0.845255244 | 0.283250396 | 0.168693421 | 0.047678936 |
| Shuffle3 / final Top1 | 0.757919802 | 0.358939588 | 0.218251402 | 0.069116660 |
| Shuffle3 / local first | 0.760653003 | 0.351342254 | 0.218745768 | 0.068581996 |
| Shuffle4 / final Top1 | 0.786710231 | 0.400166961 | 0.207172997 | 0.064241003 |
| Shuffle4 / local first | 0.781820807 | 0.386177976 | 0.207475489 | 0.064029325 |

未做未对齐的跨Shuffle列直接比较。所有样本保留，不用最漂亮用户代表总体；这些均为描述统计，无预设相似性准入线。

## 作业资源与当前进程

定向读取10个A1相关job均SUCCEEDED/exit0/cleanup verified。另按原记录的boot_id、PID、start_ticks直接核对原supervisor/worker/known members，本轮查询时没有仍属原身份的活进程；不依靠仅看PID数值，也未扫描无关账户。

训练GPU请求计时39196.161534秒（10.887822648小时），小于43200秒/12小时；三条件eval+attention共954.844502秒（0.265234584小时），小于原9000秒/2.5 GPU小时。共享调度版本及两GPU UUID保留在原request中。这里按原job started→ended×requested GPU计量，非实际独占GPU时长。CPU本审核不新增GPU额度。

## 未验证范围与结论边界

- A1是Qwen SID条件下的新双表训练方法复现，不是找回学长原Sports权重。
- 没有重新枚举所有TFRecords；全量覆盖基于原生产target index、完整预测及原validation exact-multiset事件交叉核验，另对32条原cohort做了直接TFRecord复核。
- 从零初始化通过冻结入口、有效配置、空Adam初态和实际更新事件确认；原实验没有保存完整随机初始参数dump，本审核不捏造该资产。
- 没有重放历史GPU调度；过去GPU共享事实保留原回执语义。
- 本审核不代替Toys/A2审核，Toys producer未唯一确定的限制须在总交付中保留。
- 本次reviewer脚本首轮曾把hidden/pad lookup index错误假定为raw+offset；查原代码发现mask乘法使其为0后修正审核assert。原实现/配置/产物/实验均未改，没有科研重跑、没有重置历史3/3修正次数。

本审核未安装依赖、未修改环境配置、未进行Git写操作。报告与原始轻量证据由主控纳入最终单一GPT审核ZIP；本子审核本身不声称总包已经制作或二次审核完成。

## 可复核入口

- `audit.py`：独立本地CPU全部预测/训练事件/NPZ/PNG/统计检查。
- `remote_checkpoint_audit.py`、`checkpoint_direct_audit.json`：原checkpoint CPU strict load与冻结源字节核验。
- `remote_cohort_process_audit.py`、`cohort_process_direct_audit.json`：原32条TFRecord与原process identity定向检查。
- `audit_result.json`：完整逐项机器验收，含all original metrics、paired rank/Hit、全部attention summary。
- `attention_independent_query_pairs.json`：每条件全部384条独立统计。
- `raw/`：按原job路径保存的已核hash轻量原产物；`copy_receipts.json`和`inventory.json`保存原路径及文件身份。

GPT_UPLOAD_REQUIRED: YES（原始预测与NPZ在本地和服务器，待主控合并至最终一次上传ZIP。）
