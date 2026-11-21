# 原 LC-Rec Instruments 六任务定向核对

2026-11-21 通过既有 `new-server` 连接只读检查原作者数据 ZIP、六个展开输入和原准备作业位置。作者仓库 `RUCAIBox/LC-Rec` 的默认 `main` 仍指向原固定提交 `e6f1b3b3f8b4fee55a49a7b3dfeba29dbfcf0669`，其 Dataset 入口仍指向公开 Drive。项目路径 `/data/bwwu/lc-rec/assets`、`/data/bwwu/lc-rec/relay`、`/home/bwwu/lc-rec/six-task-20261110` 及 `/data/bwwu/archives` 的定向文件清点取得的是同一作者包和展开文件，未见一套与当前 split 配对且含生成来源的辅助原件。[固定作者入口](https://github.com/RUCAIBox/LC-Rec/blob/e6f1b3b3f8b4fee55a49a7b3dfeba29dbfcf0669/README.md#dataset) · [作者 Instruments.zip](https://drive.google.com/file/d/19tsIxV_LbErh5FfYhT732M82y0V-3bxR/view) · [定向查询记录](six-task-source-search.json)

## 来源和实际读到的材料

- 作者 ZIP 的实际服务器路径为 `/data/bwwu/lc-rec/assets/Instruments.zip`，13,612,638 字节，SHA-256 `96aad0cf1496a8ffef31bf1ca3a0d1033ca49473dff46b2c2dc27763224f9453`。内部六个用途文件位于 `/data/bwwu/lc-rec/assets/data/Instruments/`，原始大小和 SHA 与冻结记录逐项相同。以公开作者入口或既有服务器许可取得该原件时，按该摘要核对版本。[本轮来源核对](six-task-pairing-source.json) · [原资产核验](../assets/ASSET_AUDIT.md)
- `item2id` 的内部编号与 `item`、`index` 的字典键及实际 interaction 内物品集合相同。`user2id` 的内部编号与当前 interaction 和三类辅助数据字典的键相同。9,922 个物品对应 9,897 个 canonical SID，23 个重复 SID 组，保留原有歧义。原 LC-Rec index SHA `b94d16398a178220a117abda5effd5508069beaccc2b8110eab25ae50f2af600`，与 LETTER Instruments 的 index 不同，尽管两边 `inter`／`item` 文件原字节相同。[原 LC-Rec index 原字节副本](../../letter/evidence/assets/B/Instruments.index.json)
- 当前 `inter` 参与 SeqRec 的时间切分。ItemSearch 读取 `user_vague_intention.train` 的辅助 item、历史和 querys，并选对应 user_explicit_preference，实际历史最多取末 20。与公开生成链训练 `[-3]/[:-3]` 逐用户比对，12,233／24,772 对不上，12,539 条精确配对。测试原辅助 `[-1]/[:-1]` 有 7,473 条不配对。训练辅助 item 对应 validation/test target 分别 3,608／1,623，训练辅助实际最后 20 个可见历史中 1,623 个用户含 validation item、0 个含 test item。分项身份可以重叠，分母是整套 24,772 用户。[公开源码与原判据](../source_audit/aux_source_followup.md)
- PreferenceObtain 的历史由当前 `inter[:-3]` 得到，随后按 Dataset 截断。监督文本由同 UID 的作者 `user_explicit_preference` 提供。每个用户都有文本候选；逐条生成输入、原评论、时间与 split 事实无法由 ID 一致判定。公开辅助脚本的 test 组装以 train 字典为筛选条件，实包 train/test 都覆盖相同 24,772 用户，仍需生成版本及装配收据才能把实包和公开代码绑定。观察证明现有发布资产的配对限制，尚不确定论文实际生成历史。

[six-task-author-pairing.jsonl.gz](six-task-author-pairing.jsonl.gz) 是从六个冻结资产生成的全用户投影，字段和顺序见 [来源回执](six-task-pairing-source.json)。保留内部 UID、完整 interaction、train/test 辅助 target 和历史及每个用户的偏好候选数，文本、原始外部用户 ID 和随机选择仍留原出处。可用 Python 标准库在本目录重算全部数字，重算输出与实测 [投影回读](six-task-pairing-replay.json) 对照。

```bash
out=$(mktemp -d)
python3 -B check_six_task_pairing.py --verify-projection six-task-author-pairing.jsonl.gz > "$out/pairing.json"
```

## 资源与实验状态

| 实验阶段 | 已核实状态 |
|---|---|
| 原作者全参 ZeRO-3 双卡无 offload | 6,746,034,176 个参数，两组 FP32 Adam moments 双卡分摊即每卡 25.1309 GiB，高于 23.9883 GiB 显存。这个静态下界还未计权重、激活与缓冲；原要求下未进入真实 GPU 更新。 [静态资源报告](../resource_estimate.json) |
| 后续 optimizer-only CPU offload | 自己独立的 BF16／ZeRO-3 工程设置及两个真实 SeqRec global128 更新，含首步 LR0 和第二步非零更新，保存/恢复、正式六任务、全量指标未运行。 [CPU offload 正式记录](../../offload/RESULTS_20261109.md) |
| 后续六任务 LoRA | 双卡 ZeRO-2、micro2、accum32 的固定安全样本完成有界真实更新及指定压力和生成检查；micro4/micro8 的长尾 OOM 原样保留。没有完整 epoch、正式评测或论文指标复现。 [有界核对](../../six_task_lora/REVIEW_ENTRY.md) |

保留目前原件和失败记录。完成可信完整六任务训练需由原发布方提供与当前 `inter`、`user2id`、`item2id` 和 `index` 同源的 `user_vague_intention`／`user_explicit_preference` 原件、逐用户上下文和生成／汇总脚本版本、实际 split 与时间口径。重建早期上游还需要同版原始交互／物品归一化来源、embedding 编码模型及量化 checkpoint 的绑定。之后才可由负责人确认资源、精度与微批语义；上述短预跑没有替正式训练验收。本轮使用既有环境做 CPU 证据核查。
