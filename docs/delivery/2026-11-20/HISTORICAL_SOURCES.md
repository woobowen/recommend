# 2026-11-21 历史原件收尾核查

本次从现行 Git、上一轮已封存的原件、Windows SSH 的授权项目目录和 A800 原档目录清单定向取证。所有研究运行、失败记录和旧显示结果保持原状。时间以各查询回执为准。下列“仍需”是继续证明历史复现时可直接转述给相应保管者的索取项。

## Parallel

- 已读四个 Git 固定版本（`f19b93d`、`800b5a4`、`7e943ea`、`84d4533`），包括后来从另一条分支移除的同名文件。四处 `tiger_generation_model_parallel.py` 与保全 Shuffle 源同为 61,418 字节、SHA-256 `e03082ad87bfc842c05467580fc479eff4345c6a0d18c5b2f7faf320d4f96c80`。两个可读服务器拷贝也匹配。源码 `generate` L952–1061 按码位循环，当前 decoder 输入依赖已生成的前缀。另一个 no-pos 源仍保留逐步 decoder cache。13 份 A800 capture manifest 的命中路径和可读服务器副本保存在 [档案范围](evidence/grid-history/a800-historical-index-probe.json)及[匹配记录](evidence/grid-history/grid-git-originals-check.json)。包内 `repository/wbw-grid/references/wmh-original/GRID/` 有冻结源码和配置。原始候选代码与 4BOS 配置见 [Parallel 入口](https://github.com/woobowen/recommend/blob/caf3915ccaeef1ed9fee9c5b81e05dd2f6bcdf37/docs/sid-extensions/experiments/parallel.md)。
- 2025-12-02 原日志确有 Parallel 初始化和测试，2025-12-03 原尝试出现加载失败。包内附两段原日志及既有独立审查原件（`evidence/closeout/parallel-originals/`）。模型参数差与多个 BOS 一致，真实 query、mask、损失和 Top-K 联合分数仍无法从这些观察还原。
- **仍需王学长提供** 2025-12-02 实际 checkout/export 的源码与 commit 或文件清单，原 launcher、Hydra overrides 及最终生效配置，checkpoint/SID 对应，以及当时合法候选组合、分数和评价源码。现有资料可审查历史现象和已保存汇总，无法复算原候选、重评真正非自回归模型或重跑原协议。

## Toys MLP

- 恢复四份 A800 中原 Hydra 配置及两个推理运行的原 log／overrides。2025-09-22 `08-44-16` 原日志记录合并 11,924 条，原 embedding tensor 实测 170,943,798 字节、SHA-256 `db7a21b3aa4389c2bfdcc5a3be81b781cf0c3b90e02e331143e46803666dcc04`。只解析容器元数据核得 float32、`[11924,3584]`，未执行 pickle 或模型加载。原配置同时记录 T5 factory、Qwen2.5-7B-Instruct 路径及 mean aggregation。推理日志确认输出 run 身份；同一尝试的 [调度原日志摘录](evidence/toys-author-originals/encoder-initialization-warning.json) 提示使用 T5EncoderModel 从 Qwen 路径实例化，部分 T5 权重未加载并重新初始化。现有资料仍无法确认完整实际权重、tokenizer revision、初始化状态、当时执行的源码快照及原 item ID 对应。
- 2025-09-22 `10-05-01` 原配置和 overrides 显式消费该 embedding 路径，用三层 RQ-VAE／STE 和 `08-50-06` quantizer checkpoint；其 `[4,11924]` int64 SID 原张量 SHA-256 `26bae0c90859e1ccf7517dbdd077adc75d3550bb9a637df9905c0dc6eca73eb0` 已实际取回并随交付提供。第四码用于同前缀区分。10 月 `21-25-31` 原训练命令显式读取同一路径。旧 `21-28-23` 是不同 R-KMeans SID 条件。可对照[原 overrides](evidence/toys-author-originals/20250922-10-05-01-overrides.yaml)、[张量身份](evidence/toys-author-originals/tensor-identity.json)和[原 SID 载荷](evidence/toys-author-originals/toys-semantic-ids.pt)。item 索引、SID 轴、目标及 checkpoint 的既有实际配对在[既有非作者记录](https://github.com/woobowen/recommend/blob/caf3915ccaeef1ed9fee9c5b81e05dd2f6bcdf37/manifests/grid-shuffle-mlp-preflight-20261015/source-conditions-review.json)。大 embedding 原件按上述记录由项目保管人提供。
- 正式复用该 SID 的 `gf-20261015-toys-mlp-formal-v1` 完整运行代码 `4495c38459971aee36b00d881eb50f50fafa3159`、导出 SHA-256 `77c012c07de3db19bc86108212ab292aabe704922f1f65d1d9c51da62b05a108`，best step 4,400，测试 19,412 用户。实时服务器与已发布 request、有效配置、训练／复算等 8 份原件全部字节一致，见[逐件比对](evidence/grid-history/grid-git-originals-check.json)。当前已恢复从 embedding 原载荷到 SID 与训练的历史文件关联，能够按发布说明复算保存输出。固定 checkpoint 重评依赖获准的真实权重与输入，本轮未执行；有源复建仍需上游身份。
- **仍需王学长提供** `08-44-16` 当时运行的编码器代码和实际权重／tokenizer revision、加载日志或有效配置覆盖及有序 item ID／索引交付记录。已有实际 T5 类与 Qwen 加载来源的日志证据。仍需上述材料确定唯一运行快照及有效权重，再验证从原文本重新生成相同行序 embedding。

## GRID 最早工作 baseline

- 10 月 7 日学弟 full/smoke 两份原命令均为 `train=true test=false`。原 CSV 有 42／11 个验证行，不能据此要求一份当时没有执行的 test 预测。王学长 A800 旧结果和 10 月 10–11 日 Qwen 工作 baseline 各有不同来源。
- Qwen 工作 baseline 三域保存的 test CSV、命令、Hydra、进度与实际 LFS 内容已回读。共同条件含 H=4、单卡、32-true、train+test、每 1600 raw batch 验证。Beauty 从 checkpoint 12,500 恢复，Sports/Toys 的该入口从空 checkpoint 起步。测试分别记录在 Beauty 13,500、Sports 10,500、Toys 7,200 更新。对应 `_运行/grid_stage.py` 的 train/test 路径只保存聚合结果，候选预测由独立 infer 路径负责。授权 live stage 与历史目录清单未发现对应 test 的逐用户预测。实际[原件目录](evidence/grid-history/review-baseline-originals/)、[命令和结果身份](evidence/grid-history/review-baseline-origins.json)及 Git 树收录。后续 B Normal 是自己的 Shuffle 对照，不能替换此处证据。
- **仍需原 baseline 保管人提供** 10 月 10–11 日三域每个测试用户的目标、顺序、候选及分数，连同原测试 split、有效配置、checkpoint、SID 版本和生成出的 job/export 路径。汇总数可直接复核。缺件限制对最早 baseline 的候选级保存输出复算；本次也未固定 checkpoint 重评或重跑。

## LC-Rec 六任务

[LC-Rec 实验入口](https://github.com/woobowen/recommend/blob/wbw-lc-rec/docs/research/LC-REC.md)说明 Instruments 辅助来源、interaction split、ItemSearch／PreferenceObtain 与 item/SID 配对，分别记录原资源阻塞、有界技术预跑和 LoRA 检查。原作者资产的全量数值配对投影、复算代码、来源清单和不利证据见包内 `repository/wbw-lc-rec/lc_rec/evidence/repository-final-closeout-20261121/`。如要开展正式六任务，还需原保管者提供与主序列对应的辅助生成和 split 原件，明确时间边界、实际排除规则及足够资源，按单独的研究决定执行。本轮完成保存证据归档，正式完整六任务没有完成。
