# 2026-11-20 SID 顺序主题补证

[主题入口](README.md) · [冻结 21 条来源](sources.json) · [GRID 逐条件证据](https://github.com/woobowen/recommend/blob/9767729a5114e21b9c949892af3d759baca6a98e/manifests/grid-a2-fd-recovery-20261018/final-results/cross-domain/cross-domain-evidence-index-with-a2.json)

旧 21 条主题来源包括 GRID 12 条、LETTER 9 条。GRID 还已发布 Toys A-5300、Toys A-5000、Sports A1、Sports A2 各三条件，补接这批后，主题下有 24 条 GRID 历史正式结果。各条件用本路线及数据域的固定权重，条件数与训练次数分别计算。

| 路线 | 已完成历史范围 | 运行源码 |
|---|---|---|
| Toys A-5300、A-5000 | 每路线 19,412 用户，Normal/S3/S4 | `b44399926c98369191c2ff5c2425377c8c07d3c8` |
| Sports A1 | 35,598 用户，Normal/S3/S4 | `b44399926c98369191c2ff5c2425377c8c07d3c8` |
| Sports A2 | 35,598 用户，Normal/S3/S4；独立 FD 恢复任务 | `85b77f5e2a5b497fdc1f620649ba488fc783a245`；快照 `22286966cd544773f21b96e4185d56c66afd7183fdc4e8888716d8ea1d77acb2` |

固定证据清单提供 job、输入、输出和审核来源。A2 保留原量化受控中断及后续 11 个阶段的对应关系。旧 Toys/Sports B 报告的 A 缺失项记录较早时点。查阅[最终科研报告](https://github.com/woobowen/recommend/blob/9767729a5114e21b9c949892af3d759baca6a98e/manifests/grid-a2-fd-recovery-20261018/final-results/FINAL_RESULTS.md)和 [24 条逐条件结果](https://github.com/woobowen/recommend/blob/9767729a5114e21b9c949892af3d759baca6a98e/manifests/grid-a2-fd-recovery-20261018/final-results/cross-domain/ORIGINAL_METRICS_WITH_A2.md)。

LETTER Shuffle V1 排除恒等排列。V2 有单独研究分支和运行记录。包含恒等排列的条件在对应分支维护，过程结果按观测时点解释。跨路线比较还须对齐模型、数据、编码、生成候选及评价协议。[V1 历史入口](https://github.com/woobowen/recommend/tree/94b79854a59f52208bc11f14fc6e3069594dab6a/lc_rec/shuffle) · [V2 运行记录起始快照](https://github.com/woobowen/recommend/tree/50507089981cb0b9d36e8977bd1d94cd9085f031)

## 四种复现起点

1. **保存输出复算。** `checks/validate.py` 对照旧 21 条固定结果和表值。GRID 额外 12 条有原始预测和汇总依据，随[现行单一审核 ZIP 和复算命令](https://github.com/woobowen/recommend/blob/wbw-grid/docs/delivery/2026-11-20/REPLAY.md)提供。GRID 可恢复原 JSONL 字节，LETTER/G3 可恢复全部已解析字段和值，二者的原始序列化身份和原件取得方法见同一入口。
2. **固定 checkpoint 重评。** 以各场运行的固定请求、权重、数据、SID 和 tokenizer 重建执行环境，推理和评价沿各自书面协议。外置载荷由获授权的项目参与者按原记录取得，本轮未运行。
3. **重跑约定流程。** 按真实 commit、有效配置、解释器和命令恢复原流程。沿已获验收的 attempt 追溯，失败及独立补算保留各自身份，本轮未训练。
4. **重建上游。** GRID 的 Qwen 工作资产、历史作者 T5／SID 和 LETTER 作者 index／交互及模型来源各自保存。未知的历史 producer、真正非自回归 Parallel 实现及外置访问条件仍有复现影响。

旧 `sources.json`、原报告和生产代码继续作为其固定历史版本使用。
