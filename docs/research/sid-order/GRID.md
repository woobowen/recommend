# GRID：SID 顺序扰动

[返回主题](README.md) · [逐条件来源和完整精度](sources.json)。固定研究快照 `9767729a5114e21b9c949892af3d759baca6a98e`；结果引用的运行版本分别记录在映射中。六组 Beauty 和 Toys/Sports B 的有效配置、request、结果路径均逐项列出；权重与 SID 为历史服务器引用，未在本次重新读取。

## 流程与干预

实际测试由 [formal_20261015/run.py](https://github.com/woobowen/recommend/blob/9767729a5114e21b9c949892af3d759baca6a98e/experiments/formal_20261015/run.py) 读取 spec、门槛和实际配置，调用 `common.py::instantiate` 创建模型/数据，再通过原测试生命周期生成和评价；原预测的评价复核与结果分别保留。序列数据使用各自 item→SID 映射，将历史按四码位展开，mask 区分有效位置；输入数据与分割路径以每个 effective config 和 request 为准，不因同名 Beauty 推定两路线 SID 相同。decoder 自回归逐码生成，使用原 beam、候选及 evaluator。

[common.py](https://github.com/woobowen/recommend/blob/9767729a5114e21b9c949892af3d759baca6a98e/experiments/beauty_extensions/common.py) 的 `bind_source` 选择 A 的 `references/wmh-original/GRID` 或 B 的 `GRID baseline`，插入 `sys.path`、切换 cwd；`load_config` 改写 Hydra model `_target_`、SID、checkpoint、分割目录和输出路径，校验资产后严格加载。A 桥接 `author/models.py::AuthorShuffleModel`；B 的 `baseline.py::BaselineShuffleModel` 在 encoder SID embedding lookup 上加 hook 并复用作者原 shuffle 方法。两条路线的参数及资产不同；不把它们当作严格同模型对照。

测试时只对每个物品的前 3 位或全部 4 位 **embedding 向量**组内重排；S3 保留第 4 位，S4 覆盖 4 位。原物品时间顺序、目标、有效 mask、padding 保留，码位对应 embedding 身份随向量移动；特殊 token 和 decoder 路径沿用原模型。作者实现使用 `torch.rand(...).argsort` 采样，允许恒等排列；不能与 LETTER 排除恒等的协议等同。评估时 loss 和 generate 的调用可分别采样，完整事件与局限见原结果回执。只作推理期干预，不将 Shuffle 解释为重训。

## 配置、资产和额外改动

| 项 | 实际依据与范围 |
|---|---|
| Beauty A | 例如 S3 [spec](https://github.com/woobowen/recommend/blob/9767729a5114e21b9c949892af3d759baca6a98e/experiments/formal_20261015/specs/beauty-a-3.json)：seed42、generation_k10；SID SHA `188af0341d8c232de73acd9aa7fc3390db8d430c8c71a94ec9816962d3d94f68`，checkpoint `21b174d051548d28eadc995c365c6c7beaf7410535fe726b8f720e9555a04e00`。 |
| Beauty B | [Normal spec](https://github.com/woobowen/recommend/blob/9767729a5114e21b9c949892af3d759baca6a98e/experiments/formal_20261015/specs/beauty-b-0.json)：SID `d41adcfb22a9a47d5e0424c068f4c74f33cc9fdbaf5db08540fbfa0e3d720e18`，checkpoint `3d8f03cf042293f2f7f02132c9d072f573f671d9550d567edd1754f50e53e605`；不能代入 A。 |
| Toys/Sports B | seed42、batch8、32-true、120 SID 输入位、TopK10；三个条件共用本域配置/SID/checkpoint。原 prefix 约束 false，非法候选不补齐、过滤或重排；逐域资产哈希和实际 effective config 在 sources 中。 |
| 原训练 | 上述 Shuffle 是继承 checkpoint 的测试；本导航不推断未唯一绑定的历史训练源码、优化器或选模。每个正式测试的 config 中训练相关字段不能自动当成该 checkpoint 训练生效值。 |
| 改动分类 | 原件 `references/wmh-original/GRID/`；工程桥接 `experiments/beauty_extensions/common.py`；研究变体 `author/models.py` 与 `baseline.py`；Toys/Sports 域 spec 在 `experiments/shuffle_20261017/specs/`。实际部署、Plan/审批及来源见每行 request 与版本图；桥接涉及路径和代码绑定，并非仅改文件名。 |

## 已有结果

表为原 SID 评价字段，0–1 比例，保留原报告的小数；未把 GRID Recall 和 LETTER item CCE 强行视为同一指标。每一行的原精度及 request 在 [sources.json](sources.json) 中。

| 路线 | 域 | 条件 | Recall@5 | NDCG@5 | Recall@10 | NDCG@10 |
|---|---|---|---:|---:|---:|---:|
| GRID A | Beauty | normal | 0.046237089 | 0.031252529 | 0.066449046 | 0.037725352 |
| GRID A | Beauty | shuffle3 | 0.046371236 | 0.031076979 | 0.066359609 | 0.037486158 |
| GRID A | Beauty | shuffle4 | 0.045968786 | 0.030933077 | 0.065733574 | 0.037285861 |
| GRID B | Beauty | normal | 0.043151636 | 0.029169954 | 0.065867729 | 0.036434427 |
| GRID B | Beauty | shuffle3 | 0.043285783 | 0.029069517 | 0.065018110 | 0.036036659 |
| GRID B | Beauty | shuffle4 | 0.042793900 | 0.028868502 | 0.064481512 | 0.035842434 |
| GRID B | Toys | normal | 0.042087369 | 0.026986862 | 0.065835565 | 0.034602936 |
| GRID B | Toys | shuffle3 | 0.040490419 | 0.025995448 | 0.065423451 | 0.034031183 |
| GRID B | Toys | shuffle4 | 0.039820731 | 0.025698222 | 0.065165877 | 0.033853244 |
| GRID B | Sports | normal | 0.024186753 | 0.015850427 | 0.035928983 | 0.019629370 |
| GRID B | Sports | shuffle3 | 0.022276532 | 0.014610196 | 0.032501824 | 0.017910449 |
| GRID B | Sports | shuffle4 | 0.022725996 | 0.014738602 | 0.034383953 | 0.018502345 |

完整 GRID Beauty 结果、原非法候选与 fixed32 观察见 [正式回执](https://github.com/woobowen/recommend/blob/9767729a5114e21b9c949892af3d759baca6a98e/manifests/grid-formal-20261015/FINAL_RECEIPT.md)；Toys/Sports 及 A 缺口见 [正式报告](https://github.com/woobowen/recommend/blob/9767729a5114e21b9c949892af3d759baca6a98e/manifests/grid-shuffle-20261017/FINAL_REPORT.md)。注意力为预定样本描述，不证明置换不变性。
