# 研究主题与实验族

本分支 `wbw-sid-order` 面向 Semantic ID 内部码字顺序对下游生成式推荐的影响：[进入 SID 顺序研究](sid-order/README.md)，或先读 [逻辑与物理结构映射](sid-order/STRUCTURE.md)。它包含 GRID/TIGER 路线和 LETTER-LC-Rec 路线，模型、数据域、Normal/S3/S4 和 seed 是主题内部的实验维度。把两条路线放在一起不等于它们已构成严格公平对照。

| 问题或材料 | 现有定位 | 维护边界 |
|---|---|---|
| SID 顺序扰动 | [主题总览](sid-order/README.md)、[GRID](sid-order/GRID.md)、[LETTER](sid-order/LETTER.md)、唯一条件来源 [sources.json](sid-order/sources.json) | 本分支是后续研究与统一阅读入口，历史结果继续指向原 RUN。 |
| MLP 聚合 | [三域已有结果](https://github.com/woobowen/recommend/blob/9767729a5114e21b9c949892af3d759baca6a98e/manifests/grid-formal-20261015/FINAL_RECEIPT.md) | 不同研究问题；本轮未创建专门分支。 |
| Attention | 上述报告及 [Toys/Sports Shuffle 报告](https://github.com/woobowen/recommend/blob/9767729a5114e21b9c949892af3d759baca6a98e/manifests/grid-shuffle-20261017/FINAL_REPORT.md) 的固定样本观察 | 是与相应实验关联的观察，不能作为严格不变性证明。 |
| Parallel | [历史源码状态](https://github.com/woobowen/recommend/blob/9767729a5114e21b9c949892af3d759baca6a98e/docs/sid-extensions/experiments/parallel.md) | 真正非自回归正文未恢复；没有以同名参考文件冒称正式结果。 |
| 原 LC 六任务及预跑 | [LETTER 文档中的历史入口](sid-order/README.md) | 保留预跑或受阻的真实状态，不作为 LETTER 单任务正式实验。 |

本分支继承的其他源码和历史材料保留原目录，不因主题范围而删除。其他实验族将来是否设专门分支、是否同步历史分支，由用户另行决定。
