# GRID 基线入口

`yjc-grid` 保留早期 GRID 通用源码供参考。当前研究采用的基线是 **Qwen2.5-7B-Instruct → 三层 RQ-KMeans（每层 256 个码字）→ 一位去重码 → 普通 TIGER**，完整 SID 为四位。

请统一使用以下固定版本：

- [当前基线代码、数据与实验产物](https://github.com/woobowen/recommend/tree/563dc50b1c7e9e2e05cdfddd13e91a3d1888001e/GRID%20baseline)
- [Beauty、Sports、Toys 最终测试指标](https://github.com/woobowen/recommend/tree/563dc50b1c7e9e2e05cdfddd13e91a3d1888001e/Baseline%20Results)
- [当前基线运行与下载说明](https://github.com/woobowen/recommend/blob/563dc50b1c7e9e2e05cdfddd13e91a3d1888001e/GRID%20baseline/README.md)

## 已清理的早期材料

2026-11-18 清理了与当前基线不一致的预计算五位 SID、配套 768 维商品向量、相关单卡短测和三卡训练/推理记录，以及旧复现命令。上述早期材料不应作为当前研究的 SID、checkpoint 或指标来源。

本分支的 `GRID/src/` 和 `GRID/configs/` 仅作历史通用实现参考；当前基线使用固定链接中的代码和实际运行配置，具体入口见上方说明。
