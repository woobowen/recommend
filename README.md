# GRID 基线：官方流程与 Qwen 流程

本分支 `yjc-grid` 分别保存作者官方 GRID 代码包和我们已经完成的 Qwen 基线实验。

| 目录 | 内容 | 实验状态 |
|---|---|---|
| [GRID official](GRID%20official/REPRODUCTION_NOTES.md) | 作者 `snap-research/GRID` 官方代码与配置；默认文本编码器为 Flan-T5-XL | 本次仅导入官方代码，未运行官方配置实验 |
| [GRID baseline](GRID%20baseline/README.md) | Qwen2.5-7B-Instruct → 三层 RQ-KMeans → 一位去重码 → 四位 SID → 普通 TIGER | Beauty、Sports、Toys 已完成训练及最佳 checkpoint 测试 |
| [Baseline Results](Baseline%20Results/README.md) | 我们的 Qwen 基线测试指标、论文参照和 checkpoint 记录 | 只对应 Qwen 流程 |

## Qwen 基线来源与下载

`GRID baseline/` 和 `Baseline Results/` 从 `main` 的固定提交 `563dc50b1c7e9e2e05cdfddd13e91a3d1888001e` 完整迁入，包含代码、完整数据、Qwen 权重传输分片、商品向量、四位 SID、量化器和推荐模型 checkpoint、配置、日志及指标。这两个目录与来源提交保持相同内容。

大文件通过 Git LFS 保存。浏览器 ZIP 或普通 Raw 链接可能只有 LFS 指针：

```bash
git lfs install
GIT_LFS_SKIP_SMUDGE=1 git clone --branch yjc-grid https://github.com/woobowen/recommend.git
cd recommend
git lfs pull --include="GRID baseline/**"
```

运行细节见 [Qwen 基线说明](GRID%20baseline/README.md)。复用 SID 或测试已保存推荐 checkpoint 时，无需加载 Qwen。重新生成商品向量前，按该说明合并 Qwen 权重分片。

## 官方代码来源

`GRID official/` 固定到作者仓库提交 `2fe3475b2d369580234093f35d52b1a2f54d0472`；92 个上游文件保持原内容。附加的来源清单、运行说明和项目根目录标记见 [官方流程说明](GRID%20official/REPRODUCTION_NOTES.md)。作者代码包不包含预计算 SID、训练 checkpoint 或本次官方配置测试结果。

原 `yjc-grid` 的早期五位 SID、768 维商品向量和旧训练记录已从分支当前内容移除。后续 Qwen 实验统一使用 `GRID baseline/experiments/` 下各数据集的四位 SID。


## Beauty G3 商品身份资料

[Beauty G3 补件入口](GRID%20baseline/codex/beauty_g3_identity/README.md)提供完整 12,101 列 SID→GRID ID→Amazon ASIN 映射、历史配置可读副本、原始预测 CSV、导出脚本与实际核验结果。固定原 G3 不变；本次仅补齐身份与来源信息，没有重新训练。
