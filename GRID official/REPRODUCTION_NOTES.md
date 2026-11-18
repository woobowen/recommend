# 作者官方 GRID 代码包

本目录来自 [snap-research/GRID](https://github.com/snap-research/GRID)，固定提交为 [`2fe3475b2d369580234093f35d52b1a2f54d0472`](https://github.com/snap-research/GRID/tree/2fe3475b2d369580234093f35d52b1a2f54d0472)，提交日期为 2025-10-15 UTC。

92 个官方文件按原内容导入，包括源码、Hydra 配置、依赖、README、许可证和第三方声明。各文件 Git blob 标识、SHA-256、文件模式和大小记录在 [UPSTREAM_SOURCE.json](UPSTREAM_SOURCE.json)。

## 流程与现有产物

- 官方配置 `configs/experiment/sem_embeds_inference_flat.yaml` 默认使用 `google/flan-t5-xl` 编码商品文本。
- 官方 README 的 RQ-KMeans 示例为三层、每层 256 个码字，生成时追加一位去重码，下游使用四位 SID。
- 本目录是官方代码包。本次没有运行文本编码、量化训练或推荐训练，没有导入作者未发布的 SID、checkpoint 或测试指标。
- 我们实际跑出的 Qwen 向量、SID、checkpoint 和指标位于相邻的 `../GRID baseline/` 与 `../Baseline Results/`，分别按该流程的说明使用。

## 运行准备

从本目录开始，按 [官方 README](README.md) 准备 Python/CUDA 依赖、下载 Amazon-P5 数据，并依次进行商品文本编码、量化器训练、SID 生成、TIGER 训练与测试。数据、Flan-T5 模型权重和运行环境需要自行准备；运行输出使用新的独立目录。

本次仅额外添加以下三个文件，上游源码和配置均未改写：

| 附加文件 | 用途 |
|---|---|
| `.project-root` | 空的项目根标记。官方训练和推理入口使用 `rootutils.setup_root(..., indicator=".project-root")`；该标记使嵌套目录中的项目根定位明确。 |
| `REPRODUCTION_NOTES.md` | 本说明，区分官方代码包和 Qwen 已完成实验。 |
| `UPSTREAM_SOURCE.json` | 固定来源版本和逐文件校验信息。 |

许可证见 [LICENSE](LICENSE)，第三方声明见 [notices.txt](notices.txt)。
