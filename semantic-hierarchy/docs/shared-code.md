# 共用代码、冻结配置和数据路径

## 上游沿用 GRID 基线

本实验的 SID 来自 **GRID 基线的 Qwen2.5-7B-Instruct → RQ-KMeans**，不是 LETTER SID，也不是对 LLM2Vec 输出再量化。Beauty/Toys 不重新训练上游；Instruments 按同协议生成新的 SID。

共用代码已存在于 `main` 分支的 `GRID baseline/`。下列链接固定到本次核对的提交 `563dc50b1c7e9e2e05cdfddd13e91a3d1888001e`，避免以后分支变更导致来源漂移。本次提交不复制这些代码或大文件。

| 用途 | GitHub 路径（固定提交） |
|---|---|
| 基线总说明 | [GRID baseline/README.md](https://github.com/woobowen/recommend/blob/563dc50b1c7e9e2e05cdfddd13e91a3d1888001e/GRID%20baseline/README.md) |
| Qwen 文本编码入口 | [GRID baseline/_运行/generate_qwen_embeddings.py](https://github.com/woobowen/recommend/blob/563dc50b1c7e9e2e05cdfddd13e91a3d1888001e/GRID%20baseline/_运行/generate_qwen_embeddings.py) |
| 原均值池化实现 | [GRID baseline/src/models/components/network_blocks/aggregation_strategy.py](https://github.com/woobowen/recommend/blob/563dc50b1c7e9e2e05cdfddd13e91a3d1888001e/GRID%20baseline/src/models/components/network_blocks/aggregation_strategy.py) |
| 原池化包装 | [GRID baseline/src/models/components/network_blocks/embedding_aggregator.py](https://github.com/woobowen/recommend/blob/563dc50b1c7e9e2e05cdfddd13e91a3d1888001e/GRID%20baseline/src/models/components/network_blocks/embedding_aggregator.py) |
| RQ-KMeans 训练配置模板 | [GRID baseline/configs/experiment/rkmeans_train_flat.yaml](https://github.com/woobowen/recommend/blob/563dc50b1c7e9e2e05cdfddd13e91a3d1888001e/GRID%20baseline/configs/experiment/rkmeans_train_flat.yaml) |
| SID 生成配置模板 | [GRID baseline/configs/experiment/rkmeans_inference_flat.yaml](https://github.com/woobowen/recommend/blob/563dc50b1c7e9e2e05cdfddd13e91a3d1888001e/GRID%20baseline/configs/experiment/rkmeans_inference_flat.yaml) |
| 残差量化与归一化实现 | [GRID baseline/src/modules/clustering/residual_quantization.py](https://github.com/woobowen/recommend/blob/563dc50b1c7e9e2e05cdfddd13e91a3d1888001e/GRID%20baseline/src/modules/clustering/residual_quantization.py) |
| 单层量化 | [GRID baseline/src/modules/clustering/vector_quantization.py](https://github.com/woobowen/recommend/blob/563dc50b1c7e9e2e05cdfddd13e91a3d1888001e/GRID%20baseline/src/modules/clustering/vector_quantization.py) |
| K-means 实现 | [GRID baseline/src/models/modules/clustering/mini_batch_kmeans.py](https://github.com/woobowen/recommend/blob/563dc50b1c7e9e2e05cdfddd13e91a3d1888001e/GRID%20baseline/src/models/modules/clustering/mini_batch_kmeans.py) |
| 原阶段与量化调度 | [GRID baseline/_运行/grid_stage.py](https://github.com/woobowen/recommend/blob/563dc50b1c7e9e2e05cdfddd13e91a3d1888001e/GRID%20baseline/_运行/grid_stage.py)、[run_pipeline.py](https://github.com/woobowen/recommend/blob/563dc50b1c7e9e2e05cdfddd13e91a3d1888001e/GRID%20baseline/_运行/run_pipeline.py) |
| 基线依赖与冻结环境 | [requirements.txt](https://github.com/woobowen/recommend/blob/563dc50b1c7e9e2e05cdfddd13e91a3d1888001e/GRID%20baseline/requirements.txt)、[codex/environment-freeze.txt](https://github.com/woobowen/recommend/blob/563dc50b1c7e9e2e05cdfddd13e91a3d1888001e/GRID%20baseline/codex/environment-freeze.txt) |

配置模板可能含默认 BF16/DDP，**实际运行配置及命令覆盖值才是复现依据**，不要直接将默认模板当作2080Ti上的最终设置。Beauty/Toys实际量化配置链接如下；Instruments配置冻结在本分支，使用FP16文本模型和32位量化训练。

## 已发布的 Beauty/Toys 资产

| 资产 | 已存在的 GitHub 路径 |
|---|---|
| Beauty SID | [GRID baseline/experiments/beauty_rkmeans_qwen_20261010_012253/semantic_ids.pt](https://github.com/woobowen/recommend/blob/563dc50b1c7e9e2e05cdfddd13e91a3d1888001e/GRID%20baseline/experiments/beauty_rkmeans_qwen_20261010_012253/semantic_ids.pt) |
| Toys SID | [GRID baseline/experiments/toys_rkmeans_qwen_20261010_093036/semantic_ids.pt](https://github.com/woobowen/recommend/blob/563dc50b1c7e9e2e05cdfddd13e91a3d1888001e/GRID%20baseline/experiments/toys_rkmeans_qwen_20261010_093036/semantic_ids.pt) |
| Beauty 实际量化配置 | [beauty_rkmeans_qwen_20261010_012253/quantizer_train/.hydra/config.yaml](https://github.com/woobowen/recommend/blob/563dc50b1c7e9e2e05cdfddd13e91a3d1888001e/GRID%20baseline/experiments/beauty_rkmeans_qwen_20261010_012253/quantizer_train/.hydra/config.yaml) |
| Toys 实际量化配置 | [toys_rkmeans_qwen_20261010_093036/quantizer_train/.hydra/config.yaml](https://github.com/woobowen/recommend/blob/563dc50b1c7e9e2e05cdfddd13e91a3d1888001e/GRID%20baseline/experiments/toys_rkmeans_qwen_20261010_093036/quantizer_train/.hydra/config.yaml) |
| Beauty 商品文本 | [GRID baseline/original_assets/data/amazon_data/beauty/items/](https://github.com/woobowen/recommend/tree/563dc50b1c7e9e2e05cdfddd13e91a3d1888001e/GRID%20baseline/original_assets/data/amazon_data/beauty/items) |
| Toys 商品文本 | [GRID baseline/original_assets/data/amazon_data/toys/items/](https://github.com/woobowen/recommend/tree/563dc50b1c7e9e2e05cdfddd13e91a3d1888001e/GRID%20baseline/original_assets/data/amazon_data/toys/items) |
| Qwen 权重与恢复工具 | [GRID baseline/models/Qwen2.5-7B-Instruct/](https://github.com/woobowen/recommend/tree/563dc50b1c7e9e2e05cdfddd13e91a3d1888001e/GRID%20baseline/models/Qwen2.5-7B-Instruct)、[codex/restore_qwen_weights.py](https://github.com/woobowen/recommend/blob/563dc50b1c7e9e2e05cdfddd13e91a3d1888001e/GRID%20baseline/codex/restore_qwen_weights.py) |

SID原张量是 `[4,N]`，转置后按显式商品ID对应，不根据商品文本排序猜测ID。这里的Beauty/Toys商品库存沿用已有GRID TFRecord，未重新指定为Amazon2018某一版本。

## 本分支专用配置

- [分类原配置](../archive/classification_beauty_v1/config.json)、[Toys版本配置](../archive/classification_toys_v7/config.json)。其中 `status=PREPARATION_ONLY...` 是保留的历史配置字段，不代表实验未完成；最终类别/计数和评估见 `results/classification/`。
- [Beauty/Toys编码配置](../results/llm2vec_beauty_toys/encoding_config.json)、[统计配置](../results/llm2vec_beauty_toys/analysis_config.json)、[launch源码哈希](../results/llm2vec_beauty_toys/launch.json)。
- [Instruments Qwen与量化配置](../results/llm2vec_instruments/qwen_config.json)、[LLM2Vec编码配置](../results/llm2vec_instruments/encoding_config.json)、[统计配置](../results/llm2vec_instruments/analysis_config.json)、[launch及共用上游源码哈希](../results/llm2vec_instruments/launch.json)。

服务器原项目根目录 `/home/jcy/GRID_wmh_Qwen_TIGER_20261009` 对应 GitHub 的 `GRID baseline/`；不是指仓库最外层目录。历史 `codex/llm2vec_sid_hierarchy_20261113` 对应本分支 `semantic-hierarchy/archive/llm2vec_beauty_toys`，Instruments同理。路径迁移方式见复现指南。

## 外部模型和数据

- [LLM2Vec官方代码](https://github.com/McGill-NLP/llm2vec)，安装版本0.2.3。
- [Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)（不同于早期分类用的3.1）。
- [MNTP适配器](https://huggingface.co/McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp/tree/31474e395ada192e8ed1586db6be79fb3b70c9c0)。
- [supervised适配器](https://huggingface.co/McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised/tree/baa8ebf04a1c2500e61288e7dad65e8ae42601a7)。
- [Amazon Reviews 2018官方页面](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/)。Instruments下载文件名为 `Musical_Instruments_5.json.gz` 和 `meta_Musical_Instruments.json.gz`，来源和传输SHA256见[下载清单](../results/llm2vec_instruments/download_manifest.json)。

LLM2Vec/Llama权重不在本次新增文件内；[模型核验清单](../provenance/llm2vec/model_download_verified.json)提供各文件SHA256。已下载但未运行的unsup-SimCSE不作为完成实验。

## 模型角色与历史部署补件

[模型、tokenizer与readiness原始收据](../delivery/README.md)区分分类用Meta-Llama-3.1-8B-Instruct与独立编码用Meta-Llama-3-8B-Instruct+MNTP+supervised。本轮补齐完整文件hash和部署收据，未将Llama3.1替换为Llama3，未重新下载/运行大模型。Meta原站受其许可及访问授权限制；原清单中的ModelScope镜像应按其可用权限取得，并按全文件SHA256核对，不能只匹配模型名。
