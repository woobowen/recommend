# SASRec Baseline Implementation

这是一个与生成式推荐系统完全兼容的SASRec基线实现，确保数据处理、数据集划分和评测方式完全一致，同时保证item和user的下标映射完全对应。

## 项目结构

```
sasrec_baseline/
├── src/
│   ├── models/
│   │   └── modules/
│   │       └── sasrec/
│   │           └── sasrec_model.py      # SASRec模型实现
│   ├── train.py                         # 训练脚本
│   ├── inference.py                     # 推理脚本
│   └── test.py                         # 测试脚本
├── configs/
│   ├── experiment/
│   │   ├── sasrec_train_flat.yaml      # 训练配置
│   │   └── sasrec_inference_flat.yaml  # 推理配置
│   ├── train.yaml                      # 主训练配置
│   └── inference.yaml                  # 主推理配置
├── scripts/
│   ├── train_sasrec.sh                 # 训练脚本
│   └── inference_sasrec.sh             # 推理脚本
└── README.md
```

## 主要特性

### 1. 完全兼容的数据处理
- 使用与生成式推荐系统相同的数据加载器和预处理管道
- 保持相同的数据集划分（训练/验证/测试）
- 支持相同的TFRecord数据格式

### 2. 一致的item/user映射
- item和user的下标映射与生成式推荐系统完全一致
- 支持从semantic_id_map自动推断item数量
- 确保后续可以使用SASRec预训练的item embedding

### 3. 相同的评测方式
- 使用相同的评测指标（NDCG、Recall、Hit Rate等）
- 支持相同的top-k评测设置
- 保持相同的评测流程和输出格式

### 4. 模型特性
- 标准的SASRec架构（Self-Attention Sequential Recommendation）
- 支持可配置的模型参数（隐藏层维度、注意力头数等）
- 包含item embedding相似度分析功能
- 支持用户embedding（可选）

## 使用方法

### 训练SASRec模型

```bash
cd sasrec_baseline
./scripts/train_sasrec.sh [数据集名称] [GPU设备ID] [层次数]

# 示例
./scripts/train_sasrec.sh amazon_toys 0 4
```

### 推理和生成item embeddings

```bash
./scripts/inference_sasrec.sh [数据集名称] [模型检查点路径] [GPU设备ID] [层次数]

# 示例
./scripts/inference_sasrec.sh amazon_toys /path/to/checkpoint.ckpt 0 4
```

### 直接使用Python脚本

```bash
# 训练
python src/train.py experiment=sasrec_train_flat \
    data_dir=/path/to/data \
    semantic_id_path=/path/to/semantic_ids.pt \
    num_hierarchies=4

# 推理
python src/inference.py experiment=sasrec_inference_flat \
    data_dir=/path/to/data \
    semantic_id_path=/path/to/semantic_ids.pt \
    model_checkpoint_path=/path/to/checkpoint.ckpt \
    num_hierarchies=4
```

## 配置说明

### 模型配置
- `hidden_size`: 隐藏层维度（默认128）
- `num_attention_heads`: 注意力头数（默认2）
- `num_hidden_layers`: Transformer层数（默认2）
- `intermediate_size`: FFN中间层维度（默认256）
- `max_seq_length`: 最大序列长度（默认120）

### 训练配置
- `batch_size_per_device`: 每设备批次大小（默认64）
- `learning_rate`: 学习率（默认0.001）
- `max_epochs`: 最大训练轮数（默认50）
- `patience`: 早停耐心值（默认15）

## 输出文件

### 训练输出
- 模型检查点：`checkpoints/sasrec_checkpoint_*.ckpt`
- 训练日志：`sasrec_csv/`
- 配置文件：`.hydra/`

### 推理输出
- 预测结果：`sasrec_predictions.pt`
- Item embeddings：`sasrec_item_embeddings.pt`（供生成式推荐系统使用）

## 与生成式推荐系统的集成

训练完成后，可以通过以下方式在生成式推荐系统中使用SASRec的item embeddings：

```python
import torch

# 加载SASRec预训练的item embeddings
sasrec_embeddings = torch.load('sasrec_item_embeddings.pt')

# 在生成式推荐模型中使用
# 确保item索引映射完全一致
generative_model.load_pretrained_item_embeddings(sasrec_embeddings)
```

## 注意事项

1. **数据路径**: 确保数据目录结构与原始项目一致
2. **语义ID文件**: 必须提供正确的semantic_ids.pt文件路径
3. **GPU内存**: 根据GPU内存调整batch_size_per_device
4. **检查点**: 推理时需要提供有效的模型检查点路径

## 依赖关系

本项目依赖原始项目的以下组件：
- `src.data.loading`: 数据加载模块
- `src.components.eval_metrics`: 评测指标
- `src.utils`: 工具函数
- `src.models.modules.base_module`: 基础模型类

确保这些模块在Python路径中可访问。
