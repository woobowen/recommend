#!/bin/bash

# SASRec推理脚本
# 使用方法: ./inference_sasrec.sh [数据集名称] [模型检查点路径] [GPU设备ID]

set -e

# 默认参数
DATASET=${1:-"amazon_toys"}
CHECKPOINT_PATH=${2:-""}
GPU_ID=${3:-"0"}
NUM_HIERARCHIES=${4:-"4"}

# 检查检查点路径
if [ -z "$CHECKPOINT_PATH" ]; then
    echo "错误: 请提供模型检查点路径"
    echo "使用方法: ./inference_sasrec.sh [数据集名称] [模型检查点路径] [GPU设备ID]"
    exit 1
fi

if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo "错误: 检查点文件不存在: $CHECKPOINT_PATH"
    exit 1
fi

# 设置环境变量
export CUDA_VISIBLE_DEVICES=$GPU_ID

# 项目根目录
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ORIGINAL_ROOT="$(cd "$PROJECT_ROOT/.." && pwd)"

echo "SASRec推理脚本"
echo "数据集: $DATASET"
echo "检查点: $CHECKPOINT_PATH"
echo "GPU设备: $GPU_ID"
echo "层次数: $NUM_HIERARCHIES"

# 切换到SASRec项目目录
cd "$PROJECT_ROOT"

# 设置数据路径
DATA_DIR="$ORIGINAL_ROOT/amazon_data/$DATASET"
SEMANTIC_ID_PATH="$DATA_DIR/semantic_ids.pt"

# 检查数据文件是否存在
if [ ! -d "$DATA_DIR" ]; then
    echo "错误: 数据目录不存在: $DATA_DIR"
    exit 1
fi

if [ ! -f "$SEMANTIC_ID_PATH" ]; then
    echo "错误: 语义ID文件不存在: $SEMANTIC_ID_PATH"
    exit 1
fi

echo "数据目录: $DATA_DIR"
echo "语义ID文件: $SEMANTIC_ID_PATH"

# 运行推理
python src/inference.py \
    experiment=sasrec_inference_flat \
    data_dir="$DATA_DIR" \
    semantic_id_path="$SEMANTIC_ID_PATH" \
    num_hierarchies=$NUM_HIERARCHIES \
    model_checkpoint_path="$CHECKPOINT_PATH" \
    trainer.devices=1 \
    trainer.accelerator=gpu \
    ++data_loading.predict_dataloader_config.dataloader.batch_size_per_device=64 \
    ++model.hidden_size=128 \
    ++model.num_attention_heads=2 \
    ++model.num_hidden_layers=2 \
    ++model.intermediate_size=256 \
    tags=["sasrec","$DATASET","inference"] \
    --config-path="$PROJECT_ROOT/configs" \
    hydra.run.dir="$PROJECT_ROOT/logs/sasrec_inference_${DATASET}_$(date +%Y%m%d_%H%M%S)"

echo "SASRec推理完成!"
echo "预测结果和item embeddings已保存到输出目录"
