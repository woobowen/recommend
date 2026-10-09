#!/bin/bash

# SASRec训练脚本
# 使用方法: ./train_sasrec.sh [数据集名称] [GPU设备ID]

set -e

# 默认参数
DATASET=${1:-"amazon_toys"}
GPU_ID=${2:-"0"}
NUM_HIERARCHIES=${3:-"4"}

# 设置环境变量
export CUDA_VISIBLE_DEVICES=$GPU_ID

# 项目根目录
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ORIGINAL_ROOT="$(cd "$PROJECT_ROOT/.." && pwd)"

echo "SASRec训练脚本"
echo "数据集: $DATASET"
echo "GPU设备: $GPU_ID"
echo "层次数: $NUM_HIERARCHIES"
echo "项目根目录: $PROJECT_ROOT"
echo "原始项目根目录: $ORIGINAL_ROOT"

# 切换到SASRec项目目录
cd "$PROJECT_ROOT"

# 设置数据路径（与原始项目保持一致）
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

# 运行训练
python src/train.py \
    experiment=sasrec_train_flat \
    data_dir="$DATA_DIR" \
    semantic_id_path="$SEMANTIC_ID_PATH" \
    num_hierarchies=$NUM_HIERARCHIES \
    trainer.devices=1 \
    trainer.accelerator=gpu \
    ++trainer.max_epochs=50 \
    ++trainer.val_check_interval=2000 \
    ++trainer.log_every_n_steps=500 \
    ++data_loading.train_dataloader_config.dataloader.batch_size_per_device=64 \
    ++data_loading.val_dataloader_config.dataloader.batch_size_per_device=32 \
    ++data_loading.test_dataloader_config.dataloader.batch_size_per_device=32 \
    ++model.hidden_size=128 \
    ++model.num_attention_heads=2 \
    ++model.num_hidden_layers=2 \
    ++model.intermediate_size=256 \
    ++optim.optimizer.lr=0.001 \
    ++optim.optimizer.weight_decay=0.0001 \
    tags=["sasrec","$DATASET","baseline"] \
    --config-path="$PROJECT_ROOT/configs" \
    hydra.run.dir="$PROJECT_ROOT/logs/sasrec_${DATASET}_$(date +%Y%m%d_%H%M%S)"

echo "SASRec训练完成!"
