#!/bin/bash
# 设置 CUDA 设备
export CUDA_VISIBLE_DEVICES=0

log_file="logs_beauty_qwen_rqkmeans_duibi/run_3_1_mask.log"

exec > >(tee "$log_file") 2>&1

export PYTHONPATH=/data/mhwang/Rec/GRID:$PYTHONPATH

python -m src.train experiment=tiger_train_flat \
    data_dir=/data/mhwang/Rec/GRID/amazon_data/beauty \
    semantic_id_path=/data/mhwang/Rec/GRID/logs/inference/runs/2025-09-15/20-54-16/pickle/merged_predictions_tensor.pt \
    num_hierarchies=4