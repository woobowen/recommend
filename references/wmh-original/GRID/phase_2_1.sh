#!/bin/bash
# 设置 CUDA 设备
export CUDA_VISIBLE_DEVICES=0,1,2,3

log_file="logs_beauty_qwen_rqkmeans_duibi/run2_1_2_hierarchies.log"

exec > >(tee "$log_file") 2>&1

export PYTHONPATH=/data/mhwang/Rec/GRID:$PYTHONPATH
python -m src.train experiment=rkmeans_train_flat  \
    data_dir=/data/mhwang/Rec/GRID/amazon_data/beauty \
    embedding_path=/data/mhwang/Rec/GRID/logs/inference/runs/2025-11-12/23-14-12/pickle/merged_predictions_tensor.pt \
    embedding_dim=3584 \
    num_hierarchies=1 \
    codebook_width=256 \