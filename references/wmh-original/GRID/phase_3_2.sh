#!/bin/bash
# 设置 CUDA 设备
export CUDA_VISIBLE_DEVICES=1

log_file="logs/run_3_2_zhengjiao.log"

exec > >(tee "$log_file") 2>&1

export PYTHONPATH=/data/mhwang/Rec/GRID:$PYTHONPATH

python -m src.inference experiment=tiger_inference_flat \
    data_dir=/data/mhwang/Rec/GRID/amazon_data/beauty \
    semantic_id_path=/data/mhwang/Rec/GRID/logs/inference/runs/2025-09-15/20-54-16/pickle/merged_predictions_tensor.pt \
    ckpt_path=/data/mhwang/Rec/GRID/logs/train/runs/2025-09-15/21-13-31/checkpoints/checkpoint_epoch_000_step_001700.ckpt \
    num_hierarchies=4