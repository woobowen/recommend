#!/bin/bash
# 设置 CUDA 设备
export CUDA_VISIBLE_DEVICES=2

log_file="logs_beauty_qwen_rqkmeans_duibi/run1_1.log"

exec > >(tee "$log_file") 2>&1


python -m src.inference experiment=sem_embeds_inference_flat data_dir=/data/mhwang/Rec/GRID/amazon_data/beauty \
    embedding_model=/data/mhwang/LLM-Research/Qwen2.5-7B-Instruct