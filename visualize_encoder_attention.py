#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
可视化Encoder Self-Attention（去除padding）

使用方法: python3 visualize_encoder_attention.py
"""

import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path


def detect_real_sequence_length(attn_matrix, threshold=1e-6):
    """
    检测真实序列长度（去除padding）
    
    Padding位置的特征：
    - 该行的attention权重总和接近0
    - 该列被关注的权重总和也接近0
    """
    row_sums = attn_matrix.sum(axis=1)
    col_sums = attn_matrix.sum(axis=0)
    
    valid_rows = row_sums > threshold
    valid_cols = col_sums > threshold
    
    if valid_rows.any():
        last_valid_row = np.where(valid_rows)[0][-1] + 1
    else:
        last_valid_row = len(row_sums)
    
    if valid_cols.any():
        last_valid_col = np.where(valid_cols)[0][-1] + 1
    else:
        last_valid_col = len(col_sums)
    
    real_length = min(last_valid_row, last_valid_col)
    
    return real_length


def visualize_encoder_attention(encoder_attn, save_path, sample_idx=0, layer_idx=-1):
    """
    可视化Encoder Self-Attention矩阵（去除padding）
    
    参数:
        encoder_attn: encoder attention权重 (tuple of tensors)
        save_path: 保存路径
        sample_idx: 样本索引
        layer_idx: 层索引，-1表示最后一层
    """
    num_layers = len(encoder_attn)
    if layer_idx == -1 or layer_idx >= num_layers:
        layer_idx = num_layers - 1
    
    # shape: [batch_size, num_heads, seq_len, seq_len]
    # 平均所有heads
    attn_full = encoder_attn[layer_idx][sample_idx].mean(dim=0).cpu().numpy()
    
    # 检测真实序列长度
    real_length = detect_real_sequence_length(attn_full)
    
    # 只保留有效部分
    attn_matrix = attn_full[:real_length, :real_length]
    
    print(f"\n{'='*70}")
    print(f"Encoder Self-Attention Visualization")
    print(f"{'='*70}")
    print(f"Sample {sample_idx}, Layer {layer_idx}")
    print(f"Full sequence length: {attn_full.shape[0]}")
    print(f"Real sequence length (without padding): {real_length}")
    print(f"Padding removed: {attn_full.shape[0] - real_length} positions")
    print(f"Attention matrix shape: {attn_matrix.shape}")
    
    # 创建图形
    fig, axes = plt.subplots(1, 2, figsize=(18, 8), 
                            gridspec_kw={'width_ratios': [1, 1], 'wspace': 0.3})
    
    # 左侧：带padding的完整矩阵（只显示一部分）
    ax_full = axes[0]
    
    # 只显示前100个位置，避免太大
    display_len = min(100, attn_full.shape[0])
    attn_display = attn_full[:display_len, :display_len]
    
    im_full = ax_full.imshow(attn_display, cmap='YlOrRd', aspect='auto', 
                            interpolation='nearest')
    
    # 标注padding边界
    if real_length < display_len:
        # 画矩形框标注有效区域
        from matplotlib.patches import Rectangle
        rect = Rectangle((0, 0), real_length-0.5, real_length-0.5, 
                         linewidth=3, edgecolor='blue', facecolor='none',
                         label='Valid Sequence')
        ax_full.add_patch(rect)
        
        # 添加分隔线
        ax_full.axhline(y=real_length-0.5, color='red', linestyle='--', 
                       linewidth=2, alpha=0.7, label='Padding Boundary')
        ax_full.axvline(x=real_length-0.5, color='red', linestyle='--', 
                       linewidth=2, alpha=0.7)
    
    ax_full.set_title(f'Full Attention Matrix (with Padding)\nDisplaying first {display_len} positions', 
                     fontsize=13, weight='bold', pad=15)
    ax_full.set_xlabel('Key Position', fontsize=11)
    ax_full.set_ylabel('Query Position', fontsize=11)
    
    # 设置刻度
    tick_step = max(1, display_len // 10)
    ticks = np.arange(0, display_len, tick_step)
    ax_full.set_xticks(ticks)
    ax_full.set_xticklabels(ticks, fontsize=10)
    ax_full.set_yticks(ticks)
    ax_full.set_yticklabels(ticks, fontsize=10)
    
    ax_full.legend(loc='upper right', fontsize=10)
    
    cbar_full = plt.colorbar(im_full, ax=ax_full, shrink=0.8)
    cbar_full.set_label('Attention Weight', fontsize=11)
    
    # 右侧：去除padding的有效矩阵
    ax_valid = axes[1]
    
    im_valid = ax_valid.imshow(attn_matrix, cmap='YlOrRd', aspect='auto', 
                              interpolation='nearest')
    
    ax_valid.set_title(f'Valid Attention Matrix (Padding Removed)\nReal sequence length: {real_length}', 
                      fontsize=13, weight='bold', pad=15)
    ax_valid.set_xlabel('Key Position', fontsize=11)
    ax_valid.set_ylabel('Query Position', fontsize=11)
    
    # 设置刻度
    tick_step = max(1, real_length // 10)
    ticks = np.arange(0, real_length, tick_step)
    ax_valid.set_xticks(ticks)
    ax_valid.set_xticklabels(ticks, fontsize=10)
    ax_valid.set_yticks(ticks)
    ax_valid.set_yticklabels(ticks, fontsize=10)
    
    # 添加网格
    ax_valid.grid(which='minor', color='gray', linestyle='-', linewidth=0.5, alpha=0.2)
    
    cbar_valid = plt.colorbar(im_valid, ax=ax_valid, shrink=0.8)
    cbar_valid.set_label('Attention Weight', fontsize=11)
    
    # 总标题
    plt.suptitle(f'Encoder Self-Attention Analysis (Sample {sample_idx}, Layer {layer_idx})', 
                 fontsize=15, weight='bold', y=0.98)
    
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"✓ Saved to: {save_path}\n")
    plt.close()


def visualize_all_layers(encoder_attn, save_dir, sample_idx=0):
    """
    可视化所有encoder层的attention（只显示有效部分）
    """
    num_layers = len(encoder_attn)
    
    # 为每一层创建一个可视化
    for layer_idx in range(num_layers):
        # 获取这一层的attention
        attn_full = encoder_attn[layer_idx][sample_idx].mean(dim=0).cpu().numpy()
        real_length = detect_real_sequence_length(attn_full)
        attn_matrix = attn_full[:real_length, :real_length]
        
        # 创建单独的图
        fig, ax = plt.subplots(1, 1, figsize=(10, 9))
        
        im = ax.imshow(attn_matrix, cmap='YlOrRd', aspect='auto', 
                      interpolation='nearest')
        
        ax.set_title(f'Encoder Layer {layer_idx} Self-Attention\n(Valid sequence: {real_length} positions)', 
                    fontsize=14, weight='bold', pad=15)
        ax.set_xlabel('Key Position', fontsize=12)
        ax.set_ylabel('Query Position', fontsize=12)
        
        # 设置刻度
        tick_step = max(1, real_length // 10)
        ticks = np.arange(0, real_length, tick_step)
        ax.set_xticks(ticks)
        ax.set_xticklabels(ticks, fontsize=10)
        ax.set_yticks(ticks)
        ax.set_yticklabels(ticks, fontsize=10)
        
        cbar = plt.colorbar(im, ax=ax, shrink=0.9)
        cbar.set_label('Attention Weight', fontsize=11)
        
        save_path = save_dir / f"encoder_layer{layer_idx}_sample{sample_idx}.png"
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"✓ Layer {layer_idx} saved to: {save_path}")
        plt.close()


def main():
    """主函数"""
    attention_dir = Path("./attention_weights")
    
    # 查找文件
    encoder_files = sorted(attention_dir.glob("encoder_attention_batch_*.pt"))
    
    if not encoder_files:
        print("❌ Encoder attention files not found")
        print("Please run inference first to generate attention weights")
        return
    
    # 创建输出目录
    output_dir = Path("./attention_visualizations_encoder")
    output_dir.mkdir(exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"Encoder Self-Attention Visualization")
    print(f"{'='*70}\n")
    print(f"Found {len(encoder_files)} encoder attention files")
    
    # 可视化前几个batch
    num_batches = min(5, len(encoder_files))
    
    for batch_idx in range(num_batches):
        print(f"\n--- Batch {batch_idx} ---")
        
        # 加载数据
        encoder_attn = torch.load(encoder_files[batch_idx])
        
        # 1. 对比图（完整 vs 去除padding）
        save_path = output_dir / f"encoder_attention_comparison_batch{batch_idx}.png"
        visualize_encoder_attention(encoder_attn, str(save_path), sample_idx=0, layer_idx=-1)
        
        # 2. 所有层的单独可视化（只显示有效部分）
        print(f"  Visualizing all {len(encoder_attn)} layers...")
        layer_dir = output_dir / f"batch{batch_idx}_all_layers"
        layer_dir.mkdir(exist_ok=True)
        visualize_all_layers(encoder_attn, layer_dir, sample_idx=0)
    
    print(f"\n{'='*70}")
    print(f"✓ Visualization Complete!")
    print(f"{'='*70}\n")
    
    print("💡 Interpretation Guide:")
    print("━" * 70)
    print("📊 Encoder Self-Attention Matrix:")
    print()
    print("Structure:")
    print("  • Square matrix: [seq_len, seq_len]")
    print("  • Bidirectional: Each position can attend to all positions")
    print("  • No causal mask: Different from decoder's lower-triangular pattern")
    print()
    print("What to Look For:")
    print("  • Diagonal dominance: Positions attend mainly to themselves")
    print("  • Off-diagonal patterns: Long-range dependencies")
    print("  • Column patterns: Some positions are attended by many (important items)")
    print("  • Block patterns: Groups of related items attending to each other")
    print()
    print("Common Patterns:")
    print("  • Local attention: Band around diagonal (like convolution)")
    print("  • Global attention: Specific positions attended by all")
    print("  • Symmetric vs Asymmetric: A[i,j] usually ≠ A[j,i]")
    print()
    print("━" * 70)
    print(f"Results saved to: {output_dir}")
    print()
    print("Files generated:")
    print(f"  • encoder_attention_comparison_batchX.png - Side-by-side comparison")
    print(f"  • batchX_all_layers/encoder_layerY_sampleZ.png - Individual layer heatmaps")


if __name__ == "__main__":
    main()

