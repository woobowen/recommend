#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
可视化完整的decoder attention矩阵
包括：
1. Decoder self-attention (生成的tokens之间的关注)
2. Cross-attention (生成的tokens对encoder输入的关注)

合并成一个大矩阵: [num_hierarchies, num_hierarchies + encoder_seq_len]

使用方法: python3 visualize_complete_decoder_attention.py
"""

import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

def build_complete_decoder_attention(decoder_self_attns, cross_attns, sample_idx=0, top_k=10):
    """
    构建完整的decoder attention矩阵
    
    参数:
        decoder_self_attns: list of tuples, 每个hierarchy的decoder self-attention
        cross_attns: list of tuples, 每个hierarchy的cross-attention
        sample_idx: 样本索引
        top_k: beam search的k值
        
    返回:
        complete_attn: [num_hierarchies, num_hierarchies + encoder_seq_len]
            前num_hierarchies列是decoder self-attention
            后encoder_seq_len列是cross-attention
    """
    num_hierarchies = len(cross_attns)
    
    print(f"\n{'='*70}")
    print(f"Building Complete Decoder Attention Matrix")
    print(f"{'='*70}\n")
    
    # 1. 处理decoder self-attention
    # 每个hierarchy的self-attention会关注之前已生成的所有tokens
    decoder_self_attn_list = []
    
    for hier_idx in range(num_hierarchies):
        if decoder_self_attns[hier_idx] is not None and len(decoder_self_attns[hier_idx]) > 0:
            # 取最后一层
            last_layer_self_attn = decoder_self_attns[hier_idx][-1]
            
            # shape可能是: [batch_size * top_k, num_heads, current_seq_len, total_seq_len]
            # current_seq_len通常是1（只生成1个新token）
            # total_seq_len是之前已生成的tokens数量（包括当前的）
            
            print(f"Hierarchy {hier_idx} decoder self-attention shape: {last_layer_self_attn.shape}")
            
            # 处理beam search扩展
            current_batch = last_layer_self_attn.size(0)
            if sample_idx == 0:
                original_batch_size = 32  # 假设原始batch size
                if current_batch > original_batch_size:
                    # 取第一个样本的第一个beam
                    last_layer_self_attn = last_layer_self_attn[0:1]
                elif current_batch > 1:
                    last_layer_self_attn = last_layer_self_attn[sample_idx:sample_idx+1]
            
            # 平均所有heads: [1, current_seq_len, total_seq_len]
            last_layer_self_attn = last_layer_self_attn.mean(dim=1)
            
            # 取最后一个生成位置的attention (对之前所有positions): [total_seq_len]
            if last_layer_self_attn.size(1) > 0:
                self_attn_vec = last_layer_self_attn[0, -1, :].cpu().numpy()
            else:
                self_attn_vec = np.array([])
            
            print(f"  → Self-attention vector shape: {self_attn_vec.shape}")
            decoder_self_attn_list.append(self_attn_vec)
        else:
            decoder_self_attn_list.append(np.array([]))
    
    # 2. 处理cross-attention (之前的代码已经处理过)
    cross_attn_list = []
    
    for hier_idx in range(num_hierarchies):
        if cross_attns[hier_idx] is not None and len(cross_attns[hier_idx]) > 0:
            last_layer_cross_attn = cross_attns[hier_idx][-1]
            
            print(f"Hierarchy {hier_idx} cross-attention shape: {last_layer_cross_attn.shape}")
            
            # 处理beam search
            current_batch = last_layer_cross_attn.size(0)
            if sample_idx == 0:
                original_batch_size = 32
                if current_batch > original_batch_size:
                    indices = torch.arange(0, current_batch, top_k, device=last_layer_cross_attn.device)
                    last_layer_cross_attn = last_layer_cross_attn[indices[0:1]]
                elif current_batch > 1:
                    last_layer_cross_attn = last_layer_cross_attn[sample_idx:sample_idx+1]
            
            # 平均所有heads
            last_layer_cross_attn = last_layer_cross_attn.mean(dim=1)
            
            # 取最后一个decoder位置: [encoder_seq_len]
            if last_layer_cross_attn.size(1) > 0:
                cross_attn_vec = last_layer_cross_attn[0, -1, :].cpu().numpy()
            else:
                cross_attn_vec = last_layer_cross_attn[0, 0, :].cpu().numpy()
            
            print(f"  → Cross-attention vector shape: {cross_attn_vec.shape}")
            cross_attn_list.append(cross_attn_vec)
        else:
            cross_attn_list.append(np.array([]))
    
    # 3. 构建完整矩阵
    # 需要处理不同hierarchy的self-attention长度不同的问题
    # 生成H0时: query在BOS位置，关注[BOS]
    # 生成H1时: query在H0位置，关注[BOS, H0]
    # 生成H2时: query在H1位置，关注[BOS, H0, H1]
    # 生成H3时: query在H2位置，关注[BOS, H0, H1, H2]
    
    # 填充成方形的self-attention部分 (下三角矩阵，包含BOS列)
    # 第一列是BOS，后面num_hierarchies-1列是H0到H(n-2)
    # 因为生成Hn时，query在H(n-1)位置，最多关注到H(n-2)
    # 例如4个hierarchies时，矩阵是4x4: [BOS, H0, H1, H2]
    decoder_self_matrix = np.zeros((num_hierarchies, num_hierarchies))
    
    for hier_idx in range(num_hierarchies):
        self_attn = decoder_self_attn_list[hier_idx]
        if len(self_attn) > 0:
            # 生成Hi时，query位置是H(i-1)（或i=0时是BOS）
            # 可以关注: BOS + H0 + H1 + ... + H(i-1)
            # 总共是 i+1 个tokens: [BOS, H0, ..., H(i-1)]
            # 例如：生成H0时query在BOS位置，关注[BOS]，长度1
            #      生成H1时query在H0位置，关注[BOS,H0]，长度2
            #      生成H3时query在H2位置，关注[BOS,H0,H1,H2]，长度4
            actual_len = min(len(self_attn), hier_idx + 1)
            decoder_self_matrix[hier_idx, :actual_len] = self_attn[:actual_len]
    
    # 拼接cross-attention
    cross_matrix = np.stack(cross_attn_list, axis=0)  # [num_hierarchies, encoder_seq_len]
    
    # 完整矩阵 - 顺序改为：先encoder输入，后已生成的tokens（符合时间顺序）
    complete_matrix = np.concatenate([cross_matrix, decoder_self_matrix], axis=1)
    
    print(f"\n✓ Complete matrix built successfully:")
    print(f"  - Decoder self-attention: {decoder_self_matrix.shape}")
    print(f"  - Cross-attention: {cross_matrix.shape}")
    print(f"  - Complete matrix: {complete_matrix.shape}")
    
    return complete_matrix, decoder_self_matrix, cross_matrix


def visualize_complete_decoder_attention(complete_matrix, decoder_self_matrix, cross_matrix, 
                                          save_path, sample_idx=0):
    """
    可视化完整的decoder attention - 简化版
    只显示两个关键热力图：Self-Attention 和 Cross-Attention
    """
    num_hierarchies = complete_matrix.shape[0]
    encoder_seq_len = cross_matrix.shape[1]
    
    # 创建图形 - 上下排列两个热力图
    fig, axes = plt.subplots(2, 1, figsize=(18, 10), 
                            gridspec_kw={'height_ratios': [1, 1.5], 'hspace': 0.3})
    
    # 上方：Self-Attention 热力图 (4x4，包含BOS)
    ax_self = axes[0]
    
    im_self = ax_self.imshow(decoder_self_matrix, cmap='Purples', aspect='equal',
                            interpolation='nearest')
    ax_self.set_title('Decoder Self-Attention Matrix\n(BOS Token + Previously Generated Hierarchies)', 
                     fontsize=14, weight='bold', pad=15)
    ax_self.set_xlabel('Key Position (What is Attended To)', fontsize=12)
    ax_self.set_ylabel('Query Position (When Generating)', fontsize=12)
    ax_self.set_xticks(range(num_hierarchies))
    # 列标签：BOS, H0, H1, H2 (生成H3时最多关注到H2)
    ax_self.set_xticklabels(['BOS'] + [f'H{i}' for i in range(num_hierarchies-1)], fontsize=11)
    ax_self.set_yticks(range(num_hierarchies))
    ax_self.set_yticklabels([f'Gen H{i}' for i in range(num_hierarchies)], fontsize=11)
    
    # 添加数值标注
    for i in range(num_hierarchies):
        for j in range(i+1):  # 生成Hi时关注[BOS, H0, ..., H(i-1)]，共i+1个tokens
            val = decoder_self_matrix[i, j]
            color = 'white' if val > 0.5 else 'black'
            ax_self.text(j, i, f'{val:.3f}', ha="center", va="center", 
                        color=color, fontsize=10, weight='bold')
    
    # 添加网格
    ax_self.set_xticks(np.arange(num_hierarchies+1)-0.5, minor=True)
    ax_self.set_yticks(np.arange(num_hierarchies+1)-0.5, minor=True)
    ax_self.grid(which='minor', color='gray', linestyle='-', linewidth=1, alpha=0.3)
    
    cbar_self = plt.colorbar(im_self, ax=ax_self, shrink=0.8, pad=0.02)
    cbar_self.set_label('Self-Attention Weight', fontsize=11)
    
    # 下方：Cross-Attention 热力图 (4x150)
    ax_cross = axes[1]
    
    im_cross = ax_cross.imshow(cross_matrix, cmap='YlOrRd', aspect='auto',
                               interpolation='nearest')
    ax_cross.set_title('Cross-Attention Matrix\n(Generated Hierarchies → Encoder Input Sequence)', 
                      fontsize=14, weight='bold', pad=15)
    ax_cross.set_xlabel('Encoder Input Position', fontsize=12)
    ax_cross.set_ylabel('Generated Hierarchy', fontsize=12)
    ax_cross.set_yticks(range(num_hierarchies))
    ax_cross.set_yticklabels([f'H{i}' for i in range(num_hierarchies)], fontsize=11)
    
    # X轴刻度
    xtick_step = max(1, encoder_seq_len // 15)
    xticks = np.arange(0, encoder_seq_len, xtick_step)
    ax_cross.set_xticks(xticks)
    ax_cross.set_xticklabels(xticks, fontsize=10)
    
    # 添加水平网格线
    ax_cross.set_yticks(np.arange(num_hierarchies+1)-0.5, minor=True)
    ax_cross.grid(which='minor', color='gray', linestyle='-', linewidth=1, alpha=0.3, axis='y')
    
    cbar_cross = plt.colorbar(im_cross, ax=ax_cross, shrink=0.8, pad=0.02)
    cbar_cross.set_label('Cross-Attention Weight', fontsize=11)
    
    # 总标题
    plt.suptitle(f'Decoder Attention Analysis (Sample {sample_idx})', 
                 fontsize=16, weight='bold', y=0.98)
    
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"✓ Saved to: {save_path}")  
    plt.close()


def main():
    """主函数"""
    attention_dir = Path("./attention_weights")
    
    # 查找文件
    decoder_self_files = sorted(attention_dir.glob("decoder_self_attention_batch_*.pt"))
    cross_files = sorted(attention_dir.glob("cross_attention_batch_*.pt"))
    
    if not decoder_self_files:
        print("❌ Decoder self-attention files not found")
        print("Please re-run inference to generate new attention weights")
        return
    
    if not cross_files:
        print("❌ Cross-attention files not found")
        return
    
    # 创建输出目录
    output_dir = Path("./attention_visualizations_complete")
    output_dir.mkdir(exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"Complete Decoder Attention Visualization")
    print(f"{'='*70}\n")
    print(f"Found {len(decoder_self_files)} decoder self-attention files")
    print(f"Found {len(cross_files)} cross-attention files")
    
    # 可视化前几个batch
    num_batches = min(10, len(decoder_self_files), len(cross_files))
    
    for batch_idx in range(num_batches):
        print(f"\n--- Batch {batch_idx} ---")
        
        # 加载数据
        decoder_self_attns = torch.load(decoder_self_files[batch_idx])
        cross_attns = torch.load(cross_files[batch_idx])
        
        # 构建完整矩阵
        complete_matrix, decoder_self_matrix, cross_matrix = build_complete_decoder_attention(
            decoder_self_attns, cross_attns, sample_idx=0
        )
        
        # 可视化
        save_path = output_dir / f"complete_decoder_attention_batch{batch_idx}.png"
        visualize_complete_decoder_attention(
            complete_matrix, decoder_self_matrix, cross_matrix,
            str(save_path), sample_idx=0
        )
    
    print(f"\n{'='*70}")
    print(f"✓ Visualization Complete!")
    print(f"{'='*70}\n")
    
    print("💡 Interpretation Guide:")
    print("━" * 70)
    print("📊 Simplified Visualization - Two Key Heatmaps:")
    print()
    print("Top: Decoder Self-Attention Matrix (4×4) - 包含BOS Token")
    print("  • Purple heatmap with numerical values")
    print("  • Lower triangular matrix (causal attention)")
    print("  • Rows: 生成哪个hierarchy (Gen H0, Gen H1, Gen H2, Gen H3)")
    print("  • Columns: Key位置 [BOS, H0, H1, H2]")
    print("  • 重要理解：生成Hi时，query位置在H(i-1)，只能关注之前的tokens")
    print("  • 例如：")
    print("    - 生成H0: query在BOS位置 → 关注[BOS]")
    print("    - 生成H1: query在H0位置 → 关注[BOS, H0]")
    print("    - 生成H2: query在H1位置 → 关注[BOS, H0, H1]")
    print("    - 生成H3: query在H2位置 → 关注[BOS, H0, H1, H2]")
    print()
    print("Bottom: Cross-Attention Matrix (4×150)")
    print("  • Yellow-Red heatmap")
    print("  • Shows: Which encoder input positions each hierarchy attends to")
    print("  • Each row = one hierarchy's attention distribution over input sequence")
    print()
    print("━" * 70)
    print("🔍 What to Look For:")
    print()
    print("Self-Attention (Top):")
    print("  • BOS column (第0列): 初始prompt的重要性 - 每个hierarchy对BOS的依赖")
    print("  • 第1行第1列 ≈ 1.0: 生成H1时，H0位置主要关注自己")
    print("  • Off-diagonal: 后面hierarchies对前面的依赖关系")
    print("  • Pattern: 生成H3时，是更依赖BOS(初始化)还是H0(粗粒度层级)?")
    print()
    print("Cross-Attention (Bottom):")
    print("  • Uniform distribution: Hierarchy关注整个输入序列")
    print("  • Sparse peaks: Hierarchy聚焦于特定输入位置")
    print("  • Compare rows: 不同hierarchies是否关注输入的不同区域?")
    print()
    print("━" * 70)
    print(f"Saved to: {output_dir}")


if __name__ == "__main__":
    main()

