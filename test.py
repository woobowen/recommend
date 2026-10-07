import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# 定义一个矩阵 A (3x2 矩阵)
A = np.array([
    [1, 2],
    [3, 4],
    [5, 6]
])
B = torch.tensor([[2.,3.], [3., 4.]])
attention_weights = F.softmax(B, dim=-2)
print("attention_weights:\n", attention_weights)

# 执行奇异值分解
# 注意：np.linalg.svd 返回的是 U, s (奇异值的一维数组), Vh (V 的转置)
U, s, Vh = np.linalg.svd(A)

print("--- 奇异值分解 (SVD) ---")
print("原始矩阵 A:\n", A)
print("\n矩阵 U (左奇异向量):\n", U)
print("\n奇异值 s (一维数组):\n", s)
print("\n矩阵 Vh (V 的转置):\n", Vh)

# 将奇异值 s 转换回对角矩阵 Sigma
# 需要根据 A 的形状 (m, n) 创建一个 (m, n) 的对角矩阵
Sigma = np.zeros(A.shape)
# 将奇异值填充到对角线上
Sigma[:A.shape[1], :A.shape[1]] = np.diag(s) 
print
print("\n对角矩阵 Sigma:\n", Sigma.shape)
# 重构原始矩阵 (验证)
# A_reconstructed = U @ Sigma @ Vh
A_reconstructed = U @ Sigma @ Vh
print("\n重构后的矩阵 A_reconstructed:\n", A_reconstructed)

# 检查重构误差（由于浮点数运算，通常会有微小的误差）
print("\n误差 (A - A_reconstructed) 的范数:", np.linalg.norm(A - A_reconstructed))





import numpy as np

# 模拟法（仅作对比，不是您的要求）
N_samples = 1_000_000 # 抽取 100 万个样本
samples = np.random.uniform(0, 1, N_samples)

# 计算样本的均值和方差
simulated_mean = np.mean(samples)
simulated_variance = np.var(samples)

print("\n--- 模拟结果（仅供参考）---")
print(f"模拟均值: {simulated_mean:.4f} (理论值: 0.5000)")
print(f"模拟方差: {simulated_variance:.4f} (理论值: 0.0833)")



