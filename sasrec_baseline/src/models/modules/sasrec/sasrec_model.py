import math
import logging
from typing import Any, Optional, Tuple, Union, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.aggregation import BaseAggregator

from src.utils import RankedLogger
from src.data.loading.components.interfaces import (
    SequentialModelInputData,
    SequentialModuleLabelData,
)
from src.models.components.interfaces import OneKeyPerPredictionOutput
from src.models.modules.base_module import BaseModule

command_line_logger = RankedLogger(__name__, rank_zero_only=True)


class SASRecModel(BaseModule):
    """
    SASRec模型实现，与生成式推荐系统保持相同的数据处理和评测方式
    确保item和user的下标映射完全一致
    """

    def __init__(
        self,
        num_items: int,
        num_users: Optional[int] = None,
        hidden_size: int = 64,
        num_attention_heads: int = 2,
        num_hidden_layers: int = 2,
        intermediate_size: int = 256,
        hidden_dropout_prob: float = 0.5,
        attention_probs_dropout_prob: float = 0.5,
        max_seq_length: int = 50,
        layer_norm_eps: float = 1e-12,
        initializer_range: float = 0.02,
        prediction_key_name: str = "user_id",
        prediction_value_name: str = "item_scores",
        **kwargs,
    ) -> None:
        """
        初始化SASRec模型

        Args:
            num_items: item总数（与生成式推荐系统保持一致）
            num_users: user总数（可选，用于user embedding）
            hidden_size: 隐藏层维度
            num_attention_heads: 注意力头数
            num_hidden_layers: Transformer层数
            intermediate_size: FFN中间层维度
            hidden_dropout_prob: 隐藏层dropout概率
            attention_probs_dropout_prob: 注意力dropout概率
            max_seq_length: 最大序列长度
            layer_norm_eps: LayerNorm的epsilon
            initializer_range: 参数初始化范围
        """
        super().__init__(**kwargs)

        self.num_items = num_items
        self.num_users = num_users
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_seq_length = max_seq_length
        self.layer_norm_eps = layer_norm_eps
        self.initializer_range = initializer_range
        
        # 预测输出的key和value名称
        self.prediction_key_name = prediction_key_name
        self.prediction_value_name = prediction_value_name

        # Item embedding（与生成式推荐系统保持一致的索引）
        self.item_embeddings = nn.Embedding(num_items + 1, hidden_size, padding_idx=0)  # +1 for padding
        
        # User embedding（可选）
        self.user_embeddings = (
            nn.Embedding(num_users, hidden_size) if num_users is not None else None
        )
        
        # Position embedding
        self.position_embeddings = nn.Embedding(max_seq_length, hidden_size)
        
        # Layer normalization and dropout
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(hidden_dropout_prob)
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                hidden_size=hidden_size,
                num_attention_heads=num_attention_heads,
                intermediate_size=intermediate_size,
                hidden_dropout_prob=hidden_dropout_prob,
                attention_probs_dropout_prob=attention_probs_dropout_prob,
                layer_norm_eps=layer_norm_eps,
            )
            for _ in range(num_hidden_layers)
        ])
        
        # 初始化权重
        self.apply(self._init_weights)
        
        # 用于累积item embedding相似度矩阵分析
        self.item_embedding_similarity_matrix_sum = None
        self.item_embedding_similarity_matrix_count = 0

    def _init_weights(self, module):
        """初始化模型权重"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        user_id: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        前向传播

        Args:
            input_ids: 输入序列 [batch_size, seq_len]
            attention_mask: 注意力掩码 [batch_size, seq_len]
            user_id: 用户ID [batch_size] (可选)

        Returns:
            序列表示 [batch_size, seq_len, hidden_size]
        """
        batch_size, seq_length = input_ids.size()
        
        # Item embeddings
        item_embeds = self.item_embeddings(input_ids)  # [batch_size, seq_len, hidden_size]
        
        # Position embeddings
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        position_embeds = self.position_embeddings(position_ids)
        
        # 组合embeddings
        embeddings = item_embeds + position_embeds
        
        # 如果有user embedding，添加到序列开头
        if user_id is not None and self.user_embeddings is not None:
            user_embeds = self.user_embeddings(user_id).unsqueeze(1)  # [batch_size, 1, hidden_size]
            embeddings = torch.cat([user_embeds, embeddings], dim=1)
            
            # 相应地扩展attention_mask
            user_mask = torch.ones(batch_size, 1, device=attention_mask.device)
            attention_mask = torch.cat([user_mask, attention_mask], dim=1)
        
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        
        # 创建因果注意力掩码
        seq_len = embeddings.size(1)
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=embeddings.device))
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]
        
        # 结合padding mask和causal mask
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1, seq_len]
        extended_attention_mask = extended_attention_mask * causal_mask
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        
        # 通过Transformer blocks
        hidden_states = embeddings
        for transformer_block in self.transformer_blocks:
            hidden_states = transformer_block(hidden_states, extended_attention_mask)
        
        return hidden_states

    def compute_item_scores(self, sequence_output: torch.Tensor) -> torch.Tensor:
        """
        计算item分数

        Args:
            sequence_output: 序列输出 [batch_size, seq_len, hidden_size]

        Returns:
            item分数 [batch_size, seq_len, num_items]
        """
        # 使用item embedding作为输出层权重
        item_embeddings_weight = self.item_embeddings.weight[1:]  # 排除padding token
        scores = torch.matmul(sequence_output, item_embeddings_weight.transpose(0, 1))
        return scores

    def _compute_item_embedding_similarity(self) -> torch.Tensor:
        """
        计算item embeddings之间的余弦相似度矩阵
        
        Returns:
            相似度矩阵 [num_items, num_items]
        """
        # 获取所有item embeddings（排除padding token）
        item_embeds = self.item_embeddings.weight[1:]  # [num_items, hidden_size]
        
        # L2归一化
        item_embeds_normalized = F.normalize(item_embeds, p=2, dim=1)
        
        # 计算余弦相似度矩阵
        similarity_matrix = torch.mm(item_embeds_normalized, item_embeds_normalized.t())
        
        return similarity_matrix

    def model_step(
        self,
        model_input: SequentialModelInputData,
        label_data: Optional[SequentialModuleLabelData] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        执行一个模型步骤

        Args:
            model_input: 模型输入数据
            label_data: 标签数据（训练时使用）

        Returns:
            模型输出和损失
        """
        # 获取输入数据
        input_ids = None
        user_id = None
        
        # 从transformed_sequences中获取input_ids
        for k, v in model_input.transformed_sequences.items():
            if k == "semantic_ids" or k == "item_ids" or "id" in k.lower():
                input_ids = v
                break
        
        if input_ids is None:
            # 如果没找到，取第一个序列
            input_ids = list(model_input.transformed_sequences.values())[0]
        
        # 获取user_id（如果有）
        if hasattr(model_input, 'user_id_list') and model_input.user_id_list is not None:
            if isinstance(model_input.user_id_list, torch.Tensor):
                user_id = model_input.user_id_list
            elif isinstance(model_input.user_id_list, list):
                user_id = torch.tensor(model_input.user_id_list, device=input_ids.device)
        
        # 前向传播
        sequence_output = self.forward(
            input_ids=input_ids,
            attention_mask=model_input.mask,
            user_id=user_id,
        )
        
        # 如果没有标签数据，返回序列输出用于推理
        if label_data is None:
            return sequence_output, torch.tensor(0.0, device=input_ids.device)
        
        # 计算损失（训练时）
        # 获取标签
        labels = None
        for label_name, label_tensor in label_data.labels.items():
            labels = label_tensor
            break
        
        if labels is None:
            return sequence_output, torch.tensor(0.0, device=input_ids.device)
        
        # 计算item分数
        item_scores = self.compute_item_scores(sequence_output)
        
        # 准备训练目标
        # 对于SASRec，我们预测下一个item
        # 输入: [item1, item2, item3, ...]
        # 目标: [item2, item3, item4, ...]
        
        batch_size, seq_len = input_ids.shape
        
        # 重塑labels以匹配序列长度
        if labels.dim() == 1:
            # 如果labels是1D，假设它是下一个item的ID
            # 创建shifted targets
            targets = torch.zeros_like(input_ids)
            targets[:, :-1] = input_ids[:, 1:]  # shift left
            targets[:, -1] = labels  # 最后一个位置是真实标签
        else:
            # 如果labels已经是序列形式
            targets = labels.reshape(batch_size, -1)
            if targets.size(1) != seq_len:
                # 如果长度不匹配，进行调整
                if targets.size(1) > seq_len:
                    targets = targets[:, :seq_len]
                else:
                    # 用padding填充
                    pad_size = seq_len - targets.size(1)
                    targets = F.pad(targets, (0, pad_size), value=0)
        
        # 计算损失
        loss = 0.0
        if self.loss_function is not None:
            # 只在非padding位置计算损失
            active_loss = model_input.mask.view(-1) == 1
            active_logits = item_scores.view(-1, item_scores.size(-1))[active_loss]
            active_labels = targets.view(-1)[active_loss]
            
            # 排除padding token (0)
            non_padding_mask = active_labels != 0
            if non_padding_mask.sum() > 0:
                final_logits = active_logits[non_padding_mask]
                final_labels = active_labels[non_padding_mask] - 1  # 调整为0-based索引
                loss = self.loss_function(final_logits, final_labels)
            else:
                loss = torch.tensor(0.0, device=input_ids.device, requires_grad=True)
        
        return item_scores, loss

    def generate_recommendations(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        user_id: Optional[torch.Tensor] = None,
        top_k: int = 10,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        生成推荐结果

        Args:
            input_ids: 输入序列
            attention_mask: 注意力掩码
            user_id: 用户ID
            top_k: 返回top-k推荐

        Returns:
            推荐item IDs和对应分数
        """
        with torch.no_grad():
            sequence_output = self.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                user_id=user_id,
            )
            
            # 取最后一个有效位置的输出
            batch_size = input_ids.size(0)
            last_positions = attention_mask.sum(dim=1) - 1  # 最后一个非padding位置
            
            last_hidden_states = []
            for i in range(batch_size):
                last_pos = last_positions[i].item()
                last_hidden_states.append(sequence_output[i, last_pos])
            
            last_hidden_states = torch.stack(last_hidden_states)  # [batch_size, hidden_size]
            
            # 计算与所有item的相似度
            item_embeddings_weight = self.item_embeddings.weight[1:]  # 排除padding token
            scores = torch.matmul(last_hidden_states, item_embeddings_weight.transpose(0, 1))
            
            # 获取top-k
            top_scores, top_indices = torch.topk(scores, k=top_k, dim=1)
            
            # 调整索引（因为我们排除了padding token）
            top_indices = top_indices + 1
            
            return top_indices, top_scores

    def predict_step(self, batch: SequentialModelInputData, batch_idx: int = 0):
        """预测步骤"""
        # 获取输入数据
        input_ids = None
        for k, v in batch.transformed_sequences.items():
            if k == "semantic_ids" or k == "item_ids" or "id" in k.lower():
                input_ids = v
                break
        
        if input_ids is None:
            input_ids = list(batch.transformed_sequences.values())[0]
        
        # 获取user_id
        user_id = None
        if hasattr(batch, 'user_id_list') and batch.user_id_list is not None:
            if isinstance(batch.user_id_list, torch.Tensor):
                user_id = batch.user_id_list
            elif isinstance(batch.user_id_list, list):
                user_id = torch.tensor(batch.user_id_list, device=input_ids.device)
        
        # 生成推荐
        recommended_items, scores = self.generate_recommendations(
            input_ids=input_ids,
            attention_mask=batch.mask,
            user_id=user_id,
            top_k=10,
        )
        
        # 计算item embedding相似度矩阵
        batch_similarity_matrix = self._compute_item_embedding_similarity()
        
        # 累积相似度矩阵
        if self.item_embedding_similarity_matrix_sum is None:
            self.item_embedding_similarity_matrix_sum = batch_similarity_matrix.detach().cpu()
        else:
            self.item_embedding_similarity_matrix_sum += batch_similarity_matrix.detach().cpu()
        self.item_embedding_similarity_matrix_count += 1
        
        # 准备输出
        ids = [
            id.item() if isinstance(id, torch.Tensor) else id
            for id in batch.user_id_list
        ] if batch.user_id_list is not None else list(range(len(recommended_items)))
        
        model_output = OneKeyPerPredictionOutput(
            keys=ids,
            predictions=recommended_items,
            key_name=self.prediction_key_name,
            prediction_name=self.prediction_value_name,
        )
        return model_output

    def eval_step(
        self,
        batch: Tuple[SequentialModelInputData, SequentialModuleLabelData],
        loss_to_aggregate: BaseAggregator,
        batch_idx: int = 0,
        save_attentions: bool = False,
    ):
        """评估步骤"""
        model_input: SequentialModelInputData = batch[0]
        label_data: SequentialModuleLabelData = batch[1]
        
        _, loss = self.model_step(model_input=model_input, label_data=label_data)
        
        # 生成推荐用于评估
        input_ids = None
        for k, v in model_input.transformed_sequences.items():
            if k == "semantic_ids" or k == "item_ids" or "id" in k.lower():
                input_ids = v
                break
        
        if input_ids is None:
            input_ids = list(model_input.transformed_sequences.values())[0]
        
        user_id = None
        if hasattr(model_input, 'user_id_list') and model_input.user_id_list is not None:
            if isinstance(model_input.user_id_list, torch.Tensor):
                user_id = model_input.user_id_list
            elif isinstance(model_input.user_id_list, list):
                user_id = torch.tensor(model_input.user_id_list, device=input_ids.device)
        
        recommended_items, scores = self.generate_recommendations(
            input_ids=input_ids,
            attention_mask=model_input.mask,
            user_id=user_id,
            top_k=10,
        )
        
        # 调用评估器
        self.evaluator(
            marginal_probs=scores,
            generated_ids=recommended_items.unsqueeze(1),  # 添加维度以匹配评估器期望
            labels=list(label_data.labels.values())[0].to(scores.device),
        )
        
        # 计算item embedding相似度矩阵
        batch_similarity_matrix = self._compute_item_embedding_similarity()
        
        # 累积相似度矩阵
        if self.item_embedding_similarity_matrix_sum is None:
            self.item_embedding_similarity_matrix_sum = batch_similarity_matrix.detach().cpu()
        else:
            self.item_embedding_similarity_matrix_sum += batch_similarity_matrix.detach().cpu()
        self.item_embedding_similarity_matrix_count += 1
        
        loss_to_aggregate(loss)

    def on_validation_start(self):
        super().on_validation_start()
        # 重置相似度矩阵累积器
        self.item_embedding_similarity_matrix_sum = None
        self.item_embedding_similarity_matrix_count = 0

    def on_validation_end(self):
        super().on_validation_end()
        # 输出平均相似度矩阵
        if self.item_embedding_similarity_matrix_count > 0:
            avg_similarity = self.item_embedding_similarity_matrix_sum / self.item_embedding_similarity_matrix_count
            command_line_logger.info("\n" + "="*80)
            command_line_logger.info(f"=== SASRec Item Embedding相似度矩阵统计信息 ===")
            command_line_logger.info(f"矩阵形状: {avg_similarity.shape}")
            command_line_logger.info(f"平均相似度: {avg_similarity.mean().item():.6f}")
            command_line_logger.info(f"相似度标准差: {avg_similarity.std().item():.6f}")
            command_line_logger.info(f"最大相似度: {avg_similarity.max().item():.6f}")
            command_line_logger.info(f"最小相似度: {avg_similarity.min().item():.6f}")
            command_line_logger.info("="*80 + "\n")

    def on_test_start(self):
        super().on_test_start()
        # 重置相似度矩阵累积器
        self.item_embedding_similarity_matrix_sum = None
        self.item_embedding_similarity_matrix_count = 0

    def on_test_end(self):
        super().on_test_end()
        # 输出平均相似度矩阵
        if self.item_embedding_similarity_matrix_count > 0:
            avg_similarity = self.item_embedding_similarity_matrix_sum / self.item_embedding_similarity_matrix_count
            command_line_logger.info("\n" + "="*80)
            command_line_logger.info(f"=== SASRec Item Embedding相似度矩阵统计信息 ===")
            command_line_logger.info(f"矩阵形状: {avg_similarity.shape}")
            command_line_logger.info(f"平均相似度: {avg_similarity.mean().item():.6f}")
            command_line_logger.info(f"相似度标准差: {avg_similarity.std().item():.6f}")
            command_line_logger.info(f"最大相似度: {avg_similarity.max().item():.6f}")
            command_line_logger.info(f"最小相似度: {avg_similarity.min().item():.6f}")
            command_line_logger.info("="*80 + "\n")

    def on_predict_start(self):
        super().on_predict_start()
        # 重置相似度矩阵累积器
        self.item_embedding_similarity_matrix_sum = None
        self.item_embedding_similarity_matrix_count = 0

    def on_predict_end(self):
        super().on_predict_end()
        # 输出平均相似度矩阵
        if self.item_embedding_similarity_matrix_count > 0:
            avg_similarity = self.item_embedding_similarity_matrix_sum / self.item_embedding_similarity_matrix_count
            command_line_logger.info("\n" + "="*80)
            command_line_logger.info(f"=== SASRec Item Embedding相似度矩阵统计信息 ===")
            command_line_logger.info(f"矩阵形状: {avg_similarity.shape}")
            command_line_logger.info(f"平均相似度: {avg_similarity.mean().item():.6f}")
            command_line_logger.info(f"相似度标准差: {avg_similarity.std().item():.6f}")
            command_line_logger.info(f"最大相似度: {avg_similarity.max().item():.6f}")
            command_line_logger.info(f"最小相似度: {avg_similarity.min().item():.6f}")
            command_line_logger.info("="*80 + "\n")

    def get_item_embeddings(self) -> torch.Tensor:
        """
        获取item embeddings，供生成式推荐系统使用
        
        Returns:
            item embeddings [num_items, hidden_size]
        """
        return self.item_embeddings.weight[1:]  # 排除padding token


class TransformerBlock(nn.Module):
    """Transformer Block"""
    
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        intermediate_size: int,
        hidden_dropout_prob: float,
        attention_probs_dropout_prob: float,
        layer_norm_eps: float,
    ):
        super().__init__()
        
        self.attention = MultiHeadSelfAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
        )
        
        self.intermediate = nn.Linear(hidden_size, intermediate_size)
        self.output = nn.Linear(intermediate_size, hidden_size)
        
        self.LayerNorm1 = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.LayerNorm2 = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(hidden_dropout_prob)
        
    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # Self-attention
        attention_output = self.attention(hidden_states, attention_mask)
        attention_output = self.dropout(attention_output)
        hidden_states = self.LayerNorm1(hidden_states + attention_output)
        
        # Feed-forward
        intermediate_output = self.intermediate(hidden_states)
        intermediate_output = F.gelu(intermediate_output)
        layer_output = self.output(intermediate_output)
        layer_output = self.dropout(layer_output)
        hidden_states = self.LayerNorm2(hidden_states + layer_output)
        
        return hidden_states


class MultiHeadSelfAttention(nn.Module):
    """Multi-Head Self-Attention"""
    
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        attention_probs_dropout_prob: float,
    ):
        super().__init__()
        
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({hidden_size}) is not a multiple of the number of attention "
                f"heads ({num_attention_heads})"
            )
        
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        
        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)
        
        self.dropout = nn.Dropout(attention_probs_dropout_prob)
        
    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)
    
    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)
        
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        
        # Compute attention scores
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        
        # Apply attention mask
        attention_scores = attention_scores + attention_mask
        
        # Normalize attention scores
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        
        # Apply attention to values
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        
        return context_layer
