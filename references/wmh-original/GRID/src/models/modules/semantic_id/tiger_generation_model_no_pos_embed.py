import logging
from typing import Any, Optional, Tuple, Union

import torch
import transformers
from torch import nn
from torchmetrics.aggregation import BaseAggregator
from transformers.cache_utils import DynamicCache, EncoderDecoderCache
from transformers.modeling_outputs import Seq2SeqModelOutput
from transformers.models.t5.modeling_t5 import T5Config, T5LayerNorm

from src.utils import RankedLogger, extras
command_line_logger = RankedLogger(__name__, rank_zero_only=True)


from src.data.loading.components.interfaces import (
    SequentialModelInputData,
    SequentialModuleLabelData,
)
from src.models.components.interfaces import OneKeyPerPredictionOutput
from src.models.components.network_blocks.mlp import MLP
from src.models.modules.huggingface.transformer_base_module import TransformerBaseModule
from src.utils.utils import (
    delete_module,
    find_module_shape,
    get_parent_module_and_attr,
    reset_parameters,
)


class SemanticIDGenerativeRecommender(TransformerBaseModule):
    """
    This is a base class for the generative recommender model.
    It is used to generate the semantic ID for the given input.
    It does not contain any specific implementation for the encoder or decoder.
    The encoder and decoder are defined in the subclasses.
    """

    def __init__(
        self,
        codebooks: torch.Tensor,
        num_hierarchies: int,
        num_embeddings_per_hierarchy: int,
        embedding_dim: int,
        should_check_prefix: bool,
        top_k_for_generation: int,
        **kwargs,
    ) -> None:
        """
        Initialize the SemanticIDGenerativeRecommender module.

        Paremeters:
        codebooks (torch.Tensor): the codebooks for the semantic ID.
            the shape of the codebooks should be (num_hierarchies, num_embeddings).
        num_hierarchies (int): the number of hierarchies in the codebooks.
        num_embeddings_per_hierarchy (int): the number of embeddings per hierarchy.
        embedding_dim (int): the dimension of the embeddings.
        top_k_for_generation (int): the number of top-k candidates for generation.
        should_check_prefix (bool): whether to check if the prefix is valid.
        """
        super().__init__(**kwargs)

        self.num_embeddings_per_hierarchy = num_embeddings_per_hierarchy
        self.embedding_dim = embedding_dim
        self.num_hierarchies = num_hierarchies
        self.should_check_prefix = should_check_prefix
        if codebooks != None:
            self.codebooks = codebooks.t()
            assert (
                self.codebooks.size(1) == num_hierarchies
            ), "codebooks should be of shape (-1, num_hierarchies)"
        else:
            logging.warning(
                "Not using pre-cached codebooks, \
            please make sure that \n \
                            1) dataset is properly pre-processed \n \
                            2) num_hierarchies and  num_embeddings_per_hierarchy are proerly set\
            "
            )

        self.top_k_for_generation = top_k_for_generation
        
        # 用于累积生成的语义ID embedding相似度矩阵 (4x4)
        self.generated_similarity_matrix_sum = None
        self.generated_similarity_matrix_count = 0
        
        # 用于累积输入序列中每个item的hierarchy embedding相似度矩阵 (4x4)
        self.input_item_similarity_matrix_sum = None
        self.input_item_similarity_matrix_count = 0

    def _inject_sep_token_between_sids(
        self,
        id_embeddings: torch.Tensor,
        attention_mask: torch.Tensor,
        sep_token: torch.Tensor,
        num_hierarchies: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Inject a separator token into the ID embeddings and attention mask.

        Parameters:
        id_embeddings (torch.Tensor): The ID embeddings of shape (batch_size, seq_len, emb_dim).
        attention_mask (torch.Tensor): The attention mask of shape (batch_size, seq_len).
        sep_token (torch.Tensor): The separator token of shape (1, emb_dim).
        num_hierarchies (int): The number of hierarchies in the codebooks.

        Returns:
        Tuple[torch.Tensor, torch.Tensor]: The modified ID embeddings and attention mask.
        id_embeddings: The ID embeddings with the separator token injected of shape (batch_size, seq_len + num_items, emb_dim).
        attention_mask: The attention mask with the separator token injected of shape (batch_size, seq_len + num_items).

        An intuitive example of the input and output:
        input:
        id_embeddings: [[1, 2, 3, 4], [5, 6, 7, 8]]
        attention_mask: [[1, 1, 1, 1], [1, 1, 1, 1], [0, 0, 0, 0]]
        output:
        id_embeddings: [[1, 2, 3, 4, sep_token], [5, 6, 7, 8, sep_token]]
        attention_mask: [[1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [0, 0, 0, 0, 0]]
        """
        batch_size, seq_len, emb_dim = id_embeddings.size()
        item_count_per_sequence = seq_len // num_hierarchies

        reshaped_id_embeddings = id_embeddings.view(
            batch_size, item_count_per_sequence, num_hierarchies, -1
        )
        reshaped_attention_mask = attention_mask.view(
            batch_size, item_count_per_sequence, num_hierarchies
        )
        reshaped_sep_token_for_concat = (
            sep_token.unsqueeze(0)
            .expand(batch_size, item_count_per_sequence, -1)
            .unsqueeze(-2)
        )
        id_embeddings = torch.cat(
            [reshaped_id_embeddings, reshaped_sep_token_for_concat], dim=-2
        )
        attention_mask = torch.cat(
            [reshaped_attention_mask, reshaped_attention_mask[:, :, [-1]]],
            dim=-1,
        )
        id_embeddings = id_embeddings.reshape(batch_size, -1, emb_dim)
        attention_mask = attention_mask.reshape(batch_size, -1)
        return id_embeddings, attention_mask

    def _spawn_embedding_tables(
        self,
        num_embeddings: int,
        embedding_dim: int,
    ) -> torch.nn.Embedding:
        """
        Spawn an embedding table with the given number of embeddings and embedding dimension.

        Parameters:
        num_embeddings (int): the number of embeddings in the table.
        embedding_dim (int): the dimension of the embeddings.
        """
        table = torch.nn.Embedding(
            num_embeddings=num_embeddings,  # type: ignore
            embedding_dim=embedding_dim,  # type: ignore
        )
        return table

    def _is_kv_cache_valid(
        self, kv_cache: Union[Tuple, DynamicCache, EncoderDecoderCache]
    ) -> bool:

        if isinstance(kv_cache, (EncoderDecoderCache, DynamicCache)):
            return len(kv_cache) > 0
        elif isinstance(kv_cache, Tuple):
            return True
        else:
            return False

    def _add_repeating_offset_to_rows(
        self,
        input_sids: torch.Tensor,
        codebook_size: int,
        num_hierarchies: int,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        """Adds repeating offsets to each element in each row of input_sids.
        we use a single embedding table for multiple code books.
        for example if each codebook has 300 embeddings and we have 3 codebooks,
        the input sequence will be transformed from [0, 1, 2] -> to [0, 301, 602]

        Parameters:
            input_sids (torch.Tensor): A 2D PyTorch tensor.
            codebook_size (int): The number of elements in the codebook.
            num_hierarchies (int): The number of hierarchy levels.
        """

        if input_sids.ndim != 2:
            raise ValueError("Input tensor must be 2-dimensional.")

        num_rows, num_cols = input_sids.shape
        offsets = (
            torch.arange(num_hierarchies, device=input_sids.device) * codebook_size
        )

        # Calculate how many times the full offset pattern needs to repeat
        num_repeats = (
            num_cols + num_hierarchies - 1
        ) // num_hierarchies  # Integer division to handle cases where num_cols is not a multiple of num_hierarchies

        # Repeat the offsets and slice to match the number of columns
        repeated_offsets = offsets.repeat(num_repeats)[:num_cols]

        # Add the repeated offsets to each row using broadcasting
        input_sids_with_offsets = input_sids + repeated_offsets
        if attention_mask is not None:
            input_sids_with_offsets = input_sids_with_offsets * attention_mask
        return input_sids_with_offsets

    def _check_valid_prefix(
        self, prefix: torch.Tensor, batch_size: int = 100000
    ) -> torch.Tensor:
        """
        Checks if a given prefix is a valid prefix of the codebooks.

        Args:
            prefix: A tensor of shape [batch_size, hierarchy_level].
            batch_size: The size of the batch to process.

        Returns:
            A boolean tensor of shape [batch_size] indicating the validity of each prefix.
        """
        # TODO (clark): this is a temporary solution, we should use a more efficient way to do this
        # like pre-sorting the codebook and implementing a tree strcture

        current_hierarchy = prefix.shape[1]
        num_prefixes = prefix.shape[0]
        results = []

        # Ensure codebooks are on the correct device.  Do this *once* outside the loop.
        if prefix.device != self.codebooks.device:
            self.codebooks = self.codebooks.to(prefix.device)

        # Trim the codebooks to the relevant hierarchy *once* outside the loop.
        trimmed_codebooks = self.codebooks[:, :current_hierarchy]

        for i in range(0, num_prefixes, batch_size):
            # Get the current batch of prefixes.
            batch_prefix = prefix[
                i : i + batch_size
            ]  # Shape: [batch_size, hierarchy_level]

            # Perform the comparison.  Broadcasting is now limited by batch_size.
            # trimmed_codebooks shape: [C, H] -> unsqueezed [C, 1, H]
            # batch_prefix shape   : [b, H] -> unsqueezed [1, b, H]
            # comparison result    : [C, b, H]
            comparison = trimmed_codebooks.unsqueeze(1) == batch_prefix.unsqueeze(0)

            # Reduce along the hierarchy dimension (H). Shape: [C, b]
            all_match = comparison.all(dim=2)

            # Reduce along the codebook dimension (C).  Shape: [b]
            any_match = all_match.any(dim=0)

            # Append the results for this batch.
            results.append(any_match)

        # Concatenate the results from all batches.
        return torch.cat(results)
    
    def _compute_semantic_id_embedding_similarity(
        self, 
        generated_ids: torch.Tensor,
        embedding_table: torch.nn.Embedding
    ) -> torch.Tensor:
        """
        计算每个item生成的语义ID embeddings之间的余弦相似度
        
        Args:
            generated_ids: 生成的语义IDs，形状为 [batch_size, top_k, num_hierarchies]
            embedding_table: 用于获取embeddings的embedding table
            
        Returns:
            平均的相似度矩阵，形状为 [num_hierarchies, num_hierarchies]
        """
        # 取top-1的生成结果
        top1_ids = generated_ids[:, 0, :]  # [batch_size, num_hierarchies]
        
        # 添加offset以匹配embedding table的索引
        shifted_ids = self._add_repeating_offset_to_rows(
            input_sids=top1_ids,
            codebook_size=self.num_embeddings_per_hierarchy,
            num_hierarchies=self.num_hierarchies,
        )  # [batch_size, num_hierarchies]
        
        # 获取embeddings
        embeddings = embedding_table(shifted_ids)  # [batch_size, num_hierarchies, embedding_dim]
        
        # 计算每个样本的语义ID embeddings之间的余弦相似度
        batch_size = embeddings.size(0)
        similarity_matrices = []
        
        for i in range(batch_size):
            item_embeddings = embeddings[i]  # [num_hierarchies, embedding_dim]
            
            # 步骤1: L2归一化 (在embedding_dim维度上对每个语义ID embedding进行L2归一化)
            # 归一化后每个向量的L2范数为1
            item_embeddings_normalized = torch.nn.functional.normalize(
                item_embeddings, p=2, dim=-1
            )
            
            # 步骤2: 计算余弦相似度矩阵
            # 对于L2归一化后的向量，余弦相似度 = 向量点积
            # 结果矩阵: [num_hierarchies, num_hierarchies]
            similarity_matrix = torch.mm(
                item_embeddings_normalized, 
                item_embeddings_normalized.t()
            )
            similarity_matrices.append(similarity_matrix)
        
        # 对batch内的所有样本求平均
        avg_similarity_matrix = torch.stack(similarity_matrices).mean(dim=0)
        
        return avg_similarity_matrix
    
    def _compute_input_item_embedding_similarity(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        embedding_table: torch.nn.Embedding
    ) -> torch.Tensor:
        """
        计算输入序列中每个item的4个hierarchy embedding之间的余弦相似度
        
        Args:
            input_ids: 输入的语义IDs，形状为 [batch_size, seq_len]
            attention_mask: 注意力掩码，形状为 [batch_size, seq_len]
            embedding_table: 用于获取embeddings的embedding table
            
        Returns:
            平均的hierarchy间相似度矩阵，形状为 [num_hierarchies, num_hierarchies]
        """
        batch_size, seq_len = input_ids.shape
        
        # 确保seq_len可以被num_hierarchies整除
        if seq_len % self.num_hierarchies != 0:
            command_line_logger.warning(f"seq_len {seq_len} 不能被 num_hierarchies {self.num_hierarchies} 整除")
            return None
        
        num_items = seq_len // self.num_hierarchies
        
        # 添加offset以匹配embedding table的索引
        shifted_ids = self._add_repeating_offset_to_rows(
            input_sids=input_ids,
            codebook_size=self.num_embeddings_per_hierarchy,
            num_hierarchies=self.num_hierarchies,
            attention_mask=attention_mask,
        )  # [batch_size, seq_len]
        
        # 获取embeddings
        embeddings = embedding_table(shifted_ids)  # [batch_size, seq_len, embedding_dim]
        
        # Reshape: [batch_size, num_items, num_hierarchies, embedding_dim]
        embeddings_reshaped = embeddings.view(batch_size, num_items, self.num_hierarchies, -1)
        attention_reshaped = attention_mask.view(batch_size, num_items, self.num_hierarchies)
        
        # 收集所有有效item的hierarchy embeddings
        all_item_similarity_matrices = []
        
        for b in range(batch_size):
            for item_idx in range(num_items):
                # 检查这个item是否有效（非全padding）
                item_mask = attention_reshaped[b, item_idx]  # [num_hierarchies]
                if item_mask.sum() == 0:
                    continue  # 跳过全padding的item
                
                # 获取当前item的hierarchy embeddings: [num_hierarchies, embedding_dim]
                item_hierarchy_embeddings = embeddings_reshaped[b, item_idx]  # [num_hierarchies, embedding_dim]
                
                # L2归一化
                item_hierarchy_embeddings_normalized = torch.nn.functional.normalize(
                    item_hierarchy_embeddings, p=2, dim=-1
                )
                
                # 计算这个item的4个hierarchy embedding之间的余弦相似度: [num_hierarchies, num_hierarchies]
                similarity_matrix = torch.mm(
                    item_hierarchy_embeddings_normalized,
                    item_hierarchy_embeddings_normalized.t()
                )
                
                all_item_similarity_matrices.append(similarity_matrix)
        
        if len(all_item_similarity_matrices) == 0:
            return None
        
        # 对所有有效items的相似度矩阵求平均，得到 [num_hierarchies, num_hierarchies]
        avg_similarity_matrix = torch.stack(all_item_similarity_matrices).mean(dim=0)
        
        return avg_similarity_matrix

    def _beam_search_one_step(
        self,
        candidate_logits: torch.Tensor,
        generated_ids: Union[torch.Tensor, None],
        marginal_log_prob: Union[torch.Tensor, None],
        past_key_values: Union[EncoderDecoderCache, None],
        hierarchy: int,
        batch_size: int,
    ):
        """
        Perform one step of beam search.

        Args:
            candidate_logits: The logits for the next token.
            generated_ids: The generated IDs so far.
            marginal_log_prob: The marginal log probabilities.
            past_key_values: The cache for past key values.
            hierarchy: The current hierarchy level.
            batch_size: The size of the batch.

        Returns:
            The updated generated IDs and the marginal probabilities.
        """

        # pruning the beams that cannot be mapped to a valid item
        if self.should_check_prefix:
            if generated_ids is None:
                valid_prefix_mask = self._check_valid_prefix(
                    torch.arange(
                        self.num_embeddings_per_hierarchy,
                        device=candidate_logits.device,
                    ).unsqueeze(1)
                )
                candidate_logits[:, ~valid_prefix_mask] = float("-inf")
            else:
                # we prune all beams with prefixes that cannot be mapped to a valid item
                valid_prefix_mask = self._check_valid_prefix(
                    torch.cat(
                        [
                            generated_ids.reshape(-1, hierarchy).repeat_interleave(
                                self.num_embeddings_per_hierarchy, dim=0
                            ),
                            torch.arange(
                                self.num_embeddings_per_hierarchy,
                                device=candidate_logits.device,
                            )
                            .repeat(self.top_k_for_generation * batch_size)
                            .unsqueeze(1),
                        ],
                        dim=1,
                    )
                ).reshape(-1, self.num_embeddings_per_hierarchy)
            candidate_logits[~valid_prefix_mask] = float("-inf")

        candidate_logits = torch.nn.functional.softmax(candidate_logits, dim=-1)
        proba, indices = torch.sort(candidate_logits, descending=True)

        if generated_ids is None:
            proba_topk, indices_topk = (
                proba[:, : self.top_k_for_generation],
                indices[:, : self.top_k_for_generation],
            )
            generated_ids = indices_topk.unsqueeze(-1)
            # we need to overwrite the cache because we expanded the beam width from bsz to bsz * beam_width
            # real KV cache starts from the first hierarchy rather than 0-th
            # this is because in 0th hierarchy, self-attention doesn't have cache.
            # and kv cache in huggingface has poor support for this corner case
            past_key_values = EncoderDecoderCache(
                self_attention_cache=DynamicCache(),
                cross_attention_cache=DynamicCache(),
            )
            replace_indices = None
        else:
            # we have beams, generating more beams from the existing beams
            proba, indices = (
                proba[:, : self.num_embeddings_per_hierarchy],
                indices[:, : self.num_embeddings_per_hierarchy],
            )
            proba, indices = proba.reshape(
                -1, self.top_k_for_generation * self.num_embeddings_per_hierarchy
            ), indices.reshape(
                -1, self.top_k_for_generation * self.num_embeddings_per_hierarchy
            )
            # calculating the marginal probability
            proba = torch.mul(
                marginal_log_prob.repeat_interleave(
                    self.num_embeddings_per_hierarchy, dim=-1
                ),
                proba,
            )
            topk_results = torch.topk(
                torch.nan_to_num(proba, nan=-1), k=self.top_k_for_generation, dim=-1
            )
            proba_topk, indices_topk = topk_results.values, topk_results.indices
            # getting indices of winning beams in the original beams
            replace_indices = (
                (indices_topk // self.num_embeddings_per_hierarchy)
                + torch.arange(indices_topk.size(0), device=proba.device).unsqueeze(1)
                * self.top_k_for_generation
            ).flatten()
            # accordingly update kv cache given the winning beams
            if past_key_values != None:
                past_key_values.reorder_cache(replace_indices)

            indices_topk = torch.gather(indices, 1, indices_topk)

        if replace_indices != None:
            generated_ids = torch.cat(
                [
                    generated_ids.reshape(-1, hierarchy)[replace_indices].reshape(
                        -1, self.top_k_for_generation, hierarchy
                    ),
                    indices_topk.unsqueeze(-1),
                ],
                dim=-1,
            )
        else:
            generated_ids = indices_topk.unsqueeze(-1)

        return generated_ids, proba_topk, past_key_values

    def eval_step(
        self,
        batch: Tuple[SequentialModelInputData, SequentialModuleLabelData],
        loss_to_aggregate: BaseAggregator,
        batch_idx: int = 0,
        save_attentions: bool = False,
    ):
        """Perform a single evaluation step on a batch of data from the validation or test set.
        The method will update the metrics and the loss that is passed.
        """
        # Batch is a tuple of model inputs and labels.
        model_input: SequentialModelInputData = batch[0]
        label_data: SequentialModuleLabelData = batch[1]
        _, loss = self.model_step(model_input=model_input, label_data=label_data)

        generated_ids, marginal_probs = self.generate(
            attention_mask=model_input.mask,
            **{
                self.feature_to_model_input_map.get(k, k): v
                for k, v in model_input.transformed_sequences.items()
            },
        )

        self.evaluator(
            marginal_probs=marginal_probs,
            generated_ids=generated_ids,
            # TODO: (lneves) hardcoded for now, will need to change for multiple features
            labels=list(label_data.labels.values())[0].to(marginal_probs.device),
        )

        loss_to_aggregate(loss.detach())

    def _make_deterministic(self, is_training: bool):
        """
        Make the model deterministic by turning off some flags.
        This is needed as the default functions in lightning such as
        on_validation_start on_predict_start cannnot properly set the flags
        for the encoder and decoder.
        (TODO) clark: in the future we can revisit this and make it more generic

        Args:
            is_training (bool): Whether the model is in training mode or not.
        """
        if is_training:
            if self.decoder != None:
                self.decoder.decoder.is_training = True
                self.decoder.decoder.train()
            if self.encoder != None:
                self.encoder.encoder.is_training = True
                self.encoder.encoder.train()
        else:
            if self.decoder != None:
                self.decoder.decoder.is_training = False
                self.decoder.decoder.eval()
            if self.encoder != None:
                self.encoder.encoder.is_training = False
                self.encoder.encoder.eval()

    def on_predict_start(self):
        super().on_predict_start()
        self._make_deterministic(is_training=False)
        # 重置相似度矩阵累积器
        self.generated_similarity_matrix_sum = None
        self.generated_similarity_matrix_count = 0
        self.input_item_similarity_matrix_sum = None
        self.input_item_similarity_matrix_count = 0

    def on_predict_end(self):
        super().on_predict_end()
        self._make_deterministic(is_training=True)
        # 输出平均相似度矩阵
        if self.generated_similarity_matrix_count > 0:
            avg_generated_similarity = self.generated_similarity_matrix_sum / self.generated_similarity_matrix_count
            command_line_logger.info("\n" + "="*80)
            command_line_logger.info(f"=== 生成的语义ID Embedding相似度矩阵 (平均自 {self.generated_similarity_matrix_count} 个batch) ===")
            command_line_logger.info(f"\n{avg_generated_similarity.cpu().numpy()}")
            command_line_logger.info("="*80 + "\n")
        
        if self.input_item_similarity_matrix_count > 0:
            avg_input_similarity = self.input_item_similarity_matrix_sum / self.input_item_similarity_matrix_count
            command_line_logger.info("\n" + "="*80)
            command_line_logger.info(f"=== 输入序列中每个Item的Hierarchy Embedding相似度矩阵 (平均自 {self.input_item_similarity_matrix_count} 个batch) ===")
            command_line_logger.info(f"\n{avg_input_similarity.cpu().numpy()}")
            command_line_logger.info("="*80 + "\n")

    def on_validation_start(self):
        super().on_validation_start()
        self._make_deterministic(is_training=False)
        # 重置相似度矩阵累积器
        self.generated_similarity_matrix_sum = None
        self.generated_similarity_matrix_count = 0
        self.input_item_similarity_matrix_sum = None
        self.input_item_similarity_matrix_count = 0
        # 基于训练集构建item流行度与Head/Body/Tail分组（若可用）
        self._maybe_init_popularity_groups()

    def on_validation_end(self):
        super().on_validation_end()
        self._make_deterministic(is_training=True)
        # 输出平均相似度矩阵
        if self.generated_similarity_matrix_count > 0:
            avg_generated_similarity = self.generated_similarity_matrix_sum / self.generated_similarity_matrix_count
            command_line_logger.info("\n" + "="*80)
            command_line_logger.info(f"=== 生成的语义ID Embedding相似度矩阵 (平均自 {self.generated_similarity_matrix_count} 个batch) ===")
            command_line_logger.info(f"\n{avg_generated_similarity.cpu().numpy()}")
            command_line_logger.info("="*80 + "\n")
        
        if self.input_item_similarity_matrix_count > 0:
            avg_input_similarity = self.input_item_similarity_matrix_sum / self.input_item_similarity_matrix_count
            command_line_logger.info("\n" + "="*80)
            command_line_logger.info(f"=== 输入序列中每个Item的Hierarchy Embedding相似度矩阵 (平均自 {self.input_item_similarity_matrix_count} 个batch) ===")
            command_line_logger.info(f"\n{avg_input_similarity.cpu().numpy()}")
            command_line_logger.info("="*80 + "\n")
        # 打印分组评测摘要（若可用）
        self._log_group_metrics_if_available()

    def on_test_start(self):
        super().on_test_start()
        self._make_deterministic(is_training=False)
        # 重置相似度矩阵累积器
        self.generated_similarity_matrix_sum = None
        self.generated_similarity_matrix_count = 0
        self.input_item_similarity_matrix_sum = None
        self.input_item_similarity_matrix_count = 0
        # 基于训练集构建item流行度与Head/Body/Tail分组（若可用）
        self._maybe_init_popularity_groups()

    def on_test_end(self):
        super().on_test_end()
        self._make_deterministic(is_training=True)
        # 输出平均相似度矩阵
        if self.generated_similarity_matrix_count > 0:
            avg_generated_similarity = self.generated_similarity_matrix_sum / self.generated_similarity_matrix_count
            command_line_logger.info("\n" + "="*80)
            command_line_logger.info(f"=== 生成的语义ID Embedding相似度矩阵 (平均自 {self.generated_similarity_matrix_count} 个batch) ===")
            command_line_logger.info(f"\n{avg_generated_similarity.cpu().numpy()}")
            command_line_logger.info("="*80 + "\n")
        
        if self.input_item_similarity_matrix_count > 0:
            avg_input_similarity = self.input_item_similarity_matrix_sum / self.input_item_similarity_matrix_count
            command_line_logger.info("\n" + "="*80)
            command_line_logger.info(f"=== 输入序列中每个Item的Hierarchy Embedding相似度矩阵 (平均自 {self.input_item_similarity_matrix_count} 个batch) ===")
            command_line_logger.info(f"\n{avg_input_similarity.cpu().numpy()}")
            command_line_logger.info("="*80 + "\n")
        # 打印分组评测摘要（若可用）
        self._log_group_metrics_if_available()

    def on_train_start(self):
        super().on_train_start()
        self._make_deterministic(is_training=True)


class SemanticIDEncoderDecoder(SemanticIDGenerativeRecommender):
    """
    This is an in-house implementation of the encoder-decoder module proposed in TIGER paper,
    See Figure 2.b in https://arxiv.org/pdf/2305.05065.
    We added some additional features and modifications to the original architecture.
    (e.g., constrained beam search, separation tokens, etc)
    """

    def __init__(
        self,
        top_k_for_generation: int = 5,
        codebooks: torch.Tensor = None,
        embedding_dim: int = None,
        num_hierarchies: int = None,
        num_embeddings_per_hierarchy: int = None,
        num_user_bins: Optional[int] = None,
        mlp_layers: Optional[int] = None,
        should_check_prefix: bool = False,
        should_add_sep_token: bool = True,
        prediction_key_name: str = "user_id",
        prediction_value_name: str = "semantic_ids",
        **kwargs,
    ) -> None:
        """
        Initialize the SemanticIDEncoderDecoder module.

        Paremeters:
        codebooks (torch.Tensor): the codebooks for the semantic ID.
            the shape of the codebooks should be (num_hierarchies, num_embeddings_per_hierarchy).
        num_hierarchies (int): the number of hierarchies in the codebooks.
        top_k_for_generation (int): the number of top-k candidates for generation.
        num_user_bins (Optional[int]): the number of bins for user in the dataset (this number equals to the number of rows in the embedding table ).
        mlp_layers (Optional[int]): the number of mlp layers in the encoder and decoder.
        embedding_dim (Optional[int]): the dimension of the embeddings.
        should_check_prefix (bool): whether to check if the prefix is valid.
        """
        command_line_logger.info('i am here')
        if num_hierarchies is None or num_embeddings_per_hierarchy is None:
            num_hierarchies, num_embeddings_per_hierarchy = (
                codebooks.shape[0],
                codebooks.max().item() + 1,
            )
        if embedding_dim is None:
            embedding_dim = (
                kwargs["huggingface_model"]
                .encoder.block[0]
                .layer[0]
                .SelfAttention.q.in_features
            )

        super().__init__(
            codebooks=codebooks,
            num_hierarchies=num_hierarchies,
            num_embeddings_per_hierarchy=num_embeddings_per_hierarchy,
            embedding_dim=embedding_dim,
            top_k_for_generation=top_k_for_generation,
            should_check_prefix=should_check_prefix,
            **kwargs,
        )

        self.encoder = SemanticIDEncoderModule(
            encoder=self.encoder,
        )

        # bos_token used to prompt the decoder to generate the first token
        bos_token = torch.nn.Parameter(
            torch.randn(1, self.embedding_dim), requires_grad=True
        )

        self.decoder = SemanticIDDecoderModule(
            decoder=self.decoder,
            bos_token=bos_token,
            decoder_mlp=torch.nn.ModuleList(
                [
                    torch.nn.Linear(
                        self.embedding_dim,
                        self.num_embeddings_per_hierarchy,
                        bias=False,
                    )
                    for _ in range(self.num_hierarchies)
                ]
            ),
        )

        if mlp_layers is not None:
            # bloating the mlp layers in both encoder and decoder
            # TODO (clark): this currently only works for T5
            for name, module in self.named_modules():
                if isinstance(module, transformers.models.t5.modeling_t5.T5LayerFF):
                    parent_module, attr_name = get_parent_module_and_attr(self, name)
                    setattr(
                        parent_module,
                        attr_name,
                        T5MultiLayerFF(
                            config=self.encoder.encoder.config, num_layers=mlp_layers
                        ),
                    )

        # generate embedding tables for each hierarchy
        # here we assume each hierarchy has the same amount of embeddings
        self.item_sid_embedding_table_encoder = self._spawn_embedding_tables(
            num_embeddings=self.num_embeddings_per_hierarchy * self.num_hierarchies,
            embedding_dim=self.embedding_dim,
        )

        # generating user embedding table
        self.user_embedding: torch.nn.Embedding = (
            self._spawn_embedding_tables(
                num_embeddings=num_user_bins,
                embedding_dim=self.embedding_dim,
            )
            if num_user_bins
            else None
        )

        # separation token for the encoder to differentiate between items
        self.sep_token = (
            torch.nn.Parameter(torch.randn(1, self.embedding_dim), requires_grad=True)
            if should_add_sep_token
            else None
        )
        # the key value names for the prediction output
        self.prediction_key_name = prediction_key_name
        self.prediction_value_name = prediction_value_name
        
        # 分组评测相关
        self.item_to_group_map = None
        self._popularity_group_percentiles = (0.2, 0.5)  # Head: top 20%, Body: next 30%, Tail: last 50%

    def _popularity_group_mapper(self, sid_tuple):
        if self.item_to_group_map is None:
            return None
        return self.item_to_group_map.get(tuple(sid_tuple), None)

    def _maybe_init_popularity_groups(self):
        """
        尝试基于训练集构建item流行度，并据此划分Head/Body/Tail分组。
        仅当训练dataloader可用时执行；否则跳过（predict-only场景通常无法获取训练集）。
        """
        # 已初始化则跳过
        if getattr(self, "item_to_group_map", None) is not None:
            return
        # 依赖Lightning的trainer与datamodule
        if not hasattr(self, "trainer") or self.trainer is None or not hasattr(self.trainer, "datamodule") or self.trainer.datamodule is None:
            command_line_logger.warning("找不到trainer/datamodule，无法构建流行度分组（通常为predict-only运行）。")
            return
        # 尝试获取训练dataloader；若训练阶段未配置，将抛出异常
        try:
            train_loader = self.trainer.datamodule.train_dataloader()
        except Exception as e:
            command_line_logger.warning(f"无法获取训练集以构建流行度分组，将跳过。原因: {e}")
            return

        command_line_logger.info("开始统计训练集中的item流行度...")
        
        # 1. 确定Item流行度 - 创建字典统计item出现次数
        item_popularity = {}  # item_popularity = {}
        num_hier = self.num_hierarchies
        total_items_processed = 0

        # 2. 遍历训练集，统计每个item(语义ID tuple)出现次数
        command_line_logger.info(f"训练集dataloader类型: {type(train_loader)}")
        
        # 检查train_loader是否是tuple，如果是则取第一个元素
        if isinstance(train_loader, tuple):
            actual_loader = train_loader[0]
            command_line_logger.info(f"实际dataloader类型: {type(actual_loader)}")
        else:
            actual_loader = train_loader
        
        try:
            for batch_idx, batch in enumerate(actual_loader):
                # 限制最多处理10000个batch
                if batch_idx >= 10000:
                    command_line_logger.info(f"已达到最大batch限制(10000)，停止统计")
                    break
                    
                # 处理不同的batch格式
                model_inputs_to_process = []
                
                if isinstance(batch, list):
                    # 当 collate_fn 返回 list of batches 时，遍历处理
                    for sub_batch in batch:
                        if isinstance(sub_batch, tuple) and len(sub_batch) >= 2:
                            model_inputs_to_process.append(sub_batch[0])
                        elif hasattr(sub_batch, 'transformed_sequences'):
                            # sub_batch直接是SequentialModelInputData对象
                            model_inputs_to_process.append(sub_batch)
                        # 跳过无法处理的sub_batch，但不输出警告以避免日志污染
                elif isinstance(batch, tuple) and len(batch) >= 2:
                    # 标准格式: (SequentialModelInputData, SequentialModuleLabelData)
                    model_input, _ = batch
                    model_inputs_to_process.append(model_input)
                elif hasattr(batch, 'transformed_sequences'):
                    # 直接是SequentialModelInputData对象
                    model_inputs_to_process.append(batch)
                else:
                    # 跳过无法处理的batch格式，但不输出警告以避免日志污染
                    continue

                # 处理所有收集到的model_input
                for model_input in model_inputs_to_process:
                    # 检查model_input是否有transformed_sequences属性
                    if not hasattr(model_input, 'transformed_sequences'):
                        # 跳过无效的model_input，但不输出警告以避免日志污染
                        continue
                    
                    # 找到语义ID序列键（与模型输入映射一致）
                    input_ids = None
                    for k, v in model_input.transformed_sequences.items():
                        mapped_key = self.feature_to_model_input_map.get(k, k)
                        if mapped_key == "input_ids":
                            input_ids = v
                            break
                    if input_ids is None:
                        # 若未找到，则取第一个非id字段作为序列
                        if len(model_input.transformed_sequences) == 0:
                            continue
                        input_ids = list(model_input.transformed_sequences.values())[0]

                    # 检查model_input是否有mask属性
                    if not hasattr(model_input, 'mask'):
                        # 跳过无效的model_input，但不输出警告以避免日志污染
                        continue
                    attn_mask = model_input.mask.to(input_ids.device)

                    bsz, seq_len = input_ids.shape
                    if seq_len % num_hier != 0:
                        # 不可整除时，按可整除长度截断
                        valid_len = (seq_len // num_hier) * num_hier
                        input_ids = input_ids[:, :valid_len]
                        attn_mask = attn_mask[:, :valid_len]
                        seq_len = valid_len
                        if seq_len == 0:
                            continue
                    num_items = seq_len // num_hier
                    # [B, num_items, H]
                    items = input_ids.view(bsz, num_items, num_hier)
                    masks = attn_mask.view(bsz, num_items, num_hier)
                    valid_items_mask = masks.sum(dim=2) > 0  # [B, num_items]

                    # 3. 对于训练集中的每一条交互记录 (user, item)，执行 item_popularity[item_id] += 1
                    for b in range(bsz):
                        valid_indices = torch.nonzero(valid_items_mask[b], as_tuple=False).flatten().tolist()
                        for idx in valid_indices:
                            sid_tuple = tuple(items[b, idx].tolist())
                            # 统计item出现次数
                            if sid_tuple in item_popularity:
                                item_popularity[sid_tuple] += 1
                            else:
                                item_popularity[sid_tuple] = 1
                            total_items_processed += 1
                
                # 每处理1000个batch打印一次进度
                if (batch_idx + 1) % 1000 == 0:
                    command_line_logger.info(f"已处理 {batch_idx + 1} 个batch，累计统计 {total_items_processed} 个item交互")
        
        except Exception as e:
            command_line_logger.error(f"遍历训练集时出错: {e}")
            command_line_logger.warning("跳过流行度分组功能")
            return

        if len(item_popularity) == 0:
            command_line_logger.warning("未能从训练集统计到任何有效item，跳过分组评测。")
            return

        command_line_logger.info(f"训练集统计完成：共 {len(item_popularity)} 个唯一item，总交互次数 {total_items_processed}")

        # 策略A：百分位（Percentile）/ 排序（Ranking）分组
        # 1. 获取所有Item列表：从item_popularity字典中，获取所有item的ID及其流行度
        sorted_items = sorted(item_popularity.items(), key=lambda kv: kv[1], reverse=True)
        n = len(sorted_items)
        
        # 2. 按流行度排序：将这个列表从高到低排序（已完成）
        
        # 3. 切分：假设你总共有 N 个item
        head_p, mid_p = self._popularity_group_percentiles  # (0.2, 0.5)
        head_end = int(n * head_p)      # Head (热门组)：取排序后的前 20% 的item
        body_end = int(n * mid_p)       # Body (中部组)：取排序后的 20% 到 50% 之间的item
        
        # Head_Items = sorted_list[0 : 0.2 * N]
        head_items = [item for item, _ in sorted_items[:head_end]]
        # Body_Items = sorted_list[0.2 * N : 0.5 * N]  
        body_items = [item for item, _ in sorted_items[head_end:body_end]]
        # Tail_Items = sorted_list[0.5 * N : N]
        tail_items = [item for item, _ in sorted_items[body_end:]]

        # 4. 创建映射：创建一个字典 item_to_group = {}，将每个 item_id 映射到它的分组
        item_to_group = {}
        for item in head_items:
            item_to_group[item] = "Head"
        for item in body_items:
            item_to_group[item] = "Body"
        for item in tail_items:
            item_to_group[item] = "Tail"
        
        self.item_to_group_map = item_to_group

        # 打印分组统计信息
        head_count = len(head_items)
        body_count = len(body_items)
        tail_count = len(tail_items)
        
        command_line_logger.info(f"流行度分组完成:")
        command_line_logger.info(f"  Head组 (前20%): {head_count} 个item")
        command_line_logger.info(f"  Body组 (20%-50%): {body_count} 个item")
        command_line_logger.info(f"  Tail组 (后50%): {tail_count} 个item")
        
        # 打印一些示例
        if head_count > 0:
            head_example = sorted_items[0]
            command_line_logger.info(f"  Head组示例: item {head_example[0]} 出现 {head_example[1]} 次")
        if tail_count > 0:
            tail_example = sorted_items[-1]
            command_line_logger.info(f"  Tail组示例: item {tail_example[0]} 出现 {tail_example[1]} 次")

        # 将group mapper注入评测器（仅对SIDRetrievalEvaluator生效）
        if hasattr(self, "evaluator") and hasattr(self.evaluator, "set_group_mapper"):
            self.evaluator.set_group_mapper(self._popularity_group_mapper)  # type: ignore
            command_line_logger.info("已启用按流行度分组的评测（Head/Body/Tail）。")
        else:
            command_line_logger.warning("评测器不支持分组mapper，跳过分组评测。")

    def _log_group_metrics_if_available(self):
        # 若评测器实现了分组汇总，则打印
        if hasattr(self, "evaluator") and hasattr(self.evaluator, "summarize_group_metrics"):
            try:
                summary = self.evaluator.summarize_group_metrics()  # type: ignore
                if summary:
                    command_line_logger.info("\n" + "="*80)
                    command_line_logger.info("=== 按流行度分组的评测指标（累计到当前阶段）===")
                    for group_name, metrics in summary.items():
                        command_line_logger.info(f"[{group_name}] " + "  ".join([f"{k}: {v:.6f}" for k, v in metrics.items()]))
                    command_line_logger.info("="*80 + "\n")
            except Exception as e:
                command_line_logger.warning(f"打印分组评测指标失败: {e}")

    def encoder_forward_pass(
        self,
        attention_mask: torch.Tensor,
        input_ids: torch.Tensor,
        user_id: torch.Tensor,
        output_attentions: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass for the encoder module.

        Parameters:
            attention_mask (torch.Tensor): The attention mask for the encoder.
            input_ids (torch.Tensor): The input IDs for the encoder.
            user_id (torch.Tensor): The user IDs for the encoder.
            output_attentions (bool): Whether to output attention weights.
        """

        # we shift the IDs here to match the hierarchy structure
        # so that we can use a single embedding table to store the embeddigns for all hierarchies
        shifted_sids = self._add_repeating_offset_to_rows(
            input_sids=input_ids,
            codebook_size=self.num_embeddings_per_hierarchy,
            num_hierarchies=self.num_hierarchies,
            attention_mask=attention_mask,
        )
        inputs_embeds_for_encoder = self.get_embedding_table(table_name="encoder")(
            shifted_sids
        )

        if self.sep_token is not None:
            (
                inputs_embeds_for_encoder,
                attention_mask,
            ) = self._inject_sep_token_between_sids(
                id_embeddings=inputs_embeds_for_encoder,
                attention_mask=attention_mask,
                sep_token=self.sep_token,
                num_hierarchies=self.num_hierarchies,
            )

        # we enter this loop if we want to use user_id
        if user_id is not None and self.user_embedding is not None:
            # preprocessing function pad user_id with zeros
            # so we only need to take the first column
            user_id = user_id[:, 0]

            # TODO (clark): here we assume remainder hashing, which is different from LSH hashing used in TIGER.
            user_embeds = self.user_embedding(
                torch.remainder(user_id, self.user_embedding.num_embeddings)
            )

            # prepending the user_id embedding to the input senquence
            inputs_embeds_for_encoder = torch.cat(
                [
                    user_embeds.unsqueeze(1),
                    inputs_embeds_for_encoder,
                ],
                dim=1,
            )
            # prepending 1 to attention mask as we introduce user embedding in the first column
            user_attention_mask = torch.ones(
                attention_mask.size(0), 1, device=attention_mask.device
            )
            attention_mask_for_encoder = torch.cat(
                [
                    user_attention_mask,
                    attention_mask,
                ],
                dim=1,
            )
        else:
            attention_mask_for_encoder = attention_mask

        encoder_result = self.encoder(
            sequence_embedding=inputs_embeds_for_encoder,
            attention_mask=attention_mask_for_encoder,
            output_attentions=output_attentions,
        )
        if output_attentions:
            encoder_output, encoder_attentions = encoder_result
            return encoder_output, attention_mask_for_encoder, encoder_attentions
        else:
            encoder_output = encoder_result
            return encoder_output, attention_mask_for_encoder

    def decoder_forward_pass(
        self,
        attention_mask: Optional[
            torch.Tensor
        ] = None,  # TODO (clark): in the future we should support variable length semantic id
        future_ids: Optional[torch.Tensor] = None,
        encoder_output: Optional[torch.Tensor] = None,
        attention_mask_for_encoder: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        past_key_values: Optional[DynamicCache] = None,
        output_attentions: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass for the decoder module.
        Parameters:
            attention_mask (torch.Tensor): The attention mask for the decoder.
            future_ids (Optional[torch.Tensor]): The future IDs for the decoder.
            encoder_output (Optional[torch.Tensor]): The output from the encoder.
            attention_mask_for_encoder (Optional[torch.Tensor]): The attention mask for the encoder.
            use_cache (bool): Whether to use cache for past key values.
            past_key_values (Optional[DynamicCache]): The cache for past key values.
            output_attentions (bool): Whether to output attention weights.
        """

        # we generated something before and we need to shift the future_ids
        if future_ids is not None:
            shifted_future_sids = self._add_repeating_offset_to_rows(
                input_sids=future_ids,
                codebook_size=self.num_embeddings_per_hierarchy,
                num_hierarchies=self.num_hierarchies,
                attention_mask=torch.ones_like(future_ids, device=future_ids.device)
                if attention_mask is None
                else attention_mask,
            )
            inputs_embeds_for_decoder = self.get_embedding_table(table_name="decoder")(
                shifted_future_sids
            )

            # we do not have valid kv cache
            # we need to prepend bos token to the decoder input
            if not self._is_kv_cache_valid(kv_cache=past_key_values):
                inputs_embeds_for_decoder = torch.cat(
                    [
                        self.decoder.bos_token.unsqueeze(0).expand(
                            future_ids.size(0), 1, -1
                        ),
                        inputs_embeds_for_decoder,
                    ],
                    dim=1,
                )
                if attention_mask is not None:
                    attention_mask = torch.cat(
                        [
                            torch.ones(future_ids.size(0), 1, device=future_ids.device),
                            attention_mask,
                        ],
                        dim=1,
                    )
            else:
                # we have valid kv cache
                # we only need the last token in the decoder input
                inputs_embeds_for_decoder = inputs_embeds_for_decoder[:, -1:, :]
        # this is the beginning of generation, we start from bos token
        else:
            inputs_embeds_for_decoder = self.decoder.bos_token.unsqueeze(0).expand(
                encoder_output.size(0), 1, -1
            )

        decoder_output = self.decoder(
            sequence_embedding=inputs_embeds_for_decoder,
            attention_mask=attention_mask,
            encoder_attention_mask=attention_mask_for_encoder,
            encoder_output=encoder_output,
            use_cache=use_cache,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
        )

        return decoder_output

    def _generate_with_chunking(
        self,
        attention_mask: torch.Tensor,
        input_ids: torch.Tensor,
        user_id: torch.Tensor = None,
        output_attentions: bool = False,
        chunk_size: int = 2,
    ) -> torch.Tensor:
        """
        Generate with memory-efficient chunking for large hierarchy numbers.
        Split the batch into smaller chunks to reduce memory usage.
        """
        batch_size = input_ids.size(0)
        all_generated_ids = []
        all_marginal_probs = []
        
        for i in range(0, batch_size, chunk_size):
            end_idx = min(i + chunk_size, batch_size)
            
            # Process chunk
            chunk_generated_ids, chunk_marginal_probs = self._generate_single_chunk(
                attention_mask=attention_mask[i:end_idx],
                input_ids=input_ids[i:end_idx],
                user_id=user_id[i:end_idx] if user_id is not None else None,
                output_attentions=output_attentions,
            )
            
            all_generated_ids.append(chunk_generated_ids)
            all_marginal_probs.append(chunk_marginal_probs)
            
            # Clear GPU cache between chunks
            torch.cuda.empty_cache()
        
        # Concatenate results
        final_generated_ids = torch.cat(all_generated_ids, dim=0)
        final_marginal_probs = torch.cat(all_marginal_probs, dim=0)
        
        if output_attentions:
            # For attention outputs, return empty lists as chunking makes attention tracking complex
            return final_generated_ids, final_marginal_probs, [], [], []
        return final_generated_ids, final_marginal_probs

    def _generate_single_chunk(
        self,
        attention_mask: torch.Tensor,
        input_ids: torch.Tensor,
        user_id: torch.Tensor = None,
        output_attentions: bool = False,
    ) -> torch.Tensor:
        """
        Generate for a single chunk with reduced top_k for memory efficiency.
        """
        # Temporarily reduce top_k for large hierarchies
        original_top_k = self.top_k_for_generation
        if self.num_hierarchies > 6:
            self.top_k_for_generation = min(3, original_top_k)  # Reduce to 3 for memory
        
        try:
            result = self._generate_core(
                attention_mask=attention_mask,
                input_ids=input_ids,
                user_id=user_id,
                output_attentions=output_attentions,
            )
        finally:
            # Restore original top_k
            self.top_k_for_generation = original_top_k
        
        return result

    def generate(
        self,
        attention_mask: torch.Tensor,
        input_ids: torch.Tensor,
        user_id: torch.Tensor = None,
        output_attentions: bool = False,
    ) -> torch.Tensor:
        """
        Generate the semantic id given the current model in the sequence using beam search.
        Parameters:
            attention_mask (torch.Tensor): The attention mask for the encoder.
            input_ids (torch.Tensor): The input IDs for the encoder.
            user_id (torch.Tensor): The user IDs for the encoder.
            output_attentions (bool): Whether to output attention weights.
        """
        
        # Clear GPU cache before generation
        torch.cuda.empty_cache()
        
        # Use chunking for large hierarchies to reduce memory usage
        if self.num_hierarchies > 6:
            command_line_logger.info(f"使用分块生成策略，num_hierarchies={self.num_hierarchies}")
            batch_size = input_ids.size(0)
            chunk_size = max(1, min(4, 32 // self.num_hierarchies))  # Adaptive chunk size
            return self._generate_with_chunking(
                attention_mask=attention_mask,
                input_ids=input_ids,
                user_id=user_id,
                output_attentions=output_attentions,
                chunk_size=chunk_size,
            )
        
        return self._generate_core(
            attention_mask=attention_mask,
            input_ids=input_ids,
            user_id=user_id,
            output_attentions=output_attentions,
        )

    def _generate_core(
        self,
        attention_mask: torch.Tensor,
        input_ids: torch.Tensor,
        user_id: torch.Tensor = None,
        output_attentions: bool = False,
    ) -> torch.Tensor:
        """
        Core generation logic without memory optimizations.
        """
        # getting encoder output
        # we only need to do this once because we have decoder
        # to do auto-regressive generation
        encoder_result = self.encoder_forward_pass(
            attention_mask=attention_mask,
            input_ids=input_ids,
            user_id=user_id,
            output_attentions=output_attentions,
        )
        
        if output_attentions:
            encoder_output, encoder_attention_mask, encoder_attentions = encoder_result
            all_cross_attentions = []
            all_decoder_self_attentions = []
        else:
            encoder_output, encoder_attention_mask = encoder_result

        # initilize cached generated ids to None
        generated_ids = None
        marginal_log_prob = None

        # initialize kv cache
        past_key_values = EncoderDecoderCache(
            self_attention_cache=DynamicCache(), cross_attention_cache=DynamicCache()
        )

        for hierarchy in range(self.num_hierarchies):
            # Memory optimization: clear intermediate tensors
            if hierarchy > 0:
                torch.cuda.empty_cache()
            
            if generated_ids is not None:
                # we generated something before
                # we need to reshape the generated ids so that
                # the number of beams equals to batch size * top_k
                squeezed_generated_ids = generated_ids.reshape(-1, hierarchy).to(
                    encoder_output.device
                )  # shape: (batch_size * top_k, hierarchy)

                repeated_encoder_output = encoder_output.repeat_interleave(
                    self.top_k_for_generation, dim=0
                )
                # shape: (batch_size * top_k, seq_len+1, hidden_dim)
                # +1 because we have user_id token

                repeated_encoder_attention_mask = (
                    encoder_attention_mask.repeat_interleave(
                        self.top_k_for_generation, dim=0
                    )
                )  # shape: (batch_size * top_k, seq_len+1)
            else:
                # we haven't generated anything yet!
                # the number of beams currently equals to batch size
                squeezed_generated_ids = None
                repeated_encoder_output = encoder_output
                repeated_encoder_attention_mask = encoder_attention_mask

            # feeding the decoder with the generated ids
            decoder_result = self.decoder_forward_pass(
                future_ids=squeezed_generated_ids,
                encoder_output=repeated_encoder_output,
                attention_mask_for_encoder=repeated_encoder_attention_mask,
                use_cache=True,
                past_key_values=past_key_values,
                output_attentions=output_attentions,
            )
            
            if output_attentions:
                decoder_output, past_key_values, self_attentions, cross_attentions = decoder_result
                all_cross_attentions.append(cross_attentions)
                all_decoder_self_attentions.append(self_attentions)
            else:
                decoder_output, past_key_values = decoder_result

            # decoder_output[:, -1, :] is the embedding for the next token
            latest_output_representation = decoder_output[:, -1, :]

            # # calculating the logits for the next token
            candidate_logits = self.decoder.decoder_mlp[hierarchy](
                latest_output_representation
            )  # shape: (batch_size * top_k, num_embeddings in the hierarchy)

            (
                generated_ids,
                marginal_log_prob,
                past_key_values,
            ) = self._beam_search_one_step(
                candidate_logits=candidate_logits,
                generated_ids=generated_ids,
                marginal_log_prob=marginal_log_prob,
                past_key_values=past_key_values,
                hierarchy=hierarchy,
                batch_size=input_ids.size(0),
            )
            
            # Memory cleanup: delete intermediate tensors for large hierarchies
            if self.num_hierarchies > 6:
                del decoder_output, latest_output_representation, candidate_logits
                if 'repeated_encoder_output' in locals():
                    del repeated_encoder_output, repeated_encoder_attention_mask

        if output_attentions:
            return generated_ids, marginal_log_prob, encoder_attentions, all_cross_attentions, all_decoder_self_attentions
        return generated_ids, marginal_log_prob

    def forward(
        self,
        attention_mask_encoder: torch.Tensor,
        input_ids: torch.Tensor,
        user_id: Optional[torch.Tensor] = None,
        future_ids: Optional[torch.Tensor] = None,
        attention_mask_decoder: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Forward pass for the encoder-decoder model.
        Parameters:
            attention_mask_encoder (torch.Tensor): The attention mask for the encoder.
            input_ids (torch.Tensor): The input IDs for the encoder.
            user_id (torch.Tensor): The user IDs for the encoder.
            future_ids (Optional[torch.Tensor]): The future IDs for the decoder.
            attention_mask_decoder (Optional[torch.Tensor]): The attention mask for the decoder.
        """

        encoder_output, attention_mask_for_encoder = self.encoder_forward_pass(
            attention_mask=attention_mask_encoder,
            input_ids=input_ids,
            user_id=user_id,
        )

        decoder_output = self.decoder_forward_pass(
            future_ids=future_ids,
            attention_mask=attention_mask_decoder,
            encoder_output=encoder_output,
            attention_mask_for_encoder=attention_mask_for_encoder,
            use_cache=False,  # we are not using cache for training
        )
        return decoder_output

    def get_embedding_table(self, table_name: str, hierarchy: Optional[int] = None):
        """
        Get the embedding table for the given table name and hierarchy.
        Args:
            table_name: The name of the table to get the embedding for.
            hierarchy: The hierarchy level to get the embedding for.
        """
        # here we assume the encoder and decoder share the same embedding table
        # we can have flexible embedding table in the future
        if table_name == "encoder":
            embedding_table = self.item_sid_embedding_table_encoder
        elif table_name == "decoder":
            embedding_table = self.item_sid_embedding_table_encoder

        if hierarchy is not None:
            return embedding_table(
                torch.arange(
                    hierarchy * self.num_embeddings_per_hierarchy,
                    (hierarchy + 1) * self.num_embeddings_per_hierarchy,
                ).to(self.device)
            )
        return embedding_table

    def predict_step(self, batch: SequentialModelInputData, batch_idx: int = 0):
        import os
        
        # 在inference模式下获取attention权重
        generated_ids, marginal_probs, encoder_attentions, cross_attentions, decoder_self_attentions = self.generate(
            attention_mask=batch.mask,
            output_attentions=True,
            **{
                self.feature_to_model_input_map.get(k, k): v
                for k, v in batch.transformed_sequences.items()
            },
        )
        
        # 保存attention权重到当前目录
        attention_save_dir = "./attention_weights"
        os.makedirs(attention_save_dir, exist_ok=True)
        
        # 保存encoder attention权重
        encoder_attn_path = os.path.join(attention_save_dir, f"encoder_attention_batch_{batch_idx}.pt")
        torch.save(encoder_attentions, encoder_attn_path)
        
        # 处理并拼接cross attention成完整的二维矩阵
        # cross_attentions是list，每个元素对应一个hierarchy
        # 由于beam search，每个hierarchy的batch_size会不同，我们只保留top-1的beam
        full_cross_attentions = []
        original_batch_size = batch.mask.size(0)
        top_k = self.top_k_for_generation
        
        for hier_idx in range(len(cross_attentions)):
            hier_attns = cross_attentions[hier_idx]  # 这个hierarchy的所有layer的attention
            if hier_attns is not None and len(hier_attns) > 0:
                # 取最后一层的attention
                last_layer_attn = hier_attns[-1]
                
                # 由于beam search，形状可能是 [batch_size * top_k, num_heads, 1, encoder_seq_len]
                # 我们只取每个样本的top-1 beam (第一个beam)
                current_batch_size = last_layer_attn.size(0)
                
                if current_batch_size > original_batch_size:
                    # 有beam search，取每个batch的第一个beam
                    # 每top_k个beam属于同一个原始样本
                    indices = torch.arange(0, current_batch_size, top_k, device=last_layer_attn.device)
                    last_layer_attn = last_layer_attn[indices]
                
                # 去掉decoder维度: [batch_size, num_heads, encoder_seq_len]
                if last_layer_attn.dim() == 4:
                    last_layer_attn = last_layer_attn.squeeze(2)
                
                full_cross_attentions.append(last_layer_attn)
        
        # 拼接所有hierarchies: [batch_size, num_hierarchies, num_heads, encoder_seq_len]
        if full_cross_attentions:
            try:
                full_cross_attention_matrix = torch.stack(full_cross_attentions, dim=1)
            except RuntimeError as e:
                command_line_logger.warning(f"Failed to stack cross attentions: {e}")
                # 打印每个tensor的shape用于调试
                for i, t in enumerate(full_cross_attentions):
                    command_line_logger.warning(f"  Hierarchy {i} shape: {t.shape}")
                full_cross_attention_matrix = None
        else:
            full_cross_attention_matrix = None
        
        # 保存原始的逐步cross attention
        cross_attn_path = os.path.join(attention_save_dir, f"cross_attention_batch_{batch_idx}.pt")
        torch.save(cross_attentions, cross_attn_path)
        
        # 保存拼接后的完整cross attention矩阵
        if full_cross_attention_matrix is not None:
            full_cross_attn_path = os.path.join(attention_save_dir, f"cross_attention_full_batch_{batch_idx}.pt")
            torch.save(full_cross_attention_matrix, full_cross_attn_path)
            command_line_logger.info(f"Saved full cross attention matrix (shape: {full_cross_attention_matrix.shape}) to {full_cross_attn_path}")
        
        # 保存decoder self-attention
        decoder_self_attn_path = os.path.join(attention_save_dir, f"decoder_self_attention_batch_{batch_idx}.pt")
        torch.save(decoder_self_attentions, decoder_self_attn_path)
        command_line_logger.info(f"Saved decoder self-attention to {decoder_self_attn_path}")
        
        command_line_logger.info(f"Saved encoder attention to {encoder_attn_path}")
        command_line_logger.info(f"Saved cross attention to {cross_attn_path}")
        
        # 计算生成的语义ID embedding相似度矩阵
        batch_generated_similarity_matrix = self._compute_semantic_id_embedding_similarity(
            generated_ids=generated_ids,
            embedding_table=self.get_embedding_table(table_name="encoder")
        )
        
        # 累积生成的语义ID相似度矩阵
        if self.generated_similarity_matrix_sum is None:
            self.generated_similarity_matrix_sum = batch_generated_similarity_matrix
        else:
            self.generated_similarity_matrix_sum += batch_generated_similarity_matrix
        self.generated_similarity_matrix_count += 1
        
        command_line_logger.info(f"当前batch的生成语义ID Embedding相似度矩阵:\n{batch_generated_similarity_matrix.cpu().numpy()}")
        
        # 计算输入序列item embedding相似度矩阵
        input_ids = None
        for k, v in batch.transformed_sequences.items():
            mapped_key = self.feature_to_model_input_map.get(k, k)
            if mapped_key == "input_ids":
                input_ids = v
                break
        
        if input_ids is not None:
            batch_input_similarity_matrix = self._compute_input_item_embedding_similarity(
                input_ids=input_ids,
                attention_mask=batch.mask,
                embedding_table=self.get_embedding_table(table_name="encoder")
            )
            
            if batch_input_similarity_matrix is not None:
                # 累积输入item的hierarchy embedding相似度矩阵
                if self.input_item_similarity_matrix_sum is None:
                    self.input_item_similarity_matrix_sum = batch_input_similarity_matrix.detach().cpu()
                else:
                    self.input_item_similarity_matrix_sum += batch_input_similarity_matrix.detach().cpu()
                self.input_item_similarity_matrix_count += 1
                
                command_line_logger.info(f"当前batch的输入Item Hierarchy Embedding相似度矩阵:\n{batch_input_similarity_matrix.cpu().numpy()}")
        
        ids = [
            id.item() if isinstance(id, torch.Tensor) else id
            for id in batch.user_id_list
        ]
        model_output = OneKeyPerPredictionOutput(
            keys=ids,
            predictions=generated_ids,
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
        """Perform a single evaluation step on a batch of data from the validation or test set.
        The method will update the metrics and the loss that is passed.
        """
        # Batch is a tuple of model inputs and labels.
        model_input: SequentialModelInputData = batch[0]
        label_data: SequentialModuleLabelData = batch[1]
        _, loss = self.model_step(model_input=model_input, label_data=label_data)

        generated_ids, marginal_probs = self.generate(
            attention_mask=model_input.mask,
            **{
                self.feature_to_model_input_map.get(k, k): v
                for k, v in model_input.transformed_sequences.items()
            },
        )

        self.evaluator(
            marginal_probs=marginal_probs,
            generated_ids=generated_ids,
            # TODO: (lneves) hardcoded for now, will need to change for multiple features
            labels=list(label_data.labels.values())[0].to(marginal_probs.device),
        )
        
        # 计算生成的语义ID embedding相似度矩阵
        batch_generated_similarity_matrix = self._compute_semantic_id_embedding_similarity(
            generated_ids=generated_ids,
            embedding_table=self.get_embedding_table(table_name="encoder")
        )
        
        # 累积生成的语义ID相似度矩阵
        if self.generated_similarity_matrix_sum is None:
            self.generated_similarity_matrix_sum = batch_generated_similarity_matrix
        else:
            self.generated_similarity_matrix_sum += batch_generated_similarity_matrix
        self.generated_similarity_matrix_count += 1
        
        # 计算输入序列item embedding相似度矩阵
        input_ids = None
        for k, v in model_input.transformed_sequences.items():
            mapped_key = self.feature_to_model_input_map.get(k, k)
            if mapped_key == "input_ids":
                input_ids = v
                break
        
        if input_ids is not None:
            batch_input_similarity_matrix = self._compute_input_item_embedding_similarity(
                input_ids=input_ids,
                attention_mask=model_input.mask,
                embedding_table=self.get_embedding_table(table_name="encoder")
            )
            
            if batch_input_similarity_matrix is not None:
                # 累积输入item的hierarchy embedding相似度矩阵
                if self.input_item_similarity_matrix_sum is None:
                    self.input_item_similarity_matrix_sum = batch_input_similarity_matrix.detach().cpu()
                else:
                    self.input_item_similarity_matrix_sum += batch_input_similarity_matrix.detach().cpu()
                self.input_item_similarity_matrix_count += 1

        loss_to_aggregate(loss)

    def model_step(
        self,
        model_input: SequentialModelInputData,
        label_data: Optional[SequentialModuleLabelData] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform a forward pass of the model and calculate the loss if label_data is provided.

        Args:
            model_input: The input data to the model.
            label_data: The label data to the model. Its optional as it is not required for inference.
        """

        # if label_data is None, we are in inference mode and doing free-form generation
        if label_data is None:
            # this is inference stage
            generated_ids, marginal_probs = self.generate(
                attention_mask=model_input.mask,
                **{
                    self.feature_to_model_input_map.get(k, k): v
                    for k, v in model_input.transformed_sequences.items()
                },
            )
            return generated_ids, 0  # returning 0 here because we don't have a loss

        fut_ids = None
        for label in label_data.labels:
            curr_label = label_data.labels[label]
            fut_ids = curr_label.reshape(model_input.mask.size(0), -1)
        # here we pass labels in to the forward function
        # because the decoder is causal and we are doing shifted prediction
        model_output = self.forward(
            attention_mask_encoder=model_input.mask,
            future_ids=fut_ids,
            **{
                self.feature_to_model_input_map.get(k, k): v
                for k, v in model_input.transformed_sequences.items()
            },
        )

        # we prepended a bos token to the decoder input
        # so we need to remove the last token in the output
        model_output = model_output[:, :-1]

        # the label locations is shared for all semantic id hierarchies
        loss = 0
        if self.loss_function is not None:
            for hierarchy in range(self.num_hierarchies):
                input = self.decoder.decoder_mlp[hierarchy](model_output[:, hierarchy])
                loss += self.loss_function(
                    input=input,
                    target=fut_ids[:, hierarchy].long(),
                )
        return model_output, loss


class SemanticIDDecoderModule(torch.nn.Module):
    """
    This is an in-house replication of the decoder module proposed in TIGER paper,
    See Figure 2.b in https://arxiv.org/pdf/2305.05065.
    """

    def __init__(
        self,
        decoder: transformers.PreTrainedModel,
        decoder_mlp: Optional[torch.nn.Module] = None,
        bos_token: Optional[torch.nn.Parameter] = None,
    ) -> None:
        """
        Initialize the SemanticIDDecoderModule.

        Parameters:
        decoder (transformers.PreTrainedModel): the encoder model (e.g., transformers.T5EncoderModel).
        decoder_mlp (torch.nn.Module): the mlp layers used to project the decoder output to the embedding table.
        bos_token (Optional[torch.nn.Parameter]):
            the bos token used to prompt the decoder.
            if None, then this means the decoder is used standalone without an encoder.
        """

        super().__init__()
        # some sanity checks
        if bos_token is not None:
            assert decoder.config.is_decoder == True, "Decoder must be a decoder model"
            assert (
                decoder.config.is_encoder_decoder == False
            ), "Decoder must be a standalone decoder model"

        self.decoder = decoder
        # this bos token is prompt for the decoder
        self.bos_token = bos_token
        self.decoder_mlp = decoder_mlp
        # deleting embedding table in the decoder to save space
        delete_module(self.decoder, "embed_tokens")
        delete_module(self.decoder, "shared")
        reset_parameters(self.decoder)
        
        # 禁用decoder的位置编码（相对位置偏置）
        # 1. 将所有block的标志设为False
        # 2. 将relative_attention_bias的权重置零，确保即使被调用也不产生影响
        for block in self.decoder.block:
            if hasattr(block.layer[0], 'SelfAttention'):
                block.layer[0].SelfAttention.has_relative_attention_bias = False
                # 将位置偏置权重置零，保证不产生实际效果
                if hasattr(block.layer[0].SelfAttention, 'relative_attention_bias'):
                    with torch.no_grad():
                        block.layer[0].SelfAttention.relative_attention_bias.weight.zero_()
                        # 冻结这个参数，防止训练时更新
                        block.layer[0].SelfAttention.relative_attention_bias.weight.requires_grad = False

    def forward(
        self,
        attention_mask: torch.Tensor,
        sequence_embedding: torch.Tensor,
        encoder_output: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        use_cache: bool = False,
        past_key_values: DynamicCache = DynamicCache(),
        output_attentions: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass for the decoder module.
        Parameters:
            attention_mask (torch.Tensor): The attention mask for the decoder.
            sequence_embedding (torch.Tensor): The input sequence embedding for the decoder.
            encoder_output (torch.Tensor): The output from the encoder.
            encoder_attention_mask (torch.Tensor): The attention mask for the encoder.
            use_cache (bool): Whether to use cache for past key values.
            past_key_values (DynamicCache): The cache for past key values.
            output_attentions (bool): Whether to output attention weights.
        """

        decoder_outputs: Seq2SeqModelOutput = self.decoder(
            attention_mask=attention_mask,
            inputs_embeds=sequence_embedding,
            encoder_hidden_states=encoder_output,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
        )

        embeddings = decoder_outputs.last_hidden_state

        if use_cache:
            if output_attentions:
                return embeddings, decoder_outputs.past_key_values, decoder_outputs.attentions, decoder_outputs.cross_attentions
            return embeddings, decoder_outputs.past_key_values
        if output_attentions:
            return embeddings, decoder_outputs.attentions, decoder_outputs.cross_attentions
        return embeddings


class SemanticIDEncoderModule(torch.nn.Module):
    """
    This is an in-house replication of the encoder module proposed in TIGER paper,
    See Figure 2.b in https://arxiv.org/pdf/2305.05065.
    """

    def __init__(
        self,
        encoder: transformers.PreTrainedModel,
    ) -> None:
        """
        Initialize the SemanticIDEncoderModule module.

        Paremeters:
        encoder (transformers.PreTrainedModel): the encoder model (e.g., transformers.T5EncoderModel).
        """
        super().__init__()

        self.encoder = encoder
        embedding_table_dim = find_module_shape(self.encoder, "embed_tokens")
        num_embeddings, embedding_dim = embedding_table_dim

        self.num_embeddings_per_hierarchy = num_embeddings
        self.embedding_dim = embedding_dim
        # TODO (clark): take care of chunky position encoding

        # deleting embedding table in the encoder to save space
        delete_module(self.encoder, "embed_tokens")
        delete_module(self.encoder, "shared")
        reset_parameters(self.encoder)

    def forward(
        self,
        attention_mask: torch.Tensor,
        sequence_embedding: torch.Tensor,
        output_attentions: bool = False,
    ) -> torch.Tensor:

        encoder_output = self.encoder(
            inputs_embeds=sequence_embedding,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )
        embeddings = encoder_output.last_hidden_state
        if output_attentions:
            return embeddings, encoder_output.attentions
        return embeddings


# TODO (clark): this is a T5 specific implementation
# this class is used for bloating the mlp layers in the encoder and decoder
# original T5 implementation only has one layer
class T5MultiLayerFF(nn.Module):
    def __init__(self, config: T5Config, num_layers: int):
        """
        Initialize the T5MultiLayerFF module.
        This module is a multi-layer feed-forward network (MLP) used in the T5 model.
        It consists of a series of linear layers with ReLU activation and dropout.
        And it also includes layer normalization and residual connections.
        Parameters:
            config (T5Config): The T5 configuration object.
            num_layers (int): The number of layers in the MLP.
        """
        super().__init__()
        self.mlp = MLP(
            input_dim=config.d_model,
            output_dim=config.d_model,
            hidden_dim_list=[config.d_ff for _ in range(num_layers)],
            activation=nn.ReLU,
            dropout=config.dropout_rate,
        )

        self.layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the T5MultiLayerFF module.
        Parameters:
            hidden_states (torch.Tensor): The input hidden states for the MLP.
        """
        forwarded_states = self.layer_norm(hidden_states)
        forwarded_states = self.mlp(forwarded_states)
        hidden_states = hidden_states + self.dropout(forwarded_states)
        return hidden_states