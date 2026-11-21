from typing import Any, Dict, List

import torch
import torch.nn.functional as F
import torchmetrics
from torchmetrics.metric import Metric
from torchmetrics.utilities.distributed import gather_all_tensors

## Custom Metrics


class CustomMeanReductionMetric(torchmetrics.Metric):
    """
    Custom metric class that uses mean reduction and supports distributed training.
    """

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self.metric_values = 0
        self.total_values = 0

    def compute(self) -> torch.Tensor:
        # Aggregates the metric accross workers and returns the final value
        metric_values_tensor = torch.tensor(self.metric_values).to(self.device)
        total_values_tensor = torch.tensor(self.total_values).to(self.device)
        # Compute final metric
        if self.total_values == 0:
            return torch.tensor(0.0, device=self.device)
        # Checks if using more than one GPU
        # If so, gather all metric values and total values from all GPUs. Else, return the current
        # worker's metric value
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            # Gather all metric values and total values from all GPUs

            metric_values_tensor_list = [
                t.unsqueeze(0) if t.dim() == 0 else t
                for t in gather_all_tensors(metric_values_tensor)
            ]
            metric_values_tensor = torch.cat(metric_values_tensor_list).sum()

            total_values_tensor_list = [
                t.unsqueeze(0) if t.dim() == 0 else t
                for t in gather_all_tensors(total_values_tensor)
            ]

            total_values_tensor = torch.cat(total_values_tensor_list).sum()

        return metric_values_tensor / total_values_tensor

    def reset(self) -> None:
        self.metric_values = 0
        self.total_values = 0

    def update(self) -> None:
        raise NotImplementedError


class CustomRetrievalMetric(CustomMeanReductionMetric):
    """
    Custom retrieval metric class to calculate ranking metrics.
    """

    def __init__(
        self,
        top_k: int,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.top_k = top_k

    def update(
        self, preds: torch.Tensor, target: torch.Tensor, indexes: torch.Tensor = None, batch_size: int = None, **kwargs
    ) -> None:
        
        # 向后兼容：如果提供了indexes，使用原来的逻辑
        if indexes is not None and batch_size is None:
            zero_count = (indexes == 0).sum().item()
            if zero_count == 0:
                # 如果没有零索引，说明所有索引都是有效的，batch_size就是总长度
                batch_size = len(indexes)
            else:
                batch_size = int(len(indexes) / zero_count)
        elif batch_size is None:
            # 如果都没有提供，默认为1
            batch_size = 1
            
        preds = preds.reshape(batch_size, -1)
        target = target.reshape(batch_size, -1).int()

        metric = self._metric(preds, target)
        self.metric_values += metric.sum().item()
        self.total_values += batch_size

    def _metric(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class NDCG(CustomRetrievalMetric):
    """
    Metric to calculate Normalized Discounted Cumulative Gain@K (NDCG@K).
    """

    def _metric(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # 确保top_k不超过预测结果的数量
        actual_k = min(self.top_k, preds.size(-1))
        topk_indices = torch.topk(preds, actual_k)[1]
        topk_true = target.gather(1, topk_indices)

        # Compute DCG
        dcg = torch.sum(
            topk_true
            / torch.log2(
                torch.arange(2, actual_k + 2, device=target.device).unsqueeze(0)
            ),
            dim=1,
        )

        # Compute IDCG
        ideal_indices = torch.topk(target, actual_k)[1]
        ideal_dcg = torch.sum(
            target.gather(1, ideal_indices)
            / torch.log2(
                torch.arange(2, actual_k + 2, device=target.device).unsqueeze(0)
            ),
            dim=1,
        )

        # Handle cases where IDCG is zero
        ndcg = dcg / torch.where(ideal_dcg == 0, torch.ones_like(ideal_dcg), ideal_dcg)
        return ndcg


class Recall(CustomRetrievalMetric):
    """
    Metric to calculate Recall@K.
    """

    def _metric(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # 确保top_k不超过预测结果的数量
        actual_k = min(self.top_k, preds.size(-1))
        topk_indices = torch.topk(preds, actual_k)[1]
        topk_true = target.gather(1, topk_indices)

        true_positives = topk_true.sum(dim=1)
        total_relevant = target.sum(dim=1)

        recall = true_positives / total_relevant.minimum(
            torch.tensor(self.top_k, device=self.device)
        ).clamp(
            min=1
        )  # Use clamp to avoid zero
        return recall
## Evaluators

class Evaluator:
    def __init__(self, metrics: Dict[str, Metric], *args, **kwargs):
        self.metrics = metrics

    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    def reset(self):
        for metric in self.metrics.values():
            metric.reset()

    def to(self, device: torch.device):
        for metric in self.metrics.values():
            metric.to(device=device)


class RetrievalEvaluator(Evaluator):
    """
    Wrapper for retrieval evaluation metrics.
    It takes model outputs and automatically calculates the retrieval metrics.
    """

    def __init__(
        self,
        metrics: Dict[str, CustomRetrievalMetric],
        top_k_list: List[int],
        should_sample_negatives_from_vocab: bool = True,
        num_negatives: int = 500,
        placeholder_token_buffer: int = 100,
    ):
        self.metrics = {
            f"{metric_name}@{top_k}": metric_object(
                top_k=top_k, sync_on_compute=False, compute_with_cache=False
            )
            for metric_name, metric_object in metrics.items()
            for top_k in top_k_list
        }
        self.should_sample_negatives_from_vocab = should_sample_negatives_from_vocab
        self.num_negatives = num_negatives
        self.placeholder_token_buffer = placeholder_token_buffer

    def __call__(
        self,
        query_embeddings: torch.Tensor,
        key_embeddings: torch.Tensor,
        labels: torch.Tensor,
    ):
        num_of_samples = query_embeddings.shape[0]
        num_of_candidates = key_embeddings.shape[0]

        if self.should_sample_negatives_from_vocab:
            inbatch_negatives = self.sample_negative_ids_from_vocab(
                num_of_samples=num_of_samples,
                num_of_candidates=num_of_candidates,
                num_negatives=self.num_negatives,
            )
            # we +1 here because we need to include the positive sample
            num_of_candidates = self.num_negatives + 1
            pos_embeddings = key_embeddings[labels]
            key_embeddings = key_embeddings[inbatch_negatives]
            # key_embeddings shape: (bsz, num_negatives+1, emb_dim)
            key_embeddings = torch.cat(
                [pos_embeddings.unsqueeze(1), key_embeddings], dim=1
            )
            # the positive index will always be 0 because the pos embedding will always be the first one.
            labels = torch.zeros(num_of_samples).long()

        # following examples from https://lightning.ai/docs/torchmetrics/stable/retrieval/precision.html
        # indexes refers to the mask of the labels
        indexes = torch.arange(0, query_embeddings.shape[0])
        expanded_indexes = (
            indexes.unsqueeze(-1).expand(num_of_samples, num_of_candidates).reshape(-1)
        )

        if self.should_sample_negatives_from_vocab:
            preds = (
                torch.mul(
                    query_embeddings.unsqueeze(1).expand_as(key_embeddings),
                    key_embeddings,
                )
                .sum(-1)
                .reshape(-1)
            )
        else:
            preds = torch.mm(query_embeddings, key_embeddings.t()).reshape(-1)

        target = torch.zeros(num_of_samples, num_of_candidates).bool()
        target[torch.arange(num_of_samples), labels] = True
        target = target.reshape(-1)

        for _, metric_object in self.metrics.items():
            metric_object.update(
                preds,
                target.to(preds.device),
                indexes=expanded_indexes.to(preds.device),
            )

    # this method samples random negative samples from the whole vocab
    def sample_negative_ids_from_vocab(
        self,
        num_of_samples: int,
        num_of_candidates: int,
        num_negatives: int,
    ) -> torch.Tensor:
        # num_of_samples: batch size
        # num_of_candidates: number of total vocabs
        # num_negatives: number of negative samples

        # we do randint to accelerate the negative sampling
        # this could have collision with positive pairs but the chance is very low

        # TODO (Clark): in the future we might need to have non-collision negative sampling
        # when K in top-k is very small (e.g., hits@1) and num_negatives is very large
        negative_candidates = torch.randint(
            self.placeholder_token_buffer,
            num_of_candidates,
            (num_of_samples, num_negatives),
        )

        return negative_candidates


class SIDRetrievalEvaluator(Evaluator):
    """
    Wrapper for retrieval evaluation metrics for semantic IDs.
    It takes model outputs in semantic IDs and automatically calculates the retrieval metrics.
    """

    def __init__(
        self,
        metrics: Dict[str, CustomRetrievalMetric],
        top_k_list: List[int],
    ):
        self.metrics = {
            f"{metric_name}@{top_k}": metric_object(
                top_k=top_k, sync_on_compute=False, compute_with_cache=False
            )
            for metric_name, metric_object in metrics.items()
            for top_k in top_k_list
        }
        # 分组评测支持：通过一个可选的group mapper将ground-truth item映射到组名（Head/Body/Tail）
        self._group_mapper = None
        # 为每个分组维护一套独立metric
        self.group_metrics: Dict[str, Dict[str, CustomRetrievalMetric]] = {}

    def set_group_mapper(self, mapper_callable) -> None:
        """
        设置一个函数: (sid_tuple) -> group_name，用于将真实item所属的语义ID tuple映射到组名。
        例如: ('Head'|'Body'|'Tail')
        """
        self._group_mapper = mapper_callable

    def __call__(
        self,
        marginal_probs: torch.Tensor,
        generated_ids: torch.Tensor,
        labels: torch.Tensor,
        **kwargs,
    ):
        batch_size, num_candidates, num_hierarchies = generated_ids.shape
        labels = labels.reshape(batch_size, 1, num_hierarchies)
        preds = marginal_probs.reshape(-1)

        # check if the generated IDs contain the labels
        # if so, we get the coordinates of the matched IDs
        matched_id_coord = torch.all((generated_ids == labels), dim=2).nonzero()

        # we initialize the ground truth as all false
        target = torch.zeros(batch_size, num_candidates).bool()

        # we set the matched IDs to true if they are in the generated IDs
        target[matched_id_coord[:, 0], matched_id_coord[:, 1]] = True
        target = target.reshape(-1)
        expanded_indexes = (
            torch.arange(batch_size)
            .unsqueeze(-1)
            .expand(batch_size, num_candidates)
            .reshape(-1)
        )

        for _, metric_object in self.metrics.items():
            metric_object.update(
                preds,
                target.to(preds.device),
                indexes=expanded_indexes.to(preds.device),
            )

        # 分组评测：对所有测试样本使用相同的预测候选集，但只统计真实标签属于特定组的样本
        if self._group_mapper is not None:
            # 提取每个样本的真实语义ID tuple
            # labels: [B,1,H] -> [B,H]
            label_tuples = [tuple(labels[i, 0].tolist()) for i in range(batch_size)]
            group_names = [self._group_mapper(sid_tuple) for sid_tuple in label_tuples]

            # 为每个group构建子样本掩码，并各自更新一套metrics
            for group_name in set([g for g in group_names if g is not None]):
                if group_name not in self.group_metrics:
                    # 为新组克隆一套metric实例
                    self.group_metrics[group_name] = {
                        metric_key: type(metric_inst)(top_k=metric_inst.top_k, sync_on_compute=False, compute_with_cache=False)  # type: ignore
                        for metric_key, metric_inst in self.metrics.items()
                    }
                    # 将新创建的metrics移动到正确的设备
                    for metric in self.group_metrics[group_name].values():
                        metric.to(preds.device)

                # 选择该组样本的掩码 - 只选择真实标签属于该组的样本
                group_sample_mask = torch.tensor(
                    [g == group_name for g in group_names], device=preds.device
                )
                if group_sample_mask.sum() == 0:
                    continue
                
                # 关键修改：对选中的样本，使用完整的候选集进行评测
                # 而不是只使用该组的候选子集
                group_sample_indices = torch.nonzero(group_sample_mask, as_tuple=False).flatten()
                
                for sample_idx in group_sample_indices:
                    # 为每个属于该组的样本，使用完整的候选集
                    sample_start = sample_idx * num_candidates
                    sample_end = (sample_idx + 1) * num_candidates
                    
                    sample_preds = preds[sample_start:sample_end]
                    sample_target = target[sample_start:sample_end]
                    sample_indexes = torch.full((num_candidates,), sample_idx.item(), device=preds.device)
                    
                    for _, metric_object in self.group_metrics[group_name].items():
                        metric_object.update(
                            sample_preds,
                            sample_target.to(sample_preds.device),
                            batch_size=1,
                        )

    def summarize_group_metrics(self) -> Dict[str, Dict[str, float]]:
        """
        返回分组评测的当前结果字典: {group_name: {metric_name: value}}
        注意：这不会重置内部状态。
        """
        results: Dict[str, Dict[str, float]] = {}
        for group_name, metrics in self.group_metrics.items():
            results[group_name] = {
                metric_name: float(metric.compute().detach().cpu().item())
                for metric_name, metric in metrics.items()
            }
        return results
