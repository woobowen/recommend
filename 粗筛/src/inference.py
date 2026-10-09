from typing import Any, Dict

import hydra
import rootutils
from omegaconf import DictConfig

rootutils.setup_root(__file__, indicator="./GRID", pythonpath=True)

from src.utils import RankedLogger, extras
from src.utils.custom_hydra_resolvers import *
from src.utils.launcher_utils import pipeline_launcher

command_line_logger = RankedLogger(__name__, rank_zero_only=True)
import os 

import torch
import torch.nn.functional as F

os.environ['CUDA_VISIBLE_DEVICES'] = '1,3'

def inference(cfg: DictConfig) -> Dict[str, Any]:   
    """Runs inference using a pre-trained model.

    :param cfg: A DictConfig configuration composed by Hydra.
    :return: A dict with all instantiated objects.
    """

    with pipeline_launcher(cfg) as pipeline_modules:
        command_line_logger.info("Starting inference!")
        ckpt_path = pipeline_modules.cfg.get("ckpt_path", None)
        if not ckpt_path:
            command_line_logger.warning(
                "No ckpt_path was provided. If using a model you trained, this is mandatory. Only leave ckpt_path=None if using a pre-trained model."
            )

        # Run predict() for inference without labels (generates predictions)
        predict_outputs = pipeline_modules.trainer.predict(
            model=pipeline_modules.model,
            datamodule=pipeline_modules.datamodule,
            ckpt_path=ckpt_path,
        )

        # 计算并记录旋转矩阵相似度：仅针对预测的target（top-1），并在末尾给出全数据集的平均值
        try:
            model = pipeline_modules.model
            num_hierarchies = getattr(model, 'num_hierarchies', None)
            num_embeddings_per_hierarchy = getattr(model, 'num_embeddings_per_hierarchy', None)
            get_table = getattr(model, 'get_embedding_table', None)
            add_offset = getattr(model, '_add_repeating_offset_to_rows', None)

            if num_hierarchies is not None and num_embeddings_per_hierarchy is not None and callable(get_table) and callable(add_offset):
                table = get_table('encoder')
                total_samples = 0
                global_sum = 0.0  # 按样本聚合后的T-相似度之和
                global_cnt = 0    # 样本计数
                sum_vec = None    # 归一化后的T向量之和（用于平均两两余弦相似度）
                cnt_vec = 0       # 有效T的数量
                for batch_out in predict_outputs:
                    # OneKeyPerPredictionOutput(keys=[...], predictions=tensor)
                    keys = getattr(batch_out, 'keys', None)
                    preds = getattr(batch_out, 'predictions', None)
                    if preds is None:
                        continue
                    # preds shape: (batch, top_k, num_hierarchies) 或 (batch, num_hierarchies)
                    if isinstance(preds, torch.Tensor):
                        preds = preds.detach().cpu()
                    else:
                        preds = torch.tensor(preds)

                    batch_size = preds.size(0)

                    for i in range(batch_size):
                        cand_ids = preds[i]  # (top_k, H) 或 (H,)
                        # 只取预测的 target（默认取 top-1）
                        if cand_ids.dim() == 1:
                            ids_vec = cand_ids  # (H,)
                        else:
                            ids_vec = cand_ids[0]  # (H,)

                        # 映射到embedding
                        shifted = add_offset(
                            input_sids=ids_vec.unsqueeze(0),
                            codebook_size=num_embeddings_per_hierarchy,
                            num_hierarchies=num_hierarchies,
                            attention_mask=None,
                        )
                        V = table(shifted.long()).squeeze(0)  # (H, D)
                        # 列向量归一化，与你Notebook一致（按向量归一化）
                        V = F.normalize(V, p=2, dim=-1)  # (H, D)

                        # 构造 X, Y 并求 T: Y = T X => T = Y @ pinv(X)
                        if V.size(0) > 1:
                            X_rows = V[:-1, :]       # (H-1, D)
                            Y_rows = V[ 1:, :]       # (H-1, D)
                            X = X_rows.T             # (D, H-1)
                            Y = Y_rows.T             # (D, H-1)
                            X_pinv = torch.linalg.pinv(X)   # (H-1, D)
                            T = Y @ X_pinv                 # (D, D)

                            # 用 T 预测下一层：v_pred_{i+1} = T @ v_i
                            v_i = X  # (D, H-1)
                            v_pred_next = (T @ v_i)   # (D, H-1)
                            v_true_next = Y           # (D, H-1)
                            # 余弦相似度（逐列）
                            v_pred_next = F.normalize(v_pred_next.T, p=2, dim=-1)  # (H-1, D)
                            v_true_next = F.normalize(v_true_next.T, p=2, dim=-1)  # (H-1, D)
                            sims = F.cosine_similarity(v_pred_next, v_true_next, dim=-1)  # (H-1,)
                            sim_T = sims.mean().item()
                            # 收集用于两两T相似度的向量
                            t_vec = T.reshape(-1)
                            t_vec = t_vec / (t_vec.norm() + 1e-12)
                            if sum_vec is None:
                                sum_vec = t_vec.clone()
                            else:
                                sum_vec = sum_vec + t_vec
                            cnt_vec += 1
                        else:
                            sim_T = 0.0

                        # 累计到全局（按样本）
                        global_sum += float(sim_T)
                        global_cnt += 1

                        # 打印当前 sample 的结果（T 的余弦相似度平均）
                        if keys is not None and i < len(keys):
                            uid = keys[i]
                            command_line_logger.info(f"RotationSim | user_id={uid} | T_top1_avg={sim_T:.4f}")
                        else:
                            command_line_logger.info(f"RotationSim | sample_index={total_samples+i} | T_top1_avg={sim_T:.4f}")

                    total_samples += batch_size

                dataset_avg = (global_sum / global_cnt) if global_cnt > 0 else 0.0
                # 两两T余弦相似度的平均值（不显式存所有T，使用单位向量求和恒等式）：
                # 若 u_i = vec(T_i)/||vec(T_i)||，S = sum u_i，则 mean_{i<j}(u_i·u_j) = (||S||^2 - N) / (N*(N-1))
                if cnt_vec > 1:
                    S_norm_sq = float((sum_vec @ sum_vec).item())
                    pairwise_avg = (S_norm_sq - cnt_vec) / (cnt_vec * (cnt_vec - 1))
                else:
                    pairwise_avg = 0.0
                command_line_logger.info(
                    f"RotationSim | T_dataset_avg={dataset_avg:.4f} | T_pairwise_avg={pairwise_avg:.4f} | samples={global_cnt}"
                )
            else:
                command_line_logger.warning("RotationSim | 模型不暴露所需接口，跳过相似度计算")
        except Exception as e:
            command_line_logger.warning(f"RotationSim | 计算失败: {e}")

        # Run test() to compute NDCG metrics using test dataloader with labels
        # pipeline_modules.trainer.test(
        #     model=pipeline_modules.model,
        #     datamodule=pipeline_modules.datamodule,
        #     ckpt_path=ckpt_path,
        # )


@hydra.main(version_base="1.3", config_path="../configs", config_name="inference.yaml")
def main(cfg: DictConfig) -> None:
    """Main entry point for inference.

    :param cfg: DictConfig configuration composed by Hydra.
    """
    # apply extra utilities
    extras(cfg)

    # run inference
    inference(cfg)


if __name__ == "__main__":
    main()
