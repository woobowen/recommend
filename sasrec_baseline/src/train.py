#!/usr/bin/env python3
"""
SASRec训练脚本
与生成式推荐系统保持相同的数据处理和评测方式
"""

import os
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(Path(__file__).parent.parent))

import hydra
import lightning as L
from omegaconf import DictConfig

from src.utils import (
    RankedLogger,
    extras,
    get_metric_value,
    instantiate_callbacks,
    instantiate_loggers,
    log_hyperparameters,
    task_wrapper,
)

log = RankedLogger(__name__, rank_zero_only=True)


@task_wrapper
def train(cfg: DictConfig) -> tuple[dict, dict]:
    """训练SASRec模型
    
    Args:
        cfg: DictConfig配置对象
        
    Returns:
        包含指标的元组
    """
    
    # 设置种子以确保可重现性
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    log.info(f"实例化数据模块 <{cfg.data_loading.datamodule._target_}>")
    datamodule: L.LightningDataModule = hydra.utils.instantiate(cfg.data_loading.datamodule)

    log.info(f"实例化模型 <{cfg.model._target_}>")
    
    # 从semantic_id_map推断num_items
    semantic_id_map = cfg.data_loading.train_dataloader_config.dataloader.dataset_config.semantic_id_map.sequence_data
    if hasattr(semantic_id_map, '_target_'):
        # 如果是torch.load，需要实际加载来获取信息
        import torch
        from src.utils.file_utils import open_local_or_remote
        with open_local_or_remote(semantic_id_map._args_[0]._args_[0], semantic_id_map._args_[0].mode) as f:
            codebooks = torch.load(f)
        num_items = codebooks.max().item() + 1
    else:
        num_items = semantic_id_map.max().item() + 1
    
    # 更新模型配置中的num_items
    cfg.model.num_items = num_items
    log.info(f"推断得到的item数量: {num_items}")
    
    model: L.LightningModule = hydra.utils.instantiate(cfg.model)

    log.info("实例化回调函数")
    callbacks: list[L.Callback] = instantiate_callbacks(cfg.get("callbacks"))

    log.info("实例化日志记录器")
    logger: list[L.pytorch.loggers.Logger] = instantiate_loggers(cfg.get("logger"))

    log.info(f"实例化训练器 <{cfg.trainer._target_}>")
    trainer: L.Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("记录超参数!")
        log_hyperparameters(object_dict)

    if cfg.get("train"):
        log.info("开始训练!")
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))

    train_metrics = trainer.callback_metrics

    if cfg.get("test"):
        log.info("开始测试!")
        ckpt_path = trainer.checkpoint_callback.best_model_path
        if ckpt_path == "":
            log.warning("找不到最佳检查点。使用当前权重进行测试...")
            ckpt_path = None
        trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
        log.info(f"最佳检查点路径:\n{ckpt_path}")

    test_metrics = trainer.callback_metrics

    # 合并训练和测试指标
    metric_dict = {**train_metrics, **test_metrics}

    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="train")
def main(cfg: DictConfig) -> None:
    """主函数"""
    
    # 应用额外的实用程序
    # (例如询问确认、强制执行标签、打印配置)
    extras(cfg)

    # 训练模型
    metric_dict, _ = train(cfg)

    # 安全地检索指标值以进行hydra-based超参数优化
    metric_value = get_metric_value(
        metric_dict=metric_dict, metric_name=cfg.get("optimized_metric")
    )

    # 返回优化指标
    return metric_value


if __name__ == "__main__":
    main()