#!/usr/bin/env python3
"""
SASRec测试脚本
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
    task_wrapper,
)

log = RankedLogger(__name__, rank_zero_only=True)


@task_wrapper
def test(cfg: DictConfig) -> dict:
    """测试SASRec模型
    
    Args:
        cfg: DictConfig配置对象
        
    Returns:
        测试指标字典
    """
    
    # 设置种子以确保可重现性
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    log.info(f"实例化数据模块 <{cfg.data_loading.datamodule._target_}>")
    datamodule: L.LightningDataModule = hydra.utils.instantiate(cfg.data_loading.datamodule)

    log.info(f"从检查点加载模型: {cfg.ckpt_path}")
    
    # 从semantic_id_map推断num_items
    semantic_id_map = cfg.data_loading.test_dataloader_config.dataloader.dataset_config.semantic_id_map.sequence_data
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
    
    log.info(f"实例化训练器 <{cfg.trainer._target_}>")
    trainer: L.Trainer = hydra.utils.instantiate(cfg.trainer)

    log.info("开始测试!")
    trainer.test(model=model, datamodule=datamodule, ckpt_path=cfg.ckpt_path)

    test_metrics = trainer.callback_metrics
    
    log.info("测试完成!")
    for key, value in test_metrics.items():
        log.info(f"{key}: {value}")

    return test_metrics


@hydra.main(version_base="1.3", config_path="../configs", config_name="train")
def main(cfg: DictConfig) -> None:
    """主函数"""
    
    # 应用额外的实用程序
    extras(cfg)

    # 测试模型
    test_metrics = test(cfg)
    
    return test_metrics


if __name__ == "__main__":
    main()
