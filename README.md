# TIGER / GRID 复现包

本目录包含基于 [Snap Research GRID](https://github.com/snap-research/GRID) 的 TIGER 生成式推荐复现材料。已验证 Beauty 数据集的单卡 smoke test 和三卡训练可以启动、保存 checkpoint，并稳定完成验证。

## 目录

- `GRID/src/`：GRID 源代码（本次训练未修改）。
- `GRID/configs/`：Hydra 配置，包括 `tiger_train_flat`。
- `GRID/artifacts/beauty/`：可直接使用的 Beauty item embedding 和 Semantic ID；`semantic_ids.pt` 的形状为 `[5, 12101]`，因此训练使用 `num_hierarchies=5`。
- `GRID/experiments/`：已完成的 smoke/full 配置、日志、指标和复现实验记录。
- `GRID/REPRODUCTION.md`：更短的复现说明。

## 环境

建议使用 Linux、Python 3.10+ 和 CUDA GPU。进入 `GRID` 后安装依赖：

```bash
cd GRID
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`requirements.txt` 来自 GRID；PyTorch 请按机器 CUDA 版本安装兼容版本。训练至少需要 1 张 GPU，复现已记录的全量配置需要 3 张 GPU。

## 数据

代码需要 Amazon-P5 Beauty 的 TFRecord 数据，目录必须包含：

```text
data/amazon_data/beauty/
├── training/
├── evaluation/
└── testing/
```

可从 GRID README 指向的 [Google Drive 数据包](https://drive.google.com/file/d/1B5_q_MT3GYxmHLrMK0-lAqgpbAuikKEz/view?usp=sharing)获取并解压。假设数据最终路径为 `/path/to/data/amazon_data/beauty`。原始数据（约 1.3 GB）没有复制进 GitHub 仓库；仓库内的两个 `.pt` 文件已经足够跳过 embedding 和 semantic-ID 生成阶段。

## 先做单卡检查

```bash
cd /path/to/recommend/GRID
python -m src.train \
  experiment=tiger_train_flat \
  data_dir=/path/to/data/amazon_data/beauty \
  semantic_id_path=artifacts/beauty/semantic_ids.pt \
  num_hierarchies=5 \
  seed=42 train=true test=false \
  trainer.accelerator=gpu trainer.devices=1 trainer.max_steps=20 \
  trainer.val_check_interval=10 \
  +trainer.limit_val_batches=1 \
  data_loading.train_dataloader_config.dataloader.batch_size_per_device=2 \
  data_loading.train_dataloader_config.dataloader.num_workers=0 \
  data_loading.train_dataloader_config.dataloader.timeout=0 \
  data_loading.val_dataloader_config.dataloader.batch_size_per_device=2 \
  data_loading.val_dataloader_config.dataloader.num_workers=0 \
  data_loading.val_dataloader_config.dataloader.timeout=0 \
  paths.root_dir=$PWD paths.data_dir=/path/to/data/amazon_data/beauty \
  hydra.run.dir=experiments/reproduction_smoke \
  ++should_skip_retry=True
```

能完成若干 step、验证并在 `experiments/reproduction_smoke/` 下生成 checkpoint，即表示环境和数据路径正确。

## 三卡 Beauty 训练

```bash
cd /path/to/recommend/GRID
CUDA_VISIBLE_DEVICES=0,1,2 python -m src.train \
  experiment=tiger_train_flat \
  data_dir=/path/to/data/amazon_data/beauty \
  semantic_id_path=artifacts/beauty/semantic_ids.pt \
  num_hierarchies=5 seed=42 train=true test=false \
  trainer.accelerator=gpu trainer.devices=3 \
  paths.root_dir=$PWD paths.data_dir=/path/to/data/amazon_data/beauty \
  hydra.run.dir=experiments/reproduction_full \
  ++should_skip_retry=True
```

配置默认值为 sequence length 120、每卡 batch size 32、gradient accumulation 16、FP32、最多 320000 steps、每 1600 steps 验证。仓库记录的运行因 early stopping 在约 5199 step 结束；最佳 checkpoint 在 step 4200。checkpoint 二进制未上传（GitHub 单文件 100 MB 限制），重新运行即可生成。

已记录的验证指标（GRID validation split）：Recall@5=0.0429、Recall@10=0.0587、NDCG@5=0.0310、NDCG@10=0.0361。由于 GRID 使用 RQ-KMeans Semantic ID、验证集划分和当前优化器配置，这些数字不等同于论文原始 test-set 设定。

## 复现检查结论

- 源代码和配置已随包提交；训练期间没有修改 `GRID/src` 或 `GRID/configs`。
- 两个预计算 `.pt` artifact 已提交并可直接被命令读取。
- 必需的原始 TFRecord 数据和 Python 环境需要按上面步骤在复现机器准备。
- `experiments/` 中的绝对路径只属于历史日志，不会被新的命令使用；命令中的 `/path/to/...` 请替换成实际路径。

## 许可证与引用

GRID 原项目许可证和第三方依赖声明见 `GRID/LICENSE`、`GRID/notices.txt`。论文：Rajput et al., *Recommender Systems with Generative Retrieval*, NeurIPS 2023；GRID 项目论文和引用信息见 `GRID/README.md`。
