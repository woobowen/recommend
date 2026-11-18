# GRID baseline：Qwen → RQ-KMeans → TIGER

这是 2026-10-10/11 已完成训练与测试的学长 TIGER 基线，供后续扩展实验复用。普通 TIGER 是本目录的下游模型，未包含 shuffle、MLP 或 parallel 实验目录。

三个数据集均完成最佳 checkpoint 的测试，结果及论文对比见 [Baseline Results](../Baseline%20Results/README.md)。上传包含源代码、完整 Amazon 数据、Qwen 预训练模型配置与权重传输分片，以及本次全部阶段输出。

## 下载与大文件

请安装 Git LFS。浏览器下载 ZIP 通常只有 LFS 指针，不是数据和权重的实际内容。

```bash
git lfs install
GIT_LFS_SKIP_SMUDGE=1 git clone --branch main https://github.com/woobowen/recommend.git
cd recommend
git lfs pull --include="GRID baseline/**"
cd "GRID baseline"
```

GitHub 此仓库的 LFS 单文件上限为 2 GiB，原始 Qwen 四个 safetensors 文件各超过该上限。传输时拆成 12 个不超过 1.5 GiB 的分片，内容没有量化或改写。若需要重新生成文本向量，先合并：

```bash
python codex/restore_qwen_weights.py
```

合并脚本会校验每个分片和还原文件的 SHA-256，保留传输分片，并拒绝覆盖内容不一致的已有文件。还原需要额外约 14.2 GiB 空间。只验证已下载分片时使用 `--verify-only`。

如果只使用保存的 SID 重训 TIGER 或测试最佳 checkpoint，无需加载 Qwen，也无需合并权重。

## 文件组成

| 路径 | 用途 |
|---|---|
| `src/`、`configs/` | 本次服务器实际使用的学长 GRID/TIGER 代码与配置，包括已修复的根目录、测试接口等问题 |
| `_运行/` | 本次 Qwen 向量生成、GRID 阶段入口、后台流水线与显式 TIGER job 入口 |
| `original_assets/data/amazon_data/` | Beauty、Sports、Toys 的完整 1,902 个 TFRecord 分片；保持原数据划分 |
| `models/Qwen2.5-7B-Instruct/` | 模型配置、tokenizer、许可证、原始模型权重的传输分片 |
| `experiments/` | 三个数据集的 Qwen 向量、码本 checkpoint、新 SID、训练/测试 checkpoint、原始日志、CSV、Hydra 配置和结果 |
| `reconfigurations/val1600_20261010_112259/` | 每 1,600 batch 验证的变更记录、显式 job、Beauty 原最佳 checkpoint 保护备份 |
| `launches/` | 本次启动记录与监督进程日志 |
| `codex/` | 权重还原、重测工具、来源清单、环境版本与归档说明 |

`_运行/run_baseline.py` 是保留的已废弃短测入口，会直接退出；请使用 `run_pipeline.py` 或下述入口。

Python 环境未打包；保存了服务器 Python 3.10.20 的依赖版本 `codex/environment-freeze.txt`，以及原始 `requirements.txt`。实验使用 Linux、PyTorch 2.6.0+cu124、Transformers 4.47.0、Lightning 2.5.0、torchmetrics 1.0.3；GPU/CUDA 包需按目标机器安装。freeze 中的 conda-pack 本地构建来源用于记录，不能直接作为其他机器上的下载路径。

## 重测已保存的最佳 checkpoint

先准备与记录兼容的 Python/CUDA 环境，再选择一张空闲 GPU。新工具会将归档配置中的旧服务器根路径替换成当前目录，并输出到新目录，不重新训练：

```bash
python codex/test_saved_checkpoint.py --dataset beauty --gpu 0 --output "$PWD/experiments/beauty_retest"
python codex/test_saved_checkpoint.py --dataset sports --gpu 1 --output "$PWD/experiments/sports_retest"
python codex/test_saved_checkpoint.py --dataset toys --gpu 2 --output "$PWD/experiments/toys_retest"
```

三个命令是分别执行的示例，使用前确认对应 GPU 空闲。新工具已经做静态核验和模型加载验证，未为了发布重新运行整份测试集。

| 数据集 | 本次新 SID 所在目录 | 最佳推荐 checkpoint |
|---|---|---|
| Beauty | `experiments/beauty_rkmeans_qwen_20261010_012253/semantic_ids.pt` | `tiger_train_test/checkpoints/checkpoint_epoch=000_step=012500.ckpt` |
| Sports | `experiments/sports_rkmeans_qwen_20261010_093035/semantic_ids.pt` | `tiger_train_test_val1600_20261010_112259/checkpoints/checkpoint_epoch=000_step=009500.ckpt` |
| Toys | `experiments/toys_rkmeans_qwen_20261010_093036/semantic_ids.pt` | `tiger_train_test_val1600_20261010_112259/checkpoints/checkpoint_epoch=000_step=006200.ckpt` |

表中 checkpoint 路径位于各自 SID 同级实验目录内。每个目录里也保留了频率调整前的输出；**最终测试以 Baseline Results 中明确指定的最佳 checkpoint 为准**。

## 复用 SID，从零训练普通 TIGER

下面以 Toys 为例，从项目根目录运行，所有参数随命令传递，避免覆盖原始结果：

```bash
CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=4 TF_CPP_MIN_LOG_LEVEL=3 \
python _运行/grid_stage.py \
  experiment=tiger_train_flat \
  "data_dir=$PWD/original_assets/data/amazon_data/toys" \
  "semantic_id_path=$PWD/experiments/toys_rkmeans_qwen_20261010_093036/semantic_ids.pt" \
  "paths.root_dir=$PWD" \
  num_hierarchies=4 trainer.devices=1 ++trainer.strategy=auto \
  trainer.precision=32-true train=true test=true ckpt_path=null \
  ++callbacks.restart_job=null extras.print_config=false \
  trainer.val_check_interval=1600 trainer.log_every_n_steps=10 \
  ++data_loading.train_dataloader_config.dataloader.num_workers=0 \
  ++data_loading.train_dataloader_config.dataloader.timeout=0 \
  ++data_loading.train_dataloader_config.dataloader.persistent_workers=false \
  ++data_loading.val_dataloader_config.dataloader.num_workers=0 \
  ++data_loading.val_dataloader_config.dataloader.timeout=0 \
  ++data_loading.val_dataloader_config.dataloader.persistent_workers=false \
  ++data_loading.test_dataloader_config.dataloader.num_workers=0 \
  ++data_loading.test_dataloader_config.dataloader.timeout=0 \
  ++data_loading.test_dataloader_config.dataloader.persistent_workers=false \
  "hydra.run.dir=$PWD/experiments/toys_new_baseline" \
  ++trainer.enable_progress_bar=false
```

Beauty/Sports 相应替换 dataset、SID 和输出目录。该命令不加载已有 TIGER checkpoint。本次归档的 Beauty 指标来自“原最佳 12,500 步 checkpoint 恢复全状态后继续训练”的流程，不能把上面的从零训练命令视为完全相同的历史过程。

## 从文本向量开始重做全流程

还原 Qwen 权重后，至少选择两张空闲 GPU：

```bash
python _运行/run_pipeline.py --dataset beauty --method rkmeans --gpus 0,1
```

此原始流水线会新建独立实验目录，重新生成 Qwen 向量、训练码本、生成 SID、训练并测试 TIGER。Qwen 是冻结的预训练文本编码器，非学长训练的推荐模型。文本向量使用长度 128 的商品文本、最后层隐藏状态与均值聚合；码本为 3 层 × 256，SID 加 1 位去重，共 4 位。

注意：原流水线读取训练模板，默认验证间隔仍为 10,000 batch；本次后期最终结果采用 1,600 batch。需要保持后期训练设置时，应使用上一节明确传入 `trainer.val_check_interval=1600` 的训练命令。

## 归档来源与校验

源项目是服务器 `/home/jcyin/GRID_wmh_Qwen_TIGER_20261009`。历史 JSON/YAML、日志和 checkpoint 中的原路径、PID、时间保持不变，以便审计；原 job.json 不适合在不同机器直接启动。新测试工具和示例命令使用当前目录。

`codex/server_manifest.json` 记录服务器输出与代码的 SHA-256，`codex/dataset_source_manifest.json` 记录数据分片的服务器来源校验，`codex/weights_manifest.json` 记录原始 Qwen 文件与传输分片的校验。来源说明 `codex/Qwen来源记录.json` 指向学长服务器的 ModelScope Qwen 预训练模型。

数据文件未发现学长额外改写证据；已有官方对比确认 1,902 个分片的路径、大小及 CRC32 相同，但官方逐文件 SHA-256 核对尚未完成。数据中的原 embedding 字段保留，**本次新向量生成忽略该字段，只读取商品文本**。

原作者代码许可证见 `LICENSE`，第三方声明见 `notices.txt`，Qwen 模型许可证位于模型目录。后续新生成的辅助脚本和记录请放入 `codex/`，实验输出放入新的 `experiments/<run>/`。
