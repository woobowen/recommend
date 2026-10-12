# All SID Experiments

本目录保存本地「学长基线实验」的代码、配置、shuffle/MLP/parallel 文件、Amazon 数据和 Qwen 权重。所有 `codex` 目录均按要求排除。
原有 [GRID baseline](../GRID%20baseline/) 和 [Baseline Results](../Baseline%20Results/) 保留，提供已完成的 Qwen → RQ-KMeans → TIGER 基线及论文对比。

## 实验状态

- 本目录的 `src/` 和 `configs/` 是已修复的普通 TIGER 基线代码。
- `shuffle/`：原始测试时 SID 嵌入顺序打乱模型及依赖。
- `mlp/`：原始 SID 嵌入拼接后通过 MLP 聚合的模型及依赖。
- `parallel/`：服务器中名为 parallel 的源文件。它与 shuffle 文件逐字节相同，仍使用自回归解码；不能据此认定真正并行 SID 实验已经实现。共用文件来自父目录或 `shuffle/`，本目录不是独立可运行项目。
- 新变体入口、模型路由与运行兼容性尚未全部修复或验证。配置模板默认目标可能仍为普通 TIGER；上传不代表各变体已经跑通。

## 下载数据和权重

大文件通过 Git LFS 保存。浏览器 ZIP 中可能只有指针，请使用：

```bash
git lfs install
GIT_LFS_SKIP_SMUDGE=1 git clone --branch SID-Order-Invariance https://github.com/woobowen/recommend.git
cd recommend
git lfs pull
cd "All SID Experiments"
```

数据在 `original_assets/data/amazon_data/{beauty,sports,toys}`。Qwen 模型在 `models/Qwen2.5-7B-Instruct/`。
GitHub 的单文件限制要求把原四个 safetensors 拆为十二个传输分片；内容保持不变，校验信息见 `weight_parts/manifest.json`。
需要生成文本向量时，在 `All SID Experiments` 目录运行以下 Bash 命令还原（额外需要约 14.2 GiB 空间）：

```bash
model_dir="models/Qwen2.5-7B-Instruct"
for name in model-00001-of-00004 model-00002-of-00004 model-00003-of-00004 model-00004-of-00004; do
  cat "$model_dir/weight_parts/$name.safetensors.part001" \
      "$model_dir/weight_parts/$name.safetensors.part002" \
      "$model_dir/weight_parts/$name.safetensors.part003" > "$model_dir/$name.safetensors"
done
```

仅复用已有 SID 训练/测试 TIGER，无需加载 Qwen。已完成基线的向量、SID、checkpoint 和历史输出保存在 `../GRID baseline/experiments/`。
历史配置中的服务器绝对路径需要按目标机器调整。`codex` 中的辅助入口、说明和配置不在此分支中。

---

以下为本地原 README，介绍上游 GRID；其中默认流程和路径不代表本分支各扩展实验均已验证。

# Generative Recommendation with Semantic IDs (GRID)
[![PyTorch](https://img.shields.io/badge/pytorch-2.0%2B-red)](https://pytorch.org/)
[![Hydra](https://img.shields.io/badge/config-hydra-89b8cd)](https://hydra.cc/)
[![Lightning](https://img.shields.io/badge/pytorch-lightning-792ee5)](https://lightning.ai/)
[![arXiv](https://img.shields.io/badge/arXiv-2507.22224-b31b1b.svg)](https://arxiv.org/abs/2507.22224)


**GRID** (Generative Recommendation with Semantic IDs) is a state-of-the-art framework for generative recommendation systems using semantic IDs, developed by a group of scientists and engineers from [Snap Research](https://research.snap.com/team/user-modeling-and-personalization.html). This project implements novel approaches for learning semantic IDs from text embedding and generating recommendations through transformer-based generative models.

## 🚀 Overview

GRID facilitates generative recommendation three overarching steps:

- **Embedding Generation with LLMs**: Converting item text into embeddings using any LLMs available on Huggingface. 
- **Semantic ID Learning**: Converting item embedding into hierarchical semantic IDs using Residual Quantization techniques such as RQ-KMeans, RQ-VAE, RVQ. 
- **Generative Recommendations**: Using transformer architectures to generate recommendation sequences as semantic ID tokens. 


## 📦 Installation

### Prerequisites
- Python 3.10+
- CUDA-compatible GPU (recommended)

### Setup Environment

```bash
# Clone the repository
git clone https://github.com/snap-research/GRID.git
cd GRID

# Install dependencies
pip install -r requirements.txt
```

## 🎯 Quick Start

### 1. Data Preparation

Prepare your dataset in the expected format:
```
data/
├── train/       # training sequence of user history 
├── validation/  # validation sequence of user history 
├── test/        # testing sequence of user history 
└── items/       # text of all items in the dataset
```

We provide pre-processed Amazon data explored in the [P5 paper](https://arxiv.org/abs/2203.13366) [4]. The data can be downloaded from this [google drive link](https://drive.google.com/file/d/1B5_q_MT3GYxmHLrMK0-lAqgpbAuikKEz/view?usp=sharing).

### 2. Embedding Generation with LLMs

Generate embeddings from LLMs, which later will be transformed into semantic IDs. 

```bash
python -m src.inference experiment=sem_embeds_inference_flat data_dir=data/amazon_data/beauty # avaiable data includes 'beauty', 'sports', and 'toys'
```

### 3. Train and Generate Semantic IDs

Learn semantic ID centroids for embeddings generated in step 2:

```bash
python -m src.train experiment=rkmeans_train_flat \
    data_dir=data/amazon_data/beauty \
    embedding_path=<output_path_from_step_2>/merged_predictions_tensor.pt \ # this can be found in the log dirs in step2
    embedding_dim=2048 \ # the model dimension of the LLMs you use in step 2. 2048 for flan-t5-xl as used in this example.
    num_hierarchies=3 \  # we train 3 codebooks
    codebook_width=256 \ # each codebook has 256 rows of centroids  
```

Generate SIDs:

```bash
python -m src.inference experiment=rkmeans_inference_flat \
    data_dir=data/amazon_data/beauty \
    embedding_path=<output_path_from_step_2>/merged_predictions_tensor.pt \ 
    embedding_dim=2048 \ 
    num_hierarchies=3 \  
    codebook_width=256 \ 
    ckpt_path=<the_checkpoint_you_just_get_above> # this can be found in the log dir for training SIDs
```


### 4. Train Generative Recommendation Model with Semantic IDs

Train the recommendation model using the learned semantic IDs:

```bash
python -m src.train experiment=tiger_train_flat \
    data_dir=data/amazon_data/beauty \ 
    semantic_id_path=<output_path_from_step_3>/pickle/merged_predictions_tensor.pt \
    num_hierarchies=4 # Please note that we add 1 for num_hierarchies because in the previous step we appended one additional digit to de-duplicate the semantic IDs we generate.
```

### 4. Generate Recommendations

Run inference to generate recommendations:

```bash
python -m src.inference experiment=tiger_inference_flat \
    data_dir=data/amazon_data/beauty \ 
    semantic_id_path=<output_path_from_step_3>/pickle/merged_predictions_tensor.pt \
    ckpt_path=<the_checkpoint_you_just_get_above> \ # this can be found in the log dir for training GR models
    num_hierarchies=4 \ # Please note that we add 1 for num_hierarchies because in the previous step we appended one additional digit to de-duplicate the semantic IDs we generate.
```

## Supported Models:

### Semantic ID:

1. Residual K-means proposed in One-Rec [2]
2. Residual Vector Quantization
3. Residual Quantization with Variational Autoencoder [3]

### Generative Recommendation:

1. TIGER [1]

## 📚 Citation

If you use GRID in your research, please cite:

```bibtex
@inproceedings{grid,
  title     = {Generative Recommendation with Semantic IDs: A Practitioner's Handbook},
  author    = {Ju, Clark Mingxuan and Collins, Liam and Neves, Leonardo and Kumar, Bhuvesh and Wang, Louis Yufeng and Zhao, Tong and Shah, Neil},
  booktitle = {Proceedings of the 34th ACM International Conference on Information and Knowledge Management (CIKM)},
  year      = {2025}
}
```

## 🤝 Acknowledgments

- Built with [PyTorch](https://pytorch.org/) and [PyTorch Lightning](https://lightning.ai/)
- Configuration management by [Hydra](https://hydra.cc/)
- Inspired by recent advances in generative AI and recommendation systems
- Part of this repo is built on top of https://github.com/ashleve/lightning-hydra-template

## 📞 Contact

For questions and support:
- Create an issue on GitHub
- Contact the development team: Clark Mingxuan Ju (mju@snap.com), Liam Collins (lcollins2@snap.com), and Leonardo Neves (lneves@snap.com).

## Bibliography 

[1] Rajput, Shashank, et al. "Recommender systems with generative retrieval." Advances in Neural Information Processing Systems 36 (2023): 10299-10315.

[2] Deng, Jiaxin, et al. "Onerec: Unifying retrieve and rank with generative recommender and iterative preference alignment." arXiv preprint arXiv:2502.18965 (2025).

[3] Lee, Doyup, et al. "Autoregressive image generation using residual quantization." Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2022.

[4] Geng, Shijie, et al. "Recommendation as language processing (rlp): A unified pretrain, personalized prompt & predict paradigm (p5)." Proceedings of the 16th ACM conference on recommender systems. 2022.
