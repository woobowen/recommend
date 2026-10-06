# TIGER reproduction bundle

This directory contains the reproducibility bundle for the Beauty experiment run on the GRID implementation.

## Included

- `src/`, `configs/`, `requirements.txt`: GRID source and experiment configuration. The source and configs were not modified for the run.
- `artifacts/beauty/beauty_item_embeddings.pt`: precomputed item embeddings.
- `artifacts/beauty/semantic_ids.pt`: Beauty Semantic IDs used by TIGER (`shape=[5,12101]`).
- `experiments/04_tiger_smoke/`: single-GPU smoke-test config, logs, and metrics.
- `experiments/05_tiger_full/`: three-GPU Beauty config, logs, and metrics.

Raw Amazon-P5 TFRecord data and the Python environment are not committed. Download the data from the link in the repository-level README and prepare `training/`, `evaluation/`, and `testing/` subdirectories.

## Reproduce

From this directory, replace `/path/to/data/amazon_data/beauty` with the prepared data directory:

```bash
python -m src.train \
  experiment=tiger_train_flat \
  data_dir=/path/to/data/amazon_data/beauty \
  semantic_id_path=artifacts/beauty/semantic_ids.pt \
  num_hierarchies=5 seed=42 train=true test=false \
  trainer.accelerator=gpu trainer.devices=3 \
  paths.root_dir=$PWD paths.data_dir=/path/to/data/amazon_data/beauty \
  hydra.run.dir=experiments/reproduction_full \
  ++should_skip_retry=True
```

For a one-GPU smoke test, use `trainer.devices=1`, `trainer.max_steps=20`, `trainer.val_check_interval=10`, and the small dataloader overrides shown in the repository-level README.

The uploaded full run used sequence length 120, per-device batch size 32, gradient accumulation 16, FP32, and validation every 1600 steps. Early stopping ended it at approximately step 5199; the best recorded checkpoint was step 4200. Checkpoint binaries are excluded because GitHub rejects files larger than 100 MB.

Recorded validation metrics: Recall@5 0.0429, Recall@10 0.0587, NDCG@5 0.0310, NDCG@10 0.0361. These are GRID validation metrics and are not a strict reproduction of the original paper's test-set protocol.
