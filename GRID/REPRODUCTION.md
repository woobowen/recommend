# TIGER reproduction bundle

This directory contains the reproducibility bundle for the Beauty experiment run on the GRID implementation.

## Included

- `src/`, `configs/`, `requirements.txt`: GRID source and experiment configuration.
- `artifacts/beauty/beauty_item_embeddings.pt`: precomputed item embeddings used by the run.
- `artifacts/beauty/semantic_ids.pt`: Beauty Semantic IDs used by TIGER (`shape=[5,12101]`).
- `experiments/04_tiger_smoke/`: single GPU smoke-test configs, logs, and metrics.
- `experiments/05_tiger_full/`: three-GPU full Beauty training configs, logs, and metrics.

Raw Amazon-P5 TFRecord data and the Python environment are not committed. Download/prepare the GRID Amazon-P5 data and install `requirements.txt` before running.

## Paths

The commands below assume the repository is checked out as `/path/to/recommend/GRID` and the data is at `/path/to/data/amazon_data/beauty`.

```bash
cd /path/to/recommend/GRID
python -m src.train \
  experiment=tiger_train_flat \
  data_dir=/path/to/data/amazon_data/beauty \
  semantic_id_path=artifacts/beauty/semantic_ids.pt \
  num_hierarchies=5 \
  seed=42 \
  trainer.accelerator=gpu \
  trainer.devices=3 \
  paths.root_dir=$PWD \
  paths.data_dir=/path/to/data/amazon_data/beauty \
  hydra.run.dir=experiments/reproduction_run
```

The uploaded full run used three GPUs, `sequence_length=120`, `max_steps=320000`, `precision=32-true`, `accumulate_grad_batches=16`, and validation every 1600 steps. The recorded run stopped early at step 5199 under the repository's validation early-stopping callback and produced:

```text
val/recall@5  = 0.0429
val/recall@10 = 0.0587
val/ndcg@5    = 0.0310
val/ndcg@10   = 0.0361
```

The best recorded checkpoint was at step 4200, but checkpoint binaries are excluded because GitHub rejects files above 100 MB. They must be generated locally by rerunning the training command.
