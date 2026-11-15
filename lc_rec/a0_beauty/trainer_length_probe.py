#!/usr/bin/env python3
"""Reproduce fixed HF Trainer 4.31 length arithmetic with Accelerate shard."""
import json
import math
import torch
from torch.utils.data import BatchSampler, SequentialSampler
from accelerate.data_loader import BatchSamplerShard
import transformers
import accelerate

N, batch, world, gas, epochs = 131413, 8, 2, 4, 4
base = BatchSampler(SequentialSampler(range(N)), batch_size=batch, drop_last=False)
shards = [BatchSamplerShard(base, num_processes=world, process_index=r,
                           split_batches=False, even_batches=True) for r in range(world)]
lengths = [len(shard) for shard in shards]
assert lengths == [8214, 8214]
assert [sum(1 for _ in shard) for shard in shards] == lengths
native_steps = math.ceil(epochs * (lengths[0] // gas))
full_microbatches = epochs * lengths[0]
native_microbatches = native_steps * gas
assert native_steps == 8212 and full_microbatches - native_microbatches == 8
print(json.dumps({'torch': torch.__version__, 'transformers': transformers.__version__,
    'accelerate': accelerate.__version__, 'dataset_rows': N, 'microbatch_per_rank': batch,
    'world_size': world, 'gradient_accumulation': gas, 'epochs': epochs,
    'shard_lengths_microbatches': lengths, 'native_trainer_max_steps': native_steps,
    'nominal_full_epoch_microbatches_per_rank': full_microbatches,
    'microbatches_at_native_max_steps_per_rank': native_microbatches,
    'omitted_final_epoch_microbatches_per_rank': full_microbatches-native_microbatches,
    'assumptions': 'Trainer 4.31 total_batched_samples starts at 0; no early stop; no resume; optimizer updates every four batches.',
    'not_a_training_run': True}, indent=2))
