"""CPU-only native DataLoaderShard order check; no model or distributed job."""
import itertools
import json
import torch
from torch.utils.data import DataLoader, RandomSampler, BatchSampler
from accelerate.data_loader import BatchSamplerShard, DataLoaderShard
from training_rows import reconstruct_native_shards, audit_two_rank_epoch

n = 131413
state = torch.Generator().manual_seed(42).get_state()
actual = []
for rank in range(2):
    generator = torch.Generator().set_state(state)
    sampler = RandomSampler(range(n), generator=generator)
    batches = BatchSampler(sampler, batch_size=8, drop_last=False)
    shard = BatchSamplerShard(batches, num_processes=2, process_index=rank,
                              split_batches=False, even_batches=True)
    loader = DataLoaderShard(range(n), batch_sampler=shard,
                             synchronized_generator=generator, rng_types=None)
    actual.append([int(x) for batch in loader for x in batch])
expected = reconstruct_native_shards(n, state, 8214)
assert expected == actual
assert reconstruct_native_shards(n, state, 10) == [x[:80] for x in actual]
report = audit_two_rank_epoch(actual, n)
print(json.dumps({'status': 'PASS', 'audit': report,
                  'scope': 'CPU installed DataLoaderShard; no distributed synchronization or model'}, indent=2))
