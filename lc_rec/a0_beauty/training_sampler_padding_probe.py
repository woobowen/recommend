#!/usr/bin/env python3
"""Inspect actual installed Accelerate batch sharding, without training."""
from collections import Counter
import hashlib
import json
import torch
import accelerate
from torch.utils.data import BatchSampler, Sampler
from accelerate.data_loader import BatchSamplerShard

N, BATCH, WORLD = 131413, 8, 2


class FixedPermutation(Sampler):
    def __init__(self, order): self.order = order
    def __iter__(self): return iter(self.order)
    def __len__(self): return len(self.order)


generator = torch.Generator().manual_seed(42)
order = torch.randperm(N, generator=generator).tolist()
base = BatchSampler(FixedPermutation(order), batch_size=BATCH, drop_last=False)
shards = [list(BatchSamplerShard(base, num_processes=WORLD, process_index=rank,
                                 split_batches=False, even_batches=True))
          for rank in range(WORLD)]
rows = [idx for shard in shards for batch in shard for idx in batch]
counts = Counter(rows)
assert len(shards[0]) == len(shards[1]) == 8214
assert len(rows) == 131424 and len(counts) == N and set(counts) == set(range(N))
assert sum(n - 1 for n in counts.values()) == 11
assert all(len(batch) == BATCH for shard in shards for batch in shard)
print(json.dumps({'torch': torch.__version__, 'accelerate': accelerate.__version__,
                  'dataset_examples': N, 'batch_size_per_rank': BATCH,
                  'ranks': WORLD, 'microbatches_per_rank': [len(s) for s in shards],
                  'example_slots': len(rows), 'unique_examples': len(counts),
                  'duplicate_slots': sum(n - 1 for n in counts.values()),
                  'duplicate_indices': sorted(k for k, n in counts.items() if n > 1),
                  'permutation_sha256': hashlib.sha256(json.dumps(order, separators=(',', ':')).encode()).hexdigest(),
                  'scope': 'Fixed synthetic index permutation; no actual user data or model loaded.'}, indent=2))
