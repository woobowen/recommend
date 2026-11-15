"""Check that row tracking preserves sampler order and collated payloads."""
import random
import torch
from torch.utils.data import DataLoader, RandomSampler
from training_rows import IndexedDataset, IndexedCollator, take_indices, audit_two_rank_epoch


class Data:
    def __len__(self): return 35
    def __getitem__(self, i): return {'value': i * 7 + random.randrange(100)}


def collator(rows):
    return {'value': torch.tensor([r['value'] for r in rows])}


def collect(tracked):
    random.seed(42)
    generator = torch.Generator().manual_seed(42)
    base = Data()
    dataset = IndexedDataset(base) if tracked else base
    sampler = RandomSampler(dataset, generator=generator)
    loader = DataLoader(dataset, sampler=sampler, batch_size=8, num_workers=0,
                        collate_fn=IndexedCollator(collator) if tracked else collator)
    payloads, indices = [], []
    for batch in loader:
        if tracked: indices.extend(take_indices(batch))
        payloads.extend(batch['value'].tolist())
    return payloads, indices


reference, _ = collect(False)
tracked, order = collect(True)
assert reference == tracked and sorted(order) == list(range(35))
audit = audit_two_rank_epoch([[0, 2, 4], [1, 3, 0]], 5, expected_repeats=1)
assert audit['duplicates'] == {'0': 1}
for bad in ([[0, 2, 4], [1, 3, 5]], [[0], [1, 2]], [[0, 2, 4], [1, 1, 0]]):
    try: audit_two_rank_epoch(bad, 5, expected_repeats=1)
    except ValueError: pass
    else: raise AssertionError('bad row stream accepted')
print('PASS: unchanged sampler and collator payloads; tracked rows and coverage checks')
