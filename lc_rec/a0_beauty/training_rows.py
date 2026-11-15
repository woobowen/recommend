"""Record consumed dataset row indices without altering the underlying sampler.

Use only with num_workers=0. The indices travel through the collator so that
DataLoader prefetch is not mistaken for actual consumption; remove them before
passing the batch to the model. This helper alone is not a training entry.
"""
from collections import Counter
import hashlib
import json

KEY = '_beauty_audit_row_indices'


class IndexedDataset:
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return index, self.dataset[index]


class IndexedCollator:
    def __init__(self, collator):
        self.collator = collator

    def __call__(self, batch):
        import torch
        result = self.collator([sample for _, sample in batch])
        if KEY in result:
            raise ValueError('audit key already present in model inputs')
        result[KEY] = torch.tensor([index for index, _ in batch], dtype=torch.int64)
        return result


def take_indices(inputs):
    if KEY not in inputs:
        raise ValueError('missing audit row indices')
    return inputs.pop(KEY).detach().cpu().tolist()


def audit_two_rank_epoch(rank_rows, n, expected_repeats=11):
    if len(rank_rows) != 2 or len(rank_rows[0]) != len(rank_rows[1]):
        raise ValueError('missing or unequal rank row streams')
    rows = [row for rank in rank_rows for row in rank]
    if len(rows) != n + expected_repeats or any(type(x) is not int or not 0 <= x < n for x in rows):
        raise ValueError('wrong row count or out-of-range index')
    counts = Counter(rows)
    if len(counts) != n or sum(value - 1 for value in counts.values()) != expected_repeats:
        raise ValueError('missing data or wrong duplicate count')
    return {'rows_per_rank': [len(x) for x in rank_rows],
            'duplicates': {str(k): v - 1 for k, v in sorted(counts.items()) if v > 1},
            'rank_order_sha256': [hashlib.sha256(json.dumps(x, separators=(',', ':')).encode()).hexdigest()
                                  for x in rank_rows]}
