"""Evaluation-only, no-padding variant of PyTorch DistributedSampler."""
import hashlib
import json


def assignment(n, rank, world_size, seed=0, epoch=0):
    import torch
    if n < 0 or world_size <= 0 or not 0 <= rank < world_size:
        raise ValueError('invalid dataset size or rank')
    generator = torch.Generator()
    generator.manual_seed(seed + epoch)
    return torch.randperm(n, generator=generator).tolist()[rank::world_size]


def validate_assignments(rows, n):
    flat = [row for rank_rows in rows for row in rank_rows]
    if len(flat) != n or sorted(flat) != list(range(n)):
        raise ValueError('assignment has missing, duplicate, or out-of-range rows')


def assignment_sha256(rows):
    return hashlib.sha256(json.dumps(rows, separators=(',', ':')).encode()).hexdigest()


if __name__ == '__main__':
    import torch
    from torch.utils.data import DataLoader, Dataset, DistributedSampler

    class Rows(Dataset):
        def __init__(self, n): self.n = n
        def __len__(self): return self.n
        def __getitem__(self, i): return i

    for n in (0, 1, 2, 10, 11, 24772, 22363):
        ds = Rows(n)
        ranks = [assignment(n, rank, 2) for rank in range(2)]
        validate_assignments(ranks, n)
        for rank, rows in enumerate(ranks):
            assert [int(x) for x in DataLoader(ds, batch_size=None, sampler=rows)] == rows
            if n % 2 == 0:
                original = DistributedSampler(ds, num_replicas=2, rank=rank,
                                              shuffle=True, seed=0, drop_last=True)
                assert rows == list(original)
        if n:
            for broken in (ranks[0][:-1] + ranks[1], ranks[0] + ranks[1] + [ranks[0][0]]):
                try: validate_assignments([broken], n)
                except ValueError: pass
                else: raise AssertionError('invalid coverage accepted')
        if n == 22363:
            print(json.dumps({'torch': torch.__version__, 'users': n,
                              'counts': [len(x) for x in ranks],
                              'assignment_sha256': [assignment_sha256(x) for x in ranks],
                              'even_matches_historical_sampler': True,
                              'dataloader_order_and_negative_tests': 'PASS'}, indent=2))
