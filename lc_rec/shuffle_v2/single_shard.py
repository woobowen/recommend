"""Bind V2 to the full single rank while preserving its frozen global axis."""
import argparse
import json
import os
from pathlib import Path

from lc_rec.shuffle_v2 import core, reference
from lc_rec.shuffle_v2.replay import dump, frozen


def main():
    from torch.utils.data.distributed import DistributedSampler
    p = argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--cpu-audit', required=True, type=Path)
    a = p.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    a.output.mkdir(parents=True, exist_ok=False)
    audit = json.loads(a.cpu_audit.read_text())
    assert audit['status'] == 'PASS_CPU_V2_FULL'
    manifest = a.cpu_audit.with_name('inputs.jsonl')
    assert frozen.sha(manifest) == audit['manifest_sha256']
    with manifest.open() as stream:
        rows = [json.loads(line) for line in stream]
    n = audit['users']
    assert n == len(rows) == 24772
    def sampler(world, rank):
        return list(DistributedSampler(range(n), num_replicas=world, rank=rank, shuffle=True, seed=0, drop_last=True))
    double = [sampler(2, rank) for rank in range(2)]
    assert [core.canonical_sha256(rows) for rows in double] == audit['full_rank_assignment_sha256']
    single = sampler(1, 0)
    assert set(single) == set(range(n)) and len(single) == len(set(single))
    assert single == [idx for pair in zip(*double, strict=True) for idx in pair]
    group_count = 0
    for i in single:
        row = rows[i]
        assert row['row_idx'] == i
        length = row['visible_history_length']
        group_count += length
        for mode, active in core.MODES.items():
            assert row['conditions'][mode]['permutation_ids'] == [reference.relative(len(active), i, j)[0] for j in range(length)]
    assignments = audit['preflight_assignments']
    selected = assignments['0'] + assignments['1']
    assert len(selected) == len(set(selected)) == 32 and set(selected) <= set(single)
    for rank in range(2):
        assert set(assignments[str(rank)]) <= set(double[rank])
    assert group_count == 172929
    dump(a.output / 'audit.json', {'status': 'PASS_SINGLE_SHARD_CPU', 'users': n, 'history_groups': group_count,
                                  'manifest_sha256': audit['manifest_sha256'], 'world_size': 1,
                                  'rank': 0, 'assignment_sha256': core.canonical_sha256(single),
                                  'normal_rank_assignment_sha256': audit['full_rank_assignment_sha256'],
                                  'exact_interleave_of_frozen_normal_shards': True,
                                  'permutation_ids_independently_verified': True, 'preflight_assignment': selected})


if __name__ == '__main__':
    main()
