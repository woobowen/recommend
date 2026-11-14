"""Full real-user tokenization and shuffle validation, without loading weights."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from core import transform, permutation


def digest(x):
    return hashlib.sha256(json.dumps(x, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    for n in ['model-key', 'adapter', 'data-root', 'source-root', 'output', 'normal-test-dir']:
        p.add_argument('--' + n, required=True)
    a = p.parse_args()
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=False)
    from transformers import AutoTokenizer, LlamaTokenizer
    if a.model_key == 'a0':
        tok = LlamaTokenizer.from_pretrained(a.adapter, local_files_only=True)
        tok.pad_token_id, tok.model_max_length = 0, 2048
    else:
        tok = AutoTokenizer.from_pretrained(a.adapter, local_files_only=True)
    tok.padding_side = 'left'
    sys.path.insert(0, a.source_root)
    from data import SeqRecDataset
    args = SimpleNamespace(dataset='Instruments', data_path=a.data_root, max_his_len=20,
                           his_sep=', ', index_file='.index.json', add_prefix=False)
    data = SeqRecDataset(args, mode='test', sample_num=-1)
    data.set_prompt(0)
    assert len(data) == 24772
    tokens = sorted({t for parts in data.indices.values() for t in parts})
    lookup = {t: tok.encode(t, add_special_tokens=False) for t in tokens}
    assert len(tokens) == 918 and all(len(v) == 1 for v in lookup.values())
    lookup = {t: v[0] for t, v in lookup.items()}
    assert len(set(lookup.values())) == 918
    axis = list(data.inters)
    assert digest(axis) == '1daf63ce4c77b0265777a46b61939006aafab0276b9106c742de602c8c7e867a'
    distributions = {n: Counter() for n in [3, 4]}
    lengths, history_lengths, manifests = [], [], []
    hashes = {n: hashlib.sha256() for n in [0, 3, 4]}
    manifest = out / 'permutations.jsonl'
    with manifest.open('x') as stream:
        for i, uid in enumerate(axis):
            seq = data.inters[uid]
            visible = seq[:-1][-20:]
            groups = [[lookup[t] for t in data.indices[str(item)]] for item in visible]
            row = data[i]
            assert row['labels'] == ''.join(data.indices[str(seq[-1])])
            text = row['input_ids']
            assert text.endswith('### Response:')
            encoded = tok(text, return_attention_mask=True)
            ids, mask = encoded['input_ids'], encoded['attention_mask']
            assert len(ids) <= 2048 and all(mask)
            lengths.append(len(ids)); history_lengths.append(len(visible))
            record = {'row_idx': i, 'user_id': uid, 'visible_history_length': len(visible),
                      'original_input_sha256': digest(ids)}
            for n in [0, 3, 4]:
                changed, evidence = transform(ids, mask, groups, set(lookup.values()), n, i)
                if n == 0:
                    assert changed == ids
                elif visible:
                    assert changed != ids
                hashes[n].update((json.dumps([i, changed], separators=(',', ':')) + '\n').encode())
                if n:
                    perms = [list(permutation(n, i, j)) for j in range(len(visible))]
                    record[str(n)] = {'permutations': perms, 'input_sha256': digest(changed)}
                    for perm in perms:
                        distributions[n][str(perm)] += 1
            # Extra padding validates group offsets and invariance to padding.
            padded, _ = transform([tok.pad_token_id]*3 + ids, [0]*3 + mask,
                                  groups, set(lookup.values()), 4, i)
            assert padded[3:] == changed and padded[:3] == [tok.pad_token_id]*3
            stream.write(json.dumps(record, separators=(',', ':')) + '\n')
    # Fixed, outcome-blind sample: strata/extremes plus each sampler shard.
    # Row parity is not sampler rank: derive the actual torch sampler below.
    import torch
    from torch.utils.data.distributed import DistributedSampler
    assignments = {}
    for rank in [0, 1]:
        assigned = list(DistributedSampler(range(len(data)), num_replicas=2, rank=rank,
                                          shuffle=True, seed=0, drop_last=True))
        chosen = []
        def add(i):
            if i not in chosen:
                chosen.append(i)
        for group in [(0, 5), (6, 10), (11, 20)]:
            eligible = [i for i in assigned if group[0] <= history_lengths[i] <= group[1]]
            for i in eligible[:2]: add(i)
        ordered = sorted(assigned, key=lambda i: (lengths[i], i))
        for i in [ordered[0], ordered[-1], ordered[-2], assigned[0], assigned[-1]]: add(i)
        repeated = [i for i in assigned if len(set(data.inters[axis[i]][:-1][-20:])) < history_lengths[i]]
        if repeated: add(repeated[0])
        for i in assigned:
            if len(chosen) >= 16: break
            add(i)
        assert len(chosen) == 16
        assignments[str(rank)] = chosen
    result = {'status': 'PASS_CPU_FULL_INPUTS', 'model_key': a.model_key, 'users': len(data),
              'user_axis_sha256': digest(axis), 'visible_history_groups': sum(history_lengths),
              'history_lengths': history_lengths, 'token_lengths': lengths,
              'preflight_assignments': assignments, 'permutation_distributions': distributions,
              'input_stream_sha256': {str(k): v.hexdigest() for k, v in hashes.items()},
              'manifest': {'path': str(manifest), 'bytes': manifest.stat().st_size,
                           'sha256': hashlib.sha256(manifest.read_bytes()).hexdigest()},
              'token_ids': lookup, 'tokenizer_vocab_sha256': digest(tok.get_vocab()),
              'normal_test_dir': a.normal_test_dir,
              'invariants': ['exact_groups', 'nonhistory_unchanged', 'multiset_conserved',
                             'OFF_exact', 'S3_fourth_fixed', 'nonidentity', 'padding', 'target_unchanged']}
    (out / 'audit.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ['status', 'model_key', 'users', 'visible_history_groups', 'manifest']}))


if __name__ == '__main__':
    main()
