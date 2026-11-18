"""Full real-user tokenization and shuffle validation, without loading weights."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'shuffle'))
from core import transform, permutation
from g3_identity import (ASSETS, NEW_TOKEN_COUNT, VOCAB_SIZE, verify_assets, tokenizer_identity,
                         EXECUTION_TOPOLOGY, EXECUTION_PROFILE)
from uneven_sampler import assignment, validate_assignments, assignment_sha256


def digest(x):
    return hashlib.sha256(json.dumps(x, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    for n in ['model-key', 'adapter', 'data-root', 'source-root', 'output', 'normal-test-dir']:
        p.add_argument('--' + n, required=True)
    a = p.parse_args()
    assert a.model_key == 'a0'
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
    args = SimpleNamespace(dataset='Beauty', data_path=a.data_root, max_his_len=20,
                           his_sep=', ', index_file='.index.json', add_prefix=False)
    data = SeqRecDataset(args, mode='test', sample_num=-1)
    data.set_prompt(0)
    asset_hashes = ASSETS
    verify_assets(a.data_root)
    for kind, expected in asset_hashes.items():
        path = Path(a.data_root)/'Beauty'/('Beauty.'+kind+'.json')
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected
    assert len(data) == 22363
    tokens = sorted({t for parts in data.indices.values() for t in parts})
    lookup = {t: tok.encode(t, add_special_tokens=False) for t in tokens}
    assert len(tokens) == NEW_TOKEN_COUNT and all(len(v) == 1 for v in lookup.values())
    lookup = {t: v[0] for t, v in lookup.items()}
    assert len(set(lookup.values())) == NEW_TOKEN_COUNT and len(tok) == VOCAB_SIZE
    tokenizer_identity(tok, data.indices)
    axis = list(data.inters)
    assert digest(axis) == '7430079e2b917dfa5e4bb8ead2702fb6c622fb48ffb684dd2889780f9480af7f'
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
    # Preserve the old fixed 32 IDs, merged into the sole rank before outputs.
    from preflight_users import validate
    from formal_evaluate import catalog_and_users
    fixed_path=Path(__file__).with_name('config')/'preflight_g3_users_seed42.json'
    fixed=json.loads(fixed_path.read_text())
    rank_shards=[assignment(len(data),0,1)]
    validate_assignments(rank_shards,len(data))
    assert [len(shard) for shard in rank_shards]==[22363]
    _,_,users,_=catalog_and_users(a.data_root)
    selected=validate(fixed,users,rank_shards)
    assignments={'0':selected[0]}
    for record in fixed['ranks']['0']:
        assert lengths[record['row_idx']]==record['input_tokens'], 'fixed G3 preflight token length'
    result = {'status': 'PASS_CPU_FULL_INPUTS', 'execution_topology':EXECUTION_TOPOLOGY,
              'execution_profile':EXECUTION_PROFILE, 'world_size':1,
              'dual_gpu_acceptance':'NOT_APPLICABLE_THIS_RUN', 'model_key': a.model_key, 'users': len(data),
              'user_axis_sha256': digest(axis), 'visible_history_groups': sum(history_lengths),
              'history_lengths': history_lengths, 'token_lengths': lengths,
              'preflight_assignments': assignments, 'permutation_distributions': distributions,
              'rank_assignment_sha256': [assignment_sha256(shard) for shard in rank_shards],
              'input_stream_sha256': {str(k): v.hexdigest() for k, v in hashes.items()},
              'manifest': {'path': str(manifest), 'bytes': manifest.stat().st_size,
                           'sha256': hashlib.sha256(manifest.read_bytes()).hexdigest()},
              'token_ids': lookup, 'tokenizer_vocab_sha256': digest(tok.get_vocab()),
              'normal_test_dir': a.normal_test_dir,
              'asset_sha256': asset_hashes,
              'saved_tokenizer_files': {p.name: {'bytes': p.stat().st_size,
                  'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
                  for p in Path(a.adapter).iterdir() if p.is_file() and
                  (p.name.startswith('tokenizer') or p.name in ['added_tokens.json', 'special_tokens_map.json', 'vocab.json', 'merges.txt'])},
              'invariants': ['exact_groups', 'nonhistory_unchanged', 'multiset_conserved',
                             'OFF_exact', 'S3_fourth_fixed', 'nonidentity', 'padding', 'target_unchanged']}
    (out / 'audit.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ['status', 'model_key', 'users', 'visible_history_groups', 'manifest']}))


if __name__ == '__main__':
    main()
