"""Small adversarial checks of the scientific counting logic (synthetic only)."""
import itertools
import tempfile
from pathlib import Path
import json
import sys

import numpy as np

from common import prefix_groups, canonical_taxonomy, write_jsonl, sha, write_json
from evaluate import contingency, scores, compare_null
from inference import parse_json
from run_stage import read_complete_labels


def synthetic_end_to_end():
    """Ensure UNKNOWN exclusion, global baselines and selections survive file I/O."""
    import torch
    import evaluate
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        prep, out = root/'preparation', root/'experiments/synthetic'
        cfg = {'datasets': {'beauty': {'expected_items': 6}}, 'prefix_lengths': [1, 2, 3],
               'top_k': [10, 20, 50, 100], 'permutations': 12, 'seed': 20261109}
        sids = [[1, 2, 3, 0], [1, 2, 3, 1], [1, 4, 5, 0],
                [1, 4, 6, 0], [2, 1, 1, 0], [2, 1, 1, 1]]
        sid_path = root/'sids.pt'
        torch.save(torch.tensor(sids).T, sid_path)
        item_path = prep/'beauty/classifier_items.jsonl'
        write_jsonl(item_path, [{'item_id': i, 'text': 'synthetic product'} for i in range(6)])
        taxonomy = canonical_taxonomy({'categories': [
            {'coarse_name': 'A', 'name': 'a', 'definition': 'a'},
            {'coarse_name': 'B', 'name': 'b', 'definition': 'b'}]})
        taxonomy_path = out/'beauty/taxonomy_frozen.json'
        write_json(taxonomy_path, taxonomy)
        for length, groups in prefix_groups(sids).items():
            ranked = sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0]))
            write_jsonl(prep/'beauty'/f'prefix_groups_l{length}.jsonl', [
                {'prefix': list(k), 'count': len(ids), 'item_ids': ids} for k, ids in ranked])
        rows = [{'item_id': i, 'category_id': c, 'needs_review': False} for i, c in enumerate(
            ['F001', 'F001', 'F002', 'UNKNOWN', 'F002', 'F002'])]
        label_path = out/'beauty/final_shard0of1.jsonl'
        write_jsonl(label_path, rows)
        config_hash = sha(Path(__file__).with_name('config.json'))
        write_json(label_path.with_suffix('.complete.json'), {
            'shards': 1, 'shard': 0, 'taxonomy_sha256': sha(taxonomy_path),
            'sha256': sha(label_path), 'items_sha256': sha(item_path), 'config_sha256': config_hash})
        write_json(prep/'data_preparation_verified.json', {'datasets': {'beauty': {
            'sid_path': str(sid_path), 'sid_sha256': sha(sid_path), 'classifier_items_sha256': sha(item_path)}}})
        saved_roots, saved_argv = evaluate.roots, sys.argv
        try:
            evaluate.roots = lambda: (cfg, root, prep, out)
            sys.argv = ['evaluate.py', '--execute']
            evaluate.main()
        finally:
            evaluate.roots, sys.argv = saved_roots, saved_argv
        records = json.loads((out/'evaluation/results.json').read_text())['records']
        actual = next(r for r in records if r['prefix_length'] == 1 and r['granularity'] == 'fine'
                      and r['label_view'] == 'known_only' and r['selection'] == 'all_shared')
        assert actual['metrics']['weighted_purity'] == .8
        assert actual['metrics']['pair_agreement'] == .5
        assert actual['global_random_pair_agreement'] == .4
        assert actual['coverage_all_items'] == 1
        assert abs(actual['evaluated_coverage_all_items'] - 5/6) < 1e-12
        # No >=5 groups exist; this must remain an empty result, never filled by singletons.
        empty = next(r for r in records if r['selection'] == 'all_min5')
        assert empty['selected_groups'] == 0 and empty['metrics']['pair_agreement'] is None


def main():
    sids = [[1, 2, 3, 0], [1, 2, 3, 1], [1, 4, 5, 0], [2, 2, 3, 0]]
    grouped = prefix_groups(sids)
    assert grouped[1][(1,)] == [0, 1, 2]
    assert grouped[2][(1, 2)] == [0, 1]
    assert grouped[3][(1, 2, 3)] == [0, 1]
    assert (2, 2, 3) in grouped[3]  # Singleton retained in inventory, excluded by analysis selection.
    labels = np.array([0, 0, 1, 1])
    groups = np.array([0, 0, 0, 1])
    table = contingency(groups, labels, 2, 2)
    actual = scores(table)
    assert actual['weighted_purity'] == .75
    assert abs(actual['pair_agreement'] - 1/3) < 1e-12
    assert scores(table[:1])['weighted_purity'] == 2/3
    # Compare algebraic pair counts with direct pair enumeration for every label placement.
    null_pair = []
    for perm in itertools.permutations(labels.tolist()):
        measured = scores(contingency(groups, np.array(perm), 2, 2))
        pairs = [(i, j) for i in range(4) for j in range(i+1, 4) if groups[i] == groups[j]]
        brute = sum(perm[i] == perm[j] for i, j in pairs)/len(pairs)
        assert abs(measured['pair_agreement'] - brute) < 1e-12
        null_pair.append(measured['pair_agreement'])
    assert abs(np.mean(null_pair) - 1/3) < 1e-12
    assert scores(np.zeros((0, 2), dtype=int))['pair_agreement'] is None
    assert compare_null(None, [None])['actual'] is None
    assert scores(np.array([[80, 15, 5]]))['weighted_purity'] == .8
    assert abs(scores(np.array([[80, 15, 5]]))['pair_agreement'] - 3275/4950) < 1e-12
    assert parse_json('```json\n{"ok":true}\n```') == {'ok': True}
    try:
        parse_json('{"name":"parent","name":"child"}')
        raise AssertionError('Duplicate JSON keys must fail rather than silently overwrite')
    except ValueError:
        pass
    try:
        parse_json('{"broken":')
        raise AssertionError('Truncated JSON must fail')
    except ValueError:
        pass
    taxonomy = canonical_taxonomy({'categories': [{'coarse_name': 'B', 'name': 'x', 'definition': 'x'},
                                                  {'coarse_name': 'A', 'name': 'y', 'definition': 'y'}]})
    assert taxonomy['categories'][0]['coarse_id'] == 'C01'
    with tempfile.TemporaryDirectory() as td:
        out = Path(td)
        path = out/'final_shard0of1.jsonl'
        write_jsonl(path, [{'item_id': 0, 'category_id': 'F001', 'needs_review': False}])
        write_json(path.with_suffix('.complete.json'), {'shards': 1, 'shard': 0,
                   'taxonomy_sha256': 'synthetic', 'sha256': sha(path)})
        assert len(read_complete_labels(out, 'final', 1)) == 1
        try:
            read_complete_labels(out, 'final', 2)
            raise AssertionError('Missing product must fail')
        except AssertionError as exc:
            assert str(exc) != 'Missing product must fail'
    synthetic_end_to_end()
    print(json.dumps({'passed': True, 'synthetic_only': True,
        'checks': ['SID-prefix membership', 'singletons', 'purity', 'pair-agreement vs brute force',
                   'exact small-sample permutation expectation', 'empty pair denominator',
                   'JSON truncation', 'taxonomy IDs', 'missing-label rejection',
                   'synthetic full evaluation including UNKNOWN and empty selections']}))


if __name__ == '__main__':
    main()
