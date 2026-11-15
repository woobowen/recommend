"""Focused synthetic checks for aggregation and Beauty asset validation."""
import json
from pathlib import Path
import tempfile

import formal_evaluate as ev


def main():
    repo = Path(__file__).resolve().parents[2]
    evaluator = ev.load_evaluator(repo/'lc_rec/letter/vendor/LETTER/LETTER-LC-Rec')
    _, catalog, users, _ = ev.catalog_and_users(
        '/home/addaswsw/recommend-support/a0-beauty-preliminary/author_data')
    assert len(users) == 22363 and len(catalog) == 12088
    small = users[:3]
    def row(i, rank):
        user = small[i]
        candidates = [{'generation_order': j, 'raw_decoded': 'Response: ' + user['target'],
                       'sequence_score': float(20-j), 'sid': user['target']} for j in range(20)]
        return {**user, 'rank': rank, 'prompt_id': 0, 'candidates': candidates,
                'sorted_generation_order': list(range(20)), 'relevance': [1]*20}
    with tempfile.TemporaryDirectory() as tmp:
        paths = [Path(tmp)/'rank0.jsonl', Path(tmp)/'rank1.jsonl']
        def save(path, records):
            path.write_text(''.join(json.dumps(x)+'\n' for x in records))
        save(paths[0], [row(0, 0), row(2, 0)])
        save(paths[1], [row(1, 1)])
        assert ev.aggregate_predictions(paths, small, catalog, evaluator, require_rank_done=False)['actual_test_users'] == 3
        for records in ([], [row(1, 1), row(1, 1)]):
            save(paths[1], records)
            try:
                ev.aggregate_predictions(paths, small, catalog, evaluator, require_rank_done=False)
            except AssertionError:
                pass
            else:
                raise AssertionError('incomplete or duplicate data accepted')
    print('PASS: Beauty catalog identity; synthetic valid, missing and duplicate aggregation')


if __name__ == '__main__':
    main()
