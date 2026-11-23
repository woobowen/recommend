"""Render public full-precision G3 tables from the exact formal paired summary."""
import argparse
import csv
import json
from pathlib import Path


def write_csv(path, fields, rows):
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def render(summary, output):
    assert (summary['status'], summary['scope'], summary['users']) == (
        'PASS_G3_V2_PAIRED', 'FORMAL', 22363)
    scores = summary['metrics']
    conditions = ('normal', 'shuffle2', 'shuffle3a', 'shuffle3b', 'shuffle4')
    keys = tuple(scores['normal'])
    assert len(keys) == 16 and set(scores) == set(conditions)
    assert all(set(scores[condition]) == set(keys) for condition in conditions)
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / 'metrics.csv', ('metric', *conditions),
              ({'metric': key, **{mode: scores[mode][key] for mode in conditions}} for key in keys))

    pairs = summary['comparisons']
    assert len(pairs) == 80
    assert {(r['comparison'], r['metric']) for r in pairs} == {
        (c, key) for c in ('shuffle2-normal', 'shuffle3a-normal', 'shuffle3b-normal',
                           'shuffle4-normal', 'shuffle3b-shuffle3a') for key in keys}
    by_key = {(r['comparison'], r['metric']): r for r in pairs}
    change = []
    for key in keys:
        baseline = scores['normal'][key]
        for mode in conditions[1:]:
            value = scores[mode][key]
            delta = value - baseline
            rel = 100 * (delta / baseline) if baseline else None
            pair = by_key[(mode + '-normal', key)]
            assert (pair['left'], pair['right'], pair['difference']) == (baseline, value, delta)
            assert rel == (None if pair['relative_change'] is None else 100 * pair['relative_change'])
            change.append(dict(metric=key, condition=mode, normal=baseline, value=value,
                               difference=delta, difference_pp=100 * delta,
                               relative_change_percent=rel))
    write_csv(output / 'changes_from_normal.csv', tuple(change[0]), change)

    annotated = []
    for row in pairs:
        subject, reference = row['comparison'].split('-')
        key = row['metric']
        assert row['left'] == scores[reference][key] and row['right'] == scores[subject][key]
        assert row['right'] - row['left'] == row['difference']
        annotated.append(dict(comparison=row['comparison'], metric=key, reference=row['left'],
                              condition=row['right'], difference=row['difference'],
                              relative_change=row['relative_change'], ci95_low=row['ci95_low'],
                              ci95_high=row['ci95_high'], ci95_includes_zero=row['ci95_low'] <= 0 <= row['ci95_high']))
    write_csv(output / 'paired.csv', tuple(annotated[0]), annotated)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('summary', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    render(json.loads(args.summary.read_text()), args.output)
