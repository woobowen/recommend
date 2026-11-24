"""Make publication CSV and exact JSON from the frozen formal evaluator outputs."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import shutil


CONDITIONS = ('normal','shuffle2','shuffle3a','shuffle3b','shuffle4')
PAIRS = tuple(name+'-normal' for name in CONDITIONS[1:])+('shuffle3b-shuffle3a',)


def read(p):
    return json.loads(p.read_bytes())


def read_csv(p):
    with p.open(newline='',encoding='utf-8') as inp:
        return list(csv.DictReader(inp))


def write_csv(p,key_names,rows):
    with p.open('x',newline='',encoding='utf-8') as stream:
        w=csv.DictWriter(stream,fieldnames=key_names,lineterminator='\n')
        w.writeheader()
        w.writerows(rows)


def convert(raw,destination):
    evidence = raw/'evidence'
    paired = evidence/'formal-paired-r0'
    summary = read(paired/'summary.json')
    metrics, comparisons = summary['metrics'], summary['comparisons']
    keys = tuple(metrics['normal'])
    assert len(keys)==16 and tuple(metrics)==CONDITIONS and tuple(comparisons)==PAIRS
    assert summary['users']==24772 and not summary['preflight']
    per_mode, anomaly = {}, {}
    for condition in CONDITIONS:
        folder = evidence/('cpu-normal-r0' if condition=='normal' else f'formal-cpu-{condition}-r0')
        got = read(folder/'summary.json')
        assert got['metrics']==metrics[condition] and got['users']==24772
        per_mode[condition] = got
        anomaly[condition] = got['candidate_anomalies']
        assert anomaly[condition]['candidate_count']==495440
    changes = []
    change_counts = {}
    for condition in CONDITIONS[1:]:
        signs = Counter()
        for key in keys:
            normal, value = metrics['normal'][key], metrics[condition][key]
            difference = value-normal
            signs['up' if difference>0 else 'down' if difference<0 else 'equal'] += 1
            changes.append({'condition':condition,'metric':key,'normal':normal,'value':value,
                            'proportion_difference':difference,'percentage_point_difference':100*difference,
                            'relative_change_percent':100*difference/normal if normal else None})
        change_counts[condition] = dict(signs)
    rankings = {key:sorted(CONDITIONS,key=lambda cond:metrics[cond][key],reverse=True) for key in keys}
    rows, interval_summary = [], {}
    for pair, comparisons_for_pair in comparisons.items():
        right, left = pair.split('-')
        coverage = {'wholly_below_zero':[],'includes_zero':[],'wholly_above_zero':[]}
        assert tuple(r['metric'] for r in comparisons_for_pair)==keys
        for item in comparisons_for_pair:
            key = item['metric']
            assert item['left']==metrics[left][key] and item['right']==metrics[right][key]
            diff=metrics[right][key]-metrics[left][key]
            assert item['absolute_difference']==diff and item['percentage_point_difference']==100*diff
            assert item['relative_change_fraction']==(diff/metrics[left][key] if metrics[left][key] else None)
            bucket = 'wholly_below_zero' if item['difference_CI95_high']<0 else (
                'wholly_above_zero' if item['difference_CI95_low']>0 else 'includes_zero')
            coverage[bucket].append(key)
            rows.append(dict(comparison=pair,**item))
        interval_summary[pair] = coverage
    for want, saved in zip(rows,read_csv(paired/'paired.csv'),strict=True):
        assert want == {key:(value if key in ('metric','comparison') else float(value) if value else None)
                        for key,value in saved.items()}
    destination.mkdir(exist_ok=False)
    write_csv(destination/'metrics.csv',('condition',*keys),[dict(condition=cond,**metrics[cond]) for cond in CONDITIONS])
    write_csv(destination/'changes_from_normal.csv',tuple(changes[0]),changes)
    write_csv(destination/'paired.csv',tuple(rows[0]),rows)
    exact = dict(status='PASS_FORMAL_PUBLICATION_TABLES',users=summary['users'],metric_keys=keys,
                 metrics=metrics,changes_from_normal=changes,comparisons=comparisons,bootstrap=summary['bootstrap'],
                 arrays_file_sha256=summary['arrays_file_sha256'],candidate_anomalies=anomaly,
                 signed_change_counts=change_counts,rankings_descending=rankings,interval_summary=interval_summary)
    with (destination/'results.json').open('x') as stream:
        json.dump(exact,stream,ensure_ascii=False,indent=2,allow_nan=False)
        stream.write('\n')
    return exact


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('raw',type=Path)
    parser.add_argument('destination',type=Path)
    args=parser.parse_args()
    convert(args.raw,args.destination)
