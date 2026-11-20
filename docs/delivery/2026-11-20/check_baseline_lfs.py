"""Read selected public Git LFS objects and compare saved baseline test rows."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    baseline = args.source / 'GRID baseline'
    paths = sorted((baseline / 'experiments').glob('*/tiger*/csv/version_*/metrics.csv'))
    paths.extend(sorted((baseline / 'experiments').glob('*/semantic_ids.pt')))
    objects = {}
    for path in paths:
        header, oid, size = path.read_text().splitlines()
        assert header == 'version https://git-lfs.github.com/spec/v1'
        objects[oid.split(':')[1]] = (int(size.split()[1]), path.relative_to(args.source))
    batch = json.dumps({'operation':'download', 'transfers':['basic'],
                        'objects':[{'oid': oid, 'size': size} for oid, (size, _) in objects.items()]}).encode()
    request = urllib.request.Request('https://github.com/woobowen/recommend.git/info/lfs/objects/batch', data=batch,
                                     headers={'Accept':'application/vnd.git-lfs+json', 'Content-Type':'application/vnd.git-lfs+json'})
    with urllib.request.urlopen(request, timeout=45) as response:
        returned = json.load(response)['objects']
    files = []
    for entry in returned:
        oid = entry['oid']
        size, path = objects[oid]
        action = entry['actions']['download']
        request = urllib.request.Request(action['href'], headers=action.get('header', {}))
        with urllib.request.urlopen(request, timeout=45) as response:
            data = response.read(size + 1)
        assert len(data) == size and hashlib.sha256(data).hexdigest() == oid
        dest = args.output / path
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        files.append({'path':str(path), 'size':size, 'sha256':oid})
    assert len(files) == len(objects)
    results = json.loads((args.source / 'Baseline Results/test_results.json').read_bytes())
    checks = {}
    for domain, info in results['datasets'].items():
        run = args.output / Path(info['checkpoint']).parents[1]
        matches = []
        for path in sorted(run.glob('csv/version_*/metrics.csv')):
            for i, row in enumerate(csv.DictReader(path.open()), start=2):
                if all(row.get(k) and float(row[k]) == v for k, v in info['test_metrics'].items()):
                    matches.append({'file':str(path.relative_to(args.output)), 'line':i})
        assert matches, domain
        checks[domain] = matches
    print(json.dumps({'actual_lfs_payloads':files, 'test_metric_rows':checks,
                      'scope':'Five saved test values per dataset and three SID files confirmed through public LFS. No per-user metrics or semantic pairings recalculated.'}, indent=2))


if __name__ == '__main__':
    main()
