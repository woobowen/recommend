"""Audit published sources, exact stored aggregates, permutations and README tables."""
import ast
import hashlib
import json
from pathlib import Path
import re
import subprocess

BASE = Path(__file__).resolve().parents[1]
REPO = BASE.parent


def load(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def lines(path):
    return [json.loads(x) for x in path.read_text(encoding='utf-8-sig').splitlines() if x.strip()]


def close(a, b):
    if abs(a - b) > 1e-10:
        raise AssertionError((a, b))


def mean(values):
    return sum(values) / len(values)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    source_manifest = load(BASE / 'provenance/source_manifest.json')
    for row in source_manifest:
        assert sha(REPO / row['published']) == row['sha256'], row['published']
    summaries = load(BASE / 'results/embedding_summary.json')
    assert len(summaries) == 6
    for root in ['llm2vec_beauty_toys', 'llm2vec_instruments']:
        run = BASE / 'results' / root
        for filename, digest in load(run / 'launch.json')['source_sha256'].items():
            assert sha(BASE / 'archive' / root / Path(filename).name) == digest
    for row in summaries:
        dataset, variant = row['dataset'], row['variant']
        root = 'llm2vec_instruments' if dataset == 'instrument' else 'llm2vec_beauty_toys'
        evaluation = BASE / 'results' / root / dataset / variant / 'evaluation'
        summary = load(evaluation / 'summary.json')
        assert sha(evaluation / 'summary.json') == load(evaluation / 'evaluation_complete.json')['summary_sha256']
        nodes = lines(evaluation / 'prefix_nodes.jsonl')
        edges = lines(evaluation / 'parent_child_edges.jsonl')
        paths = lines(evaluation / 'complete_paths.jsonl')
        gaps = lines(evaluation / 'branch_gaps.jsonl')
        nulls = lines(evaluation / 'permutation_statistics.jsonl')
        assert len(nulls) == 1998
        for depth in [1, 2, 3]:
            valid = [n for n in nodes if n['depth'] == depth and n['items'] >= 2]
            close(mean([n['mean_pairwise_cosine'] for n in valid]), row[f'S{depth}'])
            close(sum(n['items'] for n in valid) / row['items'], summary['coverage'][str(depth)]['coverage'])
        for tag in ['12', '23']:
            transition = f'{tag[0]}_to_{tag[1]}'
            valid = [e for e in edges if e['transition'] == transition and e['eligible_proper_refinement']]
            assert len(valid) == row[f'edges_{tag}']
            close(mean([e['delta'] for e in valid]), row[f'delta_{tag}'])
            close(mean([e['delta'] > 1e-6 for e in valid]), row[f'positive_{tag}'])
            close(mean([e['gap'] for e in gaps if e['transition'] == transition]), row[f'gap_{tag}'])
            parent_depth = int(tag[0])
            values = [r[f'delta_{tag}'] for r in nulls if r['parent_depth'] == parent_depth]
            comparison = summary['random_controls'][f'within_depth_{parent_depth}'][f'delta_{tag}']
            close(mean(values), comparison['null_mean'])
            close((1 + sum(x >= row[f'delta_{tag}'] for x in values)) / 1000, comparison['p_one_sided'])
        valid_paths = [p for p in paths if p['proper_both_edges']]
        assert len(valid_paths) == row['proper_complete_paths']
        close(mean([p['monotonic_as_image'] for p in valid_paths]), row['path_monotonic'])
        top = lines(evaluation / 'top20_three_code_paths.jsonl')
        assert top == sorted(paths, key=lambda p: (-p['counts'][2], p['prefix']))[:20]
    classification = load(BASE / 'results/classification/summary.json')
    assert len(classification) == 12
    for dataset, n, coarse, fine, unknown in [('beauty', 12101, 9, 41, 405), ('toys', 11924, 122, 183, 219)]:
        info = load(BASE / f'results/classification/{dataset}/taxonomy_and_counts.json')
        assert info['n'] == n and sum(info['counts'].values()) == n
        assert info['counts']['UNKNOWN'] == unknown
        assert len(info['taxonomy']['coarse_categories']) == coarse
        assert len(info['taxonomy']['categories']) == fine
    # Check every numeric result table in the root README against unrounded JSON.
    readme = (REPO / 'README.md').read_text(encoding='utf-8')
    names = {'beauty': 'Beauty', 'toys': 'Toys', 'instrument': 'Instruments'}
    for r in summaries:
        prefix = f"| {names[r['dataset']]} | {'标题（主）' if r['variant'] == 'title' else '标题＋品牌'} |"
        expected = prefix + f" {r['delta_12']:+.6f} | {r['positive_12']:.2%} | {r['delta_23']:+.6f} | {r['path_monotonic']:.2%} | {r['coverage_3']:.2%} |"
        assert expected in readme, expected
        prefix = f"| {names[r['dataset']]} | {'标题' if r['variant'] == 'title' else '标题＋品牌'} |"
        expected = prefix + ' ' + ' | '.join(f'{r[k]:.6f}' for k in ['S1', 'S2', 'S3', 'gap_12', 'gap_23']) + ' |'
        assert expected in readme, expected
    for dataset in ['beauty', 'toys']:
        for depth in [1, 2, 3]:
            fine = next(r for r in classification if r['dataset'] == dataset and r['depth'] == depth and r['granularity'] == 'fine')
            coarse = next(r for r in classification if r['dataset'] == dataset and r['depth'] == depth and r['granularity'] == 'coarse')
            expected = f'| {names[dataset]} | {depth} | ' + ' | '.join(f'{x:.2%}' for x in [fine['weighted_purity'], fine['pair_agreement'], coarse['weighted_purity'], coarse['pair_agreement']]) + ' |'
            assert expected in readme, expected
    # Python syntax and relative documentation links are checked without importing GPU code.
    for path in BASE.rglob('*.py'):
        ast.parse(path.read_text(encoding='utf-8-sig'), filename=str(path))
    for path in [REPO / 'README.md', *BASE.rglob('*.md')]:
        for target in re.findall(r'\]\(([^)]+)\)', path.read_text(encoding='utf-8')):
            if '://' in target or target.startswith('#'):
                continue
            target = target.split('#')[0].replace('%20', ' ')
            resolved = path.parent / target
            if not resolved.exists():
                # A sparse checkout may intentionally omit the inherited GRID archive.
                tracked_path = resolved.relative_to(REPO).as_posix()
                result = subprocess.run(['git', '-C', str(REPO), 'cat-file', '-e',
                                         'HEAD:' + tracked_path], capture_output=True)
                assert result.returncode == 0, (path, target)
    print('PASS: 6 runs; 12 classification rows; all source hashes; node/edge/path/gap '
          'aggregates; 11,988 permutation records; completion hashes; README values; links; Python syntax.')
    print('No full-vector re-encoding was performed by this check.')


if __name__ == '__main__':
    main()
