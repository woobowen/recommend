"""Read-only exploratory mapping of all Beauty SID prefixes to frozen Llama labels."""
import html
import json
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path('/home/jcy/GRID_wmh_Qwen_TIGER_20261009')
sys.path.insert(0, str(ROOT / 'codex/sid_prefix_semantics_v1'))
from common import sha, read_jsonl, write_json
from run_stage import read_complete_labels

out = ROOT / 'experiments/sid_prefix_semantics_v1'
prep = ROOT / 'codex/sid_prefix_semantics_v1/preparation/beauty'
taxonomy_path = out / 'beauty/taxonomy_frozen.json'
taxonomy = json.loads(taxonomy_path.read_text())
lookup = {r['id']: r for r in taxonomy['categories']}
coarse_names = {r['id']: r['name'] for r in taxonomy['coarse_categories']}
labels = {r['item_id']: r for r in read_complete_labels(out / 'beauty', 'final', 12101)}
receipt = json.loads((out / 'beauty/final_shard0of1.complete.json').read_text())
assert receipt['taxonomy_sha256'] == sha(taxonomy_path)
assert receipt['items_sha256'] == sha(prep / 'classifier_items.jsonl')
texts = {r['item_id']: r['text'] for r in read_jsonl(prep / 'classifier_items.jsonl')}
audit = json.loads((prep.parent / 'data_preparation_verified.json').read_text())['datasets']['beauty']
assert sha(audit['sid_path']) == audit['sid_sha256']
import torch
sids = torch.load(audit['sid_path'], map_location='cpu', weights_only=True).T.tolist()
nodes = []
for length in [1, 2, 3]:
    groups = read_jsonl(prep / f'prefix_groups_l{length}.jsonl')
    assert groups == sorted(groups, key=lambda r: (-r['count'], r['prefix']))
    assert sorted(i for g in groups for i in g['item_ids']) == list(range(12101))
    for rank, g in enumerate(groups, 1):
        assert g['count'] == len(g['item_ids'])
        assert all(sids[i][:length] == g['prefix'] for i in g['item_ids'])
        fine = Counter(labels[i]['category_id'] for i in g['item_ids'])
        coarse = Counter()
        brands = Counter()
        for i in g['item_ids']:
            label = labels[i]['category_id']
            coarse[lookup[label]['coarse_id'] if label != 'UNKNOWN' else 'UNKNOWN'] += 1
            match = re.search(r'; Brand: (.*?); Categories:', texts[i])
            brands[html.unescape(match.group(1)) if match else 'missing'] += 1
        nodes.append(dict(g, rank_within_length=rank, known_items=g['count'] - fine['UNKNOWN'],
            fine_counts=[{'id': k, 'name': lookup[k]['name'] if k != 'UNKNOWN' else k, 'count': v}
                         for k, v in fine.most_common()],
            coarse_counts=[{'id': k, 'name': coarse_names.get(k, k), 'count': v} for k, v in coarse.most_common()],
            brand_counts=dict(brands.most_common())))
selected_roots = [215, 199, 179]
trees = []
for root in selected_roots:
    branch = [node for node in nodes if node['prefix'][0] == root]
    trees.append({'root': [root], 'nodes': branch,
        'products': [{'item_id': i, **labels[i], 'text': texts[i]} for i in branch[0]['item_ids']]})
target = out / 'prefix_tree_inspection_beauty_20261111'
write_json(target / 'all_prefix_nodes.json', {'created_at': datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
    'provenance': {'labels_sha256': receipt['sha256'], 'taxonomy_sha256': sha(taxonomy_path),
        'sid_sha256': audit['sid_sha256'], 'script_sha256': sha(__file__)},
    'selection_note': 'Exploratory examples selected after scanning labels; not a confirmatory sample or significance test.',
    'nodes': nodes})
write_json(target / 'selected_trees_with_products.json', {'selected_roots': selected_roots, 'trees': trees})
print('SAVED', target, 'NODES', len(nodes), flush=True)
