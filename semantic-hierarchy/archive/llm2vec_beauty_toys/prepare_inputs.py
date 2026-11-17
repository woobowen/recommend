"""CPU-only item alignment and SID inventory; never performs semantic inference."""
from collections import Counter
import hashlib
import html
import json
from pathlib import Path
import re
import torch

ROOT = Path('/home/jcy/GRID_wmh_Qwen_TIGER_20261009')
HERE = ROOT / 'codex/llm2vec_sid_hierarchy_20261113'
RUNS = {'beauty': 'beauty_rkmeans_qwen_20261010_012253',
        'toys': 'toys_rkmeans_qwen_20261010_093036'}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows))


summary = {'experiment_started': False, 'datasets': {}}
for dataset, run in RUNS.items():
    source = ROOT / 'codex/sid_prefix_semantics_v1/preparation' / dataset / 'classifier_items.jsonl'
    sid_path = ROOT / 'experiments' / run / 'semantic_ids.pt'
    rows = [json.loads(line) for line in source.read_text().splitlines()]
    sids = torch.load(sid_path, map_location='cpu', weights_only=True).T.long()
    assert sids.shape == (len(rows), 4)
    assert [row['item_id'] for row in rows] == list(range(len(rows)))
    main, brand_control = [], []
    for row in rows:
        # Existing TFRecords contain Title/Brand/Categories/Price, not descriptions.
        # Parse the known serialization without copying category labels into embeddings.
        text = row['text']
        assert text.startswith('Title: ') and '; Categories: ' in text and '; Price: ' in text
        head, _ = text.rsplit('; Categories: ', 1)
        title_field, brand = head.rsplit('; Brand: ', 1)
        title = re.sub(r'\s+', ' ', html.unescape(title_field[len('Title: '):])).strip()
        assert title, row['item_id']
        clean_brand = re.sub(r'\s+', ' ', html.unescape(brand)).strip()
        main.append({'item_id': row['item_id'], 'text': 'Title: ' + title})
        control = 'Title: ' + title
        if clean_brand.lower() not in {'unknown', 'nan', 'none', ''}:
            control += '\nBrand: ' + clean_brand
        brand_control.append({'item_id': row['item_id'], 'text': control})
    out = HERE / 'inputs' / dataset
    write_jsonl(out / 'items_title.jsonl', main)
    write_jsonl(out / 'items_title_brand.jsonl', brand_control)
    write_jsonl(out / 'item_sids.jsonl', [{'item_id': i, 'sid': sid} for i, sid in enumerate(sids.tolist())])
    groups = {level: Counter(tuple(sid[:level]) for sid in sids.tolist()) for level in [1, 2, 3]}
    all_nodes = [{'prefix': list(prefix), 'level': level, 'count': count}
                 for level, mapping in groups.items() for prefix, count in mapping.items()]
    all_nodes.sort(key=lambda node: (-node['count'], node['level'], node['prefix']))
    for rank, node in enumerate(all_nodes, 1):
        node['global_frequency_rank'] = rank
    write_jsonl(out / 'prefix_inventory.jsonl', all_nodes)
    paths = []
    for prefix, count in sorted(groups[3].items(), key=lambda item: (-item[1], item[0])):
        if count < 2:
            continue
        counts = [groups[level][prefix[:level]] for level in [1, 2, 3]]
        paths.append({'prefix': list(prefix), 'counts': counts,
                      'proper_refinement_both_edges': counts[0] > counts[1] > counts[2]})
    write_jsonl(out / 'shared_paths.jsonl', paths)
    summary['datasets'][dataset] = {
        'n_items': len(rows), 'source_sha256': sha(source), 'sid_sha256': sha(sid_path),
        'title_text_sha256': sha(out / 'items_title.jsonl'),
        'blank_titles': sum(not row['text'][len('Title: '):].strip() for row in main),
        'duplicate_titles': len(main) - len({row['text'] for row in main}),
        'description_available_in_current_item_text': False,
        'shared_complete_paths': len(paths),
        'proper_paths': {str(m): sum(p['proper_refinement_both_edges'] and p['counts'][2] >= m for p in paths)
                         for m in [2, 3, 5, 10]},
        'unchanged_membership_paths': sum(not p['proper_refinement_both_edges'] for p in paths),
        'level_thresholds': {str(level): {str(m): {
            'groups': sum(c >= m for c in mapping.values()),
            'items': sum(c for c in mapping.values() if c >= m),
            'coverage': sum(c for c in mapping.values() if c >= m) / len(rows),
        } for m in [2, 3, 5, 10, 20]} for level, mapping in groups.items()},
    }
(HERE / 'preparation/input_preparation_verified.json').write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
