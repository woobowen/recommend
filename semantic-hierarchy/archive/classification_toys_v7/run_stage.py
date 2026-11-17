"""Explicit stage runner. Preparation never calls this file on real products.

Run only after the user confirms starting the experiment. Each stage is separate
so the pilot and proposed taxonomy can be inspected before full final annotation.
"""
import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import time

from common import roots, sha, write_json, read_jsonl, write_jsonl, canonical_taxonomy
from prompts import SYSTEM, DISCOVER, MERGE, REVISE, CLASSIFY
from schemas import taxonomy_schema, classification_schema
from compact_taxonomy import merge_global, classifier_taxonomy


def compact(value):
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('stage', choices=['discover', 'draft', 'revise', 'pilot', 'final'])
    ap.add_argument('--dataset', required=True, choices=['beauty', 'toys'])
    ap.add_argument('--gpus', default='0,1')
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--shards', type=int, default=1)
    ap.add_argument('--execute', action='store_true', help='Explicitly start this stage after user confirmation')
    args = ap.parse_args()
    if not args.execute:
        ap.error('No stage started. --execute is required after user confirmation.')
    cfg, root, prep, output = roots()
    assert 0 <= args.shard < args.shards
    if args.stage not in ('draft', 'final') and args.shards != 1:
        ap.error('Taxonomy/pilot stages must use a single shard')
    assert (prep / 'model_smoke_verified.json').exists(), 'Run synthetic model preflight first'
    assert json.loads((prep / 'model_smoke_verified.json').read_text())['passed']
    ds_prep = prep / args.dataset
    out = output / args.dataset
    out.mkdir(parents=True, exist_ok=True)
    audit = json.loads((prep / 'data_preparation_verified.json').read_text())['datasets'][args.dataset]
    assert sha(ds_prep / 'classifier_items.jsonl') == audit['classifier_items_sha256']
    source = ds_prep / ('discovery_items.jsonl' if args.stage == 'discover' else
                        'review_items.jsonl' if args.stage == 'pilot' else 'classifier_items.jsonl')
    items = read_jsonl(source)
    assert all(set(r) == {'item_id', 'text'} for r in items), 'Classifier input must be text-only'
    items = [r for r in items if r['item_id'] % args.shards == args.shard]
    taxonomy_path = out / ('taxonomy_draft.json' if args.stage == 'draft' else 'taxonomy_frozen.json')
    if args.stage in ('draft', 'pilot', 'final'):
        assert taxonomy_path.exists(), f'Missing taxonomy: {taxonomy_path}'
        taxonomy = json.loads(taxonomy_path.read_text())
        valid_labels = {c['id'] for c in taxonomy['categories']} | {'UNKNOWN'}
    destination = out / f'{args.stage}_shard{args.shard}of{args.shards}.jsonl'
    if args.stage == 'discover' and (out / 'taxonomy_draft.json').exists():
        raise FileExistsError('Draft taxonomy already exists; use a new version for a new discovery run')
    if args.stage == 'revise' and (out / 'taxonomy_frozen.json').exists():
        raise FileExistsError('Frozen taxonomy already exists; never alter it using evaluation results')
    manifest_path = destination.with_suffix('.manifest.json')
    manifest = {'stage': args.stage, 'dataset': args.dataset, 'shard': args.shard,
        'shards': args.shards, 'config_sha256': sha(Path(__file__).with_name('config.json')),
        'items_sha256': sha(source), 'code_sha256': {p.name: sha(p) for p in Path(__file__).parent.glob('*.py')},
        'model_manifest_sha256': sha(prep / 'model_download_verified.json')}
    if args.stage in ('draft', 'pilot', 'final'):
        manifest['taxonomy_sha256'] = sha(taxonomy_path)
    if args.stage == 'discover' and (out/'reused_discovery_provenance.json').exists():
        reuse_path = out/'reused_discovery_provenance.json'
        reuse = json.loads(reuse_path.read_text())
        for name, digest in reuse['batch_sha256'].items():
            assert sha(out/'discovery_batches'/name) == digest, 'Discovery cache changed'
        for name, digest in reuse.get('normalization_checkpoint_sha256', {}).items():
            assert sha(out/name) == digest, 'Normalization checkpoint changed'
        manifest['reused_discovery_provenance_sha256'] = sha(reuse_path)
    if manifest_path.exists():
        assert json.loads(manifest_path.read_text()) == manifest, 'Resume provenance differs'
    else:
        assert not destination.exists(), 'Results without provenance cannot be resumed'
        write_json(manifest_path, manifest)
    from inference import Engine
    engine = Engine(cfg, root / cfg['model'], [int(x) for x in args.gpus.split(',')])
    log_dir = out / 'raw' / f'{args.stage}_shard{args.shard}of{args.shards}'
    counter = 0

    def generate(user, max_new, schema=None):
        nonlocal counter
        counter += 1
        path = log_dir / f'call_{time.time_ns()}_{counter:06d}.json'
        return engine.generate(SYSTEM, user, max_new, path, schema=schema)

    def pack(rows, prompt_for, max_new, cap):
        batch = []
        for row in rows:
            candidate = batch + [row]
            if len(candidate) > cap or not engine.fits(SYSTEM, prompt_for(candidate), max_new):
                if not batch:
                    raise ValueError(f'Single record too long for context; inspect ID {row.get("item_id")}')
                yield batch
                batch = [row]
                if not engine.fits(SYSTEM, prompt_for(batch), max_new):
                    raise ValueError('Single record too long; silent truncation is prohibited')
            else:
                batch = candidate
        if batch:
            yield batch

    def taxonomy_call(prefix, rows):
        user = prefix + compact(rows)
        last_error = None
        for attempt in range(2):
            try:
                value = generate(user, cfg['taxonomy_max_new_tokens'], taxonomy_schema(len(rows)))
                return canonical_taxonomy(value)['categories']
            except (ValueError, KeyError, TypeError) as exc:
                last_error = exc
                correction = ('FORMAT CORRECTION: previous output was invalid: ' + str(exc)[:160] +
                    '. Return a flat categories array of leaf records. Each record has coarse_name, '
                    'name, definition, include, exclude exactly once; no parent-only records, '
                    'duplicate JSON keys, or duplicate category names within a parent.\n')
                user = correction + prefix + compact(rows)
                if not engine.fits(SYSTEM, user, cfg['taxonomy_max_new_tokens']):
                    break
        if len(rows) > 1:
            half = len(rows)//2
            return taxonomy_call(prefix, rows[:half]) + taxonomy_call(prefix, rows[half:])
        raise RuntimeError('Single-record taxonomy generation failed validation') from last_error

    def merge_candidates(candidates):
        return merge_global(candidates, generate, engine, cfg, out, args.stage, items)

    if args.stage == 'discover':
        candidates = []
        for batch_index, batch in enumerate(pack(items, lambda x: DISCOVER + compact(x), cfg['taxonomy_max_new_tokens'],
                          cfg['max_items_per_discovery_request'])):
            cache = out/'discovery_batches'/f'batch_{batch_index:04d}.json'
            if cache.exists():
                stored = json.loads(cache.read_text())
                assert stored['item_ids'] == [r['item_id'] for r in batch]
                batch_categories = stored['categories']
            else:
                batch_categories = taxonomy_call(DISCOVER, batch)
                write_json(cache, {'item_ids': [r['item_id'] for r in batch], 'categories': batch_categories})
            candidates.extend(batch_categories)
            write_json(out / 'discovery_candidates.json', {'categories': candidates})
            print('discover', args.dataset, 'completed_batch', batch_index+1,
                  'items_seen', sum(1 for r in items if r['item_id'] <= batch[-1]['item_id']), flush=True)
        draft = merge_candidates(candidates)
        draft['provenance'] = {'source': 'text-only discovery', 'config_sha256': manifest['config_sha256']}
        write_json(out / 'taxonomy_draft.json', draft)
        return

    if args.stage == 'revise':
        draft = json.loads((out / 'taxonomy_draft.json').read_text())
        labels = read_complete_labels(out, 'draft', audit['items'])
        item_map = {r['item_id']: r['text'] for r in items}
        grouped = defaultdict(list)
        for row in labels:
            grouped[row['category_id']].append(row['item_id'])
        review = []
        for cat in draft['categories']:
            ids = sorted(grouped[cat['id']])
            # Selection is by item ID, independent of SID/prefix or agreement.
            examples = [item_map[i] for i in ids[:3]]
            review.append(dict(cat, item_count=len(ids), product_examples=examples))
        if grouped['UNKNOWN']:
            review.append({'coarse_name': 'Unresolved product types', 'name': 'Unresolved product types',
                'definition': 'Inspect examples; retain UNKNOWN for genuinely insufficient text.',
                'item_count': len(grouped['UNKNOWN']),
                'product_examples': [item_map[i] for i in sorted(grouped['UNKNOWN'])[:20]]})
        write_json(out / 'text_only_taxonomy_review.json', review)
        proposals = []
        for batch in pack(review, lambda x: REVISE + compact(x), cfg['taxonomy_max_new_tokens'], 100000):
            proposals.extend(taxonomy_call(REVISE, batch))
        frozen = merge_candidates(proposals)
        frozen['provenance'] = {'source': 'text-only full draft distribution revision',
            'draft_taxonomy_sha256': sha(out / 'taxonomy_draft.json'),
            'review_sha256': sha(out / 'text_only_taxonomy_review.json'),
            'no_sid_or_prefix_data': True}
        write_json(out / 'taxonomy_frozen.json', frozen)
        return

    existing = read_jsonl(destination) if destination.exists() else []
    assert len({r['item_id'] for r in existing}) == len(existing)
    expected_ids = {r['item_id'] for r in items}
    assert {r['item_id'] for r in existing} <= expected_ids
    for r in existing:
        assert r['category_id'] in valid_labels and type(r['needs_review']) is bool
    done = {r['item_id'] for r in existing}
    pending = [r for r in items if r['item_id'] not in done]
    taxonomy_text = compact(classifier_taxonomy(taxonomy))

    def classification_prompt(rows):
        return CLASSIFY.replace('{taxonomy}', taxonomy_text).replace('{products}', compact(rows))

    def annotate(batch, stream):
        try:
            result = generate(classification_prompt(batch), cfg['classification_max_new_tokens'],
                              classification_schema([r['item_id'] for r in batch], valid_labels))
            labels = result['items']
            assert isinstance(labels, list) and len(labels) == len(batch)
            assert {r['item_id'] for r in labels} == {r['item_id'] for r in batch}
            for r in labels:
                assert type(r['item_id']) is int and r['category_id'] in valid_labels
                assert type(r['needs_review']) is bool
            for r in labels:
                stream.write(compact({'item_id': r['item_id'], 'category_id': r['category_id'],
                                      'needs_review': r['needs_review']}) + '\n')
            stream.flush()
        except (ValueError, KeyError, AssertionError, TypeError) as exc:
            if len(batch) == 1:
                raise RuntimeError(f'Invalid annotation for ID {batch[0]["item_id"]}; inspect raw logs') from exc
            half = len(batch) // 2
            annotate(batch[:half], stream)
            annotate(batch[half:], stream)

    with destination.open('a', encoding='utf-8') as stream:
        for batch in pack(pending, classification_prompt, cfg['classification_max_new_tokens'],
                          cfg['max_items_per_classification_request']):
            annotate(batch, stream)
            print(args.stage, args.dataset, 'last_item', batch[-1]['item_id'], flush=True)
    labels = read_jsonl(destination)
    assert len(labels) == len(items) and {r['item_id'] for r in labels} == expected_ids
    write_json(destination.with_suffix('.complete.json'), dict(manifest, rows=len(labels), sha256=sha(destination)))
    if args.stage == 'pilot':
        text_map = {r['item_id']: r['text'] for r in items}
        names = {c['id']: c['name'] for c in taxonomy['categories']}
        with (out / 'pilot_manual_review.csv').open('w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=['item_id', 'text', 'category_id', 'category_name',
                'needs_review', 'human_correct', 'human_comment'])
            writer.writeheader()
            for r in sorted(labels, key=lambda r: r['item_id']):
                writer.writerow(dict(r, text=text_map[r['item_id']], category_name=names.get(r['category_id'], 'UNKNOWN'),
                                     human_correct='', human_comment=''))


def read_complete_labels(out, stage, expected_items):
    paths = sorted(out.glob(f'{stage}_shard*of*.jsonl'))
    if not paths:
        raise ValueError('Missing labels for stage ' + stage)
    rows = []
    shard_specs = set()
    seen_shards = set()
    for path in paths:
        complete = path.with_suffix('.complete.json')
        assert complete.exists(), f'Incomplete shard {path}'
        info = json.loads(complete.read_text())
        assert sha(path) == info['sha256']
        shard_specs.add((info['shards'], info['taxonomy_sha256'], info.get('config_sha256'),
                         info.get('items_sha256'), info.get('model_manifest_sha256'),
                         json.dumps(info.get('code_sha256'), sort_keys=True)))
        seen_shards.add(info['shard'])
        rows.extend(read_jsonl(path))
    assert len(shard_specs) == 1, 'Mixed shard layouts or taxonomies'
    shards = next(iter(shard_specs))[0]
    assert seen_shards == set(range(shards))
    assert len(rows) == expected_items
    assert {r['item_id'] for r in rows} == set(range(expected_items)), 'Duplicate, missing or extra IDs'
    return sorted(rows, key=lambda r: r['item_id'])


if __name__ == '__main__':
    main()
