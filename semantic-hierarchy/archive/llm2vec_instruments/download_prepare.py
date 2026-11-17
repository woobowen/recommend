"""Fetch official Amazon 2018 Instruments and align official 5-core item ASINs."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import gzip
import html
import json
from pathlib import Path
import re
import time
import urllib.request

ROOT = Path('/home/jcy/GRID_wmh_Qwen_TIGER_20261009')
HERE = ROOT / 'codex/instrument_semantic_hierarchy_20261115'
DATA = Path('/home/jcy/data/amazon2018_instruments')
from runtime_common import sha, write_json, write_jsonl, status

FILES = {
    'Musical_Instruments_5.json.gz': ('https://mcauleylab.ucsd.edu/public_datasets/data/amazon_v2/categoryFilesSmall/Musical_Instruments_5.json.gz', 39353085),
    'meta_Musical_Instruments.json.gz': ('https://mcauleylab.ucsd.edu/public_datasets/data/amazon_v2/metaFiles2/meta_Musical_Instruments.json.gz', 173201671),
}


def clean(value):
    if value is None:
        return ''
    if isinstance(value, list):
        return ', '.join(filter(None, (clean(x) for x in value)))
    return re.sub(r'\s+', ' ', html.unescape(str(value))).strip()


def fetch(name, spec):
    url, expected = spec
    target = DATA / 'raw' / name
    if not target.exists():
        partial = target.with_suffix(target.suffix + '.partial')
        for attempt in range(5):
            try:
                cursor = partial.stat().st_size if partial.exists() else 0
                if cursor == expected:
                    partial.replace(target)
                    break
                headers = {'Range': f'bytes={cursor}-'} if cursor else {}
                with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=60) as stream:
                    append = bool(cursor and stream.status == 206)
                    if not append:
                        cursor = 0
                    with partial.open('ab' if append else 'wb') as out:
                        last = 0
                        while True:
                            chunk = stream.read(1024 * 1024)
                            if not chunk:
                                break
                            out.write(chunk); cursor += len(chunk)
                            if time.monotonic() - last > 10:
                                status(DATA / f'{name}.status.json', 'downloading', downloaded=cursor, expected=expected, url=url)
                                last = time.monotonic()
                assert partial.stat().st_size == expected, 'Unexpected byte length'
                partial.replace(target)
                break
            except Exception:
                if attempt == 4:
                    raise
                time.sleep(3)
    assert target.stat().st_size == expected
    count = 0; fields = set()
    # Reading to EOF validates gzip CRC and trailer as well as every JSON line.
    with gzip.open(target, 'rt', encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line); count += 1; fields.update(row)
    result = {'url': url, 'path': str(target), 'bytes': expected, 'sha256': sha(target),
              'json_records': count, 'fields': sorted(fields), 'gzip_crc_and_json_valid': True,
              'official_published_checksum_available': False}
    status(DATA / f'{name}.status.json', 'complete', **{('file_path' if k == 'path' else k): v for k, v in result.items()})
    return name, result


def download():
    (DATA / 'raw').mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=2) as pool:
        records = dict(pool.map(lambda pair: fetch(*pair), FILES.items()))
    assert records['Musical_Instruments_5.json.gz']['json_records'] == 231392
    result = {'complete': True, 'version': 'Amazon Reviews 2018 / amazon_v2', 'files': records}
    write_json(DATA / 'download_manifest.json', result)
    return result


def prepare():
    manifest = json.loads((DATA / 'download_manifest.json').read_text())
    assert manifest['complete']
    reviews = []
    with gzip.open(DATA / 'raw/Musical_Instruments_5.json.gz', 'rt', encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line)
            reviews.append({key: row[key] for key in ['reviewerID', 'asin', 'unixReviewTime', 'overall']})
    asins = {row['asin'] for row in reviews}
    user_counts = Counter(row['reviewerID'] for row in reviews)
    item_counts = Counter(row['asin'] for row in reviews)
    # The official file is named 5-core, but its released records include degrees <5.
    # Preserve the supplied item universe for this item-semantics experiment and audit it.
    selected = {}; record_counts = Counter(); meta_count = 0
    with gzip.open(DATA / 'raw/meta_Musical_Instruments.json.gz', 'rt', encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line); meta_count += 1
            asin = row.get('asin')
            if asin not in asins:
                continue
            record_counts[asin] += 1
            # A deterministic, SID-independent choice for duplicate metadata records.
            def score(record):
                return (bool(clean(record.get('title'))), len(clean(record.get('description'))),
                        len(clean(record.get('feature'))), json.dumps(record, sort_keys=True, ensure_ascii=True))
            if asin not in selected or score(row) > score(selected[asin]):
                selected[asin] = row
    missing = sorted(asins - selected.keys())
    blank = sorted(a for a in selected if not clean(selected[a].get('title')))
    retained = sorted(set(selected) - set(blank))
    assert len(retained) >= 3072, 'Insufficient items for the original quantizer initialization buffer'
    inputs = HERE / 'inputs/instrument'; inputs.mkdir(parents=True, exist_ok=True)
    (DATA / 'items').mkdir(exist_ok=True)
    mappings=[]; titles=[]; brands=[]; qwen=[]; metadata=[]
    for item_id, asin in enumerate(retained):
        row = selected[asin]; title = clean(row.get('title')); brand = clean(row.get('brand'))
        category = clean(row.get('category')); price = clean(row.get('price'))
        good_brand = brand if brand.lower() not in {'', 'unknown', 'none', 'nan'} else ''
        mappings.append({'item_id': item_id, 'asin': asin})
        titles.append({'item_id': item_id, 'text': 'Title: ' + title})
        brands.append({'item_id': item_id, 'text': 'Title: ' + title + ('\nBrand: ' + good_brand if good_brand else '')})
        # Matches the old Beauty/Toys serialization, including existing upstream categories.
        qwen.append({'item_id': item_id, 'text': f'Title: {title}; Brand: {brand or "Unknown"}; Categories: {category or "Unknown"}; Price: {price or "Unknown"}'})
        metadata.append({'item_id': item_id, **row})
    for name, rows in [('item_mapping.jsonl', mappings), ('items_title.jsonl', titles),
                       ('items_title_brand.jsonl', brands), ('items_qwen.jsonl', qwen),
                       ('items_metadata.jsonl', metadata)]:
        write_jsonl(inputs / name, rows)
    mapping = {row['asin']: row['item_id'] for row in mappings}
    retained_reviews = [{**row, 'item_id': mapping[row['asin']]} for row in reviews if row['asin'] in mapping]
    write_jsonl(DATA / 'interactions_5core_aligned.jsonl', retained_reviews)
    write_json(DATA / 'excluded_items.json', {'missing_metadata_asins': missing, 'blank_title_asins': blank})
    audit = {'complete': True, 'data_version': '2018', 'source_subset': 'official 5-core',
             'raw_review_records': len(reviews), 'raw_users': len(user_counts), 'raw_distinct_items': len(asins),
             'minimum_raw_user_records': min(user_counts.values()), 'minimum_raw_item_records': min(item_counts.values()),
             'raw_users_with_under_5_records': sum(n<5 for n in user_counts.values()),
             'raw_items_with_under_5_records': sum(n<5 for n in item_counts.values()),
             'metadata_records': meta_count, 'metadata_missing_items': len(missing), 'blank_title_items': len(blank),
             'duplicate_metadata_asins': sum(n>1 for n in record_counts.values()),
             'duplicate_metadata_extra_records': sum(n-1 for n in record_counts.values()),
             'retained_items': len(retained), 'retained_review_records': len(retained_reviews),
             'duplicate_user_item_extra_records': len(reviews)-len({(x['reviewerID'],x['asin']) for x in reviews}),
             'duplicate_title_extra_items': len(titles)-len({x['text'] for x in titles}),
             'title_only_primary': True, 'title_brand_secondary': True, 'qwen_uses_categories_and_price': True,
             'descriptions_preserved_but_not_encoded_for_protocol_consistency': True,
             'no_post_join_recore_for_item_semantics_experiment': True,
             'selection_rule': 'all distinct ASINs in official 5-core with matching nonblank metadata title; sort ASIN for IDs; no SID dependent selection',
             'inputs_sha256': {p.name: sha(p) for p in inputs.glob('*.jsonl')},
             'download_manifest_sha256': sha(DATA / 'download_manifest.json')}
    write_json(HERE / 'preparation/input_preparation_verified.json', audit)
    status(HERE / 'preparation/status.json', 'inputs_prepared', items=len(retained))
    return audit


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(); parser.add_argument('--download-only', action='store_true')
    args = parser.parse_args()
    download()
    if not args.download_only:
        print(json.dumps(prepare(), ensure_ascii=False), flush=True)
