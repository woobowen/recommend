"""Finite, itemized text-only taxonomy normalization with complete source mappings."""
import hashlib
import json
import random
import re
import unicodedata
from pathlib import Path
from common import canonical_taxonomy, write_json
from prompts import CLASSIFY, SYSTEM


def compact(value):
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'))


def normalize(name):
    text = unicodedata.normalize('NFKC', name).casefold().replace('&', ' and ')
    return re.sub(r'[^\w]+', ' ', text).strip()


def candidate_names(candidates):
    unique = {}
    for c in candidates:
        name = re.sub(r'\s+', ' ', unicodedata.normalize('NFKC', c['name'])).strip()
        if not name:
            raise ValueError('Empty candidate name')
        unique.setdefault(normalize(name), name)
    return [unique[k] for k in sorted(unique)]


def classifier_taxonomy(taxonomy):
    columns = ['id', 'coarse_name', 'name', 'definition']
    for optional in ['include', 'exclude']:
        if any(c.get(optional) for c in taxonomy['categories']):
            columns.append(optional)
    # A lossless table: header once, every value retained, no category truncation.
    return {'columns': columns, 'categories': [[c.get(k, '') for k in columns]
        for c in taxonomy['categories']], 'unknown': taxonomy['unknown']}


def routing_schema(ids, category_ids):
    fields = {'coarse_name': {'type': 'string', 'minLength': 1, 'maxLength': 48},
        'name': {'type': 'string', 'minLength': 1, 'maxLength': 64},
        'definition': {'type': 'string', 'minLength': 1, 'maxLength': 100}}
    entry = {'type': 'object', 'properties': fields, 'required': list(fields), 'additionalProperties': False}
    keys = {f's{i}': entry for i in ids}
    return {'type': 'object', 'properties': {'items': {'type': 'object',
        'properties': keys, 'required': list(keys), 'additionalProperties': False}},
        'required': ['items'], 'additionalProperties': False}


def decode_routes(value, ids):
    records = value['items']
    if not isinstance(records, dict) or set(records) != {f's{i}' for i in ids}:
        raise ValueError('Source keys incomplete')
    return [dict(records[f's{i}'], source_id=i, target='NEW') for i in ids]


def validate_routes(rows, ids, categories):
    if len(rows) != len(ids) or {r['source_id'] for r in rows} != set(ids):
        raise ValueError('Missing or duplicated source IDs')
    existing = {c['id'] for c in categories}
    for r in rows:
        if r['target'] != 'NEW' and r['target'] not in existing:
            raise ValueError('Invalid taxonomy target')
        if r['target'] == 'NEW' and any(not r[k].strip() for k in ['coarse_name', 'name', 'definition']):
            raise ValueError('New functional type requires a name, parent and definition')


def normalization_prompt(batch, overview):
    return ('''For EACH source below, identify its GENERAL FUNCTIONAL PRODUCT TYPE independently.
Write a concise canonical English category name, functional parent family, and definition.
Do not choose from a restricted existing taxonomy: invent the correct general type when needed.
Only actual synonyms should get the same leaf name. Distinct functions must stay distinct.
Normalize titles/brands/variants to general product types, e.g. a branded train becomes Toy Trains,
building sets with figures/vehicles become Building Sets. Never copy a brand/product title as
a category. No classes based on color, material, accessories-included, or figures-included.
Semantic checks: Card Games are NOT Dart Games; Stuffed Animals are NOT Dolls; an Excavation Kit
is a Science Kit, NOT a Dart Game; RC Vehicles are distinct from replacement batteries.
Do not replace a general source with a narrower unrelated specific type. Use natural conventional
product-type categories; do not use the universal roots Toys/Games/Miscellaneous as fine leaves.
Return EXACTLY one
record per source ID, keyed by s followed by source_id:
{"items":{"s0":{"coarse_name":"functional family",
"name":"general product type","definition":"brief function"}}}.
Repeated canonical names across synonymous sources are correct. Choose class count through
the type names, no fixed target count. No SID or agreement scores are supplied.
SOURCE PROPOSALS:\n''' + compact({f"s{r['source_id']}": r['name'] for r in batch}))


def merge_global(candidates, generate, engine, cfg, out, stage, items):
    names = candidate_names(candidates)
    digest = hashlib.sha256(compact(candidates).encode()).hexdigest()
    checkpoint = Path(out)/'local_consolidation_v6'/stage/digest
    checkpoint.mkdir(parents=True, exist_ok=True)
    write_json(checkpoint/'inputs.json', {'source_sha256': digest,
        'candidate_records': len(candidates), 'unique_names': names,
        'normalization': 'Unicode/case/punctuation/ampersand spelling only',
        'no_sid_or_metrics': True})
    source = [{'source_id': i, 'name': name} for i, name in enumerate(names)]
    pass_reports = []
    # Two finite passes, each with a mapping for EVERY source. No shrink-until-done loop.
    for pass_index in range(2):
        categories, routes = [], []
        shuffled = list(source)
        random.Random(cfg['seed'] + pass_index).shuffle(shuffled)
        for start in range(0, len(shuffled), 12):
            batch = shuffled[start:start+12]
            ids = [r['source_id'] for r in batch]
            cache = checkpoint/f'pass{pass_index}_batch{start:04d}.json'
            before = hashlib.sha256(compact(categories).encode()).hexdigest()
            if cache.exists():
                stored = json.loads(cache.read_text())
                assert stored['inputs'] == batch and stored['taxonomy_before_sha256'] == before
                rows = stored['items']
                validate_routes(rows, ids, categories)
            else:
                prompt = normalization_prompt(batch, [r['name'] for r in source])
                schema = routing_schema(ids, [c['id'] for c in categories])
                rows = None
                for attempt in range(2):
                    if not engine.fits(SYSTEM, prompt, 1536):
                        raise ValueError('Mapping context exhausted; no sources silently dropped')
                    try:
                        value = generate(prompt, 1536, schema)
                        rows = decode_routes(value, ids)
                        validate_routes(rows, ids, categories)
                        break
                    except (ValueError, KeyError, TypeError):
                        if attempt:
                            raise
                write_json(cache, {'inputs': batch, 'taxonomy_before_sha256': before, 'items': rows})
            by_source = {r['source_id']: r for r in rows}
            for item in batch:
                row = by_source[item['source_id']]
                if row['target'] == 'NEW':
                    key = normalize(row['name'])
                    matches = [c for c in categories if normalize(c['name']) == key]
                    if matches:
                        target = matches[0]['id']
                    else:
                        target = f'T{len(categories)+1:03d}'
                        parent_key = normalize(row['coarse_name'])
                        parents = [c['coarse_name'] for c in categories if normalize(c['coarse_name']) == parent_key]
                        categories.append({'id': target, 'coarse_name': parents[0] if parents else row['coarse_name'].strip(),
                            'name': row['name'].strip(), 'definition': row['definition'].strip()})
                else:
                    target = row['target']
                routes.append({'source_id': item['source_id'], 'source_name': item['name'], 'target': target})
            print('coverage_mapping', stage, 'pass', pass_index+1, 'sources', len(routes), '/', len(source),
                  'categories', len(categories), flush=True)
        assert len(routes) == len(source) and {r['source_id'] for r in routes} == {r['source_id'] for r in source}
        report = {'pass': pass_index+1, 'sources': source, 'routes': routes, 'categories': categories,
                  'every_source_mapped_once': True}
        write_json(checkpoint/f'pass{pass_index}_complete.json', report)
        pass_reports.append(report)
        source = [dict(c, source_id=i) for i, c in enumerate(categories)]
    final_rows = [dict(c, include='', exclude='') for c in categories]
    for c in final_rows:
        c.pop('id')
    result = canonical_taxonomy({'categories': final_rows})
    view = compact(classifier_taxonomy(result))
    worst = max(items, key=lambda r: engine.encode(SYSTEM, compact(r)).shape[-1])
    prompt = CLASSIFY.replace('{taxonomy}', view).replace('{products}', compact([worst]))
    if not engine.fits(SYSTEM, prompt, cfg['classification_max_new_tokens']):
        raise ValueError('Taxonomy does not fit complete item context; no silent truncation')
    second = {r['source_id']: r['target'] for r in pass_reports[1]['routes']}
    first_category_index = {c['id']: i for i,c in enumerate(pass_reports[0]['categories'])}
    final_by_name = {normalize(c['name']): c['id'] for c in result['categories']}
    last_by_id = {c['id']: c for c in categories}
    end_to_end = [dict(r, final_category_id=final_by_name[normalize(last_by_id[
        second[first_category_index[r['target']]]]['name'])]) for r in pass_reports[0]['routes']]
    assert len(end_to_end) == len(names)
    write_json(checkpoint/'complete_source_mapping.json', end_to_end)
    result['consolidation'] = {'method': 'local_keyed_functional_normalization_complete_mappings_v6',
        'source_sha256': digest, 'candidate_records': len(candidates), 'unique_names': len(names),
        'all_names_mapped_once': True, 'mapping_checkpoint': str(checkpoint/'complete_source_mapping.json'),
        'fixed_category_count': False, 'no_sid_or_metrics': True,
        'classification_context_check_passed': True,
        'worst_single_product_input_tokens': engine.encode(SYSTEM, prompt).shape[-1]}
    write_json(checkpoint/'result.json', result)
    print('global_consolidation_complete', stage, 'leaves', len(result['categories']), flush=True)
    return result
