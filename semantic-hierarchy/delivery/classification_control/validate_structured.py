"""Validate constrained decoding on synthetic products, without fixed taxonomy names."""
from pathlib import Path
import sys
import json

ROOT = Path('/home/jcy/GRID_wmh_Qwen_TIGER_20261009')
sys.path.insert(0, str(ROOT/'codex/sid_prefix_semantics_v1'))
from common import roots, canonical_taxonomy, write_json
from schemas import taxonomy_schema, classification_schema
from inference import Engine
from prompts import SYSTEM, DISCOVER, CLASSIFY


def main():
    from lmformatenforcer import JsonSchemaParser
    schema = classification_schema([1, 2], ['F001', 'F002', 'UNKNOWN'])
    valid = '{"items":[{"item_id":1,"category_id":"F001","needs_review":false},{"item_id":2,"category_id":"F002","needs_review":false}]}'
    parser = JsonSchemaParser(schema)
    for char in valid:
        assert char in parser.get_allowed_characters(), (char, valid)
        parser = parser.add_character(char)
    assert parser.can_end()
    cfg, root, prep, out = roots()
    engine = Engine(cfg, root/cfg['model'], [0, 1])
    items = [{'item_id': 1, 'text': 'A red lipstick for lip makeup.'},
             {'item_id': 2, 'text': 'A shampoo for washing hair.'}]
    result = engine.generate(SYSTEM, DISCOVER+json.dumps(items), 1024,
        prep/'structured_taxonomy_smoke.json', schema=taxonomy_schema(len(items)))
    taxonomy = canonical_taxonomy(result)
    prompt = CLASSIFY.replace('{taxonomy}', json.dumps(taxonomy)).replace('{products}', json.dumps(items))
    labels = engine.generate(SYSTEM, prompt, 256, prep/'structured_classification_smoke.json',
        schema=classification_schema([1,2], [c['id'] for c in taxonomy['categories']]+['UNKNOWN']))
    assert {r['item_id'] for r in labels['items']} == {1,2}
    assert len(labels['items']) == 2
    write_json(prep/'structured_decoding_verified.json', {
        'passed': True, 'synthetic_only': True, 'taxonomy_categories': len(taxonomy['categories']),
        'constrained_fields': ['JSON structure', 'bounded strings and array lengths', 'valid item/category IDs'],
        'lm_format_enforcer': '0.11.3', 'interegular': '0.3.3'})
    print('structured_decoding_smoke_passed', flush=True)


if __name__ == '__main__':
    main()
