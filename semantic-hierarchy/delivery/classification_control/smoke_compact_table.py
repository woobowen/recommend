import sys
import json
from pathlib import Path
ROOT = Path('/home/jcy/GRID_wmh_Qwen_TIGER_20261009')
SCIENCE = ROOT/'codex/sid_prefix_semantics_v7_compact'
sys.path.insert(0, str(SCIENCE))
from inference import Engine
from common import load_config, write_json
from prompts import SYSTEM
from compact_taxonomy import normalization_prompt, routing_schema, validate_routes, decode_routes
from compact_taxonomy import classifier_taxonomy
from common import canonical_taxonomy
from prompts import CLASSIFY
from schemas import classification_schema

names = ['Card Games', 'Dart Games', 'Stuffed Animals', 'Dolls',
         'Egypt Mummy Excavation Kit', 'RC Aircraft', 'Vehicle Batteries',
         'Thomas Wooden Railway - Thomas The Tank Engine']
names += ['Shopping Carts', 'Statues', 'Galt Toys Marble Run Toy Set',
          'Melissa &amp; Doug Hi-Rise Dollhouse and Furniture Set']
items = [{'source_id': i, 'name': name} for i, name in enumerate(names)]
cfg = load_config()
engine = Engine(cfg, ROOT/cfg['model'], [2, 3])
log = ROOT/'codex/sid_prefix_semantics_control/compact_table_smoke.json'
result = engine.generate(SYSTEM, normalization_prompt(items, names), 1536, log,
                         routing_schema(list(range(len(items))), []))
rows = decode_routes(result, list(range(len(items))))
validate_routes(rows, list(range(len(items))), [])
print(json.dumps(result, ensure_ascii=False), flush=True)
by_id = {r['source_id']: r for r in rows}
assert by_id[0]['name'].casefold() != by_id[1]['name'].casefold()
assert by_id[2]['name'].casefold() != by_id[3]['name'].casefold()
assert by_id[5]['name'].casefold() != by_id[6]['name'].casefold()
assert 'dart' not in by_id[4]['name'].casefold()
assert 'thomas' not in by_id[7]['name'].casefold()
assert 'statue' not in by_id[8]['name'].casefold()
assert 'doll' not in by_id[9]['name'].casefold()
write_json(log.with_name('compact_table_smoke_verified.json'), {
    'passed_structural_and_basic_semantic_checks': True, 'not_a_human_accuracy_estimate': True,
    'real_taxonomy_not_created': True, 'items': result['items']})
taxonomy = canonical_taxonomy({'categories': [
    {'coarse_name': 'Cosmetics', 'name': 'Lipstick', 'definition': 'Lip coloring cosmetics'},
    {'coarse_name': 'Hair Care', 'name': 'Shampoo', 'definition': 'Hair cleaning products'}]})
products = [{'item_id': 1, 'text': 'A red lipstick for lip makeup.'},
            {'item_id': 2, 'text': 'A shampoo for washing hair.'}]
prompt = CLASSIFY.replace('{taxonomy}', json.dumps(classifier_taxonomy(taxonomy))).replace('{products}', json.dumps(products))
answer = engine.generate(SYSTEM, prompt, 256, log.with_name('compact_classifier_smoke.json'),
    classification_schema([1, 2], [c['id'] for c in taxonomy['categories']] + ['UNKNOWN']))
by_item = {r['item_id']: r['category_id'] for r in answer['items']}
by_name = {r['name']: r['id'] for r in taxonomy['categories']}
assert by_item == {1: by_name['Lipstick'], 2: by_name['Shampoo']}
write_json(log.with_name('compact_classifier_smoke_verified.json'), {'passed': True, 'synthetic_only': True})
print('compact_table_smoke_passed', flush=True)
