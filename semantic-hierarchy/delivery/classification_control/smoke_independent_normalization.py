import sys
import json
from pathlib import Path
ROOT = Path('/home/jcy/GRID_wmh_Qwen_TIGER_20261009')
SCIENCE = ROOT/'codex/sid_prefix_semantics_v5_normalized'
sys.path.insert(0, str(SCIENCE))
from inference import Engine
from common import load_config, write_json
from prompts import SYSTEM
from compact_taxonomy import normalization_prompt, routing_schema, validate_routes, decode_routes

names = ['Card Games', 'Dart Games', 'Stuffed Animals', 'Dolls',
         'Egypt Mummy Excavation Kit', 'RC Aircraft', 'Vehicle Batteries',
         'Thomas Wooden Railway - Thomas The Tank Engine']
items = [{'source_id': i, 'name': name} for i, name in enumerate(names)]
cfg = load_config()
engine = Engine(cfg, ROOT/cfg['model'], [0, 1])
log = ROOT/'codex/sid_prefix_semantics_control/independent_normalization_smoke.json'
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
write_json(log.with_name('independent_normalization_smoke_verified.json'), {
    'passed_structural_and_basic_semantic_checks': True, 'not_a_human_accuracy_estimate': True,
    'real_taxonomy_not_created': True, 'items': result['items']})
print('independent_normalization_smoke_passed', flush=True)
