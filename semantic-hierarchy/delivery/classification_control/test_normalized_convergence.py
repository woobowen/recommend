import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent/'sid_prefix_semantics_v5_normalized'))
from compact_taxonomy import merge_global, validate_routes, candidate_names, decode_routes


def main():
    assert candidate_names([{'name': 'Dolls & Accessories'}, {'name': 'dolls and accessories'}]) == ['Dolls & Accessories']
    try:
        validate_routes([{'source_id': 1}, {'source_id': 1}], [1, 2], [])
    except ValueError:
        pass
    else:
        raise AssertionError('Missing sources accepted')
    calls = []
    def generate(prompt, max_new, schema):
        calls.append(prompt)
        src = json.loads(prompt.split('SOURCE PROPOSALS:\n')[1])
        assert set(schema['properties']['items']['required']) == {f"s{x['source_id']}" for x in src}
        return {'items': {f"s{x['source_id']}": {
            'coarse_name': 'Family ' + x['name'], 'name': x['name'],
            'definition': 'Functional definition of ' + x['name']} for x in src}}
    engine = SimpleNamespace(fits=lambda *a: True, encode=lambda *a: SimpleNamespace(shape=(1, 123)))
    cfg = {'seed': 99, 'classification_max_new_tokens': 1024}
    source = [{'name': 'Blocks'}, {'name': 'Rare science kit'}, {'name': 'Remote vehicles'}]
    with tempfile.TemporaryDirectory() as td:
        result = merge_global(source, generate, engine, cfg, td, 'discover', [{'item_id': 1, 'text': 'synthetic'}])
        assert len(result['categories']) == 3 and len(calls) == 2
        mapping = json.loads(Path(result['consolidation']['mapping_checkpoint']).read_text())
        assert {r['source_name'] for r in mapping} == {r['name'] for r in source}
        assert len({r['final_category_id'] for r in mapping}) == 3
        merge_global(source, generate, engine, cfg, td, 'discover', [{'item_id': 1, 'text': 'synthetic'}])
        assert len(calls) == 2
    try:
        decode_routes({'items': {'s1': {}}}, [1, 2])
    except ValueError:
        pass
    else:
        raise AssertionError('Missing required source key accepted')
    print('normalized_convergence_regressions_passed')


if __name__ == '__main__':
    main()
