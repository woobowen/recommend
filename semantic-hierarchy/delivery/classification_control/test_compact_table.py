import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent/'sid_prefix_semantics_v7_compact'))
from compact_taxonomy import merge_global, validate_routes, candidate_names, decode_routes, classifier_taxonomy


def main():
    categories = [{'id': 'F001', 'coarse_name': 'Family', 'name': 'Type',
                   'definition': 'Function', 'include': 'preserve boundary', 'exclude': ''}]
    table = classifier_taxonomy({'categories': categories, 'unknown': {'id': 'UNKNOWN'}})
    restored = [dict(zip(table['columns'], row)) for row in table['categories']]
    assert restored[0]['include'] == 'preserve boundary'
    assert all(restored[0][k] == categories[0][k] for k in table['columns'])
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
        obj = json.loads(prompt.split('SOURCE PROPOSALS:\n')[1])
        src = [{'source_id': int(k[1:]), 'name': v} for k,v in obj.items()]
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
    failed_calls = []
    def missing_source(*args):
        failed_calls.append(1)
        return {'items': {}}
    with tempfile.TemporaryDirectory() as td:
        try:
            merge_global(source, missing_source, engine, cfg, td, 'discover',
                         [{'item_id': 1, 'text': 'synthetic'}])
        except ValueError:
            assert len(failed_calls) == 2, 'Failure did not stop at the retry bound'
        else:
            raise AssertionError('Missing mappings were accepted')
    print('normalized_convergence_regressions_passed')


if __name__ == '__main__':
    main()
