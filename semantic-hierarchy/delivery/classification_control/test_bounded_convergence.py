"""Synthetic bounded consolidation and context-failure regressions."""
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent/'sid_prefix_semantics_v3_bounded'))
from compact_taxonomy import candidate_names, merge_global, validate_outline


def main():
    inputs = [{'name': 'Blocks'}, {'name': ' blocks '}, {'name': 'Rare telescope kit'}]
    assert candidate_names(inputs) == ['Blocks', 'Rare telescope kit']
    try:
        validate_outline({'categories': [{'coarse_name': 'A', 'name': 'Blocks'},
                                         {'coarse_name': 'B', 'name': 'blocks'}]})
    except ValueError:
        pass
    else:
        raise AssertionError('Duplicate names crossed parents')
    calls = []
    def generate(prompt, budget, schema):
        calls.append(prompt)
        if '"leaf_count"' in prompt:
            return {'leaf_count': 2}
        if 'CANDIDATE NAMES:' in prompt:
            assert 'Rare telescope kit' in prompt
            assert schema['properties']['categories']['minItems'] == 2
            assert schema['properties']['categories']['maxItems'] == 2
            return {'categories': [{'coarse_name': 'Construction', 'name': 'Building blocks'},
                                   {'coarse_name': 'Science', 'name': 'Telescope kits'}]}
        return {'items': [{'index': 0, 'definition': 'Blocks for toy construction'},
                          {'index': 1, 'definition': 'Optical observation kits'}]}
    engine = SimpleNamespace(fits=lambda *a: True, encode=lambda *a: SimpleNamespace(shape=(1, 123)))
    cfg = {'taxonomy_max_new_tokens': 4096, 'classification_max_new_tokens': 1024}
    with tempfile.TemporaryDirectory() as tmp:
        result = merge_global(inputs, generate, engine, cfg, tmp, 'discover', [{'item_id': 0, 'text': 'synthetic'}])
        assert len(calls) == 3 and len(result['categories']) == 2
        merge_global(inputs, generate, engine, cfg, tmp, 'discover', [{'item_id': 0, 'text': 'synthetic'}])
        assert len(calls) == 3, 'Successful checkpoints were regenerated'
    with tempfile.TemporaryDirectory() as tmp:
        fails = []
        def invalid(*a):
            if '"leaf_count"' in a[0]:
                return {'leaf_count': 2}
            fails.append(1)
            raise ValueError('synthetic truncation')
        try:
            merge_global(inputs, invalid, engine, cfg, tmp, 'discover', [{'item_id': 0, 'text': 'synthetic'}])
        except RuntimeError:
            assert len(fails) == 3
        else:
            raise AssertionError('Failure did not stop')
    print('compact_convergence_regressions_passed')


if __name__ == '__main__':
    main()
