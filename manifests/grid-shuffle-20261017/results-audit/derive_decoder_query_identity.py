"""CPU derivation of decoder query token identity from audited saved beam states.

This is a derived companion to an original attention bundle. The observer saved
complete raw beam SID prefixes, ancestry and actual cross-attention query shape;
it did not separately capture T5Stack inputs_embeds. h=0 uses the learned BOS
vector, which has no raw SID integer ID. Later query identities follow the
unchanged source mapping from the actual preselection parent's prefix.
"""
import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

REPO = Path('/home/addaswsw/recommend-worktrees/sid-extensions-p0')
sys.path.insert(0, str(REPO))
from experiments.beauty_extensions.attention.records import load_bundle


def identity(path):
    path = Path(path); raw = path.read_bytes()
    return {'path': str(path), 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--job-directory', type=Path, required=True)
    p.add_argument('--attention-review', type=Path, required=True)
    p.add_argument('--sid-dictionary', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    assert not a.output.exists()
    review = json.loads(a.attention_review.read_bytes())
    assert review['status'] == 'INDEPENDENT_FORMAL_ATTENTION_RESULT_PASS'
    invocation = json.loads((a.job_directory/'output/invocation.json').read_bytes())
    spec, deployment = invocation['spec'], invocation['code_identity']
    assert review['source_commit'] == deployment['source_commit'] and review['snapshot_sha256'] == deployment['snapshot_sha256']
    assert all(review[k] == spec[k] for k in ('domain', 'route', 'shuffle_n'))
    assert review['invocation']['sha256'] == identity(a.job_directory/'output/invocation.json')['sha256']
    evaluation_review_path = Path(review['accepted_evaluation_review']['path'])
    assert identity(evaluation_review_path) == review['accepted_evaluation_review']
    evaluation_review = json.loads(evaluation_review_path.read_bytes())
    assert evaluation_review['status'] == 'FORMAL_EVALUATION_ACCEPTANCE_PASS'
    assert all(evaluation_review[k] == spec[k] for k in ('domain', 'route', 'shuffle_n', 'config', 'sid', 'checkpoint'))
    used = json.loads((a.job_directory/'output/used_config.json').read_bytes())
    width = used['model']['decoder']['embed_tokens']['num_embeddings']
    dictionary = json.loads(a.sid_dictionary.read_bytes())
    assert dictionary['source'] == spec['sid']
    assert identity(a.sid_dictionary)['sha256'] == evaluation_review['legal_vocabulary']['sha256']
    raw_sid_maximum = max(max(row) for row in dictionary['legal_raw_sids'])
    assert raw_sid_maximum + 1 == width
    assert used['model'].get('num_embeddings_per_hierarchy') in (None, width)
    source_file = 'GRID baseline/src/models/modules/semantic_id/tiger_generation_model.py' if spec['route'] == 'B' else 'references/wmh-original/GRID/src/models/modules/semantic_id/tiger_generation_model_shuffle.py'
    source_names = (source_file, 'experiments/beauty_extensions/observation.py', 'experiments/beauty_extensions/attention/capture.py', 'experiments/beauty_extensions/attention/records.py')
    expected = {r['path']: r for r in deployment['files']}
    sources = []
    for name in source_names:
        actual = identity(REPO/name)
        assert all(actual[k] == expected[name][k] for k in ('bytes', 'sha256'))
        sources.append(actual)
    samples, records, all_keys = [], [], []
    for record in sorted((a.job_directory/'output').glob('batch-*/bundle/record.json')):
        bundle = load_bundle(record)
        assert len(bundle['steps']) == 4
        inferred_table_widths = set()
        for positions in bundle['position_map']:
            for position in positions:
                if position['role'] == 'sid_lookup_vector' and position['visible']:
                    code_position = position['lookup_code_position']
                    assert position['lookup_table_index'] == position['raw_sid_value'] + code_position*width
                    if code_position:
                        inferred_table_widths.add((position['lookup_table_index']-position['raw_sid_value'])/code_position)
        assert inferred_table_widths == {width}
        records.append(identity(record)); records.append(identity(record.parent/'arrays.npz'))
        for sample, key in enumerate(bundle['capture']['sample_keys']):
            all_keys.append(key)
            paths = {}
            final_sid = bundle['final_generated_ids'][sample, 0]
            for path in ('final_top1', 'local_first'):
                queries = []
                for h, step in enumerate(bundle['steps']):
                    assert step['hierarchy'] == h and step['last_query_index'] == step['query_length_before_last_selection']-1
                    assert step['query_identity'] == ('BOS' if h == 0 else f'last token in selected prefix of length {h}')
                    if h == 0:
                        parent, prefix, query, lookup = 0, [], None, None
                    else:
                        previous = bundle['steps'][h-1]['generated_ids'][sample]
                        if path == 'final_top1':
                            matches = np.flatnonzero((previous == final_sid[:h]).all(axis=-1))
                            assert len(matches) == 1
                            parent = int(matches[0])
                            assert parent == int(bundle['paths']['final_parent_indices'][sample, h])
                        else:
                            parent = 0
                            assert np.array_equal(previous[0], step['local_first_prefix'][sample])
                        prefix = previous[parent].tolist()
                        query = int(previous[parent, -1])
                        lookup = query + (h-1)*width
                    queries.append({'target_generated_code_1based': h+1, 'preselection_parent_beam_index': parent,
                        'raw_parent_SID_prefix': prefix, 'query_kind': 'learned decoder.bos_token vector; no SID integer' if h == 0 else 'last raw SID token in saved selected-parent prefix',
                        'raw_query_SID_integer': query, 'query_source_code_position_0based': None if h == 0 else h-1,
                        'derived_embedding_lookup_index': lookup, 'num_embeddings_per_hierarchy': width,
                        'actual_observed_cross_attention_query_length': step['query_length_before_last_selection'],
                        'actual_observed_cross_attention_last_query_index': step['last_query_index'],
                        'identity_basis': 'saved beam prefixes + independently checked parent ancestry + byte-bound decoder source mapping; not direct T5Stack inputs_embeds capture'})
                paths[path] = queries
            samples.append({'sample_key': key, 'source_record': str(record), 'final_Top1_raw_SID': final_sid.tolist(), 'paths': paths})
    assert len(records) == 8 and len(samples) == len(set(all_keys)) == 32
    assert all(len(rows) == 4 for sample in samples for rows in sample['paths'].values())
    report = {'status': 'DERIVED_DECODER_QUERY_TOKEN_IDENTITY_COMPLETE', 'author': '/root/results_audit',
        'created_at_utc': dt.datetime.now(dt.timezone.utc).isoformat(), 'job_id': review['job_id'], 'domain': spec['domain'], 'route': spec['route'], 'shuffle_n': spec['shuffle_n'],
        'source_commit': deployment['source_commit'], 'snapshot_sha256': deployment['snapshot_sha256'],
        'samples': 32, 'paths_per_sample': 2, 'queries_per_path': 4, 'total_query_identity_records': 256,
        'unchanged_raw_bundle_inputs': records, 'byte_bound_source_mapping_inputs': sources,
        'original_attention_acceptance': identity(a.attention_review), 'invocation': identity(a.job_directory/'output/invocation.json'),
        'accepted_full_evaluation_review': identity(evaluation_review_path),
        'source_SID_dictionary': identity(a.sid_dictionary), 'raw_SID_maximum': raw_sid_maximum,
        'embedding_table_width_identity': 'decoder.embed_tokens.num_embeddings equals complete source SID max+1 and all visible position_map original source-code lookup offsets; observation.py supplies the actual model.num_embeddings_per_hierarchy to that map',
        'method': 'h0 learned BOS; h>0 previous saved beam prefix selected by actual final ancestor parent or local parent0; raw query is that prefix last code; lookup index adds (h-1)*embedding-table width.',
        'direct_observer_fields': ['generated_ids', 'parent_indices', 'local_first_prefix', 'query_identity', 'query_length_before_last_selection', 'last_query_index'],
        'derived_fields': ['raw_parent_SID_prefix for each selected path', 'raw_query_SID_integer', 'derived_embedding_lookup_index'],
        'T5Stack_inputs_embeds_direct_capture_claimed': False, 'raw_NPZ_or_PNG_modified': False,
        'sample_query_identities': samples, 'program': identity(Path(__file__)), 'argv': sys.argv,
        'new_model_or_GPU_calls': 0, 'Git_or_original_evidence_writes': 0,
        'independent_review_of_this_new_derived_companion': 'Required from another actual agent; this author does not self-sign nonauthor acceptance.'}
    a.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'report': identity(a.output), 'job_id': report['job_id'], 'query_identity_records': 256}, ensure_ascii=False))


if __name__ == '__main__':
    main()
