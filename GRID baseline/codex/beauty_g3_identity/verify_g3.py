"""Read-only audit of fixed Beauty G3; all exports go to --output.

Requires Python >=3.10, torch, tfrecord (and its protobuf dependencies).
Only load the authenticated original PKL belonging to the fixed source commit.
Never runs an encoder, quantizer, recommender, or GPU operation.
"""
import argparse
import ast
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
import pickle
from pathlib import Path
import platform
import sys
import tempfile


BASE = '84f200258ab66b33bcf1fe4104fa44ad43737e4a'
RUN = 'GRID baseline/experiments/beauty_rkmeans_qwen_20261010_012253'
N = 12101
EXPECTED = {
    'semantic_ids.pt': ('d41adcfb22a9a47d5e0424c068f4c74f33cc9fdbaf5db08540fbfa0e3d720e18', 388405),
    'sid_generation/pickle/merged_predictions.pkl': ('dcbfb61369cde2627cea50cc42bb80be753e2d9854b85f1825460174486bd590', 253952),
    'sid_generation/pickle/merged_predictions_tensor.pt': ('d5241c43708cb9e5ce0c1e812b965a0b5f92e38d94aead8246bcebeea62e385f', 388534),
    'sid_generation/.hydra/config.yaml': ('142f0cdb3dc8238ed1e133b428b905f435cf372af340336da9848ebfae7b3a92', 6810),
    'quantizer_train/.hydra/config.yaml': ('ad156efabaf2f5b444c19f776e45ab6dbfb71d30c77e11ad34c6a692cba91422', 7914),
    'item_embeddings.json': ('a87c346eb48687469f059351f4e125a13d6da18e036c66f66e7dce596ced72c8', 1077),
}
P5_EXPECTED = {
    'datamaps.json': '79bce2481b59b88362621dd07843661a308819f243a00c67540c69b348123d3a',
    'sequential_data.txt': '226cce9c3105299ca0db9615d7d3fb32b3175e90da43100ae352599f0f0107b8',
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(2**20), b''):
            h.update(part)
    return h.hexdigest()


def identity(path):
    return dict(bytes=Path(path).stat().st_size, sha256=sha(path))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')


def write_csv(path, fields, records):
    with Path(path).open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fields, lineterminator='\n')
        writer.writeheader()
        writer.writerows(records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', required=True, type=Path)
    parser.add_argument('--data-root', required=True, type=Path,
                        help='Beauty TFRecord root with items/training/evaluation/testing')
    parser.add_argument('--p5-dir', required=True, type=Path,
                        help='Original P5 Beauty datamaps.json and sequential_data.txt')
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError('Choose a new empty output directory')
    output.mkdir(parents=True, exist_ok=True)
    import torch
    from tfrecord.reader import tfrecord_iterator
    from tfrecord import example_pb2
    torch.set_num_threads(2)
    report = dict(source_commit=BASE, started_at_utc=datetime.now(timezone.utc).isoformat(),
                  script=identity(__file__), argv=sys.argv,
                  environment=dict(python=platform.python_version(), torch=torch.__version__,
                                   device='cpu', gpu_used=False), checks={}, anomalies={})
    checks = report['checks']
    run = args.repo/RUN
    report['fixed_assets'] = {}
    for rel, (expected_hash, expected_size) in EXPECTED.items():
        actual = identity(run/rel)
        assert actual == dict(bytes=expected_size, sha256=expected_hash), (rel, actual)
        report['fixed_assets'][RUN+'/'+rel] = actual
    checks['A1_fixed_asset_identity'] = 'pass'
    for name, expected_hash in P5_EXPECTED.items():
        assert sha(args.p5_dir/name) == expected_hash, name
    report['p5_original_members'] = {name: identity(args.p5_dir/name) for name in P5_EXPECTED}

    # Bind all current Beauty input shard bytes to the historical published manifest.
    manifest_path = args.repo/'GRID baseline/codex/dataset_source_manifest.json'
    manifest = json.loads(manifest_path.read_text('utf-8'))
    input_files = []
    for entry in manifest['files']:
        if not entry['path'].startswith('amazon_data/beauty/'):
            continue
        local = args.data_root/entry['path'].removeprefix('amazon_data/beauty/')
        actual = identity(local)
        assert actual == dict(bytes=entry['size'], sha256=entry['sha256']), str(local)
        input_files.append(dict(path='GRID baseline/original_assets/data/'+entry['path'], **actual))
    report['historical_input_files_verified'] = len(input_files)
    assert len(input_files) > 0
    write_json(output/'beauty_g3_input_manifest.json', dict(
        source_commit=BASE, historical_dataset_manifest=identity(manifest_path), files=input_files))

    items = {}
    item_id_duplicates = []
    schemas = Counter()
    for file in sorted((args.data_root/'items').glob('*.tfrecord.gz')):
        for raw in tfrecord_iterator(str(file), compression_type='gzip'):
            row = example_pb2.Example()
            row.ParseFromString(bytes(raw))
            fields = row.features.feature
            schemas[tuple(sorted(fields))] += 1
            ids = list(fields['id'].int64_list.value)
            texts = list(fields['text'].bytes_list.value)
            assert len(ids) == len(texts) == 1
            idx = int(ids[0])
            if idx in items:
                item_id_duplicates.append(idx)
            items[idx] = texts[0].decode('utf-8')
    assert not item_id_duplicates and sorted(items) == list(range(N))
    report['item_inputs'] = dict(rows=len(items), id_type='TFRecord int64 scalar', min_id=min(items),
                                max_id=max(items), duplicate_ids=item_id_duplicates,
                                feature_schemas=[dict(fields=list(k), count=v) for k,v in schemas.items()],
                                original_id_field_present=False)

    # Validate the current P5 public numbered corpus against the complete GRID histories.
    maps = json.loads((args.p5_dir/'datamaps.json').read_text('utf-8'))
    assert len(maps['id2item']) == len(maps['item2id']) == N
    assert sorted(map(int,maps['id2item'])) == list(range(1,N+1))
    assert all(maps['item2id'][asin] == pid for pid,asin in maps['id2item'].items())
    sequences = {}
    for line in (args.p5_dir/'sequential_data.txt').read_text('utf-8').splitlines():
        vals = list(map(int,line.split()))
        assert vals[0]-1 not in sequences
        sequences[vals[0]-1] = [v-1 for v in vals[1:]]
    used_items = {i for seq in sequences.values() for i in seq}
    assert used_items == set(range(N))
    split_results = {}
    fingerprints = []
    observed_item_ids = set()
    for split, trim in [('training',2),('evaluation',1),('testing',0)]:
        found = set()
        mismatches = []
        observed_offsets = Counter()
        uncapped_mismatches = 0
        for file in sorted((args.data_root/split).glob('*.tfrecord.gz')):
            for raw in tfrecord_iterator(str(file), compression_type='gzip'):
                row = example_pb2.Example(); row.ParseFromString(bytes(raw))
                fields = row.features.feature
                uid_values = list(fields['user_id'].int64_list.value)
                assert len(uid_values)==1
                uid = int(uid_values[0])
                assert uid not in found, (split,uid)
                found.add(uid)
                observed = list(fields['sequence_data'].int64_list.value)
                uncapped = sequences[uid][:-trim] if trim else sequences[uid]
                uncapped_mismatches += observed != uncapped
                expected = uncapped if split=='training' else uncapped[-21:]
                observed_item_ids.update(observed)
                if observed != expected:
                    mismatches.append(dict(grid_user_id=uid, observed=observed, expected=expected))
                # Text identity is checked for every item occurrence in every split.
                text = list(fields['text'].bytes_list.value)
                assert len(text)==len(observed), (split,uid,len(text),len(observed))
                assert all(t.decode('utf-8')==items[i] for t,i in zip(text,observed)), (split,uid)
                if split=='testing':
                    canonical = ','.join(map(str,observed)).encode()
                    fingerprints.append(dict(grid_user_id=uid,p5_user_id=uid+1,
                        sequence_length=len(observed),zero_based_sequence_sha256=hashlib.sha256(canonical).hexdigest()))
        missing_users = sorted(set(sequences)-found)
        split_results[split] = dict(rows=len(found), missing_users=missing_users,
            mismatched_sequences=len(mismatches), mismatch_records=mismatches,
            truncation_rule=('(P5 history minus one)[:-2]; no length cap' if split=='training' else
                             '(P5 history minus one, trim last 1/0 items)[-21:]'),
            sequences_differing_before_21_item_cap=uncapped_mismatches,
            text_values_match_item_table=True)
        assert not missing_users and not mismatches, (split,len(missing_users),len(mismatches),mismatches[:2])
    report['p5_to_grid_full_sequence_binding'] = split_results
    report['mapping_basis'] = ('Current retrieval of the original public P5 numbered data. '
        'All numbered user histories in all three GRID splits equal P5 histories minus one '
        'with last-two/last-one/no-trim; training is uncapped, evaluation/testing retain the last 21 items. '
        'This is a newly recovered mapping, not an original mapping saved with training.')
    report['p5_to_grid_full_sequence_binding_item_coverage'] = dict(
        observed_items=len(observed_item_ids),unobserved_item_ids=sorted(set(range(N))-observed_item_ids))

    records = pickle.loads((run/'sid_generation/pickle/merged_predictions.pkl').read_bytes())
    assert isinstance(records,list)
    schemas = Counter(tuple(sorted(row)) for row in records)
    key_types = Counter(type(row.get('item_id')).__name__ for row in records)
    value_types = Counter(type(row.get('cluster_ids')).__name__ for row in records)
    keys = [row['item_id'] for row in records]
    counts = Counter(keys)
    duplicates = {str(k):v for k,v in counts.items() if v>1}
    missing = sorted(set(range(N))-set(keys))
    negative = sorted(set(k for k in keys if k<0))
    out_of_range = sorted(set(k for k in keys if k>=N))
    invalid_codes = []
    for row in records:
        values = row['cluster_ids']
        if (not isinstance(values,list) or len(values)!=3 or
            any(type(v) is not int or v<0 or v>255 for v in values)):
            invalid_codes.append(row)
    report['predictions'] = dict(raw_container_type=type(records).__name__, records=len(records),
        record_schemas=[dict(fields=list(k),count=v) for k,v in schemas.items()],
        key_types=dict(key_types), cluster_ids_container_types=dict(value_types),
        min_key=min(keys),max_key=max(keys),unique_keys=len(counts),duplicate_keys=duplicates,
        duplicate_extra_records=sum(v-1 for v in counts.values()),missing_keys=missing,
        negative_keys=negative,out_of_range_keys=out_of_range,invalid_code_records=invalid_codes,
        code_min=[min(row['cluster_ids'][j] for row in records) for j in range(3)],
        code_max=[max(row['cluster_ids'][j] for row in records) for j in range(3)])
    assert len(records)==N and all(type(k) is int for k in keys)
    assert not duplicates and not missing and not negative and not out_of_range and not invalid_codes
    assert set(keys)==set(items)
    checks['A5_original_prediction_keys'] = 'pass'

    # Execute the exact fixed function bodies, excluding optional unrelated imports.
    source_path=args.repo/'GRID baseline/src/utils/tensor_utils.py'
    source=source_path.read_text('utf-8')
    wanted={'merge_list_of_keyed_tensors_to_single_tensor','deduplicate_rows_in_tensor','transpose_tensor_from_file'}
    nodes=[node for node in ast.parse(source).body if isinstance(node,ast.FunctionDef) and node.name in wanted]
    assert {node.name for node in nodes}==wanted
    ns=dict(torch=torch,open_local_or_remote=lambda file_path,mode='rb': open(file_path,mode))
    module=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)]+nodes,type_ignores=[])
    exec(compile(ast.fix_missing_locations(module),str(source_path),'exec'),ns)
    merged=ns['merge_list_of_keyed_tensors_to_single_tensor'](records,'item_id','cluster_ids')
    with tempfile.TemporaryDirectory(prefix='g3_audit_') as temp:
        temp_path=str(Path(temp)/'rows.pt')
        torch.save(merged,temp_path)
        ns['deduplicate_rows_in_tensor'](temp_path)
        ns['transpose_tensor_from_file'](temp_path)
        rebuilt=torch.load(temp_path,map_location='cpu',weights_only=True)
    sid=torch.load(run/'semantic_ids.pt',map_location='cpu',weights_only=True)
    generation=torch.load(run/'sid_generation/pickle/merged_predictions_tensor.pt',map_location='cpu',weights_only=True)
    differences=int((rebuilt!=sid).sum())
    generation_differences=int((generation!=sid).sum())
    assert list(sid.shape)==[4,N] and sid.dtype==torch.int64
    assert differences==generation_differences==0
    assert torch.unique(sid.T,dim=0).shape[0]==N
    report['tensor_comparison']=dict(shape=list(sid.shape),dtype=str(sid.dtype),
        fixed_functions_source=identity(source_path),
        reconstructed_vs_g3_different_values=differences,
        generation_vs_g3_different_values=generation_differences,
        reconstructed_vs_g3_different_columns=int((rebuilt!=sid).any(dim=0).sum()),
        unique_four_code_combinations=N,
        code_min=sid.min(dim=1).values.tolist(), code_max=sid.max(dim=1).values.tolist(),
        original_files_modified=False,temporary_files_only=True)
    checks['A6_tensor_reconstruction']='pass'

    mapping=[]
    for idx in range(N):
        asin=maps['id2item'][str(idx+1)]
        assert isinstance(asin,str) and asin.strip() and len(asin)==10
        mapping.append(dict(sid_column=idx,grid_item_id=idx,original_item_id=asin,
            original_id_namespace='amazon_asin',p5_item_id=str(idx+1),
            **{f'sid_{j+1}':int(sid[j,idx]) for j in range(4)}))
    assert len({row['original_item_id'] for row in mapping})==N
    fields=['sid_column','grid_item_id','original_item_id','original_id_namespace','p5_item_id','sid_1','sid_2','sid_3','sid_4']
    write_csv(output/'beauty_g3_item_mapping.csv',fields,mapping)
    with (output/'beauty_g3_item_mapping.csv').open(encoding='utf-8',newline='') as stream:
        readback=list(csv.DictReader(stream))
    assert [r['original_item_id'] for r in readback]==[r['original_item_id'] for r in mapping]
    assert all([int(r[f'sid_{j+1}']) for j in range(4)]==sid[:,int(r['sid_column'])].tolist() for r in readback)
    report['mapping']=dict(rows=N,min_column=0,max_column=N-1,
        missing_columns=[],duplicate_columns=[],missing_grid_ids=[],duplicate_grid_ids=[],
        blank_original_ids=[],duplicate_original_ids=[],namespace='amazon_asin',
        formula='sid_column = grid_item_id = int(p5_item_id) - 1',
        original_id_normalization='none; exact P5 id2item strings retained',
        csv_readback_verified=True,redundant_sid_values_different_from_g3=0)
    checks['A2_complete_item_identity']='pass'
    checks['A3_numbering_relationship']='pass'
    checks['A4_recovered_mapping_binding']='pass'
    checks['A7_config_and_representation']='pass'
    checks['A8_remote_publication']='not_executed_by_this_script'
    ordered=sorted(records,key=lambda row:row['item_id'])
    write_csv(output/'beauty_g3_keyed_predictions.csv',['item_id','sid_1','sid_2','sid_3'],[
        dict(item_id=row['item_id'],**{f'sid_{j+1}':row['cluster_ids'][j] for j in range(3)}) for row in ordered])
    write_csv(output/'p5_grid_sequence_fingerprints.csv',
        ['grid_user_id','p5_user_id','sequence_length','zero_based_sequence_sha256'],
        sorted(fingerprints,key=lambda row:row['grid_user_id']))
    write_json(output/'p5_beauty_item_identity.json',dict(
        extraction='Exact item-only projection of original P5 Beauty datamaps.json',
        original_member=report['p5_original_members']['datamaps.json'],
        item2id=maps['item2id'],id2item=maps['id2item']))
    write_csv(output/'grid_item_text_fingerprints.csv',['grid_item_id','utf8_text_sha256'],[
        dict(grid_item_id=i,utf8_text_sha256=hashlib.sha256(items[i].encode()).hexdigest()) for i in range(N)])
    report['exports']={p.name:identity(p) for p in sorted(output.iterdir()) if p.is_file()}
    report['historical_unknowns']=[
        'Original GRID TFRecord conversion script and exact historical P5 archive version were not retained.',
        'The original GRID-to-ASIN mapping file was not found. This is a deterministic recovery using the public P5 numbered corpus, checked against every historical input shard and full histories.',
        'No claim that matching GRID/LETTER numeric IDs denote the same product; LETTER alignment remains downstream work.',
    ]
    report['finished_at_utc']=datetime.now(timezone.utc).isoformat()
    report['exit_status']=0
    write_json(output/'beauty_g3_verification.json',report)
    print(json.dumps(dict(checks=checks,mapping=report['mapping'],predictions=report['predictions'],
                         tensor=report['tensor_comparison'],splits=split_results),ensure_ascii=False))


if __name__=='__main__':
    main()
