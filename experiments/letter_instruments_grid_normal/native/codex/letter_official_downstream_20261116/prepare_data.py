"""Convert official LETTER SID tables and matching sequences to the own TIGER schema."""
import argparse
import collections
import hashlib
import json
import os
from pathlib import Path
import re

os.environ.update(CUDA_VISIBLE_DEVICES='', TF_CPP_MIN_LOG_LEVEL='3', OMP_NUM_THREADS='2',
                  TF_NUM_INTRAOP_THREADS='1', TF_NUM_INTEROP_THREADS='1')
ROOT=Path('/home/jcy/GRID_wmh_Qwen_TIGER_20261009')
HERE=ROOT/'codex/letter_official_downstream_20261116'
AUTHOR=ROOT/'third_party/LETTER-8d0154e'
NAMES={'instruments':'Instruments','yelp':'Yelp'}


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def write(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def prepare(dataset):
    import torch
    import tensorflow as tf
    tf.config.set_visible_devices([],'GPU')
    source=AUTHOR/'data'/NAMES[dataset]
    dest=HERE/'inputs'/dataset
    assert not dest.exists(),f'Fresh preparation required: {dest}'
    files={kind:source/f'{NAMES[dataset]}.{kind}.json' for kind in ['index','inter','item']}
    original_manifest=json.loads((ROOT/'codex/letter_upstream_transformer_20261115/source_manifest.json').read_text())
    assert original_manifest['commit']=='8d0154e28de37dbb6e24871c508ad8ddb1921cda'
    expected={entry['path']:entry['sha256'] for entry in original_manifest['file_manifest']}
    for path in files.values():assert digest(path)==expected[str(path.relative_to(AUTHOR))]
    official={k:json.loads(p.read_text()) for k,p in files.items()}
    indices,inters,items=official['index'],official['inter'],official['item']
    assert set(indices)==set(items)=={str(i) for i in range(len(items))}
    assert set(inters)=={str(i) for i in range(len(inters))}
    rows=[]
    for i in range(len(items)):
        codes=indices[str(i)];assert len(codes)==4
        row=[]
        for level,token in zip('abcd',codes):
            match=re.fullmatch('<'+level+r'_(\d+)>',token);assert match,token
            row.append(int(match.group(1)))
        assert all(0<=x<256 for x in row)
        rows.append(row)
    referenced=collections.Counter(i for seq in inters.values() for i in seq)
    assert set(referenced)==set(range(len(items)))
    assert min(map(len,inters.values()))>=3
    counts=collections.Counter(map(tuple,rows))
    collisions={sid for sid,count in counts.items() if count>1}
    dest.mkdir(parents=True)
    sid_path=dest/'semantic_ids.pt'
    tensor=torch.tensor(rows,dtype=torch.long).T.contiguous()
    torch.save(tensor,sid_path)
    assert torch.equal(torch.load(sid_path,weights_only=False),tensor)
    data=dest/'data'
    def feat(values):return tf.train.Feature(int64_list=tf.train.Int64List(value=values))
    options=tf.data.Options();options.threading.private_threadpool_size=1
    options.threading.max_intra_op_parallelism=1
    split_info={}
    for split in ['training','evaluation','testing']:
        folder=data/split;folder.mkdir(parents=True)
        path=folder/'part-00000.tfrecord.gz'
        expected_rows={};ambiguous=0
        with tf.io.TFRecordWriter(str(path),options='GZIP') as writer:
            for user in sorted(inters,key=int):
                seq=inters[user]
                values=seq[:-2] if split=='training' else seq[:-1][-21:] if split=='evaluation' else seq[-21:]
                expected_rows[int(user)]=values
                ambiguous+=tuple(rows[values[-1]]) in collisions
                fields={'user_id':feat([int(user)]),'sequence_data':feat(values)}
                writer.write(tf.train.Example(features=tf.train.Features(feature=fields)).SerializeToString())
        seen=set()
        for raw in tf.data.TFRecordDataset(str(path),compression_type='GZIP').with_options(options):
            features=tf.train.Example.FromString(raw.numpy()).features.feature
            user=int(features['user_id'].int64_list.value[0])
            values=list(features['sequence_data'].int64_list.value)
            assert user not in seen and values==expected_rows[user]
            seen.add(user)
        assert seen==set(expected_rows)
        split_info[split]=dict(users=len(seen),mismatched_sequences=0,
            ambiguous_sid_targets=ambiguous,readback_all_records_exact=True,
            min_saved_sequence_length=min(map(len,expected_rows.values())),
            max_saved_sequence_length=max(map(len,expected_rows.values())))
    source_hashes={str(p):digest(p) for p in files.values()}
    data_hashes={str(p.relative_to(ROOT)):digest(p) for p in sorted(data.rglob('*.tfrecord.gz'))}
    report=dict(complete=True,dataset=dataset,items=len(items),users=len(inters),
        interactions=sum(map(len,inters.values())),sid_shape=list(tensor.shape),
        unique_full_sids=len(counts),extra_colliding_items=len(items)-len(counts),
        collision_groups=len(collisions),items_in_collision_groups=sum(counts[s] for s in collisions),
        collision_size_distribution=dict(collections.Counter(counts[s] for s in collisions)),
        duplicate_user_item_interactions=sum(len(seq)-len(set(seq)) for seq in inters.values()),
        official_sources=source_hashes,data_files=data_hashes,
        sid_path=str(sid_path),sid_sha256=digest(sid_path),original_data_path=str(data),
        item_mapping_verified=True,all_split_sequences_exact_with_existing_history_policy=True,
        official_sid_values_unchanged=True,author_interaction_order_preserved=True,
        further_dedup_or_core_filtering=False,
        training_policy='Author sequence S[:-2]; own causal-duplicate collator, sequence_length=120.',
        validation_policy='S[:-1][-21:] (last 20 history items plus validation target).',
        test_policy='S[-21:] (last 20 history items plus test target).',
        metric_semantics='Existing evaluator matches complete SID; items in a collision group are indistinguishable.',
        official_commit=original_manifest['commit'],splits=split_info,
        preparation_source_sha256=digest(Path(__file__)))
    for p,h in source_hashes.items():assert digest(p)==h
    write(dest/'mapping_audit.json',report)
    print(json.dumps({k:v for k,v in report.items() if k not in ['data_files','official_sources']},indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--dataset',choices=list(NAMES),required=True)
    prepare(parser.parse_args().dataset)
