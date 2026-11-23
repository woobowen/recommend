"""Versioned Yelp alignment; preserves author order, never uses author's SIDs."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import statistics

os.environ.update(TF_CPP_MIN_LOG_LEVEL='3', TF_NUM_INTRAOP_THREADS='2', TF_NUM_INTEROP_THREADS='2')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        ap.error('Output must be new')
    expected = {'Yelp.inter.json': '22069fef8afd1d53fe9e0aaa385a6c93f7b43e43574fb10f5f5155fa2c869a97',
                'Yelp.item.json': '8f58a9d54111fc10ccd675f99e6f2afe988cd0ee70b40f1e37d04f9aea764c2d'}
    assert all(sha(args.source / n) == h for n, h in expected.items())
    inter = json.loads((args.source / 'Yelp.inter.json').read_text())
    items = json.loads((args.source / 'Yelp.item.json').read_text())
    raw_counts = dict(users=len(inter), items=len(items), interactions=sum(map(len, inter.values())))
    sequences = {}
    for uid, seq in inter.items():
        seen, rev = set(), []
        for item in reversed(seq):
            item = str(item)
            assert item in items
            if item not in seen:
                rev.append(item)
                seen.add(item)
        sequences[uid] = rev[::-1]
    dedup_count = sum(map(len, sequences.values()))
    rounds = []
    while True:
        counts = Counter(i for seq in sequences.values() for i in seq)
        next_sequences = {u: [i for i in seq if counts[i] >= 5] for u, seq in sequences.items()}
        next_sequences = {u: seq for u, seq in next_sequences.items() if len(seq) >= 5}
        rounds.append(dict(round=len(rounds)+1, users=len(next_sequences),
                           items=len(set(i for seq in next_sequences.values() for i in seq)),
                           interactions=sum(map(len, next_sequences.values()))))
        if next_sequences == sequences:
            break
        sequences = next_sequences
    old_users = sorted(sequences, key=int)
    old_items = sorted({i for seq in sequences.values() for i in seq}, key=int)
    umap = {u: n for n, u in enumerate(old_users)}
    imap = {i: n for n, i in enumerate(old_items)}
    full = {str(umap[u]): [imap[i] for i in sequences[u]] for u in old_users}
    assert len(full) == 28330 and len(imap) == 18775
    assert sum(map(len, full.values())) == 292017
    assert min(map(len, full.values())) >= 5
    assert min(Counter(i for s in full.values() for i in s).values()) >= 5
    assert all(len(s) == len(set(s)) for s in full.values())
    texts = {n: f"Title: {items[old]['title']}; Categories: {items[old]['description']}; "
             for n, old in enumerate(old_items)}
    assert all(items[i]['title'].strip() and items[i]['description'].strip() for i in old_items)
    args.output.mkdir(parents=True)
    write(args.output/'id_maps.json', dict(user_old_to_new=umap, user_new_to_old=old_users,
                                         item_old_to_new=imap, item_new_to_old=old_items))
    write(args.output/'full_sequences.json', full)
    write(args.output/'item_texts.json', texts)
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
    def feature_int(values):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=values))
    options = tf.data.Options()
    options.threading.private_threadpool_size = 1
    split_stats = {}
    for split in ('items', 'training', 'evaluation', 'testing'):
        folder = args.output/split
        folder.mkdir()
        path = folder/'part-00000.tfrecord.gz'
        expected_rows = {}
        with tf.io.TFRecordWriter(str(path), options='GZIP') as writer:
            if split == 'items':
                for idx, text in texts.items():
                    fields = {'id': feature_int([idx]), 'text': tf.train.Feature(
                        bytes_list=tf.train.BytesList(value=[text.encode('utf-8')]))}
                    writer.write(tf.train.Example(features=tf.train.Features(feature=fields)).SerializeToString())
            else:
                for uid, seq in full.items():
                    values = seq[:-2] if split == 'training' else seq[:-1][-21:] if split == 'evaluation' else seq[-21:]
                    expected_rows[int(uid)] = values
                    fields = {'user_id': feature_int([int(uid)]), 'sequence_data': feature_int(values)}
                    writer.write(tf.train.Example(features=tf.train.Features(feature=fields)).SerializeToString())
        count, lengths = 0, []
        for raw in tf.data.TFRecordDataset(str(path), compression_type='GZIP').with_options(options):
            f = tf.train.Example.FromString(raw.numpy()).features.feature
            if split == 'items':
                idx = list(f['id'].int64_list.value)[0]
                assert idx == count and f['text'].bytes_list.value[0].decode('utf-8') == texts[idx]
            else:
                uid = list(f['user_id'].int64_list.value)[0]
                seq = list(f['sequence_data'].int64_list.value)
                assert uid == count and seq == expected_rows[uid]
                lengths.append(len(seq))
            count += 1
        assert count == (len(texts) if split == 'items' else len(full))
        split_stats[split] = dict(rows=count, sha256=sha(path), fields=['id','text'] if split=='items' else ['user_id','sequence_data'])
        if lengths:
            split_stats[split].update(min_length=min(lengths), max_length=max(lengths), mean_length=statistics.mean(lengths))
    audit = dict(version='yelp_aligned_v1', created_utc=datetime.now(timezone.utc).isoformat(),
        source_repository='https://github.com/HonghuiBao2000/LETTER',
        source_commit='8d0154e28de37dbb6e24871c508ad8ddb1921cda', source_sha256=expected,
        raw=raw_counts, after_dedup_interactions=dedup_count, core_rounds=rounds,
        processed=dict(users=len(full), items=len(texts), interactions=sum(map(len,full.values())),
                       mean_sequence_length=statistics.mean(map(len,full.values())),
                       median_sequence_length=statistics.median(map(len,full.values())),
                       min_sequence_length=min(map(len,full.values())), max_sequence_length=max(map(len,full.values()))),
        dedup='keep last occurrence per user in supplied order', core='iterative user and item 5-core',
        training='S[:-2], no global history truncation', evaluation='S[:-1][-21:]', testing='S[-21:]',
        text_template='Title: {title}; Categories: {description}; ',
        schema_note='Old cached embedding fields omitted; current Qwen uses id/text and downstream uses user_id/sequence_data.',
        no_category_split=True, no_random_subsampling=True, author_sid_used=False,
        time_note='No raw timestamps/business IDs; author sequence order inherited, not independently timestamp-verified.',
        transductive_catalog_note='5-core and item-text codebooks use full retained catalog; held-out targets never used for downstream fit or checkpoint selection.',
        splits=split_stats, readback_all_records_passed=True, processor_sha256=sha(__file__))
    write(args.output/'processing_audit.json', audit)
    manifest = {str(p.relative_to(args.output)): sha(p) for p in sorted(args.output.rglob('*')) if p.is_file()}
    write(args.output/'file_manifest.json', manifest)
    assert all(sha(args.source/n) == h for n,h in expected.items())
    print(json.dumps(audit, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
