#!/usr/bin/env python3
"""Pack historical LETTER JSONL predictions into full-row, lossless NPZ evidence."""
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import datetime
import gzip
import hashlib
import json
import math
from pathlib import Path

import numpy as np

FORMAT = 'LETTER-complete-prediction-rows-v1'
BEAMS = 20
RESPONSE = 'Response:'


def canonical(value):
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def file_digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def prediction_order(scores):
    return sorted(range(len(scores)), key=scores.__getitem__, reverse=True)


def remove_duplicate(record, name, value):
    if name in record and record[name] == value:
        del record[name]


def omit_shuffle_duplicates(row):
    shuffle = row['shuffle']
    evidence = shuffle['evidence']
    original = shuffle['original_input_token_ids']
    mask = shuffle['attention_mask']
    actual = row['input_token_ids']
    positions_seen = set()
    replay = original.copy()
    for i, group in enumerate(evidence['groups']):
        positions = group['positions']
        assert len(positions) == 4 and not positions_seen.intersection(positions)
        positions_seen.update(positions)
        remove_duplicate(group, 'visible_history_item_index', i)
        remove_duplicate(group, 'start', positions[0])
        remove_duplicate(group, 'original_ids', [original[p] for p in positions])
        remove_duplicate(group, 'shuffled_ids', [actual[p] for p in positions])
        for p, source in zip(positions, group['permutation'], strict=True):
            replay[p] = original[positions[source]]
    assert replay == actual
    del row['input_token_ids']
    for key, value in (('original_sha256', original), ('shuffled_sha256', actual), ('attention_mask_sha256', mask)):
        assert evidence.pop(key) == digest(value)


def restore_shuffle_duplicates(row):
    shuffle = row['shuffle']
    evidence = shuffle['evidence']
    original = shuffle['original_input_token_ids']
    actual = original.copy()
    for i, group in enumerate(evidence['groups']):
        positions = group['positions']
        for p, source in zip(positions, group['permutation'], strict=True):
            actual[p] = original[positions[source]]
        group.setdefault('visible_history_item_index', i)
        group.setdefault('start', positions[0])
        group.setdefault('original_ids', [original[p] for p in positions])
        group.setdefault('shuffled_ids', [actual[p] for p in positions])
    row['input_token_ids'] = actual
    for key, value in (('original_sha256', original), ('shuffled_sha256', actual), ('attention_mask_sha256', shuffle['attention_mask'])):
        evidence[key] = digest(value)


def pack_row(row):
    candidates = row.pop('candidates')
    scores = [c['sequence_score'] for c in candidates]
    assert len(scores) == BEAMS and all(isinstance(s, float) and math.isfinite(s) for s in scores)
    sorted_order = prediction_order(scores)
    assert row.pop('sorted_generation_order') == sorted_order
    assert row.pop('relevance') == [int(candidates[i]['sid'] == row['target']) for i in sorted_order]
    prefix, sep, _ = candidates[0]['raw_decoded'].rpartition(RESPONSE)
    assert sep == RESPONSE
    values = []
    for i, candidate in enumerate(candidates):
        assert set(candidate) == {'generation_order', 'generated_token_ids', 'eos_position', 'raw_decoded', 'sequence_score', 'sid'}
        assert candidate['generation_order'] == i
        p, split, suffix = candidate['raw_decoded'].rpartition(RESPONSE)
        assert (p, split) == (prefix, RESPONSE)
        assert suffix.strip().replace(' ', '') == candidate['sid']
        values.append((candidate['sid'], tuple(candidate['generated_token_ids']), candidate['eos_position'], suffix))
    if 'actual_model_prefill' in row:
        prefill = row['actual_model_prefill']
        assert prefill.pop('input_ids') == row['input_token_ids']
        assert prefill.pop('attention_mask') == row['attention_mask']
        if 'shuffle' in row:
            assert row.pop('original_input_token_ids') == row['shuffle']['original_input_token_ids']
            assert row.pop('attention_mask') == row['shuffle']['attention_mask']
        else:
            assert row.pop('original_input_token_ids') == row['input_token_ids']
    if 'shuffle' in row:
        omit_shuffle_duplicates(row)
    return scores, values, prefix, row


def restore_row(scores, values, prefix, row):
    if 'shuffle' in row:
        restore_shuffle_duplicates(row)
    if 'actual_model_prefill' in row:
        if 'shuffle' in row:
            row['original_input_token_ids'] = row['shuffle']['original_input_token_ids']
            row['attention_mask'] = row['shuffle']['attention_mask']
        else:
            row['original_input_token_ids'] = row['input_token_ids']
        row['actual_model_prefill']['input_ids'] = row['input_token_ids']
        row['actual_model_prefill']['attention_mask'] = row['attention_mask']
    row['candidates'] = [dict(generation_order=i, generated_token_ids=tokens, eos_position=eos,
                              raw_decoded=prefix + RESPONSE + suffix, sequence_score=scores[i], sid=sid)
                         for i, (sid, tokens, eos, suffix) in enumerate(values)]
    row['sorted_generation_order'] = prediction_order(scores)
    row['relevance'] = [int(values[i][0] == row['target']) for i in row['sorted_generation_order']]
    return row


def encode_shard(raw, compact, sources):
    condition = Path(sources[0]['compressed_member']).parent
    output = compact / condition / 'predictions.npz'
    output.parent.mkdir(parents=True, exist_ok=True)
    source_receipts = []
    metadata = bytearray()
    d = {}
    all_scores = []
    all_codes = []
    for source in sources:
        name = source['compressed_member']
        path = raw / name
        assert file_digest(path) == source['compressed_sha256'], name
        original_sha = hashlib.sha256()
        semantic_sha = hashlib.sha256()
        row_count = 0
        total_bytes = 0
        with gzip.open(path, 'rb') as stream:
            for line in stream:
                assert line.endswith(b'\n'), name
                original_sha.update(line)
                total_bytes += len(line)
                row = json.loads(line)
                semantic_sha.update(canonical(row) + b'\n')
                scores, values, prefix, row = pack_row(row)
                codes = [d.setdefault(value, len(d)) for value in values]
                all_scores.append(scores)
                all_codes.append(codes)
                metadata += canonical([prefix, row]) + b'\n'
                row_count += 1
        assert (row_count, total_bytes, original_sha.hexdigest()) == (source['rows'], source['bytes'], source['sha256']), name
        source_receipts.append({**source, 'parsed_row_sha256': semantic_sha.hexdigest()})
    scores = np.asarray(all_scores, dtype='<f8')
    narrow_scores = scores.astype('<f4')
    if np.array_equal(scores.view('<u8'), narrow_scores.astype('<f8').view('<u8')):
        scores = narrow_scores
    codes = np.asarray(all_codes, dtype='<u2' if len(d) < 2**16 else '<u4')
    count = 0
    with output.open('xb') as stream:
        np.savez_compressed(stream, format=FORMAT, scores=scores, candidate_codes=codes,
                            candidates=np.frombuffer(canonical(list(d)), dtype=np.uint8),
                            metadata=np.frombuffer(metadata, dtype=np.uint8))
    recovered_hash = hashlib.sha256()
    shard_end = source_receipts[0]['rows']
    shard_index = 0
    for row in iter_rows(output):
        recovered_hash.update(canonical(row) + b'\n')
        count += 1
        if count == shard_end:
            source = source_receipts[shard_index]
            assert recovered_hash.hexdigest() == source['parsed_row_sha256'], source['compressed_member']
            recovered_hash = hashlib.sha256()
            shard_index += 1
            if shard_index < len(source_receipts):
                shard_end += source_receipts[shard_index]['rows']
    assert count == len(scores) == shard_end and shard_index == len(source_receipts)
    return {'format': FORMAT, 'condition': condition.as_posix(), 'member': output.relative_to(compact.parent).as_posix(),
            'bytes': output.stat().st_size, 'sha256': file_digest(output), 'rows': count,
            'candidate_rows': count * BEAMS, 'score_dtype': scores.dtype.str, 'candidate_dictionary_entries': len(d),
            'original_shards': source_receipts, 'roundtrip': 'Exact parsed JSON fields by canonical SHA-256 for each original shard.'}


def iter_rows(path, metadata_only=False):
    with np.load(path, allow_pickle=False) as archive:
        assert archive['format'].item() == FORMAT
        scores = archive['scores']
        codes = archive['candidate_codes']
        d = json.loads(archive['candidates'].tobytes())
        rows = archive['metadata'].tobytes().splitlines()
    assert scores.shape == codes.shape == (len(rows), BEAMS)
    for i, metadata in enumerate(rows):
        prefix, row = json.loads(metadata)
        row_scores = scores[i, :].tolist()
        values = [d[code] for code in codes[i, :].tolist()]
        if metadata_only:
            yield row, row_scores, values
        else:
            yield restore_row(row_scores, values, prefix, row)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    records = defaultdict(list)
    source_manifest = args.raw / 'manifest.json'
    for entry in json.loads(source_manifest.read_text()):
        records[Path(entry['compressed_member']).parent].append(entry)
    assert len(records) == 12 and sum(map(len, records.values())) == 21
    started = datetime.datetime.now(datetime.timezone.utc).isoformat()
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(encode_shard, args.raw, args.out / 'compact', sorted(rows, key=lambda x: x['compressed_member']))
                   for _, rows in sorted(records.items())]
        packed = [future.result() for future in futures]
    summary = {'format': FORMAT, 'started_at_utc': started,
               'completed_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
               'source_manifest_sha256': file_digest(source_manifest),
               'original_compressed_bytes': sum(s['compressed_bytes'] for c in packed for s in c['original_shards']),
               'compact_bytes': sum(c['bytes'] for c in packed), 'conditions': packed}
    with (args.out / 'manifest.json').open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(summary, indent=2) + '\n')


if __name__ == '__main__':
    main()
