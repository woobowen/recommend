"""Verify the released LETTER Beauty IDs against the existing TIGER data."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='', TF_CPP_MIN_LOG_LEVEL='3', OMP_NUM_THREADS='2',
                  TF_NUM_INTRAOP_THREADS='1', TF_NUM_INTEROP_THREADS='1')
import collections
import hashlib
import html
import json
from pathlib import Path
import re
import tensorflow as tf
import torch

ROOT = Path('/home/jcy/GRID_wmh_Qwen_TIGER_20261009')
HERE = ROOT / 'codex/letter_upstream_transformer_20261115'
SOURCE = ROOT / 'third_party/LETTER-8d0154e/data/Beauty'
DATA = ROOT / 'original_assets/data/amazon_data/beauty'


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def write(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def records(folder):
    files=sorted(folder.glob('*.tfrecord.gz'))
    options=tf.data.Options()
    options.threading.private_threadpool_size=1
    options.threading.max_intra_op_parallelism=1
    ds=tf.data.TFRecordDataset([str(p) for p in files],compression_type='GZIP').with_options(options)
    for raw in ds:
        yield tf.train.Example.FromString(raw.numpy()).features.feature


def normalize_title(text):
    text=re.sub(r'<[^>]*>','',html.unescape(text))
    return ' '.join(text.split()).strip(' .!?,;:`"')


def main():
    dest=HERE/'inputs/beauty'
    dest.mkdir(parents=True,exist_ok=True)
    author={k:json.loads((SOURCE/f'Beauty.{k}.json').read_text()) for k in ['index','inter','item']}
    assert set(author['index'])==set(author['item'])=={str(i) for i in range(len(author['item']))}
    ids=[]
    for i in range(len(author['index'])):
        tokens=author['index'][str(i)]
        assert len(tokens)==4
        values=[]
        for level,token in zip('abcd',tokens):
            match=re.fullmatch('<'+level+r'_(\d+)>',token)
            assert match,token
            values.append(int(match.group(1)))
        assert all(0<=v<256 for v in values)
        ids.append(values)
    item_titles={}
    mismatches=[];placeholder_differences=[]
    for f in records(DATA/'items'):
        i=int(f['id'].int64_list.value[0])
        text=f['text'].bytes_list.value[0].decode()
        assert text.startswith('Title: ') and '; Brand:' in text
        title=text[len('Title: '):].rsplit('; Brand:',1)[0]
        assert i not in item_titles
        item_titles[i]=title
        if normalize_title(title)!=normalize_title(author['item'][str(i)]['title']):
            difference={'item_id':i,'ours':title,'author':author['item'][str(i)]['title']}
            if title.strip().lower()=='nan' and author['item'][str(i)]['title']==f'title_{i+1}':
                placeholder_differences.append(difference)
            else:mismatches.append(difference)
    assert set(item_titles)==set(range(len(ids)))
    print('Title mapping checked',len(item_titles),'mismatches',len(mismatches),flush=True)
    split_audits={}
    ambiguous={tuple(row) for row,count in collections.Counter(map(tuple,ids)).items() if count>1}
    for split,trim in [('training',2),('evaluation',1),('testing',0)]:
        seen=set();differences=[];ambiguous_targets=0;truncated=0
        for f in records(DATA/split):
            uid=str(f['user_id'].int64_list.value[0]);seq=list(f['sequence_data'].int64_list.value)
            assert uid not in seen,uid
            seen.add(uid)
            expected=author['inter'][uid]
            expected=expected[:-trim] if trim else expected
            if split!='training' and len(expected)>21:
                expected=expected[-21:];truncated+=1
            if seq!=expected:
                differences.append({'uid':uid,'ours':seq,'author':expected})
            ambiguous_targets+=tuple(ids[seq[-1]]) in ambiguous
        assert seen==set(author['inter'])
        split_audits[split]={'users':len(seen),'mismatched_sequences':len(differences),
                             'mismatch_examples':differences[:3],
                             'ambiguous_sid_targets':ambiguous_targets,
                             'original_history_truncated_to_20_plus_target':truncated}
        print(split,split_audits[split],flush=True)
    report=dict(dataset='beauty',items=len(ids),users=len(author['inter']),
                interactions=sum(map(len,author['inter'].values())),
                title_mismatch_count=len(mismatches),title_mismatch_examples=mismatches[:5],
                known_missing_title_placeholder_differences=placeholder_differences,
                splits=split_audits,official_sources={str(p):digest(p) for p in sorted(SOURCE.glob('*.json'))},
                data_files={str(p.relative_to(ROOT)):digest(p) for p in sorted(DATA.rglob('*.tfrecord.gz'))},
                sid_shape=[4,len(ids)],unique_full_sids=len(set(map(tuple,ids))),
                collision_groups=[{'sid':list(s),'item_ids':[i for i,row in enumerate(ids) if tuple(row)==s]}
                                  for s in sorted(ambiguous)],
                extra_colliding_items=len(ids)-len(set(map(tuple,ids))),
                official_sid_values_unchanged=True,
                metric_semantics='Existing downstream evaluates full SID matches; items sharing an official SID are indistinguishable.',
                original_data_path=str(DATA))
    write(dest/'mapping_audit.json',report)
    assert not mismatches and not any(v['mismatched_sequences'] for v in split_audits.values()), 'Mapping mismatch; refusing launch'
    tensor=torch.tensor(ids,dtype=torch.long).T.contiguous()
    sid=dest/'semantic_ids.pt'
    if sid.exists():
        assert torch.equal(torch.load(sid,map_location='cpu',weights_only=False),tensor)
    else:torch.save(tensor,sid)
    report.update(complete=True,sid_path=str(sid),sid_sha256=digest(sid),
                  item_mapping_verified=True,all_split_sequences_exact_with_existing_history_policy=True,
                  copied_sid_numeric_values_exact=torch.equal(torch.load(sid,weights_only=False).T,torch.tensor(ids)))
    write(dest/'mapping_audit.json',report)
    print(json.dumps({k:v for k,v in report.items() if k not in ['data_files','collision_groups']},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
