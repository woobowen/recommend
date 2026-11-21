"""Capture and recheck a fixed prefix of complete formal Beauty predictions."""
import argparse
from hashlib import sha256
import json
import math
from pathlib import Path

from lc_rec.shuffle_v2 import core, reference

CAPTURE = r'''
from datetime import datetime,timezone
import hashlib,json,os
from pathlib import Path

prediction = directory/'test/predictions.rank0.jsonl'
manifest = replay/'inputs.jsonl'
rank_path = directory/'test/rank0.json'
raw_rank = rank_path.read_text()
audit = json.loads((replay/'audit.json').read_text())
shard = json.loads((replay/'single-shard.json').read_text())
rank = json.loads(raw_rank)
assert rank['assigned_users'] == shard['users'] == 22363 and rank['world_size']==1
capture = {'capture_utc':datetime.now(timezone.utc).isoformat(),
           'selection':'first six complete formal lines in the output stream',
           'rank_receipt':{'source':str(rank_path),'raw_text':raw_rank},
           'cpu_manifest_sha256':audit['manifest_sha256'],
           'frozen_sid_token_ids':audit['token_ids'],'rows':[]}

def prefix(path,count):
    rows=[]
    with path.open('rb') as f:
        limit=os.fstat(f.fileno()).st_size
        for _ in range(count):
            first=f.tell()
            raw=f.readline(min(limit-first,2**20))
            assert raw.endswith(b'\n') and f.tell()<=limit, path
            rows.append({'source':str(path),'start_byte':first,'end_byte_exclusive':f.tell(),
                         'source_size_at_capture':limit,'raw_text':raw.decode('utf-8')})
    return rows

formal = prefix(prediction,6)
normal = [prefix(normal_root/f'predictions.rank{r}.jsonl',3) for r in range(2)]
wanted = {json.loads(entry['raw_text'])['row_idx'] for entry in formal}
assert len(wanted)==6
cpu={}
with manifest.open('rb') as f:
    length=os.fstat(f.fileno()).st_size
    row_idx=-1
    while len(cpu)<len(wanted):
        start=f.tell()
        line=f.readline()
        assert line, manifest
        row_idx+=1
        if row_idx in wanted:
            assert json.loads(line)['row_idx']==row_idx
            cpu[row_idx]={'source':str(manifest),'start_byte':start,'end_byte_exclusive':f.tell(),
                          'source_size_at_capture':length,'raw_text':line.decode('utf-8')}
index = json.loads((data/'Beauty.index.json').read_text())
inter = json.loads((data/'Beauty.inter.json').read_text())
for seq,(new,old) in enumerate(zip(formal,[normal[i%2][i//2] for i in range(6)],strict=True)):
    row=json.loads(new['raw_text'])
    assert row['row_idx']==json.loads(old['raw_text'])['row_idx']
    user=inter[row['user_id']]
    relevant=user[:-1][-20:]+user[-1:]
    capture['rows'].append({'ordinal':seq,'formal':new,'normal':old,'cpu':cpu[row['row_idx']],
                            'visible_history_items':relevant[:-1],'target_item':relevant[-1],
                            'item_symbols':{str(i):index[str(i)] for i in relevant}})
print(json.dumps(capture,ensure_ascii=False))
'''


def capture(connection, folder, config):
    from lc_rec.shuffle_v2 import beauty_control as bc, control

    assert folder.startswith('gpu-formal-') and '/' not in folder and '..' not in folder
    paths = dict(directory=Path(bc.ROOT)/folder, replay=Path(bc.ROOT)/'cpu-replay-r3',
                 normal_root=Path(config['baseline']['normal_test_dir']),
                 data=Path(config['baseline']['data-root'])/'Beauty')
    source = 'from pathlib import Path, PosixPath\n' + '\n'.join(f'{k}={v!r}' for k,v in paths.items())
    return json.loads(control.remote_python(connection, source + '\n' + CAPTURE, timeout=45))


def verify(sample):
    from lc_rec.shuffle_v2.beauty_evaluate import validate_input

    expected_gates = sample['frozen_sid_token_ids']
    sid_ids = set(expected_gates.values())
    assert len(sid_ids) == 965
    rank = json.loads(sample['rank_receipt']['raw_text'])
    assert rank['assigned_users'] == 22363 and (rank['world_size'], rank['rank'], rank['local_rank']) == (1,0,0)
    assert rank.get('preflight_only') is not True
    config = json.loads(Path(__file__).with_name('a0_beauty_letter.json').read_text())
    baseline = config['baseline']
    assert rank['beauty_normal_binding'] == baseline['run-commit']
    assert rank['adapter_files']['adapter_model.bin'] == baseline['adapter_identity']
    assert rank['model']['path'] == baseline['model'] and rank['model']['revision'] == Path(baseline['model']).name
    assert rank['allowed_callback_sha256'] == baseline['allowed_callback_sha256']
    assert rank['tokenizer']['vocab_size'] == 32965 and rank['tokenizer']['added_tokens'] == expected_gates
    checked = rank['strict_adapter_reload']
    assert checked['keys'] == len(checked['tensor_checks']) == 450 and checked['lora_keys'] == 448
    assert all(tensor['finite_and_exact_after_explicit_dtype_conversion'] for tensor in checked['tensor_checks'].values())
    assert sum(tensor['loaded_dtype'] == 'torch.float32' for tensor in checked['tensor_checks'].values()) == 448
    assert sum(tensor['loaded_dtype'] == 'torch.bfloat16' for tensor in checked['tensor_checks'].values()) == 2
    assert rank['cuda_environment']['actual_rank_environment'] == config['gpu_environment']
    assert rank['cuda_environment']['llama_import_and_strict_reload'] == 'PASS'
    effective = {'do_sample':False,'num_beams':20,'num_return_sequences':20,'max_new_tokens':10,
                 'early_stopping':True,'bos_token_id':1,'eos_token_id':2,'pad_token_id':0,'use_cache':True}
    assert all(rank['shuffle']['effective_generation_config'][key] == value for key,value in effective.items())
    assert not rank['shuffle']['capture_lookup']
    assert rank['generation']['per_rank_batch_size'] == 1 and rank['generation']['filter_items'] is False
    by_level = [set() for _ in range(4)]
    for token, number in expected_gates.items():
        by_level['abcd'.index(token[1])].add(number)
    reverse = {value:key for key,value in expected_gates.items()}
    proof = []
    starts = []
    for record in sample['rows']:
        for role in ('formal','normal','cpu'):
            part = record[role]
            assert part['raw_text'].endswith('\n')
            assert part['end_byte_exclusive'] - part['start_byte'] == len(part['raw_text'].encode('utf-8'))
            assert part['end_byte_exclusive'] <= part['source_size_at_capture']
        row, old, cpu = (json.loads(record[role]['raw_text']) for role in ('formal','normal','cpu'))
        assert all(row[key] == old[key] for key in ('row_idx','user_id','target','target_item_id'))
        assert row['target_item_id'] == str(record['target_item'])
        symbols = record['item_symbols']
        assert row['target'] == ''.join(symbols[str(record['target_item'])])
        ids = old['input_token_ids']
        assert core.canonical_sha256(ids) == cpu['original_sha256']
        mode = row['shuffle']['mode']
        before = [[expected_gates[symbol] for symbol in symbols[str(item)]] for item in record['visible_history_items']]
        positions = [pos for pos,tok in enumerate(ids) if tok in sid_ids]
        assert len(positions) == 4*len(before)
        actual = list(ids)
        active = core.MODES[mode]
        for i,tokens in enumerate(before):
            where = positions[4*i:4*i+4]
            assert where == list(range(where[0],where[0]+4)) and [ids[j] for j in where] == tokens
            index,perm,_ = reference.relative(len(active),row['row_idx'],i)
            assert index == cpu['conditions'][mode]['permutation_ids'][i]
            actual[where[0]:where[0]+4] = reference.remap(tokens,active,perm)
        assert actual == row['input_token_ids']
        assert core.canonical_sha256([1]*len(ids)) == row['shuffle']['attention_mask_sha256']
        validate_input(row,cpu,mode,sample['cpu_manifest_sha256'],sid_ids,rank['allowed_callback_sha256'])
        candidates = row['candidates']
        assert len(candidates)==20 and row['prompt_id']==row['rank']==0
        for i,c in enumerate(candidates):
            assert c['generation_order']==i and math.isfinite(c['sequence_score'])
            ids_generated=c['generated_token_ids']
            assert ids_generated[4]==2 and all(token in (0,2) for token in ids_generated[5:])
            assert all(token in by_level[pos] for pos,token in enumerate(ids_generated[:4]))
            assert ''.join(reverse[x] for x in ids_generated[:4]) == c['sid']
            assert c['raw_decoded'].rsplit('Response:',1)[-1].strip().replace(' ','') == c['sid']
        order=sorted(range(20),key=lambda i:-candidates[i]['sequence_score'])
        assert order==row['sorted_generation_order']
        assert row['relevance']==[int(candidates[i]['sid']==row['target']) for i in order]
        proof.append({'ordinal':record['ordinal'],'row_idx':row['row_idx'],'user_id':row['user_id'],
                      'history_groups':len(before),'changed_tokens':sum(a!=b for a,b in zip(ids,actual,strict=True)),
                      'candidate_count':len(candidates),'mode':mode,'beam_expansion':20,'empty_initial_cache':True,
                      'raw_formal_sha256':sha256(record['formal']['raw_text'].encode()).hexdigest()})
        starts.append((record['formal']['start_byte'],record['formal']['end_byte_exclusive']))
    assert len(proof)==6 and starts[0][0]==0 and all(a[1]==b[0] for a,b in zip(starts,starts[1:]))
    return {'status':'PASS_FORMAL_PREFIX_CONTRACT','scope':'first six complete saved records',
            'rank_contract':{'assigned_users':rank['assigned_users'],'assignment_sha256':rank['assignment_sha256'],
                'run_commit':rank['RUN_COMMIT'],'adapter':baseline['adapter_identity'],'model_revision':rank['model']['revision'],
                'effective_generation':effective,'tensor_load_checks':450,'tokenizer_vocab_size':32965},
            'verified_rows':proof}


def main():
    p=argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument('sample',type=Path)
    p.add_argument('--capture',action='store_true')
    p.add_argument('--folder',default='gpu-formal-shuffle2-r0')
    args=p.parse_args()
    if args.capture:
        from lc_rec.shuffle_v2 import beauty, beauty_control as bc, control

        sample=capture(control.connect(),args.folder,json.loads(beauty.CONFIG.read_text()))
        result=verify(sample)
        bc.immutable(args.sample,sample)
    else:
        result=verify(json.loads(args.sample.read_text()))
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
