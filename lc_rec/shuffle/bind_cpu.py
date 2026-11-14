"""Add asset/source bindings to successful full CPU scans without rerunning them."""
import argparse
import hashlib
import json
from pathlib import Path


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--cpu-root',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();root=Path(__file__).resolve().parent
    cfg=json.loads((root/'config.json').read_text());old=Path(a.cpu_root)
    assert json.loads((old/'summary.json').read_text())['status']=='PASS_CPU_PREPARATION'
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    assets={'index':'33c87e28c52ae367985bdf494cb49d676d181e13a39497b4ad035f46111546dd',
            'inter':'0a11d47791d90483005da93c36d826679e20e4592d1d2b86518d5f56f218e743',
            'item':'1d9938e576a8d964e6b977a2b63aa56402840a788eaa6685f26f0d910a7133f2'}
    scans={}
    for key,b in cfg['baselines'].items():
        src=old/key/'audit.json';audit=json.loads(src.read_text())
        assert audit['status']=='PASS_CPU_FULL_INPUTS' and audit['users']==24772
        assert audit['manifest']['sha256']==sha(audit['manifest']['path'])
        for kind,expected in assets.items():
            assert sha(Path(b['data-root'])/'Instruments'/('Instruments.'+kind+'.json'))==expected
        audit['asset_sha256']=assets
        audit['saved_tokenizer_files']={q.name:{'bytes':q.stat().st_size,'sha256':sha(q)}
             for q in Path(b['adapter']).iterdir() if q.is_file() and
             (q.name.startswith('tokenizer') or q.name in ['added_tokens.json','special_tokens_map.json','vocab.json','merges.txt'])}
        audit['original_CPU_receipt']={'path':str(src),'sha256':sha(src)}
        scans[key]=audit
        (out/key).mkdir();(out/key/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    left,right=scans['a0'],scans['a1']
    assert left['history_lengths']==right['history_lengths']
    shared=out/'shared_permutations.jsonl';count=0
    with Path(left['manifest']['path']).open() as l,Path(right['manifest']['path']).open() as r,shared.open('x') as dest:
        for arow,brow in zip(l,r,strict=True):
            av,bv=json.loads(arow),json.loads(brow)
            assert av['row_idx']==bv['row_idx']==count and av['user_id']==bv['user_id']
            record={k:av[k] for k in ['row_idx','user_id','visible_history_length']}
            for n in ['3','4']:
                assert av[n]['permutations']==bv[n]['permutations']
                record[n]=av[n]['permutations']
            dest.write(json.dumps(record,separators=(',',':'))+'\n');count+=1
    assert count==24772
    result={'status':'PASS_CPU_PREPARATION','source_receipt':{'path':str(old/'summary.json'),'sha256':sha(old/'summary.json')},
            'source_and_tokenizer_bound':True,'same_user_permutations_across_models':True,
            'shared_permutation_manifest':{'path':str(shared),'bytes':shared.stat().st_size,'sha256':sha(shared)},
            'input_scans_reused_without_regeneration':True}
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()
