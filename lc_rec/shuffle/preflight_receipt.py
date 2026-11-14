"""Close real GPU acceptance with saved targets/metrics and runner time."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import datetime


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data-root',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();data=Path(a.data_root);root=Path(__file__).resolve().parent
    cfg=json.loads((root/'config.json').read_text());results={};jobs={};total=0
    for name in ['letter-shuffle-20261114-preflight-r0','letter-shuffle-20261114-preflight-r1']:
        path=Path('/home/bwwu/grid-audits/sid-extensions-p0/jobs')/name/'status.json'
        s=json.loads(path.read_text());assert s['cleanup']['verified']
        assert s['status']==('FAILED' if name.endswith('r0') else 'SUCCEEDED')
        wall=(datetime.datetime.fromisoformat(s['ended_at'])-datetime.datetime.fromisoformat(s['started_at'])).total_seconds()
        total+=2*wall;jobs[name]={'status':s['status'],'exit_code':s['exit_code'],'runner_wall_seconds':wall,'charged_device_seconds':2*wall,'status_sha256':sha(path)}
    assert total<=3600
    primary_path=root.parent/'a1_qwen3_06b/scripts/offline_metrics.py'
    spec=importlib.util.spec_from_file_location('metrics',primary_path);metrics=importlib.util.module_from_spec(spec);spec.loader.exec_module(metrics)
    for key,b in cfg['baselines'].items():
        cpu=json.loads((data/'cpu-bound-r0'/key/'audit.json').read_text())
        index=json.loads((Path(b['data-root'])/'Instruments/Instruments.index.json').read_text())
        sizes={}
        for sid in map(''.join,index.values()):sizes[sid]=sizes.get(sid,0)+1
        wanted=set(sum(cpu['preflight_assignments'].values(),[]));assert len(wanted)==32
        normal={}
        for rank in [0,1]:
            with (Path(b['normal_test_dir'])/('predictions.rank%d.jsonl'%rank)).open() as f:
                for line in f:
                    row=json.loads(line)
                    if row['row_idx'] in wanted:normal[row['row_idx']]=row
        seen=set();maxscore=0;peaks=[];references=[]
        for rank in [0,1]:
            base=data/'preflight-r1'/key
            rank_path=base/('rank%d.json'%rank);rank_report=json.loads(rank_path.read_text())
            assert rank_report['status']=='PREFLIGHT_INTERFACE_PASS' and rank_report['completed_users']==16
            assert rank_report['adapter_files']['adapter_model.bin']==b['adapter_identity']
            peaks.append(rank_report['gpu_peak_reserved_bytes']);references.append(rank_report['unmodified_reference_generation_expression'])
            accepted=json.loads((base/('acceptance.rank%d.json'%rank)).read_text())
            assert accepted['status']=='PASS' and len(accepted['samples'])==16
            with (base/('predictions.rank%d.jsonl'%rank)).open() as f:
                rows=[json.loads(line) for line in f]
            assert [r['row_idx'] for r in rows]==cpu['preflight_assignments'][str(rank)]
            for row,sample in zip(rows,accepted['samples'],strict=True):
                i=row['row_idx'];old=normal[i];assert i not in seen;seen.add(i)
                assert all(row[k]==old[k] for k in ['row_idx','user_id','target_item_id','target'])
                assert row['input_token_ids']==old['input_token_ids']
                assert row['input_nonpadding_tokens']==old['input_nonpadding_tokens']
                assert row['shuffle']['attention_mask']==[1]*len(row['input_token_ids'])
                assert sample['same_process_OFF_exact'] and sample['historical_sequences_scores_order_PASS']
                target=row['target'];lists=[]
                for item in [old,row]:
                    lists.append([item['candidates'][j]['sid'] for j in item['sorted_generation_order']])
                values=[dict(metrics.ranking(s,target)[0],**metrics.cce(s,target,sizes)) for s in lists]
                assert len(values[0])==16 and values[0]==values[1]
                for c,prior in zip(row['candidates'],old['candidates'],strict=True):maxscore=max(maxscore,abs(c['sequence_score']-prior['sequence_score']))
        assert seen==wanted
        results[key]={'status':'PASS','users':32,'original_OFF_target_all16_metrics_exact':True,
                      'maximum_OFF_historical_score_absolute_difference':maxscore,
                      'gpu_peak_reserved_bytes':peaks,'reference_generation_expressions':references}
    result={'status':'PASS','models':results,'jobs':jobs,'charged_gpu_device_seconds_including_runner':total,
            'charged_gpu_device_minutes_including_runner':total/60,'hard_device_seconds_limit':3600,
            'scope':'Saved actual native-block vs OFF/ON GPU evidence, plus CPU target/metric and full runner-time closure',
            'no_separate_original_CLI_rerun':True,'source_sha256':sha(__file__)}
    with Path(a.output).open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result))


if __name__=='__main__':main()
