"""Mechanical local delivery tables from collected, independently checked results."""
import datetime
import hashlib
import json
from pathlib import Path
from collect_results import RUN

ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[1]
NORMAL={
 'a0':REPO/'lc_rec/letter/evidence/final_audit_20261113/jobs/letter-a0-20261112-final-audit-20261113-offline-r1/results/primary',
 'a1':REPO/'lc_rec/a1_qwen3_06b/evidence/status_checks/20261114T034030Z/existing_results/offline_evaluation/primary'}


def load(path):return json.loads(path.read_text())
def job_id(model,n):
    return ('letter-a0-20261112-' if model=='a0' else 'letter-a1-q06-20261113-')+f'shuffle{n}-instruments-seed42-20261114-r0'
def table(header,rows):return '\n'.join(['| '+' | '.join(header)+' |','|'+'|'.join(['---']*len(header))+'|']+['| '+' | '.join(map(str,row))+' |' for row in rows])+'\n'
def number(x):return f'{x:.10f}'


def main():
    result={'status':'RUN_COMPLETE_METRICS_CHECKED_PENDING_NONAUTHOR_FINAL',
            'SHUFFLE_RUN_COMMIT':RUN,'EVALUATION_RUN_COMMIT':RUN,'models':{},'conditions':{},
            'scope':'Fixed trained models and single fixed Shuffle realization per condition; bootstrap is user uncertainty only.'}
    lines=['# Four-condition Shuffle results','',
           '**RUN COMPLETE / METRICS CHECKED; nonauthor final disposition is a separate report.**','',
           f'GPU and CPU evaluation RUN `{RUN}`. Fractions below use 0–1 units; relative differences use percent. All confidence intervals are user-level 5000 PCG64 draws, seed20261113, 95% linear percentile, conditional on fixed model and fixed Shuffle realization.','']
    for model in ['a0','a1']:
        normal=load(NORMAL[model]/'metrics.json');normal_groups=load(NORMAL[model]/'groups.json');normal_ci=load(NORMAL[model]/'bootstrap.json')
        keys=list(normal['metrics']);assert len(keys)==16
        result['models'][model]={'Normal':normal,'Normal_groups':normal_groups,'Normal_bootstrap':normal_ci}
        cases={}
        for n in [3,4]:
            root=ROOT/'results'/job_id(model,n);metrics=load(root/'evaluation/primary/metrics.json')
            groups=load(root/'evaluation/primary/groups.json');ci=load(root/'evaluation/primary/bootstrap.json')
            paired=load(root/'evaluation/paired/paired_comparison.json');summary=load(root/'summary.json')
            inputs=load(root/'evaluation/independent_inputs.json');statistics=load(root/'evaluation/independent_statistics.json')
            extended=load(root/'evaluation/independent_extended.json');manifest=load(root/'evaluation/prediction_manifest.json')
            state=load(root/'job/status.json');ranks=[load(root/'test'/f'rank{r}.json') for r in [0,1]]
            assert summary['status']=='PASS' and summary['RUN_COMMIT']==RUN
            assert state['status']=='SUCCEEDED' and state['exit_code']==0 and state['cleanup']['verified']
            assert metrics['run_commit']==metrics['evaluation_commit']==RUN
            assert metrics['users']==paired['users']==inputs['users']==24772
            assert inputs['status']=='PASS_INDEPENDENT_INPUT_REPLAY' and inputs['history_groups']==172929
            assert inputs['nonidentity_fraction']==inputs['input_changed_fraction']==1
            assert statistics['status']=='PASS' and len(metrics['metrics'])==16
            assert set(metrics['metrics'])==set(extended['metrics'])==set(paired['metrics'])==set(keys)
            assert all(r['status']=='PASS' and r['completed_users']==12386 for r in ranks)
            assert metrics['user_axis_sha256']==normal['user_axis_sha256']
            assert metrics['collision_catalog']==normal['collision_catalog']
            assert sum(metrics['first_target_rank_users'].values())==24772
            changes=paired['changes'];assert changes['users']==24772
            for name in ['top1_changed','top20_ordered_changed','top20_set_changed']:
                assert changes[name]/24772==changes[name+'_fraction']
            assert changes['top1_changed']<=changes['top20_ordered_changed'] and changes['top20_set_changed']<=changes['top20_ordered_changed']
            for k in [1,5,10,20]:
                transitions=changes['hit_transitions'][f'hit@{k}'];assert sum(transitions.values())==24772
                assert abs((transitions['0->1']-transitions['1->0'])/24772-paired['metrics'][f'hit@{k}']['absolute_difference'])<1e-13
                assert abs((transitions['1->0']+transitions['1->1'])/24772-normal['metrics'][f'hit@{k}'])<1e-13
                assert abs((transitions['0->1']+transitions['1->1'])/24772-metrics['metrics'][f'hit@{k}'])<1e-13
            for key in keys:
                p=paired['metrics'][key]
                assert abs(p['Normal']-normal['metrics'][key])<1e-13
                assert abs(p['Shuffle']-metrics['metrics'][key])<1e-13
                assert abs(p['Shuffle']-p['Normal']-p['absolute_difference'])<1e-13
                assert abs(metrics['metrics'][key]-extended['metrics'][key])<1e-13
            for kind,entries in normal_groups['groups'].items():
                assert set(entries)==set(groups['groups'][kind])
                assert sum(v['users'] for v in groups['groups'][kind].values())==24772
                for name,entry in entries.items():assert entry['users']==groups['groups'][kind][name]['users']
            for k in [5,10,20]:
                s=metrics['sid_statistics'][str(k)]
                assert s['valid']+s['invalid']==s['candidates']==24772*k
                assert s['coverage_denominator']==9897 and s['unique_valid_sids']/9897==s['unique_valid_sid_coverage']
            samples=[json.loads(x) for x in (root/'resources.jsonl').read_text().splitlines()]
            gpu_seconds=summary['gpu_stage_seconds']
            runner_wall=(datetime.datetime.fromisoformat(state['ended_at'])-datetime.datetime.fromisoformat(state['started_at'])).total_seconds()
            costs={'gpu_stage_wall_seconds_including_load_and_generation':gpu_seconds,
              'gpu_device_seconds':summary['gpu_device_seconds'],
              'full_generation_wall_seconds_max_rank':max(r['elapsed_generation_seconds'] for r in ranks),
              'model_load_seconds_per_rank':[r['model_load_and_strict_verification_seconds'] for r in ranks],
              'CPU_evaluation_seconds':summary['cpu_evaluation_seconds'],'runner_wall_seconds':runner_wall,
              'peak_allocated_bytes_per_rank':[r['gpu_peak_allocated_bytes'] for r in ranks],
              'peak_reserved_bytes_per_rank':[r['gpu_peak_reserved_bytes'] for r in ranks],
              'generation_latency':metrics['generation_cost'],
              'generation_users_per_second':24772/max(r['elapsed_generation_seconds'] for r in ranks),
              'minimum_MemAvailable_bytes':min(x['host']['MemAvailable'] for x in samples),
              'maximum_Swap_used_bytes':max(x['host']['SwapTotal']-x['host']['SwapFree'] for x in samples),
              'minimum_disk_available_bytes':min(x['disk_free_bytes'] for x in samples)}
            assert costs['gpu_device_seconds']==2*gpu_seconds
            cases[n]={'metrics':metrics,'groups':groups,'bootstrap':ci,'paired':paired,'inputs':inputs,
                      'prediction_manifest':manifest,'costs':costs,'job_id':job_id(model,n),'path':str(root.relative_to(REPO))}
            result['conditions'][model+'-S'+str(n)]=cases[n]
        lines += [f'## {model.upper()} Normal / Shuffle3 / Shuffle4','',table(['Metric','Normal','S3','S4'],[[k,number(normal['metrics'][k]),number(cases[3]['metrics']['metrics'][k]),number(cases[4]['metrics']['metrics'][k])] for k in keys]),'']
        for n,case in cases.items():
            lines += [f'### {model.upper()} Shuffle{n} paired change','',table(['Metric','Δ (Shuffle−Normal)','Relative %','Δ 95% CI','Shuffle 95% CI'],[[k,number(p['absolute_difference']),f"{100*p['relative_difference']:.6f}",f"[{number(p['difference_CI95_low'])}, {number(p['difference_CI95_high'])}]",f"[{number(case['bootstrap']['intervals'][k]['low'])}, {number(case['bootstrap']['intervals'][k]['high'])}]"] for k,p in case['paired']['metrics'].items()]),'']
        lines += ['### SID diagnostics','',table(['Condition','K','Valid / candidates','Invalid / users','Duplicates / users','Collision / users','Unique /9897','Coverage %'],[[condition,k,f"{s['valid']}/{s['candidates']}",f"{s['invalid']}/{s['users_invalid']}",f"{s['duplicates']}/{s['users_duplicate']}",f"{s['colliding']}/{s['users_colliding']}",s['unique_valid_sids'],f"{100*s['unique_valid_sid_coverage']:.6f}"] for condition,m in [('Normal',normal),('S3',cases[3]['metrics']),('S4',cases[4]['metrics'])] for k,s in m['sid_statistics'].items()]),'']
        lines += ['### First target rank distribution','',table(['Rank','Normal','S3','S4'],[[k,normal['first_target_rank_users'][k],cases[3]['metrics']['first_target_rank_users'][k],cases[4]['metrics']['first_target_rank_users'][k]] for k in [*map(str,range(1,21)),'miss']]),'']
        lines += ['### Fixed subgroups (four emphasized metrics)','All16 values remain in each primary/groups.json and SUMMARY.json. No subgroup CI is implied.','']
        for kind,entries in normal_groups['groups'].items():
            lines += [table(['Group','N','Metric','Normal','S3 (Δ)','S4 (Δ)'],[[kind+':'+name,v['users'],metric,number(v['metrics'][metric]),*[number(cases[n]['groups']['groups'][kind][name]['metrics'][metric])+' ('+number(cases[n]['groups']['groups'][kind][name]['metrics'][metric]-v['metrics'][metric])+')' for n in [3,4]]] for name,v in entries.items() for metric in ['hit@10','ndcg@10','ItemHit@10','ItemNDCG@10']]),'']
        lines += ['### Recommendation changes','',table(['Condition','Top1 changed','Ordered Top20 changed','Top20 set changed'],[['S'+str(n),*[f"{case['paired']['changes'][key]} ({100*case['paired']['changes'][key+'_fraction']:.6f}%)" for key in ['top1_changed','top20_ordered_changed','top20_set_changed']]] for n,case in cases.items()]),'']
        lines += [table(['Condition','K','00','01','10','11'],[['S'+str(n),k,*[trans[key] for key in ['0->0','0->1','1->0','1->1']]] for n,case in cases.items() for k,trans in case['paired']['changes']['hit_transitions'].items()]),'']
    total_runner=sum(v['costs']['runner_wall_seconds'] for v in result['conditions'].values())
    result['total_formal_conservative_runner_seconds']=total_runner
    assert total_runner<=64800
    for key,case in result['conditions'].items():assert case['costs']['gpu_stage_wall_seconds_including_load_and_generation']<=(14400 if key.startswith('a0') else 10800)
    lines += ['## Actual costs','',table(['Condition','Load max s','Generation s','GPU stage s','GPU-device h','CPU evaluation s','Runner wall s','P50 / P95 s','Users/s','Peak reserved GiB'],[[key,f"{max(c['costs']['model_load_seconds_per_rank']):.3f}",f"{c['costs']['full_generation_wall_seconds_max_rank']:.3f}",f"{c['costs']['gpu_stage_wall_seconds_including_load_and_generation']:.3f}",f"{c['costs']['gpu_device_seconds']/3600:.6f}",f"{c['costs']['CPU_evaluation_seconds']:.3f}",f"{c['costs']['runner_wall_seconds']:.3f}",f"{c['costs']['generation_latency']['p50_seconds']:.6f} / {c['costs']['generation_latency']['p95_seconds']:.6f}",f"{c['costs']['generation_users_per_second']:.6f}",f"{max(c['costs']['peak_reserved_bytes_per_rank'])/1024**3:.6f}"] for key,c in result['conditions'].items()]),'',
      'Both GPUs were reserved for each GPU stage; CPU evaluation is reported separately. Summed user latency is not end-to-end wall time. Model-size speed differences do not identify a Shuffle or architecture causal effect.','',
      'Full per-condition JSON contains permutation frequencies, resource minima, target-collision counts, all16 subgroup values, bootstrap intervals, paired intervals, prediction SHA and paths. Original large predictions and models remain on new-server.','']
    destination=ROOT/'results';destination.mkdir(exist_ok=True)
    (destination/'SUMMARY.json').write_text(json.dumps(result,indent=2)+'\n')
    (destination/'RESULTS.md').write_text('\n'.join(lines))
    print(json.dumps({'status':result['status'],'conditions':len(result['conditions']),'runner_seconds':total_runner}))


if __name__=='__main__':main()
