"""Assemble actual accepted results and original job identities; never execute models."""
import datetime as dt
import hashlib
import json
from pathlib import Path
import shutil

REPO=Path('/home/addaswsw/recommend-worktrees/sid-extensions-p0')
SUPPORT=Path('/home/addaswsw/recommend-support/grid-shuffle-20261017')
JOBS=Path('/home/addaswsw/recommend-support/sid-extensions-p0/jobs/new-server')
OUT=REPO/'manifests/grid-shuffle-20261017'
ANCHOR='d0bcacfee4bab4276861573f7746405fb63c54f2'
METRICS=['recall@5','ndcg@5','recall@10','ndcg@10']


def read(path):return json.loads(path.read_bytes())
def identity(path):return {'path':str(path),'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
def write(path,value):path.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def main():
    observation=read(SUPPORT/'final-state-query.json')
    assert observation['active_job_ids']==[] and observation['all_task_jobs_terminal_and_clean']
    conditions=[]
    for domain in ('toys','sports'):
        for n,label in [(0,'normal'),(3,'shuffle3'),(4,'shuffle4')]:
            review_path=SUPPORT/'results-audit'/f'{domain}-b-{n}-evaluation-review.json'
            review=read(review_path);assert review['status']=='FORMAL_EVALUATION_ACCEPTANCE_PASS'
            attention_path=SUPPORT/'results-audit'/f'{domain}-b-{n}-attention-review.json'
            attention_review=read(attention_path);assert attention_review['status']=='INDEPENDENT_FORMAL_ATTENTION_RESULT_PASS'
            spec=read(REPO/f'experiments/shuffle_20261017/specs/{domain}-b-{n}.json')
            expected_job=f'gs-20261017-{domain}-b-{label}-v1'
            assert (review['domain'],review['route'],review['shuffle_n'],review['job_id'])==(domain,'B',n,expected_job)
            assert (attention_review['domain'],attention_review['route'],attention_review['shuffle_n'])==(domain,'B',n)
            assert attention_review['job_id']==f'gs-20261017-{domain}-b-{label}-attention-v1'
            assert attention_review['accepted_evaluation_review']['sha256']==identity(review_path)['sha256']
            for key in ('config','sid','checkpoint'):
                assert review[key]==spec[key]
            attention_spec=read(REPO/f'experiments/shuffle_20261017/specs/attention-{domain}-b-{n}.json')
            assert (attention_spec['domain'],attention_spec['route'],attention_spec['shuffle_n'])==(domain,'B',n)
            assert all(attention_spec[k]==spec[k] for k in ('config','sid','checkpoint','attention_cohort'))
            query_identity=SUPPORT/'results-audit'/f'{domain}-b-{n}-decoder-query-identity.json'
            assert query_identity.is_file()
            conditions.append({'domain':domain,'route':'B','shuffle_n':n,'condition':label,
                'metrics':review['metrics'],'samples':review['samples'],'unique_users':review['unique_users'],
                'invalid_candidate_count':review['invalid_candidate_count'],'duplicate_candidate_count':review['duplicate_candidate_count'],
                'config':spec['config'],'sid':spec['sid'],'checkpoint':spec['checkpoint'],
                'test_index':spec['test_index'],'attention_cohort':spec['attention_cohort'],
                'evaluation_job_id':review['job_id'],'attention_job_id':f'gs-20261017-{domain}-b-{label}-attention-v1',
                'source_commit':review['source_commit'],'evaluation_snapshot_sha256':review['snapshot_sha256'],
                'prediction_sha256':review['prediction_sha256'],'intervention_trace':review['intervention_trace'],
                'independent_evaluation_review':identity(review_path),'independent_attention_review':identity(attention_path),
                'decoder_query_identity':identity(query_identity),
                'decoder_query_identity_scope':'CPU-derived from actual saved beam prefixes/ancestry and byte-bound original decoder offset/lookup/BOS source; learned BOS at step0 has no raw SID integer; stack input_ids array was not directly captured'})
    for condition in conditions:
        normal=next(r for r in conditions if r['domain']==condition['domain'] and r['shuffle_n']==0)
        condition['differences_from_own_Normal']={name:{'absolute':condition['metrics'][name]-normal['metrics'][name],
            'relative_percent':100*(condition['metrics'][name]-normal['metrics'][name])/normal['metrics'][name]} for name in METRICS}
    paired=read(SUPPORT/'results-audit/toys-sports-paired-shuffle-analysis.json')
    paired_review=read(SUPPORT/'source-audit/paired-analysis-independent-review.json')
    assert paired['status']=='DESCRIPTIVE_PAIRED_ANALYSIS_COMPLETE' and len(paired['comparisons'])==4
    assert paired_review['status']=='INDEPENDENT_COMPLETE_PAIRED_ANALYSIS_PASS'
    query_review_path=SUPPORT/'source-audit/decoder-query-identities-independent-review.json'
    query_review=read(query_review_path)
    assert query_review['status']=='INDEPENDENT_DECODER_QUERY_IDENTITIES_PASS'
    assert query_review['condition_count']==6 and query_review['total_query_identity_records']==1536
    stats_path=SUPPORT/'attention-statistics/attention-descriptive-statistics.json';statistics=read(stats_path)
    stats_review_path=SUPPORT/'results-audit/attention-statistics-independent-review.json';stats_review=read(stats_review_path)
    assert stats_review['status']=='INDEPENDENT_ATTENTION_DESCRIPTIVE_STATISTICS_PASS'
    assert len(statistics['jobs'])==6 and sum(len(r['per_sample_query_pairs']) for r in statistics['jobs'])==2304
    inherited=read(REPO/'manifests/grid-formal-20261015/results-review/beauty-six-formal-results-independent-summary.json')
    inherited_pairs=read(REPO/'manifests/grid-formal-20261015/results-review/beauty-paired-shuffle-analysis.json')
    inherited_stats=read(REPO/'manifests/grid-formal-20261015/attention-statistics/attention-descriptive-statistics.json')
    jobs=[]
    for path in sorted(JOBS.glob('gs-20261017-*')):
        root=path/'retrieved';assert root.is_dir(),path
        request,status=read(root/'request.json'),read(root/'status.json')
        current=observation['jobs'][path.name]
        assert current['status']==status['status'] and current['exit_code']==status['exit_code']
        assert current['cleanup_verified'] and not current['worker_alive'] and not current['workgroup_live']
        seconds=(dt.datetime.fromisoformat(status['ended_at'])-dt.datetime.fromisoformat(status['started_at'])).total_seconds()
        invocation=read(root/'output/invocation.json') if (root/'output/invocation.json').exists() else {}
        deployment=invocation.get('code_identity',{})
        if not isinstance(deployment,dict):deployment={}
        jobs.append({'job_id':path.name,'status':status['status'],'exit_code':status['exit_code'],
            'worker_wall_seconds_including_CPU_preparation':seconds,'request_cap_seconds':request['timeout_seconds'],
            'GPU_requested':bool(request['env'].get('CUDA_VISIBLE_DEVICES')),
            'physical_GPU':request['env'].get('CUDA_VISIBLE_DEVICES'),'capture_version':request['capture_version'],
            'source_commit':request['capture_version'].split('/')[0],
            'snapshot_sha256':deployment.get('snapshot_sha256'),'snapshot_root':request['cwd'],
            'argv':request['argv'],'local_evidence':str(root),'remote_job_root':'/home/bwwu/grid-audits/sid-extensions-p0/jobs/'+path.name,
            'request':identity(root/'request.json'),'status_record':identity(root/'status.json'),
            'export_manifest':identity(root/'export-manifest.json'),'cleanup_verified':True})
    assert {r['job_id'] for r in jobs}==set(observation['jobs'])
    def cost(kind):return sum(r['worker_wall_seconds_including_CPU_preparation'] for r in jobs if r['GPU_requested'] and kind in r['job_id'])
    checks=cost('preflight');attention=cost('attention-v1')
    formal=sum(r['worker_wall_seconds_including_CPU_preparation'] for r in jobs if r['GPU_requested'] and 'preflight' not in r['job_id'] and 'attention' not in r['job_id'])
    assert checks<=1800 and formal<=6*1800 and attention<=6*1200 and checks+formal+attention<=5.5*3600
    summary={'task':'grid-shuffle-20261017','status':'PARTIAL','qualified_B_conditions_complete':True,
        'reason_for_PARTIAL':['Toys A has two saved complete dual-table candidates, no approved unique selection and conflicting upstream producer evidence; no candidate silently selected.',
            'Sports A has no complete matching source/SID/checkpoint/run chain in the already preserved targeted archive context; no new training or SID generation.'],
        'inherited_Beauty_anchor':ANCHOR,'Beauty_rerun':False,'new_training':False,'SID_regenerated':False,
        'branch':'wbw','active_job_ids':[],'conditions':conditions,'paired_comparisons':paired['comparisons'],
        'attention_statistics':{'report':identity(stats_path),'independent_review':identity(stats_review_path),
            'conditions':[{k:r[k] for k in ['job_id','samples','summary','least_similar_three_under_declared_mean_cosine','maximum_masked_mass','maximum_raw_row_sum_error']} for r in statistics['jobs']],
            'pair_path_records':2304,'raw_NPZ_files':24,'all_formal_PNG_files':192},
        'inherited_Beauty':{'conditions':inherited['conditions'],'paired_comparisons':inherited_pairs['comparisons'],
            'attention_summaries':[{k:r[k] for k in ('job_id','summary')} for r in inherited_stats['jobs'] if '-mlp-' not in r['job_id']]},
        'jobs':jobs,'budgets':{'accounting':'Sum actual requested worker wall, including job CPU preparation; not GPU utilization time',
            'GPU_checks_seconds':checks,'formal_evaluation_seconds':formal,'attention_seconds':attention,
            'total_GPU_request_worker_wall_hours':(checks+formal+attention)/3600,'hard_limit_hours':5.5,
            'per_evaluation_cap_seconds':1800,'per_attention_cap_seconds':1200},
        'task_correction_rounds_used':3,'maximum_task_correction_rounds':3,'old_Beauty_correction_history_unchanged':True,
        'final_metadata_correction':identity(SUPPORT/'correction-round-3.json'),
        'extra_failed_CPU_evidence':['gs-20261017-toys-b-callback-audit-v1','gs-20261017-sports-b-callback-audit-v1'],
        'extra_CPU_evidence_scope':'Optional callback deserialization failed at import cycle; no model/GPU or asset change; original CSV/stage_result selection and strict checkpoint/SID binding passed. Failure retained, not reported as success.',
        'asset_audit':identity(SUPPORT/'asset-audit/runtime-asset-admission.json'),
        'paired_analysis_independent_review':identity(SUPPORT/'source-audit/paired-analysis-independent-review.json'),
        'decoder_query_identity_independent_review':identity(query_review_path),
        'reviewers':['/root/asset_audit','/root/source_audit','/root/results_audit','/root/delivery_audit'],
        'scientific_limits':['One seed42, no measured cross-seed stability or invariance proof.',
            'Every route uses its own Normal; A/B absolute differences confound upstream/model/checkpoint conditions.',
            'Aggregate close metrics do not imply unchanged user recommendations.',
            'Within-condition query-row statistics are descriptive; no unaligned cross-Shuffle column comparison or target-prefix nonuse inference.',
            'fixed32 comes from evaluation split first32 distinct users selected before model predictions; full formal recommendation metrics use testing split.'],
        'installation_log':read(SUPPORT/'INSTALLATION_LOG.json'),'GPT_UPLOAD_REQUIRED':'YES',
        'publication_and_final_container_review':'Separate content/delivery commits and external real readback/container receipts; not pre-claimed by this computation summary.'}
    write(OUT/'EXECUTION_SUMMARY.json',summary)
    lines=['# Toys / Sports Shuffle 正式结果与注意力','',
        '`PARTIAL`：两域 B 六条件及对应 fixed32、配对分析和2304条描述统计完成；Toys A未决、Sports A配套缺失。Beauty仅继承，未重跑。',
        '',f'正式评测源码：`{conditions[0]["source_commit"]}`。逐作业快照/实际 argv/输入 hash/原预测/矩阵/审核链见 EXECUTION_SUMMARY.json。',
        '', '| 域 | 路线 | 条件 | 用户 | Recall@5 | NDCG@5 | Recall@10 | NDCG@10 | 非法候选 |',
        '|---|---|---|---:|---:|---:|---:|---:|---:|']
    for r in inherited['conditions']:
        lines.append('| Beauty | '+r['route']+' | '+{0:'Normal',3:'Shuffle3',4:'Shuffle4'}[r['shuffle_n']]+' | 22363 | '+' | '.join(f'{r["metrics"][k]:.8f}' for k in METRICS)+f' | {r["invalid_candidate_count"]} |')
    for r in conditions:lines.append(f'| {r["domain"].title()} | B | {r["condition"]} | {r["samples"]} | '+' | '.join(f'{r["metrics"][k]:.8f}' for k in METRICS)+f' | {r["invalid_candidate_count"]} |')
    lines+=['','三条件各域采用同 config/SID/checkpoint、seed42、原 batch8、32-true、120 SID输入位、TopK10及原生成器/evaluator；prefix约束保持False，非法候选不补齐、不过滤、不重排。A/B不是严格同模型对照。',
        '', '| 域 | 条件 | ΔRecall@5 (相对%) | ΔNDCG@5 (相对%) | ΔRecall@10 (相对%) | ΔNDCG@10 (相对%) |',
        '|---|---|---:|---:|---:|---:|']
    for r in conditions:
        if r['shuffle_n']:
            lines.append(f'| {r["domain"].title()} | {r["condition"]} | '+' | '.join(f'{r["differences_from_own_Normal"][k]["absolute"]:+.8f} ({r["differences_from_own_Normal"][k]["relative_percent"]:+.3f}%)' for k in METRICS)+' |')
    lines+=['',
        '', '| 域 | 对比 | Top1改变 | 有序Top10改变 | Top10集合改变 |', '|---|---|---:|---:|---:|']
    for r in paired['comparisons']:
        v=r['recommendation_list_changes'];lines.append(f'| {r["domain"].title()} | Normal→Shuffle{r["Normal_to_Shuffle_n"]} | {v["top1_changed_users"]} | {v["ordered_top10_changed_users"]} | {r["paired_users"]-v["same_top10_set_users"]} |')
    lines+=['','三域 B 的 Shuffle 都改变大量逐用户推荐。Beauty B 的 Recall@10 相对变化为 Shuffle3约−1.29%、Shuffle4约−2.10%；Toys约−0.63%/−1.02%，Sports约−9.54%/−4.30%。下降幅度有域差异，Sports中Shuffle4下降小于Shuffle3，不能推出扰动数量与损失单调对应。Toys/Sports本轮四指标各条件均下降；Beauty Shuffle3的Recall@5略升，其余变化按原表保留。A只可继承Beauty现有结果，不能声称A的三域现象已覆盖。']
    lines+=['','四指标绝对/相对变化、命中迁移、逐用户分布/原始配对记录和预声明异常样本见 results-audit/toys-sports-paired-shuffle-analysis.json。','',
        '每条件32用户，共4×8，三条件共同cohort且预测前确定。原始最后层6heads、4个autoregressive query、真实最终Top1完整父链与局部首父链均保留。全部24 NPZ/192 PNG在服务器正式job及已验证本地export；轻量包图像按事先固定规则展示。',
        'Decoder query身份另以CPU从实际保存的beam前缀/父链和固定源码映射派生：首步为learned BOS（无SID整数ID），后续为对应parent前缀末码及offset lookup索引。每组256身份记录，六组1536条；这是可审查的源码推导，不宣称直接捕获未保存的T5Stack input_ids，也不是2304条注意力行对统计。','',
        '| 正式注意力条件 | 路径 | Cosine median | Pearson median | TV median | JS median (nats) |',
        '|---|---|---:|---:|---:|---:|']
    for r in statistics['jobs']:
        label=r['job_id'].removeprefix('gs-20261017-').removesuffix('-attention-v1')
        for path,v in r['summary'].items():lines.append(f'| {label} | {path} | '+ ' | '.join(f'{v[k]["median"]:.6f}' for k in ['cosine','pearson','total_variation','jensen_shannon_nats'])+' |')
    lines+=['','每路径192条，共每条件384条；各指标min/median/mean/max、全部逐pair记录、最不相似三个样本、padding质量与行和误差见 attention-statistics/attention-descriptive-statistics.json。无相似度验收阈值；不能据此证明decoder不利用目标前缀或SID严格置换不变。','',
        '资产与A缺口见 asset-source-review.md、asset-audit/runtime-asset-admission.json。Toys A需要学长指定唯一完整run及真实T5/R-Kmeans来源；Sports A需要完整匹配链。本轮不训练、不混拼、不重建SID。','',
        f'实际GPU请求worker wall总计 {summary["budgets"]["total_GPU_request_worker_wall_hours"]:.6f} 小时，含CPU准备；预算5.5小时。修正3/3（新入口bootstrap、未用Hydra元数据回读、最终审核索引读取实际六条件query回执），旧Beauty额度不变。第三轮仅修复元数据路径，不改预测、注意力或统计。历史准入门保留当时2/3记录。可选callback CPU v1两失败原件保留；CSV/stage_result选模和B严格语义门已通过。','',
        '所有本任务作业终态、进程组清理已核，无活动job。服务器恢复入口 `/home/bwwu/grid-audits/sid-extensions-p0/jobs/gs-20261017-*`；只按原job查询，不自动重启。','',
        '`GPT_UPLOAD_REQUIRED: YES`。固定GitHub发布、实际回读、最终ZIP大小/成员/SHA及真实非作者容器审核见最终交付回执。','',
        '可执行条件完成；缺失路线未执行。执行停止，等待GPT网页版二次审核。']
    (OUT/'FINAL_REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'status':'PARTIAL','qualified_conditions':6,'formal_attention_conditions':6,'statistical_records':2304,'jobs':len(jobs),'GPU_request_worker_wall_hours':summary['budgets']['total_GPU_request_worker_wall_hours']}))


if __name__=='__main__':main()
