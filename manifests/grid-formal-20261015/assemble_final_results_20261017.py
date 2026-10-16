"""Publishable integration of existing results and the accepted CPU statistics.

Only the two explicitly authorized current reports are updated. No model,
evaluator, original result, formula, or historical review is changed.
"""
from pathlib import Path
import datetime as dt
import hashlib
import json
import shutil
import zipfile

R=Path('/home/addaswsw/recommend-worktrees/sid-extensions-p0')
S=Path('/home/addaswsw/recommend-support/grid-formal-20261015')
P=Path('/home/addaswsw/recommend-support/sid-extensions-p0/jobs/new-server')
F=R/'manifests/grid-formal-20261015'

def identity(p):
    b=p.read_bytes();return {'path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
def json_bytes(x):return (json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode()
def create(p,b):
    p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists():assert p.read_bytes()==b,str(p)
    else:
        with p.open('xb') as f:f.write(b)
def copy(src,dst):create(dst,src.read_bytes())
def newjson(p,x):create(p,json_bytes(x))
def label(job):return job.removeprefix('gf-20261015-').removesuffix('-attention-v1')

facts=json.loads((S/'final-integration-source-facts-20261017.json').read_text())
stats=json.loads((S/'attention-statistics-20261017/attention-descriptive-statistics.json').read_text())
runtime=S/'attention-results-review/attention-statistics-runtime-review.json'
assert json.loads(runtime.read_text())['report']['sha256']==identity(S/'attention-statistics-20261017/attention-descriptive-statistics.json')['sha256']
resumption=json.loads((S/'final-resumption-20261017.json').read_text())
assert resumption['returncode']==0
hist=S/'finalization-input-history-20261017'
for name in ['EXECUTION_SUMMARY.json','FINAL_RECEIPT.md']:
    copy(F/name,hist/name)
copy(S/'final-integration-source-facts-20261017.json',F/'final-integration-source-facts-20261017.json')
copy(S/'final-resumption-20261017.json',F/'final-resumption-20261017.json')
for p in sorted((S/'attention-results-review').iterdir()):
    if p.is_file() and p.suffix in {'.py','.json'}:copy(p,F/'attention-results-review'/p.name)
for p in sorted((S/'attention-statistics-20261017').iterdir()):
    if p.is_file():copy(p,F/'attention-statistics'/p.name)
copy(S/'run_attention_statistics_20261017.py',F/'run_attention_statistics_20261017.py')
copy(S/'assemble_final_results_20261017.py',F/'assemble_final_results_20261017.py')

artifact_jobs=[]
for job in stats['jobs']:
    id=job['job_id'];base=P/id/'retrieved';target=F/'runs'/id
    for name in ['request.json','status.json','cleanup.json','accepted.json','attempt.json','export-manifest.json','output/RESULT.json','output/invocation.json','output/effective_config.yaml','output/used_config.json']:
        if (base/name).is_file():copy(base/name,target/name)
    original_review=S/'attention-results-review'/(id+'-review.json')
    batches=[];pngs=[]
    for i in range(4):
        b=base/'output'/f'batch-{i:02d}'
        for rel in ['off_on_checks.json','figures/plot_receipt.json']:
            copy(b/rel,target/'output'/b.name/rel)
        for product in json.loads((b/'figures/plot_receipt.json').read_text())['products']:
            pngs.append(dict(product,local_path=str(b/'figures'/product['file']),remote_path='/home/bwwu/grid-audits/sid-extensions-p0/jobs/'+id+'/output/'+b.name+'/figures/'+product['file']))
        batches.append({'batch':i,'record':identity(b/'bundle/record.json'),'arrays':identity(b/'bundle/arrays.npz'),'plot_receipt':identity(b/'figures/plot_receipt.json'),'off_on':identity(b/'off_on_checks.json')})
    # Deterministic representatives plus the least mean-cosine case for each path.
    selected={'output/batch-00/figures/sample_000_cross_attention.png'}
    for path,rows in job['least_similar_three_under_declared_mean_cosine'].items():selected.add(rows[0]['plot_relative'])
    published=[]
    for rel in sorted(selected):
        src=base/rel;parts=Path(rel).parts
        name=parts[1]+'_'+parts[-1]
        dest=F/'figures'/id/name;copy(src,dest)
        published.append({'original_relative':rel,'repository_path':str(dest.relative_to(R)),**identity(src)})
    artifact_jobs.append({'job_id':id,'main_working_baseline':label(id)=='b-normal','result':identity(base/'output/RESULT.json'),'original_export_manifest':identity(base/'export-manifest.json'),'technical_review':identity(original_review),'batches':batches,'png_inventory':pngs,'published_figures':published,'all_original_files_retained':True})
newjson(F/'ATTENTION_ARTIFACT_INDEX.json',{'jobs':artifact_jobs,'new_attention_jobs':8,'new_raw_NPZ':32,'original_PNG':256,'payload_storage':'server originals and WSL retrieved; final audit ZIP includes all raw matrices and bounded real PNGs'})

with zipfile.ZipFile(S/'deliveries/grid-formal-20261016-gpt-audit-snapshot.zip') as z:
    old_map=json.loads(z.read('references/JOB_VERSION_ASSET_MAP.json'))
newjson(F/'FINAL_VERSION_MAP.json',{'current_content_anchor':'3c4aa52d43cd145cd4cae7d93643be253ab79442','job_map':old_map,'fresh_remote_job_and_manifest_observation':resumption['result'],'statistics':{'source_commit':'3c4aa52d43cd145cd4cae7d93643be253ab79442','script':stats['invocation']['script'],'plan':stats['invocation']['plan'],'result':identity(S/'attention-statistics-20261017/attention-descriptive-statistics.json'),'actual_GPU_or_model_calls':0},'inherited_Beauty_MLP_statistics':{'included':False,'reason':'original formal RESULT.status=PASS, whereas unchanged CPU analyze contract requires COMPLETE; retained original result and old independent fixed32 review; no schema/status rewriting'}})

summary=json.loads((hist/'EXECUTION_SUMMARY.json').read_text())
summary['status']='COMPLETED_PENDING_DELIVERY_REVIEW'
summary['remaining_work']=['Normal wbw publication and actual remote readback','Final audit ZIP and nonauthor package validation','Final readiness metadata publication']
summary['formal_results_complete']=True
summary['attention_capture_and_technical_review_complete']=True
summary['attention_statistics_complete']=True
summary['attention_statistics_independent_review_complete']=True
summary['beauty_shuffle_final']=facts['beauty_shuffle']
summary['MLP_final']=facts['MLP']
summary['attention_final']=facts['attention']
summary['attention_statistics']={'conditions':8,'samples_each':32,'pairs_each_path':192,'pair_path_records':3072,'report':identity(S/'attention-statistics-20261017/attention-descriptive-statistics.json'),'runtime_review':identity(runtime),'source_provenance_review':identity(S/'attention-results-review/attention-statistics-source-provenance-review-20261017.json'),'summary':[{'job_id':j['job_id'],'summary':j['summary'],'least_similar_three_under_declared_mean_cosine':j['least_similar_three_under_declared_mean_cosine'],'maximum_masked_mass':j['maximum_masked_mass'],'maximum_raw_row_sum_error':j['maximum_raw_row_sum_error']} for j in stats['jobs']],'inherited_Beauty_exclusion':'Original PASS/COMPLETE input protocol difference; original result unmodified; existing formal fixed32 remains accepted'}
summary['finalization_20261017']={'authorized_scope':'unchanged CPU descriptive statistics, nonauthor review, integration, wbw release and final audit ZIP only','new_GPU_budget':0,'new_GPU_model_calls':0,'new_training_or_full_evaluation_or_capture_jobs':0,'new_research_correction_rounds_used':0,'original_three_plus_single_exception_unchanged':True,'stage_external_review':'User reports GPT reviewed the 20261016 snapshot and fixed 3c4aa52 content; final new output external review still pending','preserved_old_report':str(hist/'FINAL_RECEIPT.md'),'source_facts':identity(S/'final-integration-source-facts-20261017.json'),'new_packages':[],'system_configuration_changes':[]}
summary['research_limits']=['Single fixed32 cohort and original randomization seed; descriptive statistics are not a mathematical invariance proof','A/B different SID/model conditions; no unaligned column subtraction or strict cross-model structural delta','Higher cosine does not imply identical row distributions; TV/JS and low-similarity cases retained','No evidence of lossless parallel generation or complete neglect of SID structure','Toys original bundle retains upstream factory/path and unique producer binding uncertainty','Sports is an explicitly approved own-domain Qwen/RK plus original per-item MLP structure experiment, not restored complete historical Sports conditions']
(F/'EXECUTION_SUMMARY.json').write_bytes(json_bytes(summary))

metric_names=['recall@5','ndcg@5','recall@10','ndcg@10']
lines=['# GRID 正式实验统一结果与最终交付回执','',
 '## A. 总体状态','',
 '**COMPLETED_PENDING_DELIVERY_REVIEW**：六组Beauty正式全量评测、三域MLP正式成果、八组新增fixed32注意力以及本轮3072条CPU描述统计和非作者实际复算均完成。GitHub发布/回读、最终ZIP及其独立验包尚待本轮后续步骤；此处不预签这些检查。',
 '','本轮没有训练、全量测试、注意力重采或GPU模型调用。GPT已审阶段快照仅覆盖旧成果；新统计和最终发布仍等待网页二次审核。原进行中报告已在支持目录 finalization-input-history-20261017 保留原文。',
 '','## B. Beauty Shuffle','',
 '每组22363用户；A、B分别比较各自Normal。原should_check_prefix=False及候选评价不变，不过滤/补齐/重排无效SID。',
 '','| 条件 | Recall@5 | NDCG@5 | Recall@10 | NDCG@10 |','|---|---:|---:|---:|---:|']
for j in facts['beauty_shuffle']:lines.append('| '+j['condition']+' | '+' | '.join(f"{j['metrics'][m]:.9f}" for m in metric_names)+' |')
lines+=['','同路线Normal的绝对差及相对百分比（比例值差，不是百分点）：','','| 条件 | ΔR@5（相对） | ΔN@5（相对） | ΔR@10（相对） | ΔN@10（相对） |','|---|---:|---:|---:|---:|']
for j in facts['beauty_shuffle']:
    if j['condition'].endswith('Normal'):continue
    d=j['delta_vs_within_route_normal']
    lines.append('| '+j['condition']+' | '+' | '.join(f"{d[m]['absolute']:+.9f} ({d[m]['relative_percent']:+.4f}%)" for m in metric_names)+' |')
lines+=['','四项相对变化处于约−2.105%至+0.311%，但这不表示用户推荐列表不变：','','| 条件 | Top1改变用户 | 有序Top10改变用户 |','|---|---:|---:|']
for j in facts['beauty_shuffle']:
    p=j['paired_observations']
    if p:
        c=p['recommendation_list_changes'];lines.append('| '+j['condition']+' | '+str(c['top1_changed_users'])+' | '+str(c['ordered_top10_changed_users'])+' |')
lines+=['','来源、预测hash、完整候选合法性及逐用户文件见 [既有配对分析](results-review/beauty-paired-shuffle-analysis.json)、[六条件验收](results-review/beauty-six-formal-results-independent-summary.json) 和 [本次事实摘录](final-integration-source-facts-20261017.json)。',
 '','## C. 三域MLP','',
 '三域均scratch、全参数训练；模型为共享逐物品MLP聚合，decoder仍四码位自回归。原验证Recall@5最大值选best、patience10早停；不按test选模。',
 '','| 域 | raw/accum/val间隔 | 可训练参数 | final/best step | 完整验证次数 | test条数 | worker小时 |','|---|---|---:|---|---:|---:|---:|']
for j in facts['MLP']:
    p=j['approved_protocol'];lines.append(f"| {j['domain']} | {p['raw_microbatch']}/{p['accumulate_grad_batches']}/{p['validation_interval_microbatches']} | {j['parameters_trainable']} | {j['global_step']}/{j['best_global_step']} | {j['completed_full_validations']} | {j['test_samples']} | {j['worker_wall_hours']:.6f} |")
lines+=['','| 域 | Recall@5 | NDCG@5 | Recall@10 | NDCG@10 |','|---|---:|---:|---:|---:|']
for j in facts['MLP']:lines.append('| '+j['domain']+' | '+' | '.join(f"{j['test_metrics'][m]:.9f}" for m in metric_names)+' |')
lines+=['','Beauty为已批准T5/RQ-VAE+STE相关条件。Toys沿用作者固定RQ-VAE+STE bundle，上游T5 factory与Qwen路径冲突及唯一历史producer未绑定的限制仍保留，不能写成已证明某个唯一上游编码器。Sports为用户批准的本域Qwen/R-Kmeans＋学长逐物品MLP结构实验，不是已恢复学长原始Sports完整历史实验。各域SID、模型及训练条件不同，不作严格横向结构对照。单卡累积协议经批准，但不等同多卡DDP完整随机轨迹。',
 '','best checkpoint真实路径、大小、SHA、配置/SID及原训练source均见 [事实摘录](final-integration-source-facts-20261017.json)。三域训练/测试均低于各自12h；Toys/Sports加正式attention累计约6.466h/6.929h。已有源码/选模/严格加载与完整预测的真实非作者审核继续继承；未重载大checkpoint。',
 '','## D. 正式注意力与描述统计','',
 '八组新增采集均32用户、4批，每批8用户，32份原始NPZ和256张原PNG保留；B-Normal就是当前普通工作baseline，不新增重复baseline。Beauty MLP既有fixed32独立保留：旧formal RESULT.status=PASS，不满足未修改统计程序的COMPLETE入口契约，故不纳入本轮八组数值统计，不改写原件。',
 '','主路径为最终Top1完整祖先链；辅助路径为每步局部首beam。统计最后一层六heads平均的四个SID生成步骤，对actual encoder_mask==1位置计算，每路径32×6=192条，每job384条，共3072条。cosine/Pearson使用原可见权重，常数Pearson为null；TV/JS仅在运算中按可见行和概率化，JS用自然对数；不改原矩阵/热力图，不跨Shuffle原始列做未对齐相减。',
 '','| 条件 | Top1 cosine mean / median / min | 局部首beam cosine mean / min | Top1 TV mean | Top1 JS mean(nats) |','|---|---|---|---:|---:|']
for j in stats['jobs']:
    a=j['summary']['final_top1_cross'];b=j['summary']['local_first_cross'];lines.append(f"| {label(j['job_id'])} | {a['cosine']['mean']:.6f} / {a['cosine']['median']:.6f} / {a['cosine']['min']:.6f} | {b['cosine']['mean']:.6f} / {b['cosine']['min']:.6f} | {a['total_variation']['mean']:.6f} | {a['jensen_shannon_nats']['mean']:.6f} |")
lines+=['','各组padding注意力质量均为0；raw行和误差最大值见原统计JSON（约1e−7量级）。所有min/median/mean/max、逐样本六对记录、Pearson与两路径least-three均完整保留于 [实际统计](attention-statistics/attention-descriptive-statistics.json)。least-three依既定每用户六对平均cosine升序、sample key平局排序选取，未挑图。',
 '','观察：B路线及Toys的cosine摘要较高，但B-Normal/Toys的TV均值仍约0.134/0.115，不能称四行分布完全相同。A三条件的主路径cosine均值约0.876–0.888，并存在负Pearson配对。Sports主路径单对cosine最低约0.252；用户 c6001178f9652bfb338d86de 的六对平均约0.629（两者是不同统计量），局部首beam单对最低约0.227。两种路径均不能抹去这些不相似情况。',
 '','固定32用户、原seed及各自条件下的行内描述支持有限的注意力相近/差异观察；与Shuffle汇总指标较稳且用户列表明显变化可并列讨论，尚不能证明模型忽略SID结构、数学上的置换不变性或四码位可无损并行生成。不同条件的摘要比较是描述性的，不是跨模型的因果结构效应。',
 '','## E. 作业、源码和恢复入口','',
 '18个核心原job均SUCCEEDED/exit0，本轮只读确认无存活worker/supervisor、task group为空；6份不可变部署manifest与原request身份吻合。完整job_id、commit、snapshot、资产及结果/审核关联见 [版本对应](FINAL_VERSION_MAP.json)，注意各类运行不是全用当前HEAD。',
 '','服务器原job：`/home/bwwu/grid-audits/sid-extensions-p0/jobs/<job_id>/`；新增attention WSL原件：`/home/addaswsw/recommend-support/sid-extensions-p0/jobs/new-server/<job_id>/retrieved/`。原始record/NPZ、256图全清单及有限公开代表/最弱样本图见 [产物身份](ATTENTION_ARTIFACT_INDEX.json)。大型权重/数据/全预测保留原位置，不重复训练或生成。',
 '','## F. 非作者审核及修正边界','',
 '新统计实际由 `/root/implementation_review` 使用原独立程序从原始NPZ逐条复算（1e−12是计算一致性容差，不是相似性阈值），并核对fixed32、source/export、6-head均值和原件hash；详见 [runtime](attention-results-review/attention-statistics-runtime-review.json) 与 [来源复核](attention-results-review/attention-statistics-source-provenance-review-20261017.json)。旧技术审核不是本轮新采集。最终整合报告审核与包审核将另存真实回执，不预签。',
 '','原三轮3/3及一次限定resolve=False配置例外不重置；本轮新研究修正0轮、GPU预算0、新安装0，无新增研究方案。失败原件与既有CPU回归保持历史身份。',
 '','## G. 发布与交付状态','',
 '`wbw`是唯一发布目标。本文内容提交后的实际完整SHA、远端关键文件回读以及最终ZIP大小/成员/SHA在支持目录最终发布/交付回执中记录，避免自身SHA自引用。当前发布与最终验包仍待执行，不将未执行检查写成PASS。',
 '','## H. GPT审核包','',
 'GPT_UPLOAD_REQUIRED: YES。旧阶段ZIP b3627f…保留；新最终ZIP须包含本轮实际统计、原始矩阵、真实新审核和最终GitHub发布证据。路径、大小、完整SHA以本轮真实校验后回执为准。',
 '','## I. 当前剩余必做项','',
 '正式研究结果与CPU统计已验收；本轮尚须完成最终整合非作者审核、正常发布/实际回读和最终ZIP独立校验。完成后再将当前回执更新为READY_FOR_REVIEW，网页最终二次审核仍待进行。','']
(F/'FINAL_RECEIPT.md').write_text('\n'.join(lines))
print(json.dumps({'status':'FINAL_CONTENT_ASSEMBLED_PENDING_INTEGRATION_AND_DELIVERY_REVIEW','summary':identity(F/'EXECUTION_SUMMARY.json'),'report':identity(F/'FINAL_RECEIPT.md'),'attention_jobs':8,'statistics_rows':3072},ensure_ascii=False))
