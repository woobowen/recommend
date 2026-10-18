"""Produce the actual output review and append A2 into a new evidence index."""
from pathlib import Path
import copy,csv,datetime,hashlib,json,shutil
from numeric_core import ident,load
ROOT=Path(__file__).resolve().parent
CROSS=ROOT.parent/'cross_domain'
def write(p,x):
    assert not p.exists(),p
    p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def main():
    d=load(ROOT/'A2_OUTPUTS_NONAUTHOR_REVIEW.json');bindings=load(ROOT/'a2-final-output-binding-checks.json');pixels=load(ROOT/'a2-png-pixel-consistency.json')
    assert d['status']=='A2_EVALUATION_ATTENTION_STATISTICS_NONAUTHOR_REVIEW_PASS'
    assert bindings['status']=='A2_OUTPUT_SOURCE_BINDING_AND_SELECTOR_PASS' and pixels['total_images']==96
    raw=ROOT/'terminal_outputs/raw/jobs'
    train=load(raw/'ga-20261018-sportsa2-train-fd-recovery-v1/output/RESULT.json')
    name={'0':'Normal','3':'Shuffle3','4':'Shuffle4'}
    lines=['# Sports A2 正式评价、注意力与CPU统计真实非作者审核','',
      '`A2_EVALUATION_ATTENTION_STATISTICS_NONAUTHOR_REVIEW_PASS`','',
      '实际审核代理：`/root/toys_result_audit`。未编写A2量化修复、推荐模型、训练、评价、注意力或统计生产实现。本轮直接审阅实际终态原件，并执行已准备的独立CPU脚本。此PASS覆盖A2正式输出与其身份依赖；完整量化/训练验收、总预算、最终GitHub/ZIP仍由主控和相应真实审核者另行收口，不将本报告冒充全任务FINAL_INTERNAL_REVIEW_PASS。','',
      '## 固定身份与实际执行','',
      '- 科研/工程部署source commit：`85b77f5e2a5b497fdc1f620649ba488fc783a245`。snapshot：`22286966cd544773f21b96e4185d56c66afd7183fdc4e8888716d8ea1d77acb2`，实际根`/home/bwwu/grid-audits/sid-extensions-p0/grid-a-launch-20261017/snapshots/22286966cd544773`。20份实际冻结源码与manifest逐字节匹配；没有把旧b443科研原件、共享调度6daf或新发布元数据提交混成一个版本。',
      f'- 新T5/R-Kmeans SID：`{d["new_SID_sha256"]}`，原路径`/home/bwwu/grid-audits/sid-extensions-p0/jobs/ga-20261018-sportsa2-quantize-fd-recovery-v1/output/semantic_ids.pt`，588597字节，`[4,18357] int64`。本审核用限制型数值反序列化器独立读取小SID载荷，没有torch/GPU或模型调用；完整SID唯一且全量目标均合法。',
      f'- 全部评价/注意力唯一best：step{d["best_step"]}，SHA`{d["best_checkpoint"]["sha256"]}`，{d["best_checkpoint"]["bytes"]}字节。原路径`{d["best_checkpoint"]["path"]}`。每实际job严格加载178 state tensors，missing/unexpected均空、所有keys/shapes/values完全相等、embedded SID exact，完整双表不共享，参数13259136。没有下载或重新加载大checkpoint；独立训练审核负责直接载荷验收。',
      f'- 原训练依赖记录显示完整终态、final step{train["global_step"]}、best step{d["best_step"]}、{train["completed_full_validations"]}次完整验证，终止原因为`{train["termination"]}`。本报告只核准入依赖与一致性，不代签训练细节审核。',
      '- 三组评价与三组注意力、stats的新job均使用`ga-20261018-sportsa2-*-fd-recovery-v1`独立身份。7份实际request与accepted指纹精确一致；226个runner status artifact hash和24份统计input file hash相符；请求CWD、实际stage argv、capture、SID/best/配置来源均一一对应。原失败job未修改。','',
      '## 完整正式评价','',
      '| 条件 | testing用户 | Recall@5 | NDCG@5 | Recall@10 | NDCG@10 | 非法候选 | 重复候选 |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for n,e in d['evaluations'].items():
        m=e['metrics_original'];lines.append(f'| {name[n]} | {e["samples"]} | {m["recall@5"]:.17g} | {m["ndcg@5"]:.17g} | {m["recall@10"]:.17g} | {m["ndcg@10"]:.17g} | {e["invalid_candidates"]} | {e["duplicate_candidates"]} |')
    err=max(x['maximum_metric_error'] for x in d['evaluations'].values())
    lines+=['',f'表内是原evaluator正式原值，不用独立float64值替换。三条件各35598个唯一用户，sample_key/target multiset与新SID条件下的实际testing索引精确相等，共106794行预测，无遗漏或重复用户。逐用户指标独立复算，汇总对原FP32值最大差`{err:.4g}`。Normal/Shuffle3分数无并列；Shuffle4有1行非target并列，不影响命中/目标rank。非法SID保留在原十候选，不过滤、补齐、改分或重排后再评价。',
      '','每条件4450个真实CUDA test batches，raw batch8、workers8、120 SID输入位、32-true、seed42、K10、prefix check=False。三个有效配置除预定Shuffle flags和输出目录外完全相同。原loss/generate encoder各4450次：Normal的Shuffle计数0，Shuffle3/4各8900次；保存的实际排列为原rand→argsort方法，padding保持原槽位，Shuffle3第四位不变。源码确认先层级offset与embedding lookup，再在每历史occurrence内部重排完整向量，不重排整数SID或物品时间轴，不改target/decoder顺序。',
      '','## 逐用户严格配对','',
      '| 与本模型Normal比较 | Top1变化 | Top10顺序变化 | Top10集合变化 | Hit@5丢失/获得 | Hit@10丢失/获得 | 平均Top10重叠 |','|---|---:|---:|---:|---:|---:|---:|']
    for n,p in d['paired_predictions'].items():
        h=p['hit_migration'];lines.append(f'| Shuffle{n} | {p["top1_changed"]} ({p["top1_changed"]/35598*100:.4f}%) | {p["top10_order_changed"]} | {p["top10_set_changed"]} | {h["5"]["1->0"]}/{h["5"]["0->1"]} | {h["10"]["1->0"]}/{h["10"]["0->1"]} | {p["mean_top10_overlap"]:.9f} |')
    lines+=['','每组配对35598人，原target完全一致。`a2-normal-shuffle3-paired.jsonl.gz`和`a2-normal-shuffle4-paired.jsonl.gz`保留逐人变化与排名；本文件rank=0表示Top10未命中，完整rank/Hit转移计数保留于JSON。没有按照test结果挑模型、seed或样本。','',
      '## fixed32 注意力与描述统计','',
      'cohort SHA为`0a2bbfbd62ab5bbd82b3032330e1f0f411af015d674fccc6b5f23c0ddacafe01`。明确来自**evaluation/validation split**，不是testing；按实际文件/行顺序在看模型输出前固定首32 distinct users，三条件同一cohort。相同user key在两个split不代表相同history/target，未把fixed32 Top1强行等同testing Top1。','',
      '每条件4批×8人、4份NPZ＋4份record JSON＋32 PNG＋4份plot receipt；三条件合计12 NPZ/96 PNG。raw最后层6heads、4个autoregressive query、150列encoder全部保留。独立从原beam唯一prefix重建所有parent_indices，并从最终child0逆向追踪完整Top1祖先链；local-first每步parent0单独核验，两个路径的head张量、head mean、最终SID和scores逐项精确一致。Normal/Shuffle3/Shuffle4分别26/27/25个用户的Top1祖链使用过非零parent，未把局部首beam误作最终Top1。','',
      '同cohort的实际input SID、mask、item轴和120个SID位置、150个encoder位置、lookup offset、可见/隐藏组及实际shuffle source permutation全部核对。每批observer OFF/ON的all-K SID/scores完全一致、CPU/CUDA RNG结束状态相同、参数hash未变、hooks/方法正常及异常恢复均通过。四批参数hash在各条件内恒定。','',
      '96张PNG全部解码并核SHA/大小/来源/2880×1280尺寸/原始共同色阶；另外从实际黑色spine定位双panel，对全部**115200个热力格、14400个mask格**的cell-center RGB与raw NPZ/原YlOrRd LUT独立比较，全部通过（量化差不超过1/255）。未重新渲染或编辑原图，也未作逐行伸缩。对三条件固定第0批第0样本额外实际看图，四行、全150列与mask显示一致。','',
      '| 条件 | 路径 | mean Cosine | mean Pearson | mean TV | mean JS(nats) |','|---|---|---:|---:|---:|---:|']
    for n,a in d['attention'].items():
        for path,s in a['statistics'].items():lines.append(f'| {name[n]} | {path} | {s["cosine"]["mean"]:.9f} | {s["pearson"]["mean"]:.9f} | {s["total_variation"]["mean"]:.9f} | {s["jensen_shannon_nats"]["mean"]:.9f} |')
    maxstat=max(x['maximum_independent_statistic_error'] for x in d['attention'].values())
    lines+=['',f'每条件32人×6个query pair×2路径=384行，共1152行独立复算；逐条及min/median/mean/max均匹配原统计，最大差`{maxstat:.4g}`。全部masked mass=0，raw单head行和最大误差2.384185791e-7。Cosine/Pearson使用可见原值，TV/JS只在概率计算时除以可见行和，JS为自然对数，包含可见SEP，不改原注意力矩阵。每路径least-three按原声明的六对平均Cosine升序/sample_key平局规则独立复算，六个选择均相符；全32样本和全pairs保留，不只保留最漂亮或最有利案例。','',
      '## 作业终态、原件与限制','',
      '- 三eval、三attention及stats原状态均SUCCEEDED/exit0/cleanup verified、worker_may_be_live=false、最后owned-group观察为空。它们的输出内容另已验收，没有把exit0当科研证明。当前全服务器进程、控制器总终态、预算与期限由主控实际场景核验另行负责。','- 本轮终态定向收集270成员、95430018原始字节，压缩34834571字节；整体SHA`646784982c92ca2c35a0006662dbee5f7c859be34eb216adcf794b93f4e8b844`，全部成员SHA/大小与安全路径通过。不是整job export，没有复制模型checkpoint或TFRecord。先前preparation PENDING/失败0字节体积门回执保持原样。','- 本审核没有训练、推荐generation、原evaluator调用、GPU模型调用、远端写入、Git写入或新依赖；实际CPU统计来自保存原件。正式输出审核首次即通过，无研究修正或重跑。','- 本审核不代替量化和完整训练的真实非作者审核；checkpoint严格加载从实际每job原记录和固定源码核对，未在本子审核重新加载大checkpoint。','- A2属于T5/R-Kmeans SID下的新双表训练方法复现，不是找回学长原始Sports权重。A1与A2有上游SID与独立scratch训练条件差异，不能将差异归于单一因素。Toys producer非唯一披露仍在跨域交付中保留。','- 单seed42、fixed32 evaluation cohort、各条件内部行间描述统计，不构成数学严格置换不变性、跨seed稳定性或decoder忽略目标prefix的证明。Shuffle导致大量逐用户推荐变化和实际指标下降，均按原结果保留。','',
      '## 审核入口','',
      '- `A2_OUTPUTS_NONAUTHOR_REVIEW.json`：全量指标、逐用户配对、每条attention统计及身份。','- `a2-final-output-binding-checks.json`：request接受指纹、artifact/统计输入hash、least-three与scores检查。','- `a2-png-pixel-consistency.json`：全部PNG实际格点颜色与mask逐图验收。','- `terminal_outputs/raw/REMOTE_MANIFEST.json`与`collection-receipt.json`：原远端路径/逐成员身份与传输验收；原模型留服务器。','- `preparation-verification.json`和`preparation-v2/raw/`：实际新SID、metadata、cohort、20份固定源码及执行配置。','- `audit_terminal_outputs.py`、`numeric_core.py`、`final_binding_checks.py`、`verify_plot_pixels.py`：本次实际执行的可复核CPU验收程序。','',
      'GPT_UPLOAD_REQUIRED: YES（由主控并入唯一最终审核ZIP，本子任务不另要求用户上传散件）。','']
    md=ROOT/'A2_OUTPUTS_REVIEW.md';assert not md.exists();md.write_text('\n'.join(lines))
    # Produce new final-value files; the historical A2-PENDING index is untouched.
    index=copy.deepcopy(load(CROSS/'cross-domain-evidence-index.json'))
    current={}
    for file in ['A2_OUTPUTS_NONAUTHOR_REVIEW.json','A2_OUTPUTS_REVIEW.md','a2-final-output-binding-checks.json','a2-png-pixel-consistency.json']:
        src=ROOT/file;dest=CROSS/'current_review_sources/a2_outputs'/file;dest.parent.mkdir(parents=True,exist_ok=True);assert not dest.exists();shutil.copyfile(src,dest)
        current[file]=ident(src)|{'included_copy':'current_review_sources/a2_outputs/'+file,'review_inheritance':'actual A2 nonauthor output review after terminal artifacts'}
    a2=next(r for r in index['routes'] if r['route_id']=='Sports-A2');a2['review_mode']='REVIEWED_DURING_THIS_RECOVERY_TASK'
    a2['reviewer']=d['reviewer'];a2['review_status']=d['status'];a2['review_source']=current['A2_OUTPUTS_NONAUTHOR_REVIEW.json']
    a2['human_report']=current['A2_OUTPUTS_REVIEW.md'];a2['PNG_pixel_review']=current['a2-png-pixel-consistency.json']
    a2['output_binding_review']=current['a2-final-output-binding-checks.json'];a2['paired_analysis']=d['paired_predictions']
    a2['separate_quantization_and_training_nonauthor_review_required_for_overall_final_pass']=True
    a2['conditions']=[]
    for n,e in d['evaluations'].items():
        a=d['attention'][n];a2['conditions'].append({'condition':name[n],'shuffle_n':int(n),'result_status':'COMPLETE',
            'review_status':d['status'],'review_inheritance':'REVIEWED_DURING_THIS_RECOVERY_TASK','samples':e['samples'],'unique_users':e['samples'],
            'metrics_original':e['metrics_original'],'invalid_candidate_count':e['invalid_candidates'],'duplicate_candidate_count':e['duplicate_candidates'],
            'candidate_treatment':e['candidate_treatment'],'prediction_sha256':e['prediction']['sha256'],'checkpoint':e['identity']['checkpoint'],
            'sid':e['identity']['sid'],'config':e['identity']['config'],'job_id':e['job_id'],
            'scientific_evaluation_execution':{'commit':d['scientific_capture'].split('/')[0],'snapshot_sha256':d['scientific_capture'].split('/')[1]},
            'evaluation_review_source':current['A2_OUTPUTS_NONAUTHOR_REVIEW.json'],
            'attention':{'review_status':a['status'],'fixed_users':32,'batches':4,'raw_NPZ':4,'PNG':32,'cohort_split':'evaluation',
                         'cohort':a['fixed32'],'summary':a['statistics'],'PNG_cells_verified':38400,'mask_cells_verified':4800},
            'result_metrics_recomputed_during_this_index':False})
    index['status']='CROSS_DOMAIN_ORIGINAL_OUTPUT_EVIDENCE_COMPLETE';index['available_condition_count']=24;index['pending_condition_count']=0
    index['overall_final_review_pass']=False
    index['A2_completion_addendum']={'actual_output_acceptance':d['status'],'source_index_with_pending_preserved':ident(CROSS/'cross-domain-evidence-index.json'),
        'note':'A2 output evidence is now filled from real accepted terminal artifacts. Whole-task quantization/training/resource/GitHub/ZIP acceptance remains separately owned by the main controller.'}
    index['current_A2_review_sources']=current
    index['historical_scope_context']['interpretation']='Old PARTIAL and the initial A2-PENDING index remain untouched historical evidence. Current Toys/A1/A2 accepted original outputs are separately linked in this new derived index.'
    a2['limitations']=['This output-only index does not itself sign quantization, full training, total budgets, publication, ZIP or overall final acceptance.',
                      'A1/A2 differ in upstream SID and separate scratch training; no single-factor causal attribution.']
    assert sum(len(r['conditions']) for r in index['routes'])==24 and all(c['metrics_original'] is not None for r in index['routes'] for c in r['conditions'])
    write(CROSS/'cross-domain-evidence-index-with-a2.json',index)
    rows=[]
    for r in index['routes']:
        for c in r['conditions']:
            rows.append({'route':r['route_id'],'domain':r['domain'],'family':r['family'],'condition':c['condition'],'samples':c['samples'],
                'review_mode':r['review_mode'],'result_status':c['result_status'],**{k:c['metrics_original'][k] for k in ['recall@5','ndcg@5','recall@10','ndcg@10']},
                'invalid_candidates':c['invalid_candidate_count'],'duplicate_candidates':c['duplicate_candidate_count'],'job_id':c['job_id']})
    write(CROSS/'cross-domain-original-metrics-with-a2.json',{'status':'24_ORIGINAL_CONDITIONS_INDEXED','rows':rows,
        'old_fixed_commit':index['old_evidence_fixed_commit'],'A2_output_review':current['A2_OUTPUTS_NONAUTHOR_REVIEW.json'],
        'metric_semantics':'Original accepted evaluator proportion values; inherited original metrics never recomputed in this index builder.',
        'overall_final_internal_review_pass':False})
    with (CROSS/'cross-domain-original-metrics-with-a2.csv').open('x',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    table=['# 四条A路线及固定跨域对照原指标','',
       '真实A2终态输出已补齐；历史PENDING索引保持原样。Beauty A/B、Toys/Sports B固定继承9ab3已审证据，不重新审计或复算。Toys A双候选、Sports A1、Sports A2使用本轮真实非作者输出审核。此表不单独签整个任务最终PASS。','',
       '| 路线 | 条件 | testing用户 | Recall@5 | NDCG@5 | Recall@10 | NDCG@10 | 非法/重复 |','|---|---|---:|---:|---:|---:|---:|---:|']
    for route in ['Toys-A-5300','Toys-A-5000','Sports-A1','Sports-A2','Beauty-A','Beauty-B','Toys-B','Sports-B']:
        for row in [x for x in rows if x['route']==route]:
            table.append(f'| {route} | {row["condition"]} | {row["samples"]} | {row["recall@5"]:.15f} | {row["ndcg@5"]:.15f} | {row["recall@10"]:.15f} | {row["ndcg@10"]:.15f} | {row["invalid_candidates"]}/{row["duplicate_candidates"]} |')
    table+=['','各Shuffle只与其具体模型自己的Normal比较。A/B、A1/A2、不同域不是严格单因素因果对照；Toys历史producer非唯一和新Sports方法复现身份继续保留。fixed32来自预先固定evaluation cohort，不能与testing历史/目标混用。描述统计不证明数学严格不变性或Decoder忽略prefix。',
      '','完整SHA/固定GitHub路径/原审核结论和所有实际资产/预测身份见`cross-domain-evidence-index-with-a2.json`；本表不替换原始精度JSON，也不变更原候选处理。','']
    p=CROSS/'ORIGINAL_METRICS_WITH_A2.md';assert not p.exists();p.write_text('\n'.join(table))
    outputs=[ROOT/'A2_OUTPUTS_REVIEW.md',ROOT/'A2_OUTPUTS_NONAUTHOR_REVIEW.json',ROOT/'a2-final-output-binding-checks.json',ROOT/'a2-png-pixel-consistency.json',
             CROSS/'cross-domain-evidence-index-with-a2.json',CROSS/'cross-domain-original-metrics-with-a2.json',CROSS/'cross-domain-original-metrics-with-a2.csv',CROSS/'ORIGINAL_METRICS_WITH_A2.md']
    receipt={'status':'A2_OUTPUT_REVIEW_AND_CROSS_DOMAIN24_DELIVERY_READY','reviewer':'/root/toys_result_audit',
             'scientific_output_review':d['status'],'original_condition_rows':24,'old_pending_index_modified':False,
             'new_GPU_model_or_evaluator_calls':0,'Git_or_remote_mutations':0,'outputs':[ident(p) for p in outputs],
             'separate_training_quantization_and_overall_final_acceptance_not_signed_by_this_receipt':True,
             'visual_spotchecks':'Actual first sample of batch00 in all three A2 conditions inspected; all 96 figures also received complete numeric cell checks.'}
    write(ROOT/'final-review-products.json',receipt)
    print(json.dumps({'status':receipt['status'],'outputs':receipt['outputs']}))
if __name__=='__main__':main()
