#!/usr/bin/env python3
"""Assemble original scientific results; package/publication are separate gates."""
from pathlib import Path
import csv
import datetime as dt
import hashlib
import json

BASE = Path(__file__).resolve().parent
OUT = BASE/'final-results'

def read(relative):
    return json.loads((BASE/relative).read_text())

def identity(relative):
    path = BASE/relative
    return dict(path=str(path), bytes=path.stat().st_size,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest())

def write(name, value):
    path = OUT/name
    assert not path.exists(), 'Preserve the prior result assembly: '+str(path)
    path.write_text(value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')

def main():
    primary = read('primary-result-acceptance/all-four.json')
    scene = read('final-scene/scene-20261018T043428336611Z.json')
    quant = read('reviews/fd/quantization-result-review.json')
    train = read('reviews/a2_training/final_training_review.json')
    toys = read('reviews/toys/toys-independent-review.json')
    a1 = read('reviews/sports_a1/audit_result.json')
    a2 = read('reviews/a2_outputs/A2_OUTPUTS_NONAUTHOR_REVIEW.json')
    combined = read('reviews/cross_domain/cross-domain-original-metrics-with-a2.json')
    assert primary['a2_included'] and len(primary['routes']) == 4
    assert quant['status'] == 'QUANTIZATION_AND_SID_INDEPENDENT_REVIEW_PASS'
    assert train['status'] == 'A2_TRAINING_NONAUTHOR_REVIEW_PASS'
    assert a2['status'] == 'A2_EVALUATION_ATTENTION_STATISTICS_NONAUTHOR_REVIEW_PASS'
    assert a1['status'] == 'SPORTS_A1_NONAUTHOR_REVIEW_PASS'
    assert toys['status'] == 'INDEPENDENT_TOYS_SAVED_RESULT_REVIEW_PASS'
    assert scene['activity']['active_project_workers'] == scene['activity']['active_known_supervisors'] == 0
    metrics, paired, attention = [], [], []
    for route in primary['routes']:
        for condition, result in route['conditions'].items():
            row = dict(route=route['route'], condition=condition, samples=route['samples'],
                       **result['original_metrics'], invalid_candidates=result['invalid_candidates'],
                       duplicate_candidates=result['duplicate_candidates'], job_id=result['job_id'],
                       prediction_sha256=result['raw_prediction_sha256'])
            counterpart = next(r for r in combined['rows'] if r['route'] == row['route'] and r['condition'] == condition)
            for metric in ['recall@5', 'ndcg@5', 'recall@10', 'ndcg@10']:
                assert row[metric] == counterpart[metric]
            metrics.append(row)
        for condition, result in route['paired'].items():
            paired.append(dict(route=route['route'], condition=condition, **result))
    for route, record in [('Toys-A-5300', toys['workflows']['toys5300']),
                          ('Toys-A-5000', toys['workflows']['toys5000']),
                          ('Sports-A1', a1), ('Sports-A2', a2)]:
        for condition, n in [('Normal', '0'), ('Shuffle3', '3'), ('Shuffle4', '4')]:
            if route.startswith('Toys'):
                r = record['attention'][n]; cohort = r['cohort']['sha256']; summary = r['statistics']
                npzs, pngs = len(r['npz_files']), r['png_count']; error = r['maximum_stat_absolute_error']
            elif route == 'Sports-A1':
                r = record['attention']['conditions'][n]; cohort = record['attention']['fixed32_sha256']; summary = r['summary']
                npzs, pngs = r['arrays_npz'], r['pngs']; error = r['maximum_independent_statistic_delta']
            else:
                r = record['attention'][n]; cohort = r['fixed32']['sha256']; summary = r['statistics']
                npzs, pngs = len(r['NPZ_files']), r['PNG_count']; error = r['maximum_independent_statistic_error']
            assert npzs == 4 and pngs == 32
            for path, values in summary.items():
                attention.append(dict(route=route, condition=condition, path=path, users=32, NPZ_per_condition=npzs,
                     PNG_per_condition=pngs, heads=6, queries=4, cohort_sha256=cohort, cohort_split='evaluation',
                     query_pairs=192, maximum_independent_error=error,
                     **{metric: value['mean'] for metric, value in values.items()}))
    assert len(metrics) == 12 and len(paired) == 8 and len(attention) == 24
    sources = ['primary-result-acceptance/all-four.json','primary-result-acceptance/a2-training.json',
               'primary-result-acceptance/a2-attention.json','primary-quantization-acceptance/acceptance.json',
               'reviews/fd/source-repair-admission.json','reviews/fd/quantization-result-review.json',
               'reviews/a2_training/final_training_review.json','reviews/a2_training/FINAL_TRAINING_REVIEW.md',
               'reviews/a2_outputs/A2_OUTPUTS_NONAUTHOR_REVIEW.json','reviews/a2_outputs/A2_OUTPUTS_REVIEW.md',
               'reviews/a2_outputs/A2_REVIEW_DELIVERY_VALIDATION.json',
               'reviews/toys/toys-independent-review.json','reviews/toys/TOYS_INDEPENDENT_REVIEW.md',
               'reviews/sports_a1/audit_result.json','reviews/sports_a1/REPORT.md',
               'reviews/cross_domain/cross-domain-evidence-index-with-a2.json',
               'reviews/cross_domain/cross-domain-original-metrics-with-a2.json',
               'reviews/delivery_tools/terminal-scene-cross-review.json',
               'final-scene/scene-20261018T043428336611Z.json']
    OUT.mkdir(exist_ok=True)
    report = dict(status='SCIENTIFIC_RESULT_ASSEMBLY_ALL_REQUIRED_SCOPED_REVIEWS_PASS',
                  assembled_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                  overall_delivery_final_pass=False,
                  remaining_delivery_gates=['non-author combined report and final staging review', 'fixed GitHub result publication/readback', 'one final ZIP and actual independent ZIP integrity review'],
                  A_route_original_metrics=metrics, paired_predictions=paired, attention_means=attention,
                  resources=scene['resources'], activity=scene['activity'], source_commit=scene['source_commit'],
                  source_snapshot=scene['source_snapshot'], source_snapshot_sha256=scene['source_snapshot_sha256'],
                  source_members=262, frozen_evidence_members=17,
                  original_failure_preserved=True, original_three_correction_rounds_exhausted=True,
                  new_special_formal_quantizer_attempts=1,
                  actual_non_author_reviewers=['/root/fd_repair_reviewer','/root/sports_a1_result_audit','/root/toys_result_audit'],
                  source_records=[identity(p) for p in sources],
                  new_dependencies_installed=[], global_configuration_changes=[])
    write('scientific-results.json', report)
    for name, rows in [('original-metrics.csv', metrics), ('paired-changes.csv', paired), ('attention-means.csv', attention)]:
        destination = OUT/name; assert not destination.exists()
        with destination.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(k for row in rows for k in row)))
            writer.writeheader(); writer.writerows(rows)
    metric_table = ['| 路线 | 条件 | 用户 | Recall@5 | NDCG@5 | Recall@10 | NDCG@10 | 非法/重复 |', '|---|---|---:|---:|---:|---:|---:|---:|']
    for r in metrics:
        metric_table.append('| '+r['route']+' | '+r['condition']+' | '+str(r['samples'])+' | '+
            ' | '.join(format(r[k], '.17g') for k in ['recall@5','ndcg@5','recall@10','ndcg@10'])+
            f" | {r['invalid_candidates']}/{r['duplicate_candidates']} |")
    pair_table = ['| 路线 | 与自己Normal比较 | Top1变化 | Top10顺序/集合变化 | Hit@5丢失/获得 | Hit@10丢失/获得 | rank改善/恶化/相同 |', '|---|---|---:|---:|---:|---:|---:|']
    for r in paired:
        pair_table.append(f"| {r['route']} | {r['condition']} | {r['top1_changed']} | {r['top10_order_changed']}/{r['top10_set_changed']} | {r['hit5_1to0']}/{r['hit5_0to1']} | {r['hit10_1to0']}/{r['hit10_0to1']} | {r['rank_improved_miss11']}/{r['rank_worsened_miss11']}/{r['rank_unchanged_miss11']} |")
    attention_table = ['| 路线 | 条件 | 路径 | mean Cosine | mean Pearson | mean TV | mean JS(nats) |', '|---|---|---|---:|---:|---:|---:|']
    for r in attention:
        attention_table.append('| '+r['route']+' | '+r['condition']+' | '+r['path']+' | '+
            ' | '.join(format(r[k], '.9f') for k in ['cosine','pearson','total_variation','jensen_shannon_nats'])+' |')
    jobs_table = ['| 新A2阶段（均以ga-20261018-sportsa2-开头、-fd-recovery-v1结尾） | 状态/退出码 | GPU请求秒 |', '|---|---|---:|']
    for row in scene['resources']['stage_details']:
        phase = row['job_id'][len('ga-20261018-sportsa2-'):-len('-fd-recovery-v1')]
        jobs_table.append(f"| {phase} | {row['status']}/{row['exit_code']}，cleanup verified | {row['gpu_requested_seconds']:.6f} |")
    text = '''# GRID / Semantic ID：Sports A2 FD专项恢复与A路线正式结果

全部已授权计算及各项真实非作者科研审核已闭合。本报告是科研结果固定材料；最终发布回读、轻量ZIP字节校验及交付状态由独立的交付回执继续绑定，不把程序exit0或本次汇总当作最终整包验收。

## A. 范围与原失败

仅执行一次Sports A2专用FD工程恢复及其原10个下游阶段；原T5编码直接复用。Toys A-5300、Toys A-5000、Sports A1只读审核，没有重训或重新评价；旧Beauty A/B、Toys B、Sports B仅继承固定来源。本轮没有Parallel、MLP、额外数据域、seed搜索、测试集调参、额外生成器或预算扩张。

原任务3/3集中修正耗尽历史保留。本次专项仅一次正式量化attempt；旧ga-20261017-sportsa2-quantize-v1仍INTERRUPTED/-9。24次EMFILE来自DataLoader/Tensor storage IPC路径；原-9是受控中断与清理，不是已确认GPU OOM。旧34阶段历史始终23 SUCCEEDED、1旧量化失败、10旧模板未提交；新11阶段拥有独立身份，不覆盖旧请求。

## B. 工程修复与数据等价

保全原件references/wmh-original/GRID不修改。原collate保留`list(field_value)`的独立小Tensor；新A2专用adapter先调用原collate，再仅将item_ids改为`torch.stack(result.item_ids, dim=0)`。原train/predict两个collate target被替换，其余DataModule配置精确还原对比。量化model_step仍读取同一个input_embedding；predict_step对每ID调用.item()，writer键值保持。

固定真实2048样本逐项核对ID数量/值/顺序/dtype、512维embedding及全部feature、ID配对、model_step、predict/writer及RNG状态，均完全一致；全18357 item无重复遗漏。batch内ID storage由2048变1，包含embedding的总storage由2049变2。512样本×2worker的有界IPC原FD527/527/1040/527，新16/18/16/16；2048×12worker CPU两rank顺序模拟另有原始回执，不冒充DDP实测。

仅本作业RLIMIT_NOFILE soft提高至524288且不超过hard。真实两DDP rank、每rank12训练worker、随后每rank2预测worker全部记录soft/hard524288、file_descriptor sharing strategy；rank0原soft1024、rank1继承524288。新5秒采样FD范围78–168，采样峰168，零EMFILE；这是采样上界，不是连续内核最大值。防失控门为每进程min(8192,soft×0.5)，异常仅结束本rank，由已验收runner精准清理；没有全局ulimit或sudo改动。

原FD表有一项应纠正：实际原件中rank0 worker候选soft/hard524288，rank1候选soft1024/hard524288、FD1023；历史rank归属来自argv推导，缺完整PPID/worker_id原件。storage数量级支持碎片IPC放大，但没有FD类别原件能证明它是唯一历史根因。原失败无有效训练产物，不能声称新训练结果与一个不存在的旧checkpoint逐值相等。

FD_ENGINEERING_EQUIVALENCE_PASS由非作者/root/fd_repair_reviewer在正式提交前签署；当时明确SCIENTIFIC_RESULT_NOT_YET_AVAILABLE。新A2-only controller还经另一非作者审核。旧长supervisor白名单曾在只读预提交检查拒绝c31候选，未提交未运行；85版本只为新CPU controller精确ID/module加入同期限白名单，不改旧runner或科研条件。

## C. A2量化、SID与新双表训练

实际执行source85b77f5e2a5b497fdc1f620649ba488fc783a245；不可变服务器snapshot22286966cd544773f21b96e4185d56c66afd7183fdc4e8888716d8ea1d77acb2。不是旧b443源码或6daf共享调度提交，也不是后续结果发表提交。

复用T5-small FP32 [18357,512]，embedding SHA b137cb8a006a8b950396524ed5be3bbc91d711b36068d671c01da3742d7cda6f。原ResidualQuantization+MiniBatchKMeans、3层×256、残差归一化、双4090 DDP、bf16-mixed、每卡raw2048/12worker、seed42、30 optimizer steps均保持。原初始化额外消耗raw batch，实际每rank36 raw batch、30优化步；三个量化层初始化真实完成。72份真实batch ID/embedding哈希与原T5严格对应。

quantizer.ckpt：151976576 B，SHA bdf869dcd188da6a97cc544c2e401117c4fd6b6040f1ebe748264a1ac831c309。非作者CPU严格加载原模型，三个[256,512]有限质心、global_step30。全量rank预测与合并精确覆盖18357 item；codes [18357,3]为原float32存储的精确整数0..255。新SID [4,18357] int64、588597 B，SHA 06cfa6ba0f64d6c68e0e94c4111edd51d086796e8cdb474ac3fcd424a4a2adf0。四位均合法且唯一；15647个不同三元组、1672碰撞组、最大第四位25。第四位沿实际原函数：单例0、碰撞组所有成员按item轴1..N，不是第4个可训练量化层。

量化job为ga-20261018-sportsa2-quantize-fd-recovery-v1；训练job为ga-20261018-sportsa2-train-fd-recovery-v1。新SID经metadata/data门后，从头随机初始化完整双表生成器，不加载A1/B推荐权重。实际有效配置与A1逐项核对，除7项SID/输出路径差异外协议相同：seed42，32-true，Adam lr0.001/weight_decay0.0001，无scheduler，raw16×accumulate16，1600 loader batch验证，val/recall@5最大选模、patience10，max_steps320000/max_epochs10，K10，prefix check=False。双表独立，13259136参数、178 state tensors。

训练真实7900次Adam更新、126400 microbatch、39426025窗口；79次完整validation，每次35598目标/4450 batch/目标multiset精确。best step6900，val R@5=0.031434349715709686；7000–7900十次不改善后正常patience早停，final step7900。loss有限且监测梯度有限非零。耗时32909.66135秒=9.141572597小时，低于12小时；没有将预算超时或临时checkpoint冒充完整训练。

best：checkpoint_epoch=000_step=006900.ckpt，161660326 B，SHA1287c3f60404726dbda165d5242ba17d40ecaf6520ded69d6e3165170586513d。final：recovery/training-final/full-state.ckpt，161653798 B，SHA74770b7d1166fa16df1a931b6ffa221bbf9e6f9b3eb84fa2a9d2ad8055b55676。实际评价严格加载178项且SID精确；非作者另在CPU直接严格载入best/final，128个Adam state步数6900/7900，finite/完整键值/独立表均通过。weights/RNG保存不等于逐batch精确断点恢复，未授权自动resume。

Sports A1保持final9600/best8600、正常早停和全部原结果。Toys两候选都保留，不按testing分数择优删除。

## D. 正式原指标与逐用户配对

以下为原evaluator原值；独立float64复算仅用来验收，不替换原值。全12条件完整用户/target配对，总计330060行原预测；原十候选全部保留，非法候选不额外过滤、补齐、重排或改分。

METRIC_TABLE

每个Shuffle仅与同路线自身Normal比较。原向量重排在层级offset+embedding lookup之后、每历史item occurrence内部进行；不改整数SID、时间轴、目标SID或decoder顺序。主控独立原预测复算最大差1.937127889e-9，非作者另用独立实现核完整原件。

PAIR_TABLE

rank汇总将Top10未命中编码11；独立审核逐人gzip用0表示未命中，两种约定在各自元数据中明示。完整Hit四格、rank迁移、逐人改变、目标与原预测均保留。A2 Shuffle3/4 Top1分别变化50.1348%/47.7892%，聚合值和逐人变化共同报告。

## E. 注意力与CPU统计

四路线×三条件，每条件固定32人、4×8batch、4 NPZ/32 PNG；总48 NPZ/384 PNG。每份保留最后层六heads、四autoregressive query、head mean、mask、shuffle source permutation、item/SID位置图、plot receipt、最终Top1完整祖先链及每步局部首parent路径。每一路线三条件同一预固定evaluation cohort，正式指标是完整testing；不将两个split的同用户强行视为相同history/target。

cohort SHA：Toys两候选共同3a82afccb0d944d537039ae3da51c971c1b6abba2a869052d3197b2e6bbb2ab6；Sports A1为3935dc0c085ad6134d6b5da50c387ee59e7b990e7d21226afcc3655ad211f565；Sports A2为0a2bbfbd62ab5bbd82b3032330e1f0f411af015d674fccc6b5f23c0ddacafe01。样本在输出前固定，无结果挑选。

下表每格是同条件、同路径内32人×六query pair的192项均值，不是Normal与Shuffle之间的attention距离。Cosine/Pearson用原始可见权重（含可见SEP），Pearson常量行null；TV/JS仅在计算时除可见行和，JS自然对数；原矩阵不修改。每条件384行，两路径合计全12条件4608行；原raw mask mass0、row-sum误差最大2.384185791e-7。Toys独立统计最大差5.56e-16，A1 6.67e-16，A2 7.78e-16。

ATTENTION_TABLE

A2的12 NPZ/96 PNG另由非作者从实际beam prefix独立重建父链，1152统计逐条复算；96图全部115200热力格和14400 mask格RGB与原NPZ/共同色阶相符。PNG检查没有重新渲染或修改原图。全fixed32及least-three规则均保留，未只展示有利案例；两种路径不可混为同一条最终SID轨迹。

## F. 终态、共享资源与预算

JOBS_TABLE

新controller ga-20261018-a2-fd-queue-v1为SUCCEEDED/0、cleanup；内部queue状态TERMINAL_PENDING_RESULT_REVIEW仅是计算结束时的原始标记，保留不改，科研审核由本次独立回执另签。旧ga-20261017-queue-v1 FAILED/2、ga-20261018-shared-queue-v1 SUCCEEDED/0，均未重启。45个已接收known job全部终态/cleanup、10个旧模板无job目录，活动本任务worker/supervisor均0，两GPU锁释放。

原历史计算41837.856885 GPU请求秒＋先前三个GPU工程门58.650573；本次新33992.349971；总75888.857429秒=21.080238175小时。原总143100秒=39.75小时，剩67211.142571秒=18.669761825小时。量化原失败302.049476＋新93.09782=395.147296秒，不重获7200秒；新正式预留6840秒/3420wall内完成。新训练32909.66135/43200、新eval+attention960.457251/9000、新GPU工程门29.13355/2641均合规。CPU等价与独立审核不请求GPU。保守配置秒数向下取整比精确余量少0.349427秒，实际预算表保留精确起止独立重算。

原绝对截止2026-10-19 13:40:30.012007北京时间未重置；本链2026-10-18 12:32:11北京时间结束。共享准入按free显存+峰值+余量，其他用户任务允许共存，没有抢占或杀他人进程，GPU/batch/precision/steps不因资源改变。GPU UUID dfff0e95-1db6-e2f4-3431-55bb852fa0ae 与59cac3bd-3645-668b-79c8-839e0d878893均真实核对。最终/home剩617717571584 B（575.29 GiB），/data剩4320230797312 B（4023.53 GiB）。

## G. 源码与实际运行身份

原科研b44399926c98369191c2ff5c2425377c8c07d3c8对应b6203414ee58a400快照；共享调度6dafbc7215b58a2dbb47905a687c930caecf7b44对应cdfd283823bd1928。上次交付9ab3d0ea5d32db00e278f74c599a1943885453db保持历史。新修复0d18820025e7a28675ed7112be0b1b4ff0110025；正式执行85b77f5e2a5b497fdc1f620649ba488fc783a245对应22286966cd544773完整快照；接续请求/已完成旧结果审核发表acb2f395c62af074a9649e99af17a99aa60dd83e。最终科研结果发表与ZIP回执提交将另外给完整SHA及实际回读证据，不能把元数据发表提交冒充执行版本。

真实执行根/home/bwwu/grid-audits/sid-extensions-p0/grid-a-launch-20261017/snapshots/22286966cd544773；解释器/data/bwwu/envs/grid-20261009-e61baf3/bin/python；非.git源码export按262文件与Git字节映射，并非声称服务器checkout。部署manifest SHA05defd537c43f410493ea3e1d312d1daae8d1d7c9317353e09036c6ff8b439b3；有效queue配置SHAddc231d33f739ec8f5282c60ad2a670023d057f8609bcca1d9526129fdfb711e。17旧冻结文件及262新源码最终实际重新核对无变化。独立runner、argv/cwd/请求/cleanup/UUID/源资产哈希均有原件。

## H. 真实非作者审核与依据

/root/fd_repair_reviewer直接读旧失败、collate/数据流、diff、CPU原始等价输出、真实rank/worker/FD、正式quantizer CPU载荷/SID、源码与预算、请求和终态；/root/sports_a1_result_audit审核原A1，并直接CPU严格核新A2 best/final训练载荷、7900步/79完整验证及选模；/root/toys_result_audit独立审两Toys全原结果及新A2完整预测、实际Shuffle、全部fixed32/两条beam路径/统计/PNG。不是作者自评或只复述主控表。主控也先独立验收原预测、量化数值、训练事件和注意力原数组。审核源码、原始回执和逐文件SHA均随包。

必要非科研采集/包装工具排错记录如实保留：通用quantizer fetch已完成真实传输后因本地自动child请求镜像缺失而失败，后逐成员验证并以实际admitted request补齐镜像，无研究重跑；初始静态stage因负向fixture目录遗漏被非作者拒绝，保留旧stage后新v2排除173个合成fixture成员并重新核全部真实输入。没有把假fixture状态作为实际job。此类本地取证/包装问题不改原科研3/3历史或专项正式attempt计数。

## I. 交付与可复核入口

GPT_UPLOAD_REQUIRED: YES。唯一最终轻量ZIP包含原失败材料、新修复/等价性、真实FD、quantizer/SID小数值、7900步/79验证、全12组未修改原预测、48 NPZ/384 PNG、两路径统计、逐人配对、真实各项非作者审核、终态/预算/版本映射。大型模型checkpoint、T5权重、TFRecord与完整环境留服务器；assets-not-copied.json提供成员身份、真实路径/大小/SHA与只读恢复入口，目录不伪造单一hash。

ZIP每payload成员有大小/SHA/CRC32、安全路径/重复/敏感信息扫描；manifest不自哈希，整体ZIP SHA置包外，实际序列化后再做主控与真实非作者完整ZIP验收。WSL/Windows UNC、字节、成员数与整包SHA以实际最终交付回执为准。本报告不编造尚未产生的整包SHA或Git提交。

scientific-results.json、original-metrics.csv、paired-changes.csv、attention-means.csv是本报告逐值来源；cross-domain-original-metrics-with-a2.*与cross-domain-evidence-index-with-a2.json给8路线24条件新完整入口。旧pending index保留其历史时间点，明确不用作最终A2覆盖。旧Beauty/B固定来源仅继承，没有重复全套大档案审计。

## J. 研究解释边界

实现正确、运行完成、原件完整、现象是否复现和研究推论分别判定。不同或不利结果保留，不以指标接近证明实现正确。Toys历史producer仍非唯一：Qwen/3584配置与保全512维producer之间历史绑定缺口不补造；本A2是新T5/R-Kmeans条件下方法复现，不是找回学长原Sports权重。学长当前方案与历史实际源码绑定是不同事实。

单seed42、固定32人样本和行间描述统计不能证明数学严格置换不变性、跨seed稳定性或Decoder不使用目标prefix。Shuffle已有大量逐人推荐变化；Sports A1与A2的SID及独立scratch训练均不同，不能把差异单独归于一种SID。旧FD缺完整归属/类别直接证据，采样峰不是连续峰；完整环境目录没有虚构统一SHA；精确逐batchresume未验证。

## K. 停止与依赖变更

计算已终态，停止新增实验；最终真实交叉审核、固定wbw发表和一次ZIP验收完成后停止本任务。未同步main/yinjingchen、不新分支、不清理正式证据、不进入Parallel/机制探索。新增系统包、语言包、工具链均0；未改.bashrc、全局环境/资源配置，无安装项需要清理。本轮专用源码/请求/参数文件和证据按用户授权保留。
'''
    text = text.replace('METRIC_TABLE', '\n'.join(metric_table)).replace('PAIR_TABLE', '\n'.join(pair_table)).replace('ATTENTION_TABLE', '\n'.join(attention_table)).replace('JOBS_TABLE', '\n'.join(jobs_table))
    # Do not carry a hand-typed sample total into the fixed report.
    total = sum(r['samples'] for r in metrics)
    assert total == 330060
    write('RESULTS.md', text)
    print(json.dumps(dict(status=report['status'], output=str(OUT), metrics=len(metrics), paired=len(paired),
                         attention_rows=len(attention), original_prediction_rows=total,
                         total_GPU_hours=scene['resources']['total_gpu_requested_seconds']/3600,
                         final_delivery_not_yet_approved=True), ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
