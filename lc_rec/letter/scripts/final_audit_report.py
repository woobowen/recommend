#!/usr/bin/env python3
"""Render final audit tables from preserved CPU/GPU receipts, without model work."""
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]
E=ROOT/'evidence/final_audit_20261113'
JOB='letter-a0-20261112-final-audit-20261113-offline-r1'
R=E/'jobs'/JOB/'results'


def read(p):
    return json.loads(p.read_bytes())


def table(head,rows):
    return '\n'.join(['| '+' | '.join(head)+' |','| '+' | '.join(['---']*len(head))+' |']+
                     ['| '+' | '.join(map(str,row))+' |' for row in rows])+'\n'


def main():
    m=read(R/'primary/metrics.json');orig=read(E/'metrics.json');g=read(R/'primary/groups.json')
    ci=read(R/'primary/bootstrap.json');cost=read(E/'original_cost.json');budget=read(E/'cpu_budget.json')
    assert m['status']==read(R/'independent_extended.json')['status']==read(R/'independent_statistics.json')['status']==read(R/'crosscheck.json')['status']=='PASS'
    paper={'hit@5':.0913,'hit@10':.1115,'ndcg@5':.0789,'ndcg@10':.0854}
    comparisons={k:{'formal':orig['metrics'][k],'paper':v,'absolute_difference':orig['metrics'][k]-v,'relative_difference':orig['metrics'][k]/v-1} for k,v in paper.items()}
    summary={'status':'PASS','original_run_commit':m['run_commit'],'evaluation_run_commit':m['evaluation_commit'],
             'original_gpu_job':'letter-a0-20261112-normal-det-seed42-r0','cpu_job':JOB,'users':m['users'],
             'paper_comparison':comparisons,'original_formal_metrics':orig['metrics'],'derived_metrics':m['metrics'],
             'cce_absolute_change':{f'{name}@{k}':m['metrics'][f'{name}@{k}']-m['metrics'][f'{base}@{k}'] for name,base in [('ItemHit','hit'),('ItemNDCG','ndcg')] for k in (5,10,20)},
             'substatuses':{k:'PASS' for k in ['A0_FORMAL_TRAINING','A0_ORIGINAL_FULL_TEST','ORIGINAL_METRICS_INDEPENDENT_AUDIT','EXTENDED_METRICS','CCE_EVALUATION','FINAL_NONAUTHOR_REVIEW','A0_FINAL_AUDIT']},
             'limits':['Single training seed; user bootstrap is not training-seed variability.','Published SID downstream reproduction, not upstream SID training.','Approved FA2 deterministic adaptation changes backward numerical execution; no proven bitwise equivalence with author historical CUDA/four-GPU execution.','No token re-decoding, new generation, model reload or checkpoint tensor re-audit in this CPU task.','Supplementary groups and CCE have no direct LETTER paper baseline.','Overall bootstrap intervals do not apply to subgroups; UNSEEN has only35 users.'],
             'correction_rounds':read(E/'corrections.json')['rounds_used'],'new_gpu_device_seconds':0}
    (E/'final_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    body='''# A0 Instruments Normal：最终科研审核与离线扩展评价

**A0 FINAL AUDIT: PASS**（2026-11-13）。正式四轮训练及完整测试已结束，原四指标、扩展指标、CCE和用户统计均完成真实独立审核。科研结论限于本次 seed42、固定模型与公开SID下游复现；不是A1/Shuffle授权。

| 分项 | 状态 |
| --- | --- |
'''
    body+='\n'.join('| '+k+' | '+v+' |' for k,v in summary['substatuses'].items())+'\n\n'
    body+=f'''## 固定身份与执行证据

- 原GPU RUN：`{m['run_commit']}`；原job：`letter-a0-20261112-normal-det-seed42-r0`；4/4 Epoch、8240/8240 updates、四次全量Validation；best=`checkpoint-8240`，eval_loss=1.5409067869186401；原runner SUCCEEDED/exit0/cleanup verified，单attempt，无训练/测试重跑。
- CPU评价RUN：`{m['evaluation_commit']}`；job：`{JOB}`；[真实request/status/log/pipeline与结果](evidence/final_audit_20261113/jobs/{JOB}/)。服务器固定CPU export `/home/bwwu/lc-rec/letter-a0-final-audit-20261113/releases/{m['evaluation_commit']}`，[七文件逐字节映射](evidence/final_audit_20261113/deployment-{m['evaluation_commit']}.json)，目录没有.git。
- 模型：huggyllama/llama-7b revision `4782ad278652c7c71b72204d462d6d01eaaf7549`；[模型既有身份](evidence/formal_det_20261112/model_identity.json)。原adapter SHA `f4651c5c6e6ed841cd3384c0c8ebfab30880e94597e515ffa21916b876001ed5`，位于 `/data/bwwu/lc-rec/a0-normal-20261111/formal_seed42/training/best_adapter`。
- 2×RTX4090 / Int8主干 BF16 / LoRA r8 alpha16 dropout.05 / 七投影模块+embed/head / AdamW 2e-4 wd.01 cosine warmup.01 / micro8×GAS4×2=64 / ZeRO2 / GC / response-only CE temperature.8。两rank真实TA/DS/optimizer 450 trainable tensors/289652736 elements及224 A-only模块已[独立核验](review/final_audit_20261113/terminal_effective_check.json)。save_total_limit作者1/实际存储2，保留best及最新完整恢复点，不改选模。
- FA2 2.7.4.post1 deterministic backward正式适配，两rank各1,054,720次native_varlen与CUDA extension deterministic=True；未扩大torch/cuBLAS deterministic。原源码/生成/配置/作者文件与GPU RUN完全一致。compat元数据有旧“probe-only”文字，真实正式审批/源码/调用已核，保留原文字不改写历史。
- 原预测仍在 `/data/bwwu/lc-rec/a0-normal-20261111/letter-a0-20261112-normal-det-seed42-r0/test/`。rank0 190712530 bytes，SHA `bdcbdde15d3ac6850708888e055aa67ae61212dbb675d9f915b9947b2a208bde`；rank1 190154918 bytes，SHA `f5d2ce538bfe86b024c38070f74cce3411a457554ae08ec610ec26987570c629`。本轮StageA完整流式SHA一次；主实现及另一非作者实现各一次必要评价读取，不重复大型哈希。
- SID `33c87e28c52ae367985bdf494cb49d676d181e13a39497b4ad035f46111546dd`；inter `0a11d47791d90483005da93c36d826679e20e4592d1d2b86518d5f56f218e743`；item `1d9938e576a8d964e6b977a2b63aa56402840a788eaa6685f26f0d910a7133f2`。

## 第一层：原论文指标及独立门槛

[原始metrics.json只读副本](evidence/final_audit_20261113/metrics.json)，[完整独立复算回执](evidence/final_audit_20261113/independent_original.json)。24772用户轴逐行对应，零失败/跳过，每人20候选，原relevance/target/原始解析/稳定排序一致。HR完全相同；NDCG均值差@5=4.3021142204224816e-16、@10=3.469446951953614e-16，源于求和次序，沿用原1e-13均值/1e-9总和容差，**不声称浮点逐位一致**；未改变原metrics。

[LETTER v2 Table1](https://arxiv.org/html/2405.07314v2)的Instruments参照：

'''
    body+=table(['指标','原正式结果','论文','绝对差','相对差'],[(k,f"{v['formal']:.12f}",v['paper'],f"{v['absolute_difference']:+.12f}",f"{100*v['relative_difference']:+.4f}%") for k,v in comparisons.items()])
    body+='\n接近论文不单独证明实现正确；单seed不能代表多seed稳定性。本轮不通过测试选模或追数值。\n\n## 第二层：扩展排名与生成诊断\n\n原SID匹配口径，HR=Recall；NDCG保留所有匹配折扣相加，MRR只取首次命中。新增指标 `NO_DIRECT_PAPER_REFERENCE`。\n\n'
    body+=table(['K','HR/Recall','NDCG','MRR'],[(k,f"{m['metrics'][f'hit@{k}']:.12f}",f"{m['metrics'][f'ndcg@{k}']:.12f}" if k>1 else '—',f"{m['metrics'][f'mrr@{k}']:.12f}" if k>1 else '—') for k in (1,5,10,20)])
    body+='\n'+table(['K','有效/候选','有效率','非法数/含非法用户','重复数/用户','碰撞候选/用户','有效SID Coverage'],[(k,f"{v['valid']}/{v['candidates']}",f"{v['valid_rate']:.9%}",f"{v['invalid']}/{v['users_invalid']}",f"{v['duplicates']}/{v['users_duplicate']}",f"{v['colliding']}/{v['users_colliding']}",f"{v['unique_valid_sids']}/9897={v['unique_valid_sid_coverage']:.9%}") for k,v in m['sid_statistics'].items()])
    body+='\n所有候选保留，未过滤、去重或改顺序。各rate及其分母均在[完整指标JSON](evidence/final_audit_20261113/jobs/'+JOB+'/results/primary/metrics.json)。Coverage是SID覆盖，不是真实物品覆盖。目录9922 items/9897 SIDs，23碰撞组（21组×2、2组×3），48items，最大组3；122名用户的目标在碰撞组。\n\n首次目标排名1–20人数：`'+', '.join(str(m['first_target_rank_users'][str(r)]) for r in range(1,21))+'`；miss='+str(m['first_target_rank_users']['miss'])+'，总和24772。\n\n## 第三层：独立CCE\n\n'
    body+=table(['K','ItemHit','相对SID-HR绝对变化','ItemNDCG','相对SID-NDCG绝对变化'],[(k,f"{m['metrics'][f'ItemHit@{k}']:.12f}",f"{summary['cce_absolute_change'][f'ItemHit@{k}']:+.12f}",f"{m['metrics'][f'ItemNDCG@{k}']:.12f}",f"{summary['cce_absolute_change'][f'ItemNDCG@{k}']:+.12f}") for k in (5,10,20)])
    body+='''
定义依据：[CCE v2 §3.1, Eq4–7/Fig2](https://arxiv.org/html/2605.25330v2)，[官方固定源码6dd5d314f62d3379e664a52c9007e7645556970c](https://github.com/Nishikata97/CollisionGenRec/blob/6dd5d314f62d3379e664a52c9007e7645556970c/model/cce.py)，4544bytes/SHA `173c257a8711a885e8f6d5abcdb0ad2510ad4fab01c1700631b5ec261f50cd2d`，MIT原件保留。P为首次目标前各SID展开大小之和，g为目标组大小，m=min(g,max(0,K-P))；ItemHit=m/g，ItemNDCG=sum(i=1..m,1/log2(P+i+1))/g。非法SID占1位置，前置重复各自展开、目标只取首次。每名正式用户与官方函数逐项比对一致；独立实现显式展开位置，266合成/随机cases含论文Fig2通过。CCE是平均身份不确定性credit，不是模型恢复出了唯一item；本轮未执行ZCR或修改原SID。

## 补充分析与不确定性

`SUPPLEMENTARY_ANALYSIS`。history=`items[:-1][-20:]`，padding不计数；热度只数原训练可见`items[:-2]`每次交互，156609条，不数展开训练记录的重复前缀，排除heldout验证/测试。正频catalog item nearest-rank tertiles得q1=5/q2=11，ties不拆；UNSEEN=0、LONG_TAIL1–5、MID6–11、POPULAR>=12，先固定规则再看指标。

'''
    for field,groups in g['groups'].items():
        body+='\n'+table([field,'用户数','HR@10','NDCG@10','ItemHit@10','ItemNDCG@10'],[(label,v['users'],*[f"{v['metrics'][key]:.9f}" for key in ('hit@10','ndcg@10','ItemHit@10','ItemNDCG@10')]) for label,v in groups.items()])
    body+='\nUNSEEN仅35用户（catalog9items），小样本不稳定；分组仅描述，不据此改组/选模。总体Bootstrap不能冒充各组CI。\n\n整体用户级95% percentile CI：5000次有放回、seed20261113、PCG64、linear分位，仅反映固定模型下测试用户不确定性，不是跨训练seed方差。\n\n'
    body+=table(['指标','点估计','95% CI'],[(k,f"{v['estimate']:.9f}",f"[{v['low']:.9f}, {v['high']:.9f}]") for k,v in ci['intervals'].items()])
    body+=f'''
稳定用户轴/CSV/float64 NPZ留服务器：`/data/bwwu/lc-rec/a0-final-audit-20261113/{JOB}/primary/`。CSV {m['per_user_files'][0]['bytes']}bytes/SHA `{m['per_user_files'][0]['sha256']}`；NPZ {m['per_user_files'][1]['bytes']}bytes/SHA `{m['per_user_files'][1]['sha256']}`。不含候选重复副本。用户轴SHA `{m['user_axis_sha256']}`表示ordered user_id列表，与原测试完整user/target records SHA的序列化范围不同，并非轴冲突。[全结果manifest](evidence/final_audit_20261113/derived_manifest.json)。未来同用户A1/Shuffle可配对，但本轮无比较数据。

## 时间、资源与审核范围

原train stage={cost['stages'][0]['wall_seconds']:.3f}s（含Validation/checkpoint/best恢复），Trainer runtime={cost['trainer_runtime_seconds']:.4f}s；Validation合计={cost['validation_total_seconds']:.4f}s；四次checkpoint保存={cost['checkpoint_save_seconds']:.3f}s；最终CP完成到strict恢复的包含开销区间={cost['best_restore_interval_upper_bound_seconds']:.3f}s，不能冒称纯恢复独占时长。训练runtime扣Validation约{cost['trainer_runtime_minus_validation_seconds']:.3f}s，仍含存储/恢复开销。

完整test stage={cost['stages'][1]['wall_seconds']:.3f}s；原总runner wall={cost['runner_wall_seconds']:.3f}s（12h35m22s）；charged GPU-device-time={cost['charged_gpu_device_seconds']:.3f}s（25.179h双卡预留计费，非kernel活跃时长）。训练peak allocated最大{max(cost['training_gpu_peak_allocated_bytes'])/2**30:.3f}GiB/reserved{max(cost['training_gpu_peak_reserved_bytes'])/2**30:.3f}GiB；推理allocated{max(cost['inference_gpu_peak_allocated_bytes'])/2**30:.3f}GiB/reserved{max(cost['inference_gpu_peak_reserved_bytes'])/2**30:.3f}GiB。原RAM最低可用{cost['resources']['min_host_available_bytes']/2**30:.2f}GiB，swap恒定约145.14MiB，盘最低可用{cost['resources']['min_disk_available_bytes']/2**40:.3f}TiB，无资源保护触发。

单用户generate P50={m['generation_cost']['p50_seconds']:.6f}s/P95={m['generation_cost']['p95_seconds']:.6f}s/mean={m['generation_cost']['mean_seconds']:.6f}s；双卡全test吞吐={24772/cost['stages'][1]['wall_seconds']:.4f}users/s。不能把两rank用户时延相加当端到端墙钟。

本轮CPU runner累计{budget['cumulative_runner_wall_seconds']:.6f}s（含一次失败）；主评价15.029s、独立扩展7.326s、独立统计14.065s，总pipeline36.893s；主峰RSS{m['cpu_peak_rss_bytes']/2**20:.2f}MiB；持久派生结果{budget['new_result_logical_bytes']}bytes，新增GPU=0、无依赖安装/系统环境修改。原指标/SID/权重/预测/checkpoint未改变，受保护原元数据前后指纹一致。

真实非作者报告：[训练及原四指标](review/final_audit_20261113/training_review.md)、[指标/CCE](review/final_audit_20261113/evaluation_review.md)、[分组/Bootstrap](review/final_audit_20261113/statistics_review.md)。训练审核者独立读原件并补一次只读SSH有效字段核验；另外两名通过主控桥接服务器原始文件、独立编写程序执行，没有声称独立SSH/GPU复现。24772×16=396352逐用户值差0；独立5000×16=80000 Bootstrap均值核对，CI最大差4.996e-16；原均值容差不放宽。

任务级修正2/3轮：[保留的失败、原因与修正](evidence/final_audit_20261113/corrections.json)：一次CPU数据路径错误（扫描预测前失败）；一次deploy违反既有transport写路径边界（上传前失败）。未发生模型/评价方法修正，原实验一次自然完成。[机器最终摘要](evidence/final_audit_20261113/final_summary.json)。

## 固定交付与停止

[复用评价定义](EVALUATION_PROTOCOL_V1.md)及机器配置已冻结。最终DELIVERY取本文件固定GitHub链接中的完整SHA；与CPU RUN、原GPU RUN明确分离。GitHub发布逐文件回读及ZIP CRC/完整解压/每成员大小SHA/安全路径/秘密模式/Windows实测由外部交付回执绑定最终提交，避免自引用。审核ZIP不含模型/adapter/checkpoint/完整预测/原始全量数据；用户Downloads保留正式备份。

GPT_UPLOAD_REQUIRED: NO（必要源码、原始轻量证据及终审均发布至本固定GitHub提交）。本轮授权终点完成后停止；未重训/重生成/改SID/改原论文指标/启动GPU/A1/Shuffle/额外seed/其他数据集。

<details><summary>以下为启动及运行中历史入口原文，仅保留当时事实，不代表当前状态</summary>

'''
    validation_table=table(['Epoch','Step','UTC完成','eval_loss','runtime(s)'],[(e['epoch'],e['step'],e['utc'],e['values']['eval_loss'],e['values']['eval_runtime']) for e in cost['validation_history']])
    body=body.replace('## 第一层：原论文指标及独立门槛', '四次正式Validation记录（均COMPLETE，仅按eval_loss选模）：\n\n'+validation_table+'\n## 第一层：原论文指标及独立门槛',1)
    old=subprocess.check_output(['git','show','1d1db1c2aff35177dd72e28fdd265a7ef213fec4:lc_rec/letter/A0_NORMAL_FINAL_REVIEW_ENTRY.md'],cwd=ROOT.parents[1],text=True)
    (ROOT/'A0_NORMAL_FINAL_REVIEW_ENTRY.md').write_text(body+old+'\n</details>\n')


if __name__=='__main__':
    main()
