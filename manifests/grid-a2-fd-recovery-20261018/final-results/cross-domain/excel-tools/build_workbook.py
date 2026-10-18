#!/usr/bin/env python3
"""Build the one-sheet GRID delivery from immutable Git objects; stdlib only.

No model, dataset, prediction, remote job or scientific code is executed.
Run from any directory: python3 build_workbook.py [--output /path/file.xlsx]
"""
import argparse
import csv
from decimal import Decimal, getcontext
import hashlib
import io
import json
import math
from pathlib import Path
import re
import subprocess
import unicodedata
import zipfile
import xml.etree.ElementTree as ET

getcontext().prec = 50
ROOT = Path(subprocess.check_output(['git', '-C', str(Path(__file__).parent), 'rev-parse', '--show-toplevel'], text=True).strip())
REF = 'd5c0db4203d6226a7aa5a73583da65f2b00d125d'
BASE = 'manifests/grid-a2-fd-recovery-20261018/final-results/cross-domain/'
INDEX = BASE + 'cross-domain-evidence-index-with-a2.json'
CSV = BASE + 'cross-domain-original-metrics-with-a2.csv'
FILE = 'GRID_Shuffle_MLP_Results_AllInOne_20261018.xlsx'
SHEET = 'GRID实验总表'
METRICS = ['recall@5', 'ndcg@5', 'recall@10', 'ndcg@10']
DOMAINS = ['Beauty', 'Toys', 'Sports']
COUNTS = {'Beauty': 22363, 'Toys': 19412, 'Sports': 35598}
ROUTES = {'Beauty': ['Beauty-A', 'Beauty-B'], 'Toys': ['Toys-B', 'Toys-A-5300', 'Toys-A-5000'], 'Sports': ['Sports-B', 'Sports-A1', 'Sports-A2']}
MLP_REFS = {'Beauty': ['Beauty-A', 'Beauty-B'], 'Toys': ['Toys-B', 'Toys-A-5300', 'Toys-A-5000'], 'Sports': ['Sports-B']}
EXEC = {'Beauty-A': '2cf4c793c2e0b2e9dc7e7a9269f5e54d5c667896', 'Beauty-B': '2cf4c793c2e0b2e9dc7e7a9269f5e54d5c667896', 'Toys-B': 'c8acac42d5e13ca80af75faac5d1f7bc33fb589f', 'Sports-B': 'c8acac42d5e13ca80af75faac5d1f7bc33fb589f', 'Toys-A-5300': 'b44399926c98369191c2ff5c2425377c8c07d3c8', 'Toys-A-5000': 'b44399926c98369191c2ff5c2425377c8c07d3c8', 'Sports-A1': 'b44399926c98369191c2ff5c2425377c8c07d3c8', 'Sports-A2': '85b77f5e2a5b497fdc1f620649ba488fc783a245'}
MLP_EXEC = {'Beauty': '89b6d961c81333b0a44c4eb17527ced11897849f', 'Toys': '4495c38459971aee36b00d881eb50f50fafa3159', 'Sports': '4495c38459971aee36b00d881eb50f50fafa3159'}
SHUFFLE = 'references/wmh-original/GRID/src/models/modules/semantic_id/tiger_generation_model_shuffle.py'
MLP_SOURCE = 'references/wmh-original/GRID/src/models/modules/semantic_id/tiger_generation_model_mlp.py'
AUTHOR = 'experiments/beauty_extensions/author/models.py'
B_SOURCE = 'experiments/beauty_extensions/baseline.py'
PAPER_URL = 'https://arxiv.org/html/2507.22224v1#S4'
# Transcribed Table 1 in requested R5,N5,R10,N10 order; all 36 values were
# checked against the saved GRID.pdf and the official arXiv v1 Table 1.
PAPER = {
 'Beauty': {'RK-Means': '.0422 .0277 .0639 .0347', 'R-VQ': '.0422 .0282 .0638 .0351', 'RQ-VAE': '.0404 .0268 .0593 .0329'},
 'Toys': {'RK-Means': '.0376 .0243 .0577 .0308', 'R-VQ': '.0327 .0209 .0493 .0262', 'RQ-VAE': '.0342 .0224 .0514 .0280'},
 'Sports': {'RK-Means': '.0236 .0153 .0353 .0191', 'R-VQ': '.0234 .0151 .0352 .0189', 'RQ-VAE': '.0205 .0132 .0312 .0166'},
}
ROUTE_META = {
 'Beauty-A': ('T5-small / 512维', '双独立', 4800, '学长双表派生路线；与B同时改变文本表示、SID与权重。'),
 'Beauty-B': ('Qwen2.5-7B-Instruct / 3584维', '共享', 12500, '既有共享表Baseline；Encoder lookup作用域hook。'),
 'Toys-B': ('Qwen2.5-7B-Instruct / 3584维', '共享', 6200, 'B类适配；不是学长双表完整生成器。'),
 'Toys-A-5300': ('历史producer未唯一确定', '双独立', 5300, '量化保存配置支持R-Kmeans；T5 factory/Qwen路径/维度冲突；保留候选。'),
 'Toys-A-5000': ('历史producer未唯一确定', '双独立', 5000, '与A-5300同SID、不同checkpoint；不按test分数删除任一候选。'),
 'Sports-B': ('Qwen2.5-7B-Instruct / 3584维', '共享', 9500, 'B类适配；与MLP同SID及基础配置，但不同权重/结构。'),
 'Sports-A1': ('Qwen2.5-7B-Instruct / 3584维', '双独立', 8600, '同Sports-B SID；另行scratch双表训练；final9600。'),
 'Sports-A2': ('T5-small / 512维', '双独立', 6900, '新SID+scratch双表；量化FD工程恢复30 steps；final7900。'),
}
INPUTS = []

def url(path, ref=REF):
    return 'https://github.com/woobowen/recommend/blob/' + ref + '/' + path

def read(path, ref=REF):
    data = subprocess.check_output(['git', 'show', ref + ':' + path], cwd=ROOT)
    INPUTS.append((ref, path, hashlib.sha256(data).hexdigest()))
    return data.decode('utf-8')

def load(path):
    return json.loads(read(path), parse_float=Decimal)

def build_records():
    source = list(csv.DictReader(io.StringIO(read(CSV))))
    assert len(source) == 24
    indexed = {(r['route'], r['condition']): r for r in source}
    assert len(indexed) == 24
    evidence = load(INDEX)
    routes = {r['route_id']: r for r in evidence['routes']}
    assert len(routes) == 8
    for rid, route in routes.items():
        conditions = route['conditions']
        assert len(conditions) == 3
        assert {c['condition'] for c in conditions} == {'Normal', 'Shuffle3', 'Shuffle4'}
        for identity in ['checkpoint', 'sid']:
            assert len({c[identity]['sha256'] for c in conditions}) == 1
        for c in conditions:
            s = indexed[(rid, c['condition'])]
            assert int(s['samples']) == c['samples'] == c['unique_users'] == COUNTS[rid.split('-')[0]]
            assert c['result_status'] == s['result_status'] == 'COMPLETE'
            assert c['scientific_evaluation_execution']['commit'] == EXEC[rid]
            assert all(Decimal(s[k]) == c['metrics_original'][k] for k in METRICS)
    mlp = {}
    for domain in DOMAINS:
        if domain == 'Beauty':
            path = 'manifests/beauty-extensions/runs/beauty-mlp-formal-v1/test/metrics.json'
            values = load(path)
            train = load('manifests/beauty-extensions/runs/beauty-mlp-formal-v1/training_result.json')
            metrics = {m: values['metrics'][0]['test/' + m] for m in METRICS}
            assert values['formal'] and values['complete_test_coverage'] and values['samples'] == 22363
        else:
            path = f'manifests/grid-formal-20261015/runs/gf-20261015-{domain.lower()}-mlp-formal-v1/output/RESULT.json'
            train = load(path)
            metrics = train['test_metrics']['metrics']
        assert train['status'] == 'COMPLETE' and train['formal'] and train['training_complete'] and train['test_complete']
        assert train['fresh_initialization'] and train['full_parameter_training'] and not train['engineering_only']
        assert train['test_metrics']['coverage_exact'] and train['test_metrics']['samples'] == COUNTS[domain]
        mlp[domain] = {'train': train, 'metrics': metrics, 'path': path}
    receipt = read('manifests/grid-formal-20261015/FINAL_RECEIPT.md')
    for d in DOMAINS:
        line = next(l for l in receipt.splitlines() if l.startswith('| ' + d.lower() + ' | 0.'))
        vals = [Decimal(s.strip()) for s in line.split('|')[2:6]]
        assert [mlp[d]['metrics'][m].quantize(Decimal('.000000001')) for m in METRICS] == vals

    records = []
    for domain in DOMAINS:
        for quant, string in PAPER[domain].items():
            rid = domain + '.Paper.' + quant
            records.append(dict(id=rid, result_id=rid, domain=domain, category='论文', route=quant, condition='Paper', samples=None,
                ref_id='', reference='无对照（论文参考值）', grade='论文5-seed汇总', metrics=dict(zip(METRICS, map(Decimal, string.split()))),
                encoder='Flan-T5-XL（默认）', quant=quant, table='论文未对应', checkpoint='5 seeds均值；无单一checkpoint',
                invalid=None, duplicate=None, job='', commit='arXiv:2507.22224v1', source=PAPER_URL, asset=PAPER_URL,
                note='论文原列R5/R10/N5/N10已交换为R5/N5/R10/N10；未把本域目标数冒充论文独立记录。', note_key='paper'))
        for rid in ROUTES[domain]:
            meta = ROUTE_META[rid]
            for condition in ['Normal', 'Shuffle3', 'Shuffle4']:
                c = next(c for c in routes[rid]['conditions'] if c['condition'] == condition)
                s = indexed[(rid, condition)]
                record_id = rid.replace('-', '.', 1) + '.' + condition
                normal_id = rid.replace('-', '.', 1) + '.Normal'
                src = c.get('evaluation_result_source') or {}
                records.append(dict(id=record_id, result_id=record_id, domain=domain,
                    category='Baseline' if condition == 'Normal' else 'Shuffle', route=rid[len(domain)+1:], condition=condition, samples=int(s['samples']),
                    ref_id=domain+'.Paper.RK-Means' if condition == 'Normal' else normal_id,
                    reference='本域论文 RK-Means' if condition == 'Normal' else rid+'-Normal',
                    grade='论文参考；非严格同配置复现' if condition == 'Normal' else '同checkpoint/SID测试时扰动',
                    metrics={m: Decimal(s[m]) for m in METRICS}, encoder=meta[0], quant='R-Kmeans（配置支持）' if rid.startswith('Toys-A') else 'R-Kmeans',
                    table=meta[1], checkpoint='step '+str(meta[2])+' / '+c['checkpoint']['sha256'][:12],
                    invalid=int(s['invalid_candidates']), duplicate=int(s['duplicate_candidates']), job=s['job_id'], commit=EXEC[rid],
                    source=src.get('github_url', url(INDEX)), asset=url(INDEX),
                    asset_label='SID '+c['sid']['sha256'][:12]+'\nCFG '+((c.get('config') or {}).get('sha256') or '见下方证据')[:12],
                    note=meta[3], note_key=rid))
        for ref_route in MLP_REFS[domain]:
            t = mlp[domain]['train']
            sid = t.get('sid_asset') or t['sid']
            config = t.get('approved_config_asset') or t['config']
            records.append(dict(id=domain+'.MLP.vs.'+ref_route[len(domain)+1:]+'.Normal', result_id=domain+'.MLP.Formal',
                domain=domain, category='MLP', route='MLP', condition='MLP', samples=COUNTS[domain],
                ref_id=ref_route.replace('-', '.', 1)+'.Normal', reference=ref_route+'-Normal', grade='参考性能差值；非严格单因素增益',
                metrics=mlp[domain]['metrics'], encoder={'Beauty':'T5（已批准条件）','Toys':'历史producer未唯一确定','Sports':'Qwen2.5-7B-Instruct / 3584维'}[domain],
                quant='R-Kmeans' if domain == 'Sports' else 'RQ-VAE+STE', table='共享', checkpoint='best step '+str({'Beauty':2600,'Toys':4400,'Sports':4900}[domain])+' / '+t['best_checkpoint']['sha256'][:12],
                invalid=t['test_metrics']['invalid_candidate_count'], duplicate=t['test_metrics']['duplicate_candidate_count'],
                job='beauty-mlp-formal-v1' if domain=='Beauty' else 'gf-20261015-'+domain.lower()+'-mlp-formal-v1',
                commit=MLP_EXEC[domain], source=url(mlp[domain]['path']), asset=url(INDEX) if domain=='Sports' else url(mlp[domain]['path']),
                asset_label='SID '+sid['sha256'][:12]+'\nCFG '+config['sha256'][:12],
                note='同一MLP模型的不同参考行；不计为额外实验。'+('与B同SID/基础配置，独立权重与聚合结构仍不同。' if domain=='Sports' else '无唯一匹配Normal；SID/结构/训练协议有差异。'), note_key=domain+'-MLP'))
    assert len(records) == 39 and len({r['id'] for r in records}) == 39
    assert len({r['result_id'] for r in records}) == 36
    assert [sum(r['domain']==d for r in records) for d in DOMAINS] == [11,15,13]
    return records, routes, mlp


def method_sections(routes, mlp):
    """Paragraphs, not a second metric table. Each tuple is label/body/link."""
    sections = []
    def heading(key, text): sections.append(('heading', key, text, ''))
    def p(label, body, link=''): sections.append(('paragraph', label, body, link))
    heading('methods', '详细实验方法说明｜公共流程、A/B差异与11条路线')
    p('阅读与范围', '本表只整理已完成并验收的历史正式结果；不训练、不量化、不推理、不重新评价、不调用GPU、不访问new-server。主表39条比较记录，但只有36条不同结果：9论文、24正式Normal/Shuffle、3正式MLP。点击主表“路线”可跳到对应说明；MLP多个参考行共享实验结果ID。')
    heading('baseline', '公共方法：正常 GRID / TIGER 流程')
    p('文本→SID', '物品文本先经语义编码器得到连续embedding；相应量化方法产生前三位量化码，再按既有确定性规则添加第四位碰撞消歧。第四位不是第四个独立残差量化层。八条Normal所对应的保存配置均支持R-Kmeans（Toys A历史producer限制见各段）；Beauty/Toys MLP则沿用相应RQ-VAE+STE资产。')
    p('历史→推荐', '每个历史物品由四位SID表示。按码位加hierarchy offset，再lookup成向量；常规基线沿用SEP及原mask组织，Transformer Encoder编码按时间排列的物品历史。Decoder按照目标四位SID顺序逐位自回归生成，接续原Beam Search（正式生成K=10、should_check_prefix=False）和原evaluator。无额外过滤、补齐、重评分或重排原候选。')
    p('共享 / 双独立', '共享表：Encoder与Decoder访问同一组SID embedding参数。双独立表：码位语义和词表体系仍配套，但Encoder与Decoder各有独立、可训练的SID表参数。A系为双独立表，B系和三域MLP为共享表。表结构不是Shuffle开关造成的，跨A/B不能解释为单独Shuffle算法的区别。')
    heading('shuffle', '学长原始 Shuffle：查表后、物品内部、测试时随机重排')
    p('具体张量路径', '原SID整数→hierarchy offset→embedding lookup→按[B,M,4,E]组织→在同一历史物品内重排前N个完整E维向量→恢复展平→后续Encoder。padding组保持不变。Shuffle3：[e1,e2,e3,e4]→[eπ1,eπ2,eπ3,e4]；Shuffle4：四位均可排列。不是把整数SID改成另一组量化码，也不跨物品交换、不改变历史时间顺序。', url(SHUFFLE,EXEC['Beauty-A']))
    p('真实随机机制', '冻结原函数用torch.rand(B,M,N).argsort(dim=2)为各物品occurrence、每次Encoder调用采样排列，允许恒等排列；同一batch中的loss与generate调用也可能得到不同排列。正式路径依赖is_testing和shuffle_embedding_on_test / hierarchies_to_shuffle；正常训练不施加该测试扰动。追踪使用独立generator重建排列，不额外消耗或重置正式全局随机流。')
    p('保持的部分', 'Normal与其Shuffle3/4严格复用同一checkpoint、同一SID及本域完整testing目标。量化器、目标SID、Decoder四位自回归顺序、Beam与evaluator不因Shuffle改变。指标接近不证明逐用户推荐不变，更不证明数学上的严格置换不变。')
    heading('route-a', 'A 方案：继承学长双表 Shuffle 源码并工程适配')
    p('继承与适配', 'AuthorShuffleModel(_RuntimeBridge, OriginalShuffleModel)以学长双独立SID表模型为依据。运行桥接、完整物品组校验、采样追踪和受控回放用于兼容与验证；正式Shuffle仍通过super()调用冻结原始随机排列函数。受控回放是工程检查入口，不是正式结果的替代采样方案。不能把“继承并适配”写成未做修改的完整历史执行。', url(AUTHOR,EXEC['Beauty-A']))
    heading('route-b', 'B 方案：共享表 Baseline 的 Encoder lookup hook')
    p('接入与作用域', 'BaselineShuffleModel以已有共享SID表的SemanticIDEncoderDecoder为基础，并未把完整生成器替换成学长双表架构。在encoder_forward_pass作用域记录mask，对Encoder的SID lookup输出设置hook；离开该作用域恢复状态，Decoder lookup不受重排。OFF/Normal直接返回原lookup张量。', url(B_SOURCE,EXEC['Toys-B']))
    p('相同随机逻辑', 'B从冻结学长文件提取唯一_shuffle_hierarchical_embeddings函数体，正式路径复用同一rand/argsort逻辑；校验完整四码位组、padding不变、Shuffle3第四位不变。没有新增替代Decoder、Beam、评分或evaluator。B不是另一种Shuffle算法，而是对共享表Baseline的接入适配。')
    for domain in DOMAINS:
        for rid in ROUTES[domain]:
            c = next(c for c in routes[rid]['conditions'] if c['condition']=='Normal')
            meta=ROUTE_META[rid]
            heading(rid, rid + '｜独立路线说明')
            intro = {
              'Beauty-A':'目的：检查学长双表派生模型对历史码位向量顺序的敏感性。使用T5-small的512维文本embedding与R-Kmeans SID，保留双独立表；使用既有step4800 checkpoint，未在本次统计中重新训练。A适配位于查表后、Encoder前。与Beauty B不同之处包含文本编码器、SID、表结构和权重，不能把A/B差异只归因于Shuffle。',
              'Beauty-B':'目的：在已有共享表Baseline上执行相同学长Shuffle。上游Qwen2.5-7B-Instruct的3584维文本表示经R-Kmeans生成SID，使用step12500 checkpoint。通过Encoder lookup hook接入，Decoder共享表本身保留；与Beauty A的T5/双表/SID/权重不同。',
              'Toys-B':'目的：在Toys共享表Baseline中测试同物品内向量重排。上游Qwen2.5-7B-Instruct、3584维、R-Kmeans；保留既有step6200 checkpoint。B类Encoder作用域hook复用学长排列逻辑。它与Toys A两候选使用不同SID、表结构和权重，不能跨路线计算Shuffle主差值。',
              'Toys-A-5300':'目的：保留并检查学长双表候选step5300。原保存配置明确ResidualQuantization→MiniBatchKMeans；但T5EncoderModel factory、Qwen模型路径和声明3584维/实际512维的记录存在冲突，历史上游producer未唯一绑定，不能写成已证实T5或Qwen。A适配复用原随机函数；与A-5000共用SID但checkpoint不同。',
              'Toys-A-5000':'目的：保留第二个学长双表候选step5000，而非按test分数从两候选中择优。与A-5300使用完全相同SID，但来自不同checkpoint及保存训练配置。量化保存配置支持R-Kmeans，文本encoder/producer冲突同样保留。A类查表后Shuffle；不重新训练或改变原目标。',
              'Sports-B':'目的：在Sports既有共享表Qwen/R-Kmeans Baseline上测试Shuffle。Qwen2.5-7B-Instruct文本表示3584维，step9500 checkpoint，通过B类Encoder lookup hook复用原Shuffle函数。与Sports A1共享同一SID，但A1是另行scratch训练的双表模型。',
              'Sports-A1':'目的：在Sports Qwen/R-Kmeans SID条件下复现学长双独立表生成器。复用Sports B的同一SID，完整模型从零初始化训练；best step8600、final step9600，按验证Recall@5选模、patience10正常早停。正式三条件均使用best8600。它与B同时涉及表结构及独立训练，差异不只是启用Shuffle。',
              'Sports-A2':'目的：在T5-small的512维embedding所产生的新R-Kmeans SID下，从零训练学长双表生成器。保留三层量化与第四位确定性碰撞消歧。原量化遇EMFILE后，仅通过A2专用item-ID collate把小Tensor列表堆叠为连续Tensor并作FD保护；不改变ID/embedding配对或量化计算。复用已有T5编码，双卡量化真实完成30 optimizer steps；新SID训练best6900、final7900，patience10正常早停。A1/A2同时改变SID来源并独立训练，不是单因素SID因果实验。',
            }[rid]
            p('目的与实际处理', intro)
            p('测试与对照', '三条件：Normal、Shuffle3、Shuffle4。共同使用step'+str(meta[2])+' checkpoint及下列同一SID，完整testing覆盖'+str(COUNTS[domain])+'个目标；各Shuffle仅对比本路线Normal。接入为'+('A类双表继承适配' if meta[1]=='双独立' else 'B类共享表Encoder lookup hook')+'，Decoder仍四位自回归；候选处理沿用原evaluator。',url(INDEX))
            p('checkpoint身份', c['checkpoint']['path']+'\nSHA-256: '+c['checkpoint']['sha256'])
            p('SID身份', c['sid']['path']+'\nSHA-256: '+c['sid']['sha256'])
            config=c.get('config')
            p('有效配置与源码', ('原配置资产: '+config['path']+'\n配置SHA-256: '+config['sha256'] if config else '本索引未复制配置资产字段；有效训练配置、选模及严格加载证据见Sports A1固定独立审核。')+'\n正式评价执行commit: '+EXEC[rid]+'\n部署snapshot SHA-256: '+c['scientific_evaluation_execution']['snapshot_sha256'],url('manifests/grid-a2-fd-recovery-20261018/launch-v2/completed-result-reviews/sports_a1/audit_result.json') if not config else url(AUTHOR if meta[1]=='双独立' else B_SOURCE,EXEC[rid]))
            if rid=='Sports-A1':
                p('实际训练有效配置', 'seed42、32-true；Adam lr=.001、weight_decay=.0001、scheduler=None；raw16、accum16、名义有效batch256 raw records、val1600 loader batches；96次完整验证，每次35598目标。base配置SHA6f336505677ded65b16023e0cb798df2497f9c6c8fb34ff400dcb5ccc8e0dc9d；实际展开used_config.json是另一文件：/home/bwwu/grid-audits/sid-extensions-p0/jobs/ga-20261017-sportsa1-train-v1/output/used_config.json，14202字节，SHA-256 1e3b608a2037bce5c2934a6cd220fd2ffc1c5b494fa617b4314fc672f4c6320a。训练协议SHA-256 9d3823c2ada022f479fb3b8ac4e36bc2cf09267281db5b15edc636bec0a45c7e。',url('manifests/grid-a-launch-20261017/compiled/specs/sportsa1-train.json'))
            if rid=='Sports-A2':
                p('A2恢复边界', '原失败job ga-20261017-sportsa2-quantize-v1保留；恢复job ga-20261018-sportsa2-quantize-fd-recovery-v1，新训练ga-20261018-sportsa2-train-fd-recovery-v1。原T5编码执行b44399926c98369191c2ff5c2425377c8c07d3c8，修复后量化/训练/评价执行85b77f5e2a5b497fdc1f620649ba488fc783a245。属于已完成的受控工程恢复与方法复现，不是找回学长原始Sports权重。',url('manifests/grid-a2-fd-recovery-20261018/final-results/FINAL_RESULTS.md'))
    heading('mlp-common', 'MLP 公共实现：有序逐物品聚合，Decoder保留自回归')
    p('张量与网络', 'SID按码位offset后lookup；[B,M,4,E]依原顺序reshape/Concat为[B,M,4E]，执行Linear(4E,2E)→ReLU→Dropout(0.1)→Linear(2E,E)。本次E=128，即512→256→128，同一MLP作用于各历史物品，输出[B,M,E]。这是有序拼接，不是求和、平均或无序池化，本身不保证数学上的置换不变。',url(MLP_SOURCE,MLP_EXEC['Sports']))
    p('mask / SEP / 表', '实际原件设置sep_token=None，不再插入SEP；mask reshape[B,M,4]后取[:,:,−1]，即第四码位mask，而非OR/max。适配层拒绝部分有效的四码位组。MLP encoder/decoder SID表共享。历史四个码位输入聚合成一个物品级Encoder token；Encoder长度与普通保留SEP的四码位基线不同。',url(AUTHOR,MLP_EXEC['Sports']))
    p('训练与生成', '三域均独立scratch、全参数学习新MLP及生成器参数；不是已有Normal模型的零参数测试开关。Decoder继续按四位目标SID逐位自回归预测，保留前缀依赖、Beam及原evaluator，因此MLP不等于Parallel。Shuffle是测试时输入扰动；MLP是改变Encoder物品表示并重新训练。')
    for domain in DOMAINS:
        t=mlp[domain]['train']; sid=t.get('sid_asset') or t['sid']; cfg=t.get('approved_config_asset') or t['config']; best=t['best_checkpoint']
        heading(domain+'-MLP', domain+' MLP｜一个正式模型，多参考比较不增加实验数')
        p('条件与比较边界', {
          'Beauty':'已批准T5/RQ-VAE+STE SID，学长逐物品MLP、共享表，独立从零训练。其SID不同于Beauty A的T5/R-Kmeans和Beauty B的Qwen/R-Kmeans；A只部分匹配文本族，B只部分匹配表结构，均非唯一匹配Normal。两行展示同一MLP相对A/B的参考性性能差值，不是严格单因素结构增益。',
          'Toys':'沿用作者固定RQ-VAE+STE bundle（beta .25），学长逐物品MLP、共享表，独立从零训练。保存配置的T5 factory与Qwen路径/维度冲突，历史encoder/producer未唯一确定。其SID不同于Toys B及A两候选；相对三个Normal均为参考比较，三行共享同一实验结果ID。',
          'Sports':'使用与Sports B完全相同的Qwen/R-Kmeans SID，学长逐物品MLP、共享表，独立从零训练。与B的SID及基础配置SHA相同，raw/accum/验证间隔更匹配，因此以B-Normal为主要数值参考；MLP聚合、SEP、参数和独立权重仍有差异，不能解释成严格单因素增益。不是恢复学长历史Sports MLP权重。',
        }[domain])
        raw,acc,val={'Beauty':(32,64,6400),'Toys':(32,48,4800),'Sports':(16,16,1600)}[domain]
        p('实际训练与选模', f'seed=42；precision=32-true；Adam，lr=.001、weight_decay=.0001；scheduler=None。raw microbatch={raw}；accumulate_grad_batches={acc}；单卡名义有效batch={raw*acc} raw records；val_check_interval={val} loader batches。按完整验证Recall@5最大选best，patience10正常早停。best step='+str({'Beauty':2600,'Toys':4400,'Sports':4900}[domain])+'；final step='+str(t['global_step'])+'；完成完整验证'+str(t['completed_full_validations'])+'次；正式testing精确覆盖'+str(COUNTS[domain])+'个目标。此处batch按raw records计，collator可进一步展开历史滑窗，不等于实际expanded-window数；单卡累积协议不是多卡DDP完全相同随机轨迹。',url('manifests/grid-formal-20261015/FINAL_RECEIPT.md'))
        p('checkpoint身份',best['path']+'\nSHA-256: '+best['sha256'])
        p('SID身份',sid['path']+'\nSHA-256: '+sid['sha256'])
        p('配置与执行身份',cfg['path']+'\n配置SHA-256: '+cfg['sha256']+'\n正式训练/测试执行commit: '+MLP_EXEC[domain],url(mlp[domain]['path']))
    heading('paper', '论文、对照关系、公式与证据说明')
    p('论文原表与协议', 'GRID: Generative Recommendation with Semantic IDs: A Practitioner’s Handbook，arXiv:2507.22224v1，Section4/Table1。原列R@5/R@10/N@5/N@10已改为R@5/N@5/R@10/N@10。论文报告5个seed平均，默认Flan-T5-XL；生成器Adam lr5e−4、weight_decay1e−6、batch256，选best验证Recall@10。本表单checkpoint历史结果的文本编码器、训练/选模等不同，所有Normal对论文均为参考比较，非严格同配置复现；差值不单独证明复现成败。',PAPER_URL)
    p('论文对应与未知', '八条Normal据量化源码/保存配置对应本域论文RK-Means，而非根据A/B名称或分数匹配。Toys A只能写“配置支持R-Kmeans，producer未唯一绑定”；不强行补全编码器。R-VQ与RQ-VAE论文行完整保留，但没有虚构对应的Normal。论文未单列本表逐路线测试计数，因此论文计数留空。')
    p('指标定义', '每个完整testing目标对应一个真实目标物品。Recall@K为目标是否进入前K候选的用户平均；NDCG@K为命中时按1/log2(rank+1)折扣后的用户平均（单目标）。沿用正式原evaluator对候选的处理；非法候选数、重复数照原结果披露，没有在本表修补或重算预测。')
    p('差值与单位', 'Δabs=current−reference，是0到1指标比例之差；Δrel=(current−reference)/reference，在Excel底层保存为小数比率，用百分比格式显示。指标六位、绝对差值八位、相对变化两位；颜色仅区分正负号，不暗示研究成功/失败。论文显示其公布的四位小数。不先舍入再计算。')
    p('公式与排序', '主表使用唯一“比较记录ID”，对照列存确切ID。INDEX/MATCH按该ID查找本表对应指标，不按相邻行或路线名字猜测。筛选不改变配对；整张表排序后仍按ID找对照。没有对照的论文差值留空；相对分母为零时IFERROR返回空。重复MLP结果不作为额外模型计数。')
    p('缓存与重算', 'OOXML保存数值指标、真实Excel公式及独立Decimal计算的数值缓存；workbook设置auto/fullCalcOnLoad/forceFullCalc。缓存是本地CPU算术结果，不冒充历史科研重算。原生表格软件只读打开、重算、排序和有限渲染的实际结果另见excel-tools/excel-validation.json；未完成的检查不以缓存冒充通过。')
    p('固定身份边界', '输入/已发表指标固定于d5c0db4203d6226a7aa5a73583da65f2b00d125d；较早跨域证据继承9ab3d0ea5d32db00e278f74c599a1943885453db；A2科研发表42ac9acf23ae0bdc65ed6505089e1b88101755cd；A2实际执行85b77f5e2a5b497fdc1f620649ba488fc783a245。当前Excel发布commit是交付身份，不是所有历史实验的执行版本。',url(INDEX))
    p('生成入口', '本文件由同目录excel-tools/build_workbook.py生成，仅Python标准库读取固定Git对象。复现：python3 manifests/grid-a2-fd-recovery-20261018/final-results/cross-domain/excel-tools/build_workbook.py --output /tmp/GRID-reproduced.xlsx 。仅创建统计工作簿，不访问服务器或加载科研模型。')
    p('固定结果入口', url(CSV)+'\n'+url(INDEX)+'\n'+url('manifests/grid-formal-20261015/FINAL_RECEIPT.md'),url(CSV))
    p('来源内容身份', '\n'.join(path+'\nSHA-256: '+sha for _,path,sha in dict.fromkeys(INPUTS)),url(INDEX))
    return sections

NS='http://schemas.openxmlformats.org/spreadsheetml/2006/main'
REL='http://schemas.openxmlformats.org/officeDocument/2006/relationships'
ET.register_namespace('',NS); ET.register_namespace('r',REL)
def E(tag, attrs=None, **kw): return ET.Element('{'+NS+'}'+tag, {**(attrs or {}),**{k:str(v) for k,v in kw.items()}})
def sub(parent, tag, **attrs):
    e=E(tag, **attrs);parent.append(e);return e
def xml(e): return ET.tostring(e,encoding='utf-8',xml_declaration=True)
def col(n):
    s=''
    while n: n,k=divmod(n-1,26);s=chr(65+k)+s
    return s

class Workbook:
    def __init__(self):
        self.rows={};self.merges=[];self.links=[];self.styles=[];self.smap={};self.notes={}
        self.widths=[18,22,11,12,17,13,13,28,24,28]+[14]*12+[32,24,16,33,12,12,47,23,25,27,49]
    def style(self,font=0,fill=0,border=1,num=0,align='left'):
        key=(font,fill,border,num,align)
        if key not in self.smap:self.smap[key]=len(self.styles);self.styles.append(key)
        return self.smap[key]
    def cell(self,r,c,value='',style=None,formula=None,link=None):
        item={'value':value,'style':self.style() if style is None else style,'formula':formula}
        self.rows.setdefault(r,{'height':38,'cells':{}})['cells'][c]=item
        if link:self.links.append((col(c)+str(r),link))
    def merge(self,r,c1,c2,text,style,height):
        self.cell(r,c1,text,style);self.rows[r]['height']=height
        self.merges.append(col(c1)+str(r)+':'+col(c2)+str(r))
    def styles_xml(self):
        s=E('styleSheet')
        nf=sub(s,'numFmts',count=4)
        for i,code in [(164,'0.000000'),(165,'[Blue]+0.00000000;[Red]-0.00000000;0.00000000'),(166,'[Blue]+0.00%;[Red]-0.00%;0.00%'),(167,'0.0000')]:sub(nf,'numFmt',numFmtId=i,formatCode=code)
        fonts=sub(s,'fonts',count=5)
        for size,bold,color in [(10,False,'263445'),(20,True,'FFFFFF'),(10,False,'526174'),(10,True,'FFFFFF'),(12,True,'FFFFFF')]:
            f=sub(fonts,'font');sub(f,'sz',val=size);sub(f,'name',val='Noto Sans CJK SC');sub(f,'family',val=2);sub(f,'color',rgb='FF'+color)
            if bold:sub(f,'b')
        fills=sub(s,'fills',count=8)
        sub(sub(fills,'fill'),'patternFill',patternType='none');sub(sub(fills,'fill'),'patternFill',patternType='gray125')
        for color in ['17324D','EAF2F8','F7F9FB','F0EAF7','EEF0F3','FFFFFF']:
            p=sub(sub(fills,'fill'),'patternFill',patternType='solid');sub(p,'fgColor',rgb='FF'+color);sub(p,'bgColor',indexed=64)
        borders=sub(s,'borders',count=3)
        for kind in [0,1,2]:
            b=sub(borders,'border')
            for side in ['left','right','top','bottom','diagonal']:
                x=sub(b,side)
                if kind and side=='bottom':x.set('style','hair');sub(x,'color',rgb='FFD7DFE8')
                if kind==2 and side=='top':x.set('style','medium');sub(x,'color',rgb='FF607D96')
        cs=sub(s,'cellStyleXfs',count=1);sub(cs,'xf',numFmtId=0,fontId=0,fillId=0,borderId=0)
        xs=sub(s,'cellXfs',count=len(self.styles))
        for font,fill,border,num,align in self.styles:
            x=sub(xs,'xf',numFmtId=num,fontId=font,fillId=fill,borderId=border,xfId=0,applyAlignment=1,applyNumberFormat=1)
            sub(x,'alignment',horizontal=align,vertical='center',wrapText=1)
        css=sub(s,'cellStyles',count=1);sub(css,'cellStyle',name='Normal',xfId=0,builtinId=0)
        sub(s,'dxfs',count=0);sub(s,'tableStyles',count=0,defaultTableStyle='TableStyleMedium2',defaultPivotStyle='PivotStyleLight16')
        return xml(s)
    def sheet_xml(self,end):
        s=E('worksheet');sub(s,'sheetPr');sub(s,'dimension',ref=f'A1:AG{end}')
        view=sub(sub(s,'sheetViews'),'sheetView',showGridLines=0,zoomScale=85,workbookViewId=0)
        sub(view,'pane',xSplit=6,ySplit=8,topLeftCell='G9',activePane='bottomRight',state='frozen')
        sub(view,'selection',pane='bottomRight',activeCell='K9',sqref='K9')
        sub(s,'sheetFormatPr',defaultRowHeight=24)
        cols=sub(s,'cols')
        for i,width in enumerate(self.widths,1):sub(cols,'col',min=i,max=i,width=width,customWidth=1)
        data=sub(s,'sheetData')
        for r,record in sorted(self.rows.items()):
            row=sub(data,'row',r=r,ht=min(409,record['height']),customHeight=1)
            for c,it in sorted(record['cells'].items()):
                v=it['value'];cell=sub(row,'c',r=col(c)+str(r),s=it['style'])
                if it['formula']:
                    sub(cell,'f').text=it['formula'];sub(cell,'v').text=str(v)
                elif isinstance(v,(Decimal,int,float)) and not isinstance(v,bool):sub(cell,'v').text=str(v)
                elif v!='':
                    cell.set('t','inlineStr');t=sub(sub(cell,'is'),'t');t.set('{http://www.w3.org/XML/1998/namespace}space','preserve');t.text=str(v)
        if self.merges:
            merges=sub(s,'mergeCells',count=len(self.merges))
            for ref in self.merges:sub(merges,'mergeCell',ref=ref)
        if self.links:
            links=sub(s,'hyperlinks')
            for i,(ref,target) in enumerate(self.links,1):
                if target.startswith('#'):sub(links,'hyperlink',ref=ref,location=target[1:],tooltip='跳转到对应方法说明')
                else:
                    h=sub(links,'hyperlink',ref=ref,tooltip=target[:250]);h.set('{'+REL+'}id','rId'+str(i))
        sub(s,'printOptions',horizontalCentered=0)
        sub(s,'pageMargins',left=.25,right=.25,top=.4,bottom=.4,header=.2,footer=.2)
        sub(s,'pageSetup',paperSize=8,orientation='landscape',scale=75,fitToWidth=0,fitToHeight=0)
        hf=sub(s,'headerFooter');sub(hf,'oddFooter').text='GRID / Semantic ID | 固定结果统计 &P / &N'
        part=sub(sub(s,'tableParts',count=1),'tablePart');part.set('{'+REL+'}id','rIdTable')
        return xml(s)
    def write(self,path,end,headers):
        # Stable IDs are unique; formulas refer to exact ID instead of adjacency.
        table=E('table',id=1,name='GRIDResults',displayName='GRIDResults',ref='A8:AG47',totalsRowShown=0)
        sub(table,'autoFilter',ref='A8:AG47');columns=sub(table,'tableColumns',count=len(headers))
        for i,h in enumerate(headers,1):sub(columns,'tableColumn',id=i,name=h)
        sub(table,'tableStyleInfo',name='TableStyleMedium2',showFirstColumn=0,showLastColumn=0,showRowStripes=0,showColumnStripes=0)
        wb=E('workbook');sub(wb,'bookViews').append(E('workbookView',activeTab=0))
        sh=sub(sub(wb,'sheets'),'sheet',name=SHEET,sheetId=1);sh.set('{'+REL+'}id','rId1')
        dn=sub(wb,'definedNames');sub(dn,'definedName',name='_xlnm.Print_Area',localSheetId=0).text=f"'{SHEET}'!$A$1:$AG$47,'{SHEET}'!$A$51:$P${end}"
        sub(dn,'definedName',name='_xlnm.Print_Titles',localSheetId=0).text=f"'{SHEET}'!$7:$8"
        sub(wb,'calcPr',calcId=191029,calcMode='auto',fullCalcOnLoad=1,forceFullCalc=1)
        pkg='http://schemas.openxmlformats.org/package/2006/relationships'
        def rels(items):
            root=ET.Element('Relationships',xmlns=pkg)
            for ident,typ,target,external in items:
                a={'Id':ident,'Type':REL+'/'+typ,'Target':target}
                if external:a['TargetMode']='External'
                ET.SubElement(root,'Relationship',a)
            return xml(root)
        contents=ET.Element('Types',xmlns='http://schemas.openxmlformats.org/package/2006/content-types')
        for ext,mime in [('rels','application/vnd.openxmlformats-package.relationships+xml'),('xml','application/xml')]:ET.SubElement(contents,'Default',Extension=ext,ContentType=mime)
        for part,mime in [('xl/workbook.xml','spreadsheetml.sheet.main'),('xl/worksheets/sheet1.xml','spreadsheetml.worksheet'),('xl/styles.xml','spreadsheetml.styles'),('xl/tables/table1.xml','spreadsheetml.table')]:ET.SubElement(contents,'Override',PartName='/'+part,ContentType='application/vnd.openxmlformats-officedocument.'+mime+'+xml')
        parts={'[Content_Types].xml':xml(contents),'_rels/.rels':rels([('rId1','officeDocument','xl/workbook.xml',False)]),'xl/workbook.xml':xml(wb),'xl/_rels/workbook.xml.rels':rels([('rId1','worksheet','worksheets/sheet1.xml',False),('rId2','styles','styles.xml',False)]),'xl/worksheets/sheet1.xml':self.sheet_xml(end),'xl/styles.xml':self.styles_xml(),'xl/tables/table1.xml':xml(table)}
        linkrels=[('rId'+str(i),'hyperlink',target,True) for i,(_,target) in enumerate(self.links,1) if not target.startswith('#')]
        parts['xl/worksheets/_rels/sheet1.xml.rels']=rels(linkrels+[('rIdTable','table','../tables/table1.xml',False)])
        path.parent.mkdir(parents=True,exist_ok=True)
        with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED,compresslevel=9) as z:
            for name,body in parts.items():
                info=zipfile.ZipInfo(name,date_time=(2026,10,18,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED;info.external_attr=0o644<<16;z.writestr(info,body)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=ROOT/BASE/FILE)
    args=parser.parse_args();records,routes,mlp=build_records();sections=method_sections(routes,mlp)
    w=Workbook();title=w.style(font=1,fill=2,border=0);subtle=w.style(font=2,border=0);band=w.style(font=3,fill=2,border=0,align='center');head=w.style(font=3,fill=2,align='center')
    w.merge(1,1,22,'GRID / Semantic ID｜Shuffle & MLP 正式实验总表',title,44)
    w.merge(2,1,22,'统计日期 2026-10-18（北京时间）｜固定输入 '+REF+'｜单文件 · 单工作表 · CPU整理 · GPU 0',subtle,28)
    w.merge(3,1,22,'39条比较记录 = 9论文 + 8 Normal + 8 Shuffle3 + 8 Shuffle4 + 6 MLP参考行；36条不同结果，其中MLP仅3个模型。',subtle,28)
    w.merge(4,1,22,'Δabs = 当前−对照；Δrel = (当前−对照)/对照（底层小数、百分比显示）。原始精度入表，显示再舍入；蓝/红仅表示正/负号。',subtle,28)
    w.merge(5,1,22,'论文为5-seed均值；Normal对论文及全部MLP差值均属参考比较，非严格单因素增益。Toys历史producer不确定性完整保留。',subtle,28)
    w.merge(6,1,6,'↓ 点击跳转：方法与资产说明',w.style(font=2,fill=3,border=0),30);w.links.append(('A6',"#'"+SHEET+"'!A51"))
    w.merge(6,7,22,'使用：筛选第8行；冻结前6列和前8行；点击路线查看说明；右侧保留源码、job、候选与来源。',subtle,30)
    for a,b,t in [(1,10,'实验身份与对照'),(11,13,'Recall@5'),(14,16,'NDCG@5'),(17,19,'Recall@10'),(20,22,'NDCG@10'),(23,33,'模型、资产与证据')]:w.merge(7,a,b,t,band,25)
    headers=['比较记录ID','实验结果ID','数据集','实验类别','实验路线','条件','测试目标数','对照记录ID','比较基准','可比性等级','Recall@5','R5 Δabs','R5 Δrel','NDCG@5','N5 Δabs','N5 Δrel','Recall@10','R10 Δabs','R10 Δrel','NDCG@10','N10 Δabs','N10 Δrel','文本embedding模型','SID量化方法','SID表结构','模型 / checkpoint','非法候选数','重复候选数','job ID','实际执行commit','原始结果链接','SID / 配置证据','备注 / 限制']
    for c,h in enumerate(headers,1):w.cell(8,c,h,head)
    w.rows[8]['height']=52
    byid={r['id']:r for r in records}
    for row,r in enumerate(records,9):
        fill={'论文':6,'Baseline':3,'Shuffle':7,'MLP':5}[r['category']];border=2 if row==9 or records[row-10]['domain']!=r['domain'] else 1
        text=w.style(fill=fill,border=border);integer=w.style(fill=fill,border=border,align='right');values=[r[k] for k in ['id','result_id','domain','category','route','condition','samples','ref_id','reference','grade']]
        for c,v in enumerate(values,1):w.cell(row,c,'' if v is None else v,integer if c==7 else text)
        for j,m in enumerate(METRICS):
            c=11+3*j;current=r['metrics'][m];w.cell(row,c,current,w.style(fill=fill,border=border,num=167 if r['category']=='论文' else 164,align='right'))
            if r['ref_id']:
                base=byid[r['ref_id']]['metrics'][m];delta=current-base;letter=col(c)
                lookup=f'INDEX(${letter}$9:${letter}$47,MATCH($H{row},$A$9:$A$47,0))'
                w.cell(row,c+1,delta,w.style(fill=fill,border=border,num=165,align='right'),formula=f'IFERROR({letter}{row}-{lookup},"")')
                w.cell(row,c+2,delta/base if base else '',w.style(fill=fill,border=border,num=166,align='right'),formula=f'IFERROR(({letter}{row}-{lookup})/{lookup},"")')
            else:
                for c2 in [c+1,c+2]:w.cell(row,c2,'',w.style(fill=fill,border=border,align='right'))
        tails=[r['encoder'],r['quant'],r['table'],r['checkpoint'],r['invalid'],r['duplicate'],r['job'],r['commit'],'原始结果 ↗',r.get('asset_label','论文固定原文 ↗'),r['note']]
        for c,v in enumerate(tails,23):w.cell(row,c,'' if v is None else v,integer if c in [27,28] else text)
        w.links.extend([(f'AE{row}',r['source']),(f'AF{row}',r['asset'])])
        if re.fullmatch('[0-9a-f]{40}',r['commit']):w.links.append((f'AD{row}','https://github.com/woobowen/recommend/commit/'+r['commit']))
        w.rows[row]['height']=84
    w.merge(49,1,22,'主表结束｜39条记录；筛选仅覆盖A8:AG47。以下是同一工作表内的方法附录，不属于第二张数据表。',subtle,30)
    row=51
    labelstyle=w.style(font=2,fill=3,border=1);notestyle=w.style(border=1);sectionstyle=w.style(font=4,fill=2,border=0)
    for kind,key,body,link in sections:
        if kind=='heading':
            w.notes[key]=row;w.merge(row,1,16,body,sectionstyle,32)
        else:
            w.merge(row,1,2,key,labelstyle,42)
            width=sum(w.widths[2:16])*.94
            lines=sum(max(1,math.ceil(sum(2 if unicodedata.east_asian_width(ch) in 'WF' else 1 for ch in para)/width)) for para in body.split('\n'))
            # Long identity manifests are split into paragraphs rather than one giant cell.
            if lines>23:
                pieces=body.split('\n');chunk=[];size=0
                for piece in pieces:
                    chunk.append(piece);size+=max(1,math.ceil(len(piece)/width))
                    if size>=14:
                        w.merge(row,3,16,'\n'.join(chunk),notestyle,min(360,(size+2)*14));row+=1;w.merge(row,1,2,key+'（续）',labelstyle,42);chunk=[];size=0
                body='\n'.join(chunk);lines=size
            w.merge(row,3,16,body,notestyle,max(42,min(360,(lines+1)*14)))
            if link:w.links.append((f'C{row}',link))
        row+=1
    for i,r in enumerate(records,9):w.links.append((f'E{i}',"#'"+SHEET+"'!A"+str(w.notes[r['note_key']])))
    w.merge(row+1,1,16,'↑ 返回主数据表｜生成脚本与独立验收回执随Excel发布；本文件不改变任何科研原始结果。',subtle,30)
    w.links.append((f'A{row+1}',"#'"+SHEET+"'!A8"));w.write(args.output,row+1,headers)
    with zipfile.ZipFile(args.output) as z:
        assert z.testzip() is None
        for name in z.namelist():
            if name.endswith('.xml') or name.endswith('.rels'):ET.fromstring(z.read(name))
        sheet=ET.fromstring(z.read('xl/worksheets/sheet1.xml'))
        assert len(sheet.findall('.//{'+NS+'}f'))==240
    print(json.dumps({'status':'GENERATED_AND_XML_PARSED','path':str(args.output.resolve()),'bytes':args.output.stat().st_size,'sha256':hashlib.sha256(args.output.read_bytes()).hexdigest(),'sheet_count':1,'table_records':39,'numeric_metrics':156,'abs_formulas':120,'relative_formulas':120,'route_note_count':11,'notes_end_row':row+1,'source_commit':REF},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
