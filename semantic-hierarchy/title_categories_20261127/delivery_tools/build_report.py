"""Generate the report directly from unrounded archived results."""
from pathlib import Path
import json
HERE=Path(__file__).resolve().parent;DEST=HERE/'repo/semantic-hierarchy/title_categories_20261127'
def main():
 data=json.loads((DEST/'run/all_results.json').read_text(encoding='utf8'));rows=data['primary']
 lines=['# 语义层级分析报告：Title＋Categories','',
 '结果支持统计意义上的语义细分：四组父子平均增量与分支Gap均为正，多数有效路径递进；不支持每条路径都单调或已恢复显式人类类别树。第一层到第二层为主要证据，第三层受覆盖率限制。',
 '', '## 五项指标与分母','',
 '| 数据集/SID | S1→S2→S3 | Δ12 / Δ23 | 正增量12 / 23 | 单调路径 | Gap12 / Gap23 |',
 '|---|---|---|---|---|---|']
 for r in rows:
  o=r['observed'];mono=round(o['path_monotonic']*r['proper_paths'])
  lines.append('| '+r['dataset']+'/'+r['method']+' | '+'→'.join(f'{r["node_macro_cosines"][str(d)]:.6f}' for d in [1,2,3])+f' | {o["delta_12"]:.6f} / {o["delta_23"]:.6f} | {o["positive_12"]:.2%} / {o["positive_23"]:.2%} | {mono}/{r["proper_paths"]} ({o["path_monotonic"]:.2%}) | {o["gap_12"]:.6f} / {o["gap_23"]:.6f} |')
 lines+=['','S是每层可评估节点的宏平均，不把相邻层均值差当逐边增益。完整路径须两次真正缩小，且第三层至少两件商品；正增量阈值1e−6。','',
 '## 覆盖范围','',
 '| 数据集/SID | 主分析n | L2覆盖商品/比例 | L3覆盖商品/比例 | 有效边12/23 |','|---|---:|---|---|---|']
 for r in rows:
  c=r['coverage'];lines.append(f'| {r["dataset"]}/{r["method"]} | {r["items"]} | {c["2"]["covered_items"]} / {c["2"]["coverage"]:.2%} | {c["3"]["covered_items"]} / {c["3"]["coverage"]:.2%} | {r["proper_edges"]["1_to_2"]}/{r["proper_edges"]["2_to_3"]} |')
 lines+=['','Qwen第三层覆盖约19–21%，LETTER约5.57%。例如Instruments LETTER第三层均值0.969653只涉及538件共享三位前缀的商品，不能推广到全部9661件。单商品节点没有配对对象，不等于语义不相似。',
 '', '## 随机对照与区间','',
 '| 数据集/SID | Δ12观察 | 条件置换均值 | Δ12簇bootstrap95%CI | Holm p12 |','|---|---:|---:|---|---:|']
 for r in rows:
  ci=r['cluster_bootstrap_95ci']['delta_12'];null=r['conditional_controls']['1']['delta_12']['null_mean']
  lines.append(f'| {r["dataset"]}/{r["method"]} | {r["observed"]["delta_12"]:.6f} | {null:.6f} | [{ci[0]:.6f}, {ci[1]:.6f}] | {r["holm_p_value"]:.3f} |')
 lines+=['','999次置换在父集合内保持组大小；观察增量明显高于接近0的随机增量，不能只用集合变小解释。2000次bootstrap按第一层子树重采样，区间均高于0。p=0.001是置换分辨率下限，不是p=0。Holm只对四个预设Δ12主检验；不是Qwen−LETTER差值检验。','',
 '## 与同商品集title-only比较','',
 '| 数据集/SID | title Δ12 → 融合Δ12 | title Gap12 → 融合Gap12 | 单调比例变化（百分点） |','|---|---|---|---:|']
 for r in rows:
  t=next(x for x in data['all_variants'] if x['dataset']==r['dataset'] and x['method']==r['method'] and x['variant']=='title_complete');o=r['observed'];b=t['observed']
  lines.append(f'| {r["dataset"]}/{r["method"]} | {b["delta_12"]:.6f} → {o["delta_12"]:.6f} | {b["gap_12"]:.6f} → {o["gap_12"]:.6f} | {(o["path_monotonic"]-b["path_monotonic"])*100:+.2f} |')
 lines+=['','融合后绝对余弦更高，但Δ12下降约41–48%、Gap12下降约41–48%、路径单调比例下降约1.46–3.79个百分点。这与共同类目方向拉近总体商品、压缩组间差距相符；不能把绝对余弦变高解释成层级变强。title-only的正Δ/Gap仍存在，层级现象不是只在加入类目后出现。这里只作固定样本描述，没有对两输入差值额外显著性检验。',
 '', '## Qwen与LETTER如何比较','',
 'Qwen在两个域的1→2正增量比例更高，LETTER的平均Δ和Gap更大；LETTER可评估完整路径的单调比例也更高，但路径更少、第三层覆盖显著较低。它们的树形、分组大小和宏平均分母不同，不能据这些表宣称LETTER整体显著优于Qwen。两者共用当前商品集合/向量，但Qwen上游原先在更大商品宇宙中训练，此历史差异仍保留。',
 '', '## 高频路径（预按第三层商品数排序，不筛掉反例）','',
 '| 数据集/SID | 前缀 | 商品数 | S1→S2→S3 | 单调 |','|---|---|---|---|---|']
 for r in rows:
  path=DEST/'run/analysis'/r['dataset']/r['method']/'fusion_complete/top20_proper_paths.jsonl'
  for line in path.read_text(encoding='utf8').splitlines()[:3]:
   p=json.loads(line);lines.append('| '+r['dataset']+'/'+r['method']+' | '+','.join(map(str,p['prefix']))+' | '+'→'.join(map(str,p['counts']))+' | '+'→'.join(f'{v:.6f}' for v in p['cosines'])+' | '+('是' if p['monotonic'] else '否')+' |')
 lines+=['','Beauty Qwen高频[181,51,198]为反例：0.923266→0.971916→0.970244，第三层略降。完整Top20和全部路径保留，不以成功个案替代全域统计。',
 '', '## 全量回退与旧实验','',
 'Beauty主分析排除7件无有效title商品，回退补充覆盖12101；Instruments主分析排除261件无categories商品，回退补充覆盖9922。四组全量回退Δ12仍为正（详见metrics.csv），无方向翻转；回退向量语义组成不同，因此与双字段主分析分开。旧title/title+brand的Instruments10611商品结果不与当前9661主分析绝对值混称同条件变化。',
 '', '## 结论','',
 '固定LLM2Vec编码空间中，两套SID均有多数父子边的语义集中增益和同分支优势，且超过保组大小的随机对照，支持统计语义细分结构。Qwen上游已用过Categories，本轮是包含平台类目信息的一致性验证；单独不足以证明脱离平台类目的独立语义树。覆盖、单种子和不同树的分母限制必须与主结果一并报告。',
 '', '本报告由原all_results.json及原路径明细生成；发布阶段CPU复算范围见verification/cpu_verification.json，不是重新生成embedding或重新训练SID。']
 (DEST/'analysis.md').write_text('\n'.join(lines)+'\n',encoding='utf8');print('analysis report generated')
if __name__=='__main__':main()
