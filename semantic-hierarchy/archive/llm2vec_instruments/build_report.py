"""Write a reproducible Chinese Markdown report with aligned full-border tables."""
import argparse
from pathlib import Path
from wcwidth import wcswidth
from runtime_common import read_json,read_jsonl,write_json


def table(headers,rows):
    rows=[list(map(str,headers)),*[list(map(str,row)) for row in rows]]
    widths=[max(wcswidth(row[i]) for row in rows) for i in range(len(headers))]
    def border(a,b,c):return a+b.join('─'*(w+2) for w in widths)+c
    def rowline(row):return '│'+'│'.join(' '+value+' '*(widths[i]-wcswidth(value)+1) for i,value in enumerate(row))+'│'
    result=['```text',border('┌','┬','┐'),rowline(rows[0]),border('├','┼','┤')]
    for i,row in enumerate(rows[1:]):
        result.append(rowline(row))
        if i<len(rows)-2:result.append(border('├','┼','┤'))
    result.extend([border('└','┴','┘'),'```',''])
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',required=True,type=Path);args=parser.parse_args();run=args.run
    assert read_json(run/'final_verification.json')['passed']
    prep=read_json(run/'input_audit.json');n=prep['retained_items']
    summaries={v:read_json(run/'instrument'/v/'evaluation/summary.json') for v in ['title','title_brand']}
    write_json(run/'combined_results.json',{'dataset':'instrument','results':summaries,
               'primary_p_value':summaries['title']['main_p_value'],
               'primary_test_family':'one newly specified instrument/title/delta12 test; no multiplicity adjustment',
               'no_retroactive_pooling_with_beauty_toys':True})
    main=summaries['title'];o=main['observed'];ci=main['cluster_bootstrap_95ci']
    lines=['# Musical Instruments：SID 语义层级实验结果','','实验数据版本：Amazon Reviews 2018 / amazon_v2；来源：UCSD 官方 5-core 与商品元数据。',
           '',f'运行编号：`{run.name}`。全部编码、SID 生成、条件随机对照、bootstrap 和结果复核已完成。',
           '', '## 1. 实验范围和数据对齐','',
           '第一层→第二层为主要证据；第三层只评价至少两件商品的节点，并明确报告覆盖率。不运行商品数门槛敏感性实验。',
           '',f"官方 5-core 原始评论 {prep['raw_review_records']:,} 条，用户 {prep['raw_users']:,} 个，不同商品 {prep['raw_distinct_items']:,} 件。元数据实际读取 {prep['metadata_records']:,} 条（记录数不自动视为不同商品数）。",
           '',f"官方名为 5-core 的文件实际仍有 {prep['raw_users_with_under_5_records']} 个用户、{prep['raw_items_with_under_5_records']} 件商品不足 5 条记录（最小分别为 {prep['minimum_raw_user_records']} 和 {prep['minimum_raw_item_records']}）。本次是商品语义评价，保留官方文件商品范围，不额外重做迭代 5-core。",
           '',f"按 ASIN 关联后保留 {n:,} 件商品、{prep['retained_review_records']:,} 条交互；缺少元数据 {prep['metadata_missing_items']} 件，空标题 {prep['blank_title_items']} 件。重复元数据额外记录 {prep['duplicate_metadata_extra_records']} 条，按是否有标题、描述/特征丰富度及固定字典序选择，选择不依赖 SID。",
           '',f"原评论中重复用户–商品交互额外记录 {prep['duplicate_user_item_extra_records']} 条；清洗后重复标题额外商品 {prep['duplicate_title_extra_items']} 件。商品语义实验保留不同 item_id，不为去重或重新筛选 5-core 另启实验。",
           '', '## 2. 编码、量化与 GPU','',
           '- Qwen2.5-7B-Instruct：使用已有本地完整 FP16 模型，最后一层隐藏状态、原 WMH MeanAggregation、右填充、长度 128、3584 维 float32 输出，不微调模型。',
           '- Qwen 文本延续 Beauty/Toys 的 Title、Brand、Categories、Price 序列化。描述和特征字段下载后保留，本轮不编码，以保持标题主实验口径。',
           '- 原始 RQ-KMeans：3 层语义码，每层宽度 256，seed=42，30 步，逐层归一化输入/残差；第四位仅消除重复 SID。新数据集重新训练码本，未复用其他数据集 SID。',
           '- 独立 LLM2Vec：Meta-Llama-3-8B-Instruct + MNTP + supervised LoRA，双向注意力、正文 mean pooling；标题为主输入，标题+品牌为补充。4096 维向量转 float32 后 L2 归一化，统计聚合用 float64。',
           '- LLM2Vec 编码阶段只读取 item_id/text 和编码配置，不读取 SID、类别、价格或旧分类；完成后由独立评估进程关联 SID。Llama 输出不做离散量化。',
           '- LLM2Vec 固定指令：`Represent this product for semantic similarity:`，池化排除指令；max_length=256，doc_max_length=224，batch=2。',
           '- GPU 0/1 生成 Qwen 向量；GPU 2/3 同时生成 LLM2Vec 的标题及标题+品牌向量；Qwen 完成后 GPU 0 执行码本训练和 SID 生成；评估使用 CPU。未训练下游推荐模型。',
           '', '## 3. 五项主要展示指标','',
           '设归一化向量为 zᵢ，组内平均两两余弦 S(G)=2Σ(i<j)zᵢᵀzⱼ/[n(n−1)]；排除自比较，单商品节点记 NA。实际用向量和恒等式精确计算全部商品对。',
           '', '父子增量 Δ=S(child)−S(parent)。有效边要求 2≤n_child<n_parent，宏平均按有效边等权；父子同集合的边单列。正增量要求 Δ>10⁻⁶。完整路径主分母要求 n₁>n₂>n₃≥2，判定按截图 S₁<S₂≤S₃（Δ₁₂>10⁻⁶ 且 Δ₂₃≥−10⁻⁶）。覆盖率为至少两件商品节点中的不同商品数/N。','']
    rows=[]
    for label,key,pct in [('第一→第二层平均增量','delta_12',False),('第一→第二层正增量比例','positive_12',True),
                          ('第二→第三层平均增量','delta_23',False),('完整路径递增比例','path_monotonic',True)]:
        rows.append([label,*[(f"{summaries[v]['observed'][key]:.2%}" if pct else f"{summaries[v]['observed'][key]:+.6f}") for v in summaries]])
    rows.append(['第三层可评价商品覆盖率',*[f"{s['coverage']['3']['coverage']:.2%}" for s in summaries.values()]])
    lines+=table(['指标','标题主实验','标题+品牌补充'],rows)
    lines+=['## 4. 组内余弦与分支区分度','','Gap=W−B：父集合内同一直接子节点商品对的平均相似度 W，减不同直接子节点间商品对的平均相似度 B。W/B 内商品对等权，最后父节点等权汇总。Gap>0 表示同分支比跨分支更相似。','']
    rows=[]
    for variant,s in summaries.items():
        rows.append(['标题' if variant=='title' else '标题+品牌',*[f"{s['node_macro_cosines'][str(d)]:.6f}" for d in [1,2,3]],
                     f"{s['observed']['gap_12']:.6f}",f"{s['observed']['gap_23']:.6f}",f"{s['observed']['positive_23']:.2%}"])
    lines+=table(['输入','第一层 S','第二层 S','第三层 S','Gap₁₂','Gap₂₃','2→3 正增量比例'],rows)
    lines+=['各层 S 汇总的节点/商品集合不同，相邻层绝对均值的差不等于平均父子增量；逐边 Δ 才是主要细化证据。','', '## 5. 覆盖率和实际分母','']
    rows=[]
    for d in ['1','2','3']:
        c=main['coverage'][d];rows.append([d,c['all_nodes'],c['evaluable_nodes'],c['covered_distinct_items'],f"{c['coverage']:.2%}",c['maximum_node_items']])
    lines+=table(['前缀长度','全部节点','可评价节点','覆盖商品数','覆盖率','最大组商品数'],rows)
    rows=[]
    for transition,e in main['edge_summary'].items():
        rows.append([transition,e['eligible_edges'],e['positive_edges'],e['ties'],e['negative_edges']])
    lines+=table(['转移','有效真细分边','正增量','持平','负增量'],rows)
    paths=read_jsonl(run/'instrument/title/evaluation/complete_paths.jsonl');proper=[x for x in paths if x['proper_both_edges']]
    lines += [f"全部第三层共享路径 {main['shared_three_code_paths']} 条，两次均真细分的有效完整路径 {len(proper)} 条，其中满足 S₁<S₂≤S₃ 的 {sum(x['monotonic_as_image'] for x in proper)} 条。第三层节点覆盖率与真细分完整路径采用不同筛选条件，不能混用分母。",'', '## 6. 区间与随机对照','',
              '95% 区间按第一层整棵子树为簇 bootstrap 2000 次；每种条件置换 999 次。1→2 在第一层父集合内置换，保持第二/三层嵌套和节点大小；2→3 在第二层父集合内置换，保持第三层大小。','']
    rows=[]
    for key,label in [('delta_12','Δ₁₂'),('positive_12','R₊₁₂'),('delta_23','Δ₂₃'),('path_monotonic','Rmono'),('gap_12','Gap₁₂'),('gap_23','Gap₂₃')]:
        low,high=ci[key];pct=key in ['positive_12','path_monotonic']
        rows.append([label,f'{o[key]:.2%}' if pct else f'{o[key]:.6f}',f'[{low:.2%}, {high:.2%}]' if pct else f'[{low:.6f}, {high:.6f}]'])
    lines+=table(['标题主实验指标','观察值','簇 bootstrap 95% 区间'],rows)
    rows=[]
    for tag,depth in [('12',1),('23',2)]:
        c=main['random_controls'][f'within_depth_{depth}']['delta_'+tag]
        rows.append(['Δ'+tag,f"{c['observed']:.6f}",f"{c['null_mean']:.6f}",f"{c['observed_minus_null_mean']:.6f}",f"{c['p_one_sided']:.3f}"])
    lines+=table(['统计量','观察均值','随机均值','观察−随机','单侧 p'],rows)
    lines += [f"本次新增单一预设主检验为 Instrument/标题/Δ₁₂，p={main['main_p_value']:.3f}；无需在单个检验内做 Holm 校正，不将本次结果追溯并入先前 Beauty/Toys 的检验族。其他指标与输入为支持性结果。p 的计算分辨率为 0.001，p=0.001 不等于 p=0。",'', '## 7. 高频三位前缀路径','',
              '按三位前缀包含的不同商品数降序展示 Top-10，再回溯祖先；不按相似度筛选。全部 Top-20 和所有节点/边/路径另存明细。','']
    rows=[]
    for p in read_jsonl(run/'instrument/title/evaluation/top20_three_code_paths.jsonl')[:10]:
        rows.append([','.join(map(str,p['prefix'])),'→'.join(map(str,p['counts'])),'→'.join(f'{v:.6f}' for v in p['cosines']),'是' if p['proper_both_edges'] else '否'])
    lines+=table(['三位前缀','逐层商品数','S₁→S₂→S₃','两段真细分'],rows)
    lines += ['## 8. 质量复核与解释','',
              '- 下载原始文件已通过长度、gzip CRC 和逐行 JSON 解析检查，记录 SHA256 用于传输和追溯；来源未发布可独立比对的官方 SHA256。',
              '- 原 Qwen/RQ-KMeans 核心源码和配置哈希在运行前后核验不变；SID 与连续 item_id 一一对应，四位完整 SID 唯一。',
              '- 向量有限且单位范数误差小于 10⁻⁵；节点/边/路径/Gap 全量重新汇总一致；两种输入每层各取三个大节点显式计算商品对余弦核对。',
              '- 两种输入各含 1998 条条件置换记录；公式、Gap、置换、单商品与路径条件测试通过。',
              '',f"标题主实验 Δ₁₂={o['delta_12']:+.6f}，正增量比例 {o['positive_12']:.2%}，Gap₁₂={o['gap_12']:+.6f}。请结合区间和同父集合内随机对照判断，不仅依据跨层 S 的总体均值。",
              '',f"第三层可评价覆盖 {main['coverage']['3']['coverage']:.2%} 的商品；第三层结果仅适用于这一范围，不能外推到单商品节点。结果检验的是独立 LLM2Vec 语义空间中的细化，不直接证明人类类别树，也不保证每条路径递增。上游输入包含类别；独立编码器不等于训练语料完全独立。未根据结果挑选编码器、门槛或输入。",'',
              '## 9. 文件位置','','- 原始数据：`/home/jcy/data/amazon2018_instruments/raw/`。',
              f'- 本次运行：`{run}`',
              '- 配置与代码：`encoding_config.json`、`qwen_config.json`、`analysis_config.json`、`launch.json`、`source/`。',
              '- 新 SID 和 Qwen 向量：`upstream/semantic_ids.pt`、`upstream/item_embeddings.pt`。',
              '- 两种输入的 LLM2Vec 向量和结果：`instrument/{title,title_brand}/` 与其下 `evaluation/`。',
              '- 完整结果：`combined_results.json`；复核：`final_verification.json`。','']
    (run/'Instrument_语义层级实验结果.md').write_text('\n'.join(lines),encoding='utf-8-sig')
    print({'report':str(run/'Instrument_语义层级实验结果.md'),'main_p_value':main['main_p_value']},flush=True)


if __name__=='__main__':main()
