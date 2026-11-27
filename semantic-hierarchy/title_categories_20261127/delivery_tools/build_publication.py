"""Build a readable, scoped overlay on the existing semantic hierarchy branch."""
from pathlib import Path
import csv,hashlib,json,shutil
HERE=Path(__file__).resolve().parent;REPO=HERE/'repo';DEST=REPO/'semantic-hierarchy/title_categories_20261127'
TAG='sid-title-categories-20261127'
def load(p):return json.loads(p.read_text(encoding='utf8'))
def copy(src,dest):dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dest)
def main():
 DEST.mkdir(parents=True,exist_ok=True)
 for p in (HERE/'metadata').rglob('*'):
  if not p.is_file() or '__pycache__' in p.parts or p.name.startswith('._') or p.suffix in ['.pyc','.safetensors']:continue
  rel=p.relative_to(HERE/'metadata')
  if rel.as_posix()=='MANIFEST.json':continue
  copy(p,DEST/rel)
 for n in ['verify_outputs.py','fetch_assets.py','prepare_runtime.py']:copy(HERE/n,DEST/n)
 (DEST/'verification').mkdir(exist_ok=True)
 for n in ['cpu_verification.json','export_receipt.json']:copy(HERE/n,DEST/'verification'/n)
 (DEST/'delivery_tools').mkdir(exist_ok=True)
 copy(HERE/'export_server.py',DEST/'delivery_tools/export_server.py')
 specs=[]
 for name in ['fusion-metadata-originals.zip','fusion-beauty-vectors.zip','fusion-instruments-vectors.zip']:
  s=load(HERE/(name+'.json'));s['url']=f'https://github.com/woobowen/recommend/releases/download/{TAG}/'+name;specs.append(s)
 (DEST/'assets.json').write_text(json.dumps(dict(release_tag=TAG,assets=specs),ensure_ascii=False,indent=2)+'\n',encoding='utf8')
 allresults=load(DEST/'run/all_results.json');primary=allresults['primary'];keys=['delta_12','positive_12','delta_23','positive_23','path_monotonic','gap_12','gap_23']
 with (DEST/'metrics.csv').open('w',encoding='utf8',newline='') as f:
  w=csv.writer(f);w.writerow(['dataset','method','variant','items','S1','S2','S3']+keys+['L2_coverage','L3_coverage','proper_paths','holm_p12'])
  for r in allresults['all_variants']:w.writerow([r['dataset'],r['method'],r['variant'],r['items']]+[r['node_macro_cosines'][str(d)] for d in [1,2,3]]+[r['observed'][k] for k in keys]+[r['coverage']['2']['coverage'],r['coverage']['3']['coverage'],r['proper_paths'],r.get('holm_p_value','')])
 copy(DEST/'run/指标.md',DEST/'metrics.md')
 (DEST/'README.md').write_text('''# Title＋Categories：共同商品轴上的 SID 语义层级实验

维护：Jingchen Yin（Justin-hub-coder）。正式run `20261127_014125`，北京时间2026-11-27 01:41启动、01:54完成，退出0。Beauty原GPU0/1、Instruments原GPU2/3；本次发布只做CPU保存向量复算和归档，不重新编码、训练或改SID。

研究问题：同一个SID共享前缀加深时，商品语义是否更集中，同父节点的不同分支是否能区分。Qwen和LETTER共用同商品、文本、向量及有效掩码，仅SID树不同。Instruments已从Qwen原10611商品筛选/重排至LETTER共同9922商品，不再把两种数据宇宙作纯方法比较。

## 结果导航

- [指标表](metrics.md)、[12组全精度CSV](metrics.csv)、[完整原始汇总](run/all_results.json)、[分析报告](analysis.md)。
- [所有节点、父子边、完整路径、分支Gap、Top20及置换记录](run/analysis)：4个主分析＋8个补充。
- [输入身份与缺失掩码](run/input_audit.json)、[冻结实际配置](run/effective_config.json)、[编码配置](run/encoding_config.json)、[真实命令](run/beauty/command.json)。
- [CPU全量核验](verification/cpu_verification.json)、[资产清单及下载](assets.json)、[复现四种起点](reproduction.md)。

| 范围 | 共同商品 | 双字段主分析 | 单字段回退 |
|---|---:|---:|---:|
| Beauty | 12101 | 12094 | 7 |
| Instruments | 9922 | 9661 | 261 |

## 方法与实际调用

1. [prepare_experiment.py](run/source/prepare_experiment.py)：先构建title/categories、缺失标记、唯一文本缓存，再打开SID；同域两树共用相同文本。Beauty原TFRecord提取类别并通过已核ASIN/商品编号；Instruments使用[作者item2id与ASIN全量映射](alignment/README.md)，Qwen筛选码值逐件不变。
2. [encode_fields.py](run/source/encode_fields.py)调用[original_encoder.py::load_encoder](run/source/original_encoder.py)：Meta-Llama-3-8B-Instruct＋LLM2Vec MNTP adapter合并＋supervised adapter；双向注意力、正文masked mean、4096维、FP16两卡切层，float32保存。未用Llama3.1、未做本数据指令微调、未读SID、未量化向量。
3. title与categories分别编码得到未经字段L2的t、c，融合z=normalize((t+c)/2)。缺字段回退仅用于fusion_all_available；fusion_complete只用两字段有效商品。title_complete使用同一主分析掩码，隔离字段变化；完整序列化文本见[inputs](run/inputs)。相同类别文本只编码一次，不合并商品。类别信息来自平台原件，未人工生成标签。
4. [analyze.py](run/source/analyze.py)→[semantic_metrics.py::TreeMetrics](run/source/semantic_metrics.py)：前3位、组内真实不同商品平均两两余弦；n>=2、排除自比较，float64向量和恒等式精确计算。Qwen第4位去重码与LETTER第4量化位都不用于本次三层统计。
5. [launch_fusion.py](run/source/launch_fusion.py)：冻结源/输入/模型hash、预跑、两GPU worker双锁、服务器独立监督，完成后CPU统计。状态/子进程真实退出码保留，旧绝对路径为历史证据；新机器不得直接运行旧launcher，应按[可移植入口](reproduction.md)。

## 指标与预先固定口径

S(G)=sum(i!=j)z_i·z_j/[n(n−1)]。每层S对可评估节点宏平均；不同层节点集合不同，S2−S1不等于平均父子增益。Δ=S(child)−S(parent)，仅2<=nchild<nparent；正增量Δ>1e−6。完整路径要求n1>n2>n3>=2，S1<S2<=S3（epsilon1e−6）。Gap是父集合内同子分支商品对均值减跨子分支商品对均值，商品对内部加权、最终父节点宏平均。

覆盖率是位于至少两件商品前缀组中的不同商品数/当前分析商品数，不是推荐准确率。第三层主分析覆盖：Beauty Qwen18.98%/LETTER5.57%，Instruments Qwen21.36%/LETTER5.57%；主要证据放1→2，第三层只代表可评估子集。高频路径按第3位商品数预排序，不按余弦好坏筛选，反例保留。

主分析999次父节点内保组大小条件置换，2000次第一层子树簇bootstrap；4个dataset×SID的Δ12主检验统一Holm（均0.004）。这些p检验相对各自随机对照，不是Qwen与LETTER差值检验；相同数据/掩码不等于相同分组大小/可评估范围，不宣称某方法整体更优或跨训练种子显著。

## 研究结论边界

四组均表现出正的平均父子增益、正Gap和多数路径递进，支持编码空间中的统计语义细化，不证明显式的人类类别树，也不是每条路径都单调。Qwen上游用过Categories，故本轮包含共享类目信息的保留效应，不能单独当作完全独立于上游输入的证据；title-only同掩码对照随包保留。加入Categories提高绝对余弦，但Δ/Gap较title-only缩小，不能把高余弦等同于更强层级。

Qwen是在原10611商品宇宙训练后筛到共同9922；LETTER是在作者数据宇宙学习的固定SID。本轮不重训上游，因此只控制当前评估商品集合，不声称上游训练数据也相同。LETTER完整SID碰撞保留，评价单位仍是不同商品。

## 资产、来源、三阶段记录

原run是服务器无Git导出的固定快照；[launch.json](run/launch.json)15文件SHA256是实际执行身份，本次Git提交仅为发布版本。基座/adapter实际核验见[model_verified.json](run/model_verified.json)。基座不重托管；固定HF/ModelScope来源、取得条件和hash见[reproduction.md](reproduction.md)。两个LLM2Vec adapter及实际tokenizer/配置放metadata Release，所有原NPY按域发布并逐成员校验。

前置独立审核[launch_review](history/launch_review.md)，启动实际非作者复核[launch_runtime_review](history/launch_runtime_review.md)，实验中[启动记录](history/启动记录.md)与原console/status，实验后[status](run/status.json)、原全结果和CPU复算。本轮旁支不允许调用子代理，因此没有新发布阶段非作者审核；不将前置独立审核冒称最终发布审核，结果发布由主控自检。该范围限制不会改写任何原始记录。

失败尝试[014041](attempts/failed_014041/status.json)因原Beauty标题换行导致准备正则断言，尚未加载GPU；修正只允许跨换行，新attempt014125成功。原失败源码/日志保留。

归档输入/输出/映射及必要调用均提供；旧Qwen metadata264MB未重复入包，原件仍留服务器，实际新编码文本与完整ASIN链已提供。原模型与Amazon/作者数据各自遵守来源许可和权限；本次不自行给第三方材料声明额外许可证。
''',encoding='utf8')
 (DEST/'reproduction.md').write_text('''# 复现与资产取得

## 1. 保存输出复算（本轮已实际完成）

在本目录运行`python fetch_assets.py --output /ABS/RESTORED`。下载3个ZIP：metadata约185MB、Beauty向量约679MB、Instruments约559MB；包含原字段/raw/归一化/缓存/掩码/norm，ZIP展开后约2.1GB。完整缓存＋展开＋独立runtime建议预留7GB。每ZIP和每成员的长度/SHA256、CRC和安全路径均核验，拒绝覆盖不同原件。

`python verify_outputs.py --run /ABS/RESTORED/run --output /ABS/check.json`，CPU重新计算12组所有节点/边/路径/Gap、覆盖、2000次簇bootstrap区间，核保存的7992条置换统计与p/Holm。此脚本不重新加载大模型、不重新编码，不重新产生置换随机分配。依赖numpy2.0.2兼容环境。

`python prepare_runtime.py --archive /ABS/RESTORED --output /ABS/NEW_STATS`创建独立源/输入/向量副本；`python /ABS/NEW_STATS/source/analyze.py --run /ABS/NEW_STATS`会从原向量重新执行999次条件置换及全部分析，输出新analysis目录。该全置换重跑入口静态/原方法已验证，本次发布没有为凑验证重复运行同样全部置换；原实验已实际执行过。本轮CPU核验边界见verification/cpu_verification.json。

## 2. 固定模型重新编码（材料具备，本轮未重评）

原基座`meta-llama/Meta-Llama-3-8B-Instruct`，原兼容镜像`LLM-Research/Meta-Llama-3-8B-Instruct`的ModelScope revision `ccdc34064d2ac9f2d99d1a6885333773a6f12c09`；HF镜像`NousResearch/Meta-Llama-3-8B-Instruct` revision `53346005fb0ef11d3b6a83b12c895cca40156b6c`。原辅助配置存在不同镜像提交，必须以run/model_verified.json逐文件身份为准，不能只按模型同名视为字节相同。基座4个权重分片约16GB，需来源访问/许可；如需HF token只在自己环境提供，不写入仓库。实际使用的非权重配置及tokenizer已放metadata ZIP的encoder_assets/models，取得权重后按清单组合核验。固定公开模型来源可读不代表所有地区/账户都无权限限制。

LLM2Vec MNTP：`McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp@31474e395ada192e8ed1586db6be79fb3b70c9c0`；supervised：`McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised@baa8ebf04a1c2500e61288e7dad65e8ae42601a7`。两adapter实际bytes均在metadata ZIP，不必另下载；hash见model_verified。

关键包实际版本见run/environment.json：Python3.10.20、torch2.6.0+cu124、transformers4.44.2、peft0.12、llm2vec0.2.3、numpy2.0.2、accelerate1.8.1。CUDA/硬件组件按目标Linux安装；不要直接装旧本地路径freeze。FP16需两张足够空闲的11GiB设备，不能将8B模型塞到单张2080Ti。

`python prepare_runtime.py --archive /ABS/RESTORED --output /ABS/NEW_ENCODER --base-model /ABS/BASE`全量核验基座，复制实际adapter、文本和原配置，并只在新source修改ROOT/基座路径，记录relocation.json，原运行不改。选两张空闲卡（按服务器规约持锁），例如`CUDA_VISIBLE_DEVICES=0,1 python /ABS/NEW_ENCODER/source/encode_fields.py --run /ABS/NEW_ENCODER --dataset beauty`，Instruments同理。脚本会先最长文本/一致性预跑再编码。两域完成后`python /ABS/NEW_ENCODER/source/analyze.py --run /ABS/NEW_ENCODER`。不得直接运行旧prepare_experiment/launch_fusion的服务器绝对路径。

本轮移植源只做路径/配置验证，不启动新的GPU编码；不承诺跨硬件浮点逐位相同。原两域并行编码各约10分钟，CPU分析约2分钟，属本次观测，不能当别的设备保证。

## 3. 重跑实验

固定SID＋新编码＋原统计构成完整本实验；没有推荐训练/分类器微调阶段。按第2节从保存的实际文本开始，配置与输入hash足以复现本次起点。原始商品文本重建脚本保留，但依赖原Beauty TFRecords、映射及旧上游目录，不能直接在其他机器执行旧绝对路径。

## 4. 上游重建边界

本次未重新学习Qwen/LETTER SID；精确原SID表已随run/inputs提供。Qwen原上游代码和权重/TFRecord可从本分支既有GRID baseline与固定yjc-grid@6200d5875e264b5e6ec9841cf2cdb3b4a12d71a4读取；LETTER固定公开SID来自HonghuiBao2000/LETTER@8d0154e28de37dbb6e24871c508ad8ddb1921cda。不能把对齐后Qwen SID说成在9922商品上重训结果。

Instruments严格映射所用作者原item2id/user2id、下载来源与全图核验随alignment提供；不按标题猜配。Beauty全量sid-column/grid-item/P5-item/ASIN映射在run/source/beauty_g3_item_mapping.csv，历史完整交互轴核验依据在identity/beauty_mapping_audit.json。本轮从保存向量复算与商品轴检查，不冒称重新训练上游或完整重跑历史推荐测试。
''',encoding='utf8')
 # Append navigation to exact current branch originals; preserve pre-existing bytes.
 for name,line in [('README.md','\n## 2026-11-27 Title＋Categories 语义层级实验\n\n[完整实验与结果](semantic-hierarchy/title_categories_20261127/README.md)：Beauty及对齐9922商品的Instruments，Qwen/LETTER共用向量；12组分析、原始向量、身份映射、配置和复现材料。\n'),('semantic-hierarchy/README.md','\n- [Title＋Categories：Qwen/LETTER共同商品轴实验](title_categories_20261127/README.md)（2026-11-27，4主＋8补充分析）\n')]:
  dest=REPO/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes((HERE/'before'/name).read_bytes()+line.encode())
 print(DEST)
if __name__=='__main__':main()
