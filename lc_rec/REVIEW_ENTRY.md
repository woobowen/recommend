# 本轮审核入口：PARTIAL，原版 GPU 预跑 BLOCKED

日期：2026-11-08。一次性任务，已完成当前可独立推进的准备、真实 CPU 核查与失败定位；不把未执行的 GPU 项目标为通过。发布与 ZIP 的最终完整 SHA、回读和解包结果在包内 `final_receipts/` 及 ZIP 外侧回执给出，本文不嵌入自己的最终提交。

## 结论与停止边界

| 项目 | 真实状态与依据 |
|---|---|
| 三条 GRID 分支迁移 | PASS，GitHub 原生 Rename；完整前后 SHA 和真实 ref 集合见 `evidence/branch_migration.json`、独立 `review/github_independent_readback.json`。 |
| LC 分支与原件准备 | PASS（身份/静态范围），从迁移前远端 wbw 精确基点创建；41 份官方原件逐 Git blob 对照，无原件修改。 |
| 官方 Instruments 资产 | 文件齐备且内容身份 PASS；语义配对 PARTIAL/阻塞：辅助数据与已发布交互 split 不一致，不能直接进入干净复现。 |
| CPU 数据/tokenizer/监督 | 六任务各3条真实样本监督通过；整体作业 FAILED，退出2，360项检查中2项辅助配对失败。不是完整 CPU PASS。 |
| 原版全参 GPU 预跑 | BLOCKED，模型状态下界超双4090；0次模型加载/前后向/optimizer update/保存/重载/生成，GPU计算时间0。 |
| 独立审核 | 真实非作者已查原件、代码、资产与日志；核查证据一致性通过不等于完整复现通过，也不是网页 GPT 二次审核。 |
| 正式五 seed | 仅候选配置与预算推导；0次启动、0次排队，没有自动后续训练任务。 |

严格保留本轮范围：Instruments＋huggyllama/llama-7b。未运行 Qwen、其他域、Shuffle、MLP、注意力图、RL、CoT、新 Loss 或上游重建。发现资产和资源问题后不通过删任务、改映射、量化/冻结/LoRA、offload 或减少 beam 换取通过。

## 最重要的阻塞：作者资产配对

官方 Drive 实际提供 `Instruments.zip`，13,612,638 bytes，SHA-256 `96aad0cf1496a8ffef31bf1ca3a0d1033ca49473dff46b2c2dc27763224f9453`。本轮真实下载、CRC校验、SCP传输及服务器逐文件hash闭环，不再只是一个入口。六个必要载荷的路径、大小、hash、映射轴和任务消费关系见 `evidence/cpu_audit/output/real_data_audit.json`、`evidence/cpu_input_asset_hashes.json`、`evidence/assets/ASSET_AUDIT.md`。

实际有24,772用户、9,922物品、206,153交互；与论文24,773/9,923的细小数量差异保留，不调整数据凑论文。title/description/brand/categories四字段均保留；description为空421、brand为空83、categories为空261。最短定向样本存在仅监督EOS的空description回答，保留原始行为，不填内容或删样本。

官方六任务构造共221,225条：seqrec131,837、item2index9,922、index2item9,922、fusionseqrec20,000、itemsearch24,772、preferenceobtain24,772。采样仍是 `1,1,1,1,1,1` / `0,0,0,20000,0,0`；0保留全量语义。全量构造验证字符串49,544条仅为原loader行为，实际只collate32条固定验证样本，没有loss验证或全量模型计算。

逐uid与公开处理源码对照：train intention应为目标`inter[-3]`、原始历史`inter[:-3]`，test应为`[-1]/[:-1]`。训练辅助12,233/24,772不匹配；测试辅助7,473/24,772不匹配。训练辅助目标中3,608为当前split验证目标，1,623为测试目标；这些目标没有在相同用户SeqRec训练prefix中重复出现。实际ItemSearch可见最近20件历史仍有1,623条包含该split验证物品。两个独立实现得到同样结果；所有用户完整inter均无重复item。

这些是本发布包与该split的字段/输入身份事实，不证明论文历史run的因果原因。inter无原始timestamp；自然语言辅助生成manifest、作者历史版本和上游embedding/量化checkpoint绑定缺失。不能由这份包推断历史泄漏原因或确认所有标签文字内容来自什么上下文。公开辅助生成脚本还有test字典条件问题，而实包test非空，进一步说明不能逐字绑定公开处理链。详见 `evidence/aux_pairing/output/auxiliary_pairing.json`、`review/auxiliary_pairing_independent_round1.json`、`review/auxiliary_visible_input_independent_round1.json`、`evidence/source_audit/aux_source_followup.md`。

SID均四位，实际词表按位置162/256/256/256，共930个新增token，tokenizer从32,000扩至32,930；全映射有记录。9,922item只对应9,897个canonical SID，有23个碰撞组、25个额外item。保留原映射，SID反解返回其所有item，不追加第五码、不任意选一个item冒称唯一解码。作者发布资产来源可确认，但没有可唯一核验的上游LLaMA embedding/RQ-VAE生成记录；不重训上游。

## 模型、环境与资源

固定模型/tokenizer revision `4782ad278652c7c71b72204d462d6d01eaaf7549`。`tokenizer.model`真实SHA与HF LFS `9e556afd44213b6bd1be2b850ebbbd98f5481437a8021afaf58ee7fb1818d347`一致。配置/许可/模型卡已取；权重只核API/HEAD身份，没有下载或加载。两片safetensors总13,476,876,661 bytes，SHA在 `evidence/assets/hf_summary.json`；未取重复bin或作者delta。公开访问不构成额外许可授权。

现场：bwwu@zwlab-01，2×RTX4090，每卡24,564MiB，初末均空闲；CPU为双Xeon Gold6133/80逻辑核，RAM总265,490,894,848 bytes，结束可用256,957,255,680 bytes。/data容量7,937,302,376,448、结束可用3,324,502,056,960 bytes，远高于10%预留。原始输出见 `evidence/remote_initial.txt`、`evidence/remote_final_resources.txt`。

按实际930新增token，静态全参N=6,746,034,176。只算两个FP32 Adam状态的ZeRO3两卡分片已25.13GiB/卡，超过实际23.99GiB；12bytes/param下界37.70GiB/卡，含FP32主副本16bytes约50.26GiB/卡，DS18bytes＋largest-layer估计约57.05GiB/卡，尚未覆盖激活/通信/碎片/保存/原代码FP32整模上卡加载。没有必要制造OOM。`evidence/resource_estimate.json`是静态推导，绝非GPU峰值实测；micro1×accum64只能保全global128，不能消除参数状态不足。

环境为 `/data/bwwu/envs/lc-rec-20261108`，Python3.10.22；只读复用GRID的torch2.6.0/CUDA12.4组件，venv内固定transformers4.31.0/tokenizers0.13.3/huggingface-hub0.16.4/sentencepiece0.1.99/numpy1.24.4/safetensors0.3.3。accelerate/deepspeed未安装，未声称其实际展开/Trainer训练接口已通过。作者只明确torch1.13.1+cu117，其余依赖未锁；本轮是CPU审核环境，不冒称作者历史训练环境。原GRID的transformers4.47.0/numpy2.0.2等仍保留，pip日志明确没有卸载外部环境文件；精确import位置、freeze和前后版本见 `evidence/asset_preparation/output/`、`evidence/final_details.json`。

本轮服务器受控目录初始不存在，主要新增目录结束逻辑合计约367MB，实际分配约386MB（不含另列的轻量runner job目录）。这是结束占用观测，不是逐秒采样的峰值；保存的payload/安装流程均远低于200GiB上限，WSL新增亦低于5GiB。没有模型/大checkpoint写入。安装/下载计时独立于GPU预算。未删除旧GRID、档案或临时件。

## 真实作业与版本

| job_id | source commit | 结果 |
|---|---|---|
| lc-rec-20261108-assets-r0 | `89e10760f5cac6b29249633347c98f2be35b760d` | FAILED exit1；服务器Google Drive网络不可达，约90.7秒；失败原样保留。 |
| lc-rec-20261108-assets-r1 | `53a0f618fc68dc2c936b4e5e10e1b5f0bbf282ad` | SUCCEEDED exit0；同源WSL有界下载＋SCP、离线wheel安装，准备程序32.88秒。 |
| lc-rec-20261108-cpu-audit-r1 | `53a0f618fc68dc2c936b4e5e10e1b5f0bbf282ad` | FAILED exit2；CPU审核68.37秒，2个真实配对断言失败，其余358项通过。 |
| lc-rec-20261108-aux-pairing-r1 | `247adcee4ae881ce6df264b61da66db1d61c65ba` | SUCCEEDED exit0，仅只读分类完成；不能覆盖数据配对失败。 |

主CPU审核的 `RUN_COMMIT=53a0f618fc68dc2c936b4e5e10e1b5f0bbf282ad`；其他辅助run身份逐项如上。服务器固定release位于 `/home/bwwu/lc-rec/releases/<完整commit>/`，为无.git导出，`RUN_VERSION.json`逐文件映射已核，非checkout。持久job位于 `/home/bwwu/grid-audits/sid-extensions-p0/jobs/<job_id>/`，复用已验收runner（SHA398768a732bbfebde4308486a0cedd924595157fda6cbff6a88f0bfaa512beff），不是新调度平台。所有本轮作业终态且cleanup_verified，没有后台训练队列。

每个证据子目录含真实request、accepted/status、work.log、output、export-manifest；外部local retrieval回执和完整部署清单随包交付。服务器首次断网后未重试原ID或改网络，保留失败并建立r1。同源relay不改变数据或方法。

六任务各3条真实定向记录共18条，选择处理后字符最长/最短及固定首条，不按指标选样；实际最长被查full input为841tokens，不保证其是全局token最长，也没有2048长序列GPU验收。BOS1/EOS2/pad=UNK0、response mask、padding、SID单token和最近20件历史均留IDs/labels/mask证据。32固定验证只collate；固定16用户为numeric uid最前16，覆盖17模板、1,360次原callback位置检查。构造的非canonical组合只是逻辑probe，不是模型生成。CPU峰值RSS1,644,838,912 bytes，CUDA上下文未初始化。梯度/更新、真实trainable范围、embedding/head指针、完整权重保存重载与指标都未执行，不能标PASS。

## 官方实现差异与正式候选

原件仍逐字不改；新增的是CPU审计、资产准备、只读分类、候选argv渲染与ZIP验证脚本。`evidence/engineering_adaptations.patch`列新增脚本/配置diff，官方正文的身份见manifest。不把 `run.sh`多余空格、初始化顺序、OOM降beam等问题静默修回原件；候选argv规避了整份三域shell及错误赋值，但没有声称训练加载适配已实现或验收。

严格全目录合法SID与原position-token集合约束冲突已经真实callback/probe确认；原evaluator对SID字符串计分、默认filter_items=False、不做canonical item去重；不能暗中换trie或指标规则。下一阶段须明确沿用原实现还是批准有标注的严格目录偏离。全目录/beam20/17模板原指标候选写在配置，实际没有beam search或评价数值。

`config/formal_five_seed_candidate.json`保留4epoch、AdamW5e-5/wd.01/cosine/warmup.01、六任务、BF16/ZeRO3/checkpointing、global128、20items/2048tokens、按epoch验证保存、按较低eval_loss选best；两卡布局micro1/accum64为候选，不是数值轨迹逐位等同。依赖静态推导beta=.9/.999、eps1e-8、clip1，新增SID行采用4.31旧Normal初始化（非mean_resizing）；实际模型权重初始化/DS auto解析未验证。当前CPU dataset参数dump中的lr2e-5等是未使用的parser默认，不能当实际训练值。

DS auto静态展开micro1/accum64/global128，BF16true、clip1、reduce_bucket16,777,216元素、prefetch15,099,494.4（DS最终整数解析未知）、persist40,960；原DS没有offload/optimizer/scheduler节，原profile_step10仍保留。正式当前数据长度条件下4.31推导每epoch1728updates、每seed6912、warmup70、五seed34,560，须真实dataloader/Accelerate回读，不是已运行scheduler。耗时为未知；预算式为5×(6912×实测update秒数＋4轮验证/保存＋最终保存/17模板test)，没有用短跑承诺正式总时长。

单套16bytes/param状态约107.94GB，单状态＋FP32导出＋base权重静态约148.40GB；五套状态约539.68GB，原save_total_limit5也可能超本轮200GiB。因此未来正式保存/空间策略需要另审；本轮未保存checkpoint，不改正式候选来挤预算。有界预跑候选最多10updates、32validation、16users×17templates×beam20、单次60分钟/累计120分钟，全部尚未执行。

后续方向仅记录，不执行：Instruments原LLaMA seeds42–46，同一套确认资产，每seed先汇总17模板，再算五seed均值与样本标准差ddof1；审核后再Qwen3-8B，其他四域主干随后决定；原论文三域优先作者资产，Beauty/Toys优先旧GRID对应Amazon2014；未来Shuffle只在推荐时打乱每件历史物品内部全部SID位，保留物品时间顺序与canonical目标、训练不扰动；MLP/注意力图暂缓。

## 分支、规则与审核

| 原真实分支 → 新分支 | 改名前与改名后相同的完整 SHA |
|---|---|
| wbw → wbw-grid | `9767729a5114e21b9c949892af3d759baca6a98e` |
| wmh → wmh-grid | `7e943ea9a7d92f16eb10c6c1b27c27761d3f6c20` |
| yjc → yjc-grid | `c92e1576210b3daf2ce6f63f77b65f651f6f7127` |

LC创建基点精确为上述迁移前wbw SHA。main仍为 `563dc50b1c7e9e2e05cdfddd13e91a3d1888001e`，default=main、visibility=public；没有修改安全设置或往GRID分支附加提交。本地原GRID工作树路径/HEAD及6个原未跟踪pycache保留，wbw-grid upstream更新；原无本地wmh/yjc，未补建。历史本地缓存origin/yinjingchen未作为本次yjc改名对象操作；仅精确清掉已验证失效的origin/wbw、origin/wmh缓存。详见 `evidence/local_branch_tracking.json`。

权威原文件 `/home/addaswsw/recommend/{AGENTS.md,PROJECT_RULES.md,REMOTE_EXECUTION.md}` 已原处最小更新，并有2026-11-08分支条款修订注记。`rules/`只存冻结审核快照/patch，不作为另一套持续维护规则。网页项目源未被修改，用户需随后替换三份副本。

真实审核者：`/root/independent_review`非实现作者，独立回读GitHub refs、重新计算41原件blob，从同源ZIP另行读取并重算6资产hash/配对/SID统计，审运行request/status/日志/18样本mask与32验证/16×17callback证据；`/root/official_audit`另查原源码/论文/依赖/处理链。`/root/asset_audit`是CPU审计脚本作者，其元数据报告不当作该脚本非作者验收。最终独立报告见 `review/`，不声称网页GPT已二审。

round0：现场/官方原件/初始服务器下载，网络失败保留。round1：最小同源离线传输适配、CPU真实核查及只读不一致分类，未修研究方法或资产。round2：交付审核要求的ZIP字节对HEAD绑定、秘密pattern与候选阻塞说明完善，运行代码/资产不变，因此不重复CPU执行。没有round3，也没有消耗GPU预算去“修到通过”。

## 下一步首选与安装记录

首选先向作者取得/确认与本次交互split、item轴一致的一整套辅助资产和生成来源，原包保留作失败证据；禁止用重排、删用户、填空或换模型重生成来消除错配。确认资产后，优先申请足够的无offload资源（例如2×80GiB）保持方法；当前双4090备选CPU optimizer/master offload需另批准，约100.52GiB状态、1.5倍余量约150.79GiB CPU，PCIe/编译/耗时未测且执行路径不同。还需集中确认position约束与严格合法目录要求如何处理。上述任何批准都不在本轮代为做出。

新增系统包/驱动/全局框架/工具链：无。venv内新增六个固定包以及ensurepip的pip23.0.1/setuptools79.0.1，完整安装日志保留；WSL仅下载wheel，无pip安装。配置改动仅本任务文件、venv pyvenv.cfg和三份已授权规则；无.bashrc/VPN/代理/路由/SSH修改。保留环境、同源暂存和正式审核包，未执行清理；用户可另行指定是否清理一次性组件。
