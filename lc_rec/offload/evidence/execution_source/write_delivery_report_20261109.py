import json,datetime
from pathlib import Path
repo=Path('/home/addaswsw/recommend-worktrees/lc-rec');base=repo/'lc_rec/offload'
s=json.loads((base/'results_summary_20261109.json').read_text())
c=next((x for x in s['candidates'] if x['complete_real_update_chain']),s['candidates'][0]);r=c['ranks'];peak=c.get('resource_summary',{}).get('peaks',{})
core=s['real_model_update_status'];batch=s['batch128_observed_seqrec_status'];G=1024**3
max_a=max((x.get('gpu_allocator_peak',{}).get('max_allocated_bytes',0) for x in r if x.get('gpu_allocator_peak')),default=0)/G
max_reserved=max((x.get('gpu_allocator_peak',{}).get('max_reserved_bytes',0) for x in r if x.get('gpu_allocator_peak')),default=0)/G
cpu=peak.get('cpu_rss_sum_peak_bytes',0)/G;avail=peak.get('min_mem_available_bytes',0)/G
conditional=c.get('conditional_time_substitution');tau=conditional['same_seqrec_path_seconds_per_global128_update'] if conditional else None
cost=(f"第二次真实SeqRec global128更新的两rank较慢者为 **{tau:.3f}s**（64个micro-step墙钟之和，含探针审核开销）；以完全相同SeqRec路径代入6912次，约 **{conditional['one_seed_6912_updates_hours']:.2f}小时／一个seed**、**{conditional['five_seeds_34560_updates_hours']:.2f}小时／五seed顺序运行**。这只是条件替代量，不是六任务实测吞吐、性能承诺或数学下界。" if conditional else '没有可用的完整第二次更新计时；不虚构正式耗时。')
updates=[]
for q in r:
 for u in q['optimizer_updates']:
  updates.append(f"| {q['rank']} | {u['index']} | {u['global_steps_before']}→{u.get('global_steps_after')} | {u['lr_before']} | {u['all_rank_gradients_finite']} | {u.get('global_grad_norm')} | {u.get('engine_step_wall_seconds')} | {u.get('sample_witnesses')} |")
real_pass=core=='PASS'
pressure='; '.join(f"rank{x['rank']}: {(x.get('synthetic_pressure') or {}).get('status','NOT TESTED')}" for x in r)
models=all(x.get('model',{}).get('trainable_numel')==6746034176 for x in r) and len(r)==2
forward=all(x['local_forwards']>=1 for x in r) and len(r)==2
backward=all(any(u['all_rank_gradients_finite'] for u in x['optimizer_updates']) for x in r) and len(r)==2
chain=s.get('chain_status') or {}
text=f'''# 双 RTX 4090 全参数 CPU Offload：一次性结果（2026-11-09）

**核心真实模型更新：{core}。观测到的SeqRec batch128布局：{batch}。严格1:1正式六任务复现仍BLOCKED。** 本轮调查交付与正式训练成功是不同结论；没有正式四轮/五seed、Qwen训练或推荐指标。执行合规不能标全PASS：Qwen收集器超修正额度的一次中断复跑及外层runner marker限制已明确披露。

## 验收与范围

| 项目 | 真实状态 | 依据与边界 |
|---|---|---|
| 两片LLaMA权重／revision | PASS，继承全文件SHA＋本轮轻量复查 | 13,476,876,661 bytes；原完整回执与当前size/mtime/inode、小文件身份见[evidence](evidence/preparation/model_identity_readback_20261109.json)；没有再次下载或全量hash |
| 官方源码保留 | PASS | 41份原件与已审2e0e266原字节一致；[manifest](evidence/preparation/official_preservation_20261109.json) |
| Optimizer-only完整模型加载 | {'PASS' if models else 'PARTIAL'} | 两rank真实LLaMA，930个SID token，32930词表，6,746,034,176 trainable参数，291个BF16参数对象，embedding/head untied |
| 真实forward | {'PASS' if forward else 'NOT TESTED'} | 两rank各记录 {[x['local_forwards'] for x in r]} 次；原prompt/Collator/response-only mask；训练期真实SeqRec |
| 真实backward | {'PASS' if backward else 'NOT TESTED'} | 真实累计梯度全分片finite scan、SID及backbone代表梯度；不是单看loss |
| 全参AdamW optimizer update | {core} | 两rank原始before/after、CPUAdam step/m/v、global steps、全参数optimizer集合；首步lr0，第二步才验非零master变化 |
| SID／主体证据 | {core} | 所有预期参数均trainable且在optimizer集合；记录新SID embedding/head及早/晚backbone的CPU FP32 master/梯度小样本；不声称每个元素单样本均非零 |
| effective batch128 | {batch}（观测SeqRec路径） | 2×micro1×accum64；256个真实训练prefix分为两更新；不是六任务完整训练验收 |
| CPU parameter offload | {chain.get('parameter_offload','NOT TESTED')} | 只有Optimizer-only真实GPU OOM才可进入；未为对比额外运行 |
| 人工2048tokens压力 | {pressure} | 重复真实token模式的人工输入，原CE，零额外optimizer step；不等于真实最长六任务数据路径 |
| 保存／恢复／导出／beam | NOT TESTED；静态审核完成 | 固定探针显式禁save/eval/generation；没有大型checkpoint；不能称最佳checkpoint实际重载成功 |
| 六任务正式配方 | 静态保留／运行NOT TESTED | 同一ConcatDataset、模型与optimizer；没有六次顺序微调；见忠实性和独立静态报告 |
| 数据清洁与历史1:1 | BLOCKED | 原辅助错配、SID残余碰撞、逐位置约束、历史绑定缺口保留，不改数据 |
| Qwen资产准备 | PARTIAL；完整权重BLOCKED | 两候选身份/小文件已核，Base仅待批准推荐；没有Qwen权重/模型/GPU任务 |
| 正式四轮／五seed | NOT TESTED，未启动 | 技术seed42不等于正式seed42 |

## 运行身份、参数与原件对比

- RUN_COMMIT：`{s['RUN_COMMIT']}`。服务器固定源码export：`/home/bwwu/lc-rec/releases/{s['RUN_COMMIT']}`，无`.git`；[部署清单](evidence/deploy/deploy-{s['RUN_COMMIT']}.json)与[七份运行文件映射](evidence/preparation/runtime_source_mapping_20261109.json)。DELIVERY_COMMIT由GitHub最终ref／包外回执给出，不向本文件写自引用SHA。
- 独立根job：`{s['job_id']}`；runner原始终态`{s['runner_status']}`、exit`{s['runner_exit_code']}`；两子候选的实际argv、配置、log/exit和预算在[原始证据](evidence/gpu_20261109/)。输出目录：`/home/bwwu/grid-audits/sid-extensions-p0/jobs/{s['job_id']}/output`。
- 原作者提交`e6f1b3b3f8b4fee55a49a7b3dfeba29dbfcf0669`；基础模型`huggyllama/llama-7b@4782ad278652c7c71b72204d462d6d01eaaf7549`；上一轮准备交付`2e0e26633b58d44cc4595417f4a054f6bb6b7e77`。旧`lc_rec/REVIEW_ENTRY.md`保持历史原字节。
- Python3.10.22，Torch2.6.0+cu124，Transformers4.31.0，Accelerate0.21.0，DeepSpeed0.16.9；独立overlay `/data/bwwu/envs/lc-rec-offload-20261108`只读复用GRID Torch，不修改GRID环境。实际版本、源码、native CPUAdam、全部freeze和前期安装日志见[ENVIRONMENT](ENVIRONMENT.md)及两rank raw report。
- ZeRO3在`from_pretrained`之前生效；不混`device_map`。基础权重严格载入后legacy normal resize；全部预期参数进optimizer。BF16直接初始化新增行与原晚转换的随机/舍入轨迹不同，不能声称逐位一致；同版本DS晚转换的master也可来自BF16，作者历史master来源未知，不制造对比。
- 实际Trainer第0组291参数、weight_decay.01；第1组为空。当前TF4.31分组包含RMSNorm，CPUAdam只保留非空组；没有按现代习惯另改免衰减。lr5e-5、betas(.9,.999)、eps1e-8、decoupled AdamW、bias correction、clip1.0；CPU FP32 master/m/v，CPU BF16累计buffer。实际设置与source保存在原报告。
- 技术max_steps2，但cosine保持条件正式总6912/warmup70；首lr0、次lr5e-5/70；窄no_sync适配与scheduler单tick，未改数据/loss。六任务变长回答下micro8→micro1会改变token-mean归组权重，属于正式重要未决，不只浮点舍入。

完整逐项对比见[FAITHFULNESS](FAITHFULNESS.md)、[loss推导](review/loss_reduction_review.json)与[正式候选](config/formal_candidate.json)。两Offload配置均发布，实际生效配置见raw report，不能从日志通用“Parameter Offload”类名推断参数已卸CPU。

## 真实更新证据

| rank | update | global step | lr前 | 双rank梯度finite | 原始全局grad norm | engine.step秒 | 跨rank代表样本证据 |
|---|---|---|---|---|---|---|---|
'''+'\n'.join(updates)+f'''

梯度scan覆盖每rank3,373,017,088元素。检查值来自更新前真实CPU梯度、更新后master及CPUAdam状态，不是合成模型或空step。首步状态推进但lr0无master delta属预期。FP32 master变化不意味着每个BF16存储元素此小步都越过舍入阈值；没有据小样本推断每个参数的非零变化。训练期prefix与canonical SID映射、真实input/labels/mask、每次loss、参数与native扩展身份在两个raw rank报告中。程序化摘要为[results_summary](results_summary_20261109.json)，其实际入口是[summarize_probe_evidence.py](scripts/summarize_probe_evidence.py)。

## 资源与条件性预算

- PyTorch allocator累计最高allocated **{max_a:.3f} GiB**、reserved **{max_reserved:.3f} GiB**；外部5/15秒采样各卡MiB最大`{c['sampled_gpu_max_mib']}`。分开记录real-update峰值与人工2048峰值，不把采样当不可漏检的瞬时峰值。
- 任务树RSS求和最高 **{cpu:.3f} GiB**（可能重复计共享页，不是唯一物理RSS）；主机最小MemAvailable **{avail:.3f} GiB**，仍高于40GiB。task VmSwap峰值`{peak.get('task_vm_swap_peak_bytes')}`bytes；host swap变动与实际pswpout分别保留。
- 双4090 UUID：`GPU-dfff0e95-1db6-e2f4-3431-55bb852fa0ae`、`GPU-59cac3bd-3645-668b-79c8-839e0d878893`，CUDA0/1；双路Xeon Gold6133、80逻辑CPU、两NUMA；负载下PCIe Gen3×8。[真实负载回执](evidence/preparation/pcie-under-real-model-load.json)与拓扑原件。
- GPU账本保守累计 **{s.get('conservative_gpu_wall_seconds',0):.3f}s**，含前期305s；未重置预算。组合包装收紧至5370s、外层5400s，两类各最多2更新，未到7200s总上限。资源守卫每5秒检查CPU/swap、每15秒采GPU/磁盘。
- [逻辑大小独立盘点](evidence/preparation/logical-storage-20261109.json)对本任务全部已知目录、release/tar及原有13.48GB权重保守计入；不是按st_blocks冒充逻辑大小。runtime guard原为st_blocks抽样，最终另核逻辑字节与home/data两文件系统安全余量。没有清理任何原档案。

{cost}

正式六任务总时间仍未知，需另加加载、四轮validation/native checkpoint（选定版本可重复写同tag）、best reload、导出及17模板beam评价；正式数据长度与回答长度分布未测。静态内存下界：CPU master+m/v主载荷12N≈75.393GiB，另有梯度/累积buffer/工作区；Optimizer-only GPU BF16分片约6.283GiB/卡，另有激活、gather、通信与allocator。不得把各buffer盲目相加当真实峰值。

静态native checkpoint单套约75.393GiB；四轮+root+shared base约389.515GiB/seed，五seed共享base约1897.371GiB，另加开销。单native裸载荷并非自动超100GiB；本轮核心优先，固定探针禁止保存且正式最佳轮重载兼容缺口未解决，未为附加验证制造大文件。完整空间情景、旧转换器RAM描述纠正、best-model发现门槛和same-engine reload限制见[独立报告](analysis_20261109/report.md)。原旧静态报告被保留，纠正位置明确。

## 数据、Qwen、排错与独立执行限制

旧真实审计：六任务221225条；train辅助错配12233/24772、test7473/24772；9922item/9897uniqueSID，23组碰撞、25额外item；官方逐位置约束不等于合法SID trie。部分train辅助目标处于当前validation/test位置，辅助生成来源不完整。没有修复/重排/删用户/重生成标签；本次仅training-period SeqRec更新，不摄入auxiliary。未来优先取得作者绑定同版inter/user/SID的资产与处理/生成输入说明；数据门槛未解决不启动正式五seed。

[Qwen报告](qwen_20261109/QWEN_ASSET_REVIEW.md)固定Base与后训练两候选revision、各5片共16381516776bytes、不同OID；13份已取小文件身份只读PASS。Base为与预训练LLaMA比较的首选建议，等待用户选择；不得默认下载。两份大型tokenizer JSON仅留WSL缓存，按publish=false排除Git/ZIP。Transformers>=4.51、padding/EOS、vocab/SID行及现代resize等未来方法差异未实施。

整个同任务沿用环境round0–round3，失败原回执保留；CPUAdam小张量真实通过不冒充7B模型更新，r3最后standalone-ldd失败不重写为成功。本接续未新增安装或修正原probe/guard；新包装只是首次GPU独立提交。**Qwen子链一次collector修正复跑在获知已耗尽round3前发出，通知后立即中断exit130；原exit1、中断、未绑定早期源码的限制如实记录，属于执行偏离，不能称全流程合规PASS。** 未因此获得第四轮或增加GPU预算。

包装为子候选改GRID_JOB_ID，旧runner根job marker因此不能证明部分子进程ownership；内层guard仍按PID+create-time管理本任务、执行硬时限与资源停止。不能声称外层可完美兜底。最终必须用真实终态query、空进程组、GPU进程和清理回执确认收尾，不以主worker退出代替。没有修改旧runner、SSH/驱动/全局配置。独立执行只承诺已部署脚本的有界行为，不承诺关闭AI后仍会新决策。

## 审核、发布与停止

真实非作者`/root/independent_audit`直接读取原件、适配、实际配置、两rank原始report与资源/runner证据，并核Qwen和C链；[最终审核](review/independent_task_review_20261109.json)状态为准，不能把早期静态PASS当真实GPU通过。`/root/checkpoint_budget`独立静态读32份源码/材料并验证算术。

RUN源码已推送并[GitHub完整内容回读](evidence/publication/RUN_READBACK.json)。最终仅发布wbw-lc-rec；DELIVERY完整SHA、最终远端ref/关键文件回读与ZIP CRC/解包/成员hash/Windows可访问性在包外最终回执及最终答复。无模型/数据/optimizer大资产或秘密进入Git/ZIP。旧入口/其他分支原样。

最终根job若已终态，按原job查询收尾，不重复提交：`python3 /home/addaswsw/lc-rec-offload-20261108/control/runtime/gridctl.py --target new-server query {s['job_id']}`。本轮授权终点后立即停止，不启动正式训练、Qwen加载或新方法。

GPT_UPLOAD_REQUIRED: YES（ZIP补充最终发布及包自身的外部验收回执；代码与关键GPU证据优先在固定GitHub提交读取）。
'''
(base/'RESULTS_20261109.md').write_text(text)
entry=f'''# 网页GPT单一审核入口：双4090 LLaMA全参CPU Offload（2026-11-09）

**真实模型更新 {core}；观测SeqRec batch128 {batch}；严格1:1正式六任务复现仍BLOCKED。** 本次只做技术预跑。先读[完整结果与验收](RESULTS_20261109.md)，再按下面顺序读原始材料。最终非作者结论与发布/ZIP回执必须一并核；不能从本入口的技术PASS推断全流程无偏离。

1. [机器摘要](results_summary_20261109.json)和[原始GPU job文件](evidence/gpu_20261109/)：request/status/cleanup/chain、两个rank完整记录、resource samples/summary、账本。核真实before/after、CPUAdam m/v/step、SID/backbone梯度及delta，区分首lr0、第二非零lr与人工2048零update段。
2. 实际运行源码：[probe](scripts/offload_probe.py)、[资源保护](scripts/resource_guard.py)、[独立候选包装](scripts/run_bounded_candidates.py)；实际派生diff在[review](review/official_to_probe.diff)。RUN_COMMIT `{s['RUN_COMMIT']}`；[服务器固定文件映射](evidence/preparation/runtime_source_mapping_20261109.json)和[完整部署清单](evidence/deploy/deploy-{s['RUN_COMMIT']}.json)。实际snapshot不是Git checkout。
3. 两候选[optimizer-only](config/ds_optimizer_only.json)、[optimizer+parameter](config/ds_optimizer_parameter.json)；[正式候选](config/formal_candidate.json)仅静态不可运行；[环境／前期安装](ENVIRONMENT.md)。最终生效数值以raw report为准。
4. [原版对照](FAITHFULNESS.md)、[联合六任务源码证据](review/JOINT_TRAINING_SOURCE_EVIDENCE.md)、[micro regrouping loss差异](review/loss_reduction_review.json)。不批准新Loss/LoRA/量化，不能用相同global batch替代严格等价证明。
5. [权重完整校验继承与轻量回读](evidence/preparation/model_identity_readback_20261109.json)、[小输入身份](evidence/preparation/small_input_identity.json)、[原件未修改](evidence/preparation/official_preservation_20261109.json)。旧CPU/数据审计沿用上一轮[原审核入口](../REVIEW_ENTRY.md)，该历史入口不被本轮改写。
6. [checkpoint/预算/数据依赖独立复核](analysis_20261109/report.md)与[源码身份/算术](analysis_20261109/findings.json)。保存/恢复/导出/生成未实测；新的报告明确纠正旧报告的转换RAM描述，不能任选旧结论。
7. [Qwen资产调查](qwen_20261109/QWEN_ASSET_REVIEW.md)、[API候选身份](qwen_20261109/candidate_summary.json)、[小文件hash](qwen_20261109/asset_checks.json)、[真实失败/中断与预算偏离](qwen_20261109/collection_attempts.json)。没有Qwen权重/训练；Base只是待批准推荐；大tokenizer cache排除发布。
8. [真实非作者最终审核](review/independent_task_review_20261109.json)。旧pre_run_closeout、旧PARTIAL资产报告是其写作截点历史，不能用来覆盖新的真实job，也不能反向伪造旧PASS。
9. [RUN GitHub回读](evidence/publication/RUN_READBACK.json)。最终DELIVERY_COMMIT等于本入口固定GitHub URL中的提交，并由ZIP `publication/DELIVERY_RECEIPT.json`及最终回复给出；避免SHA自引用。ZIP含全部必要轻量源码/证据及外部回执，`MEMBER_SHA256.json`与包外SHA/解包/Windows验证需核对。

本轮是一次性交付，不是持续维护看板。技术seed42不等于正式seed42；未跑四轮、五seed、Qwen或新研究实验。原job查询入口在完整结果中；收尾后不重启。

GPT_UPLOAD_REQUIRED: YES。请将已交付ZIP交给网页GPT核对外部回执；无需上传模型、原始数据或checkpoint。
'''
(base/'REVIEW_ENTRY.md').write_text(entry)
print(json.dumps({'core':core,'reports_written':['RESULTS_20261109.md','REVIEW_ENTRY.md']}))
