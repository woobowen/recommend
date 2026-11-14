# Shuffle 非作者正式运行前审核

**NONAUTHOR PRELAUNCH TECHNICAL REVIEW: PASS（固定发布、部署和runner清收补充门落实后准入）。** 本结论覆盖实现、预验收及正式设计，不代表四组正式计算、结果终审或最终交付完成。由主控按真实非作者最终回复整理落盘。

审核者 `/root/shuffle_non_author_review` 非实现/运行/主评价作者，实际分析主控逐段桥接的冻结源码、有效配置、真实机器回执及回归结果，并独立编写 [independent_inputs.py](independent_inputs.py)。审核者没有可用SSH、文件读取或GPU工具，未独立登录服务器/访问磁盘/重跑GPU；访问及执行均由主控桥接，不扩大独立性。

已审core、A0/A1冻结入口最小适配、CPU全轴/共同manifest、runtime/Allowed/GPU acceptance、condition/series、评价/独立回放/配对、local launch摘要。GPU R1实际源码 `85c9d4473fb408c9635d904972910e8fffb7b26b`，job `letter-shuffle-20261114-preflight-r1`。A0/A1原GPU源码分别 `4c6a83d9ba42dc2fe737792759e410ef1874da76` / `0e3e2c6f5c1a76152ea0228390d94ffc1ff75ba5`；正式RUN须另经发布固定。

方法：完整Prompt原tokenization后、原生模型及KV Cache前置换，真实连续4Token边界逐历史物品核对。Token身份/物品时间顺序/组外Token/mask/长度/target不变；S3仅前三位且d不动，S4四位参加，排除恒等；不改embedding/attention/模型参数/目标生成码位。SHA256 canonical JSON/full256-bit/rejection/字典序非恒等集合，不依赖rank/batch/worker/Python hash。

全CPU实测：每模型24772用户、172929历史组、同用户轴和逐项S3/S4排列。共同manifest4952762字节/SHA `f936c87e07a48413fb3012d0afd1c05fad8f50823b0c0bd5ea6048bb6bb37be1`，tokenizer专属摘要不同但共同排列逐行相同。

OFF参照执行冻结源码唯一原generate AST表达式、原collate张量/原生回调/同一严格加载模型，适配OFF走真实入口；**并非完整冻结CLI另跑**。R1每模型32真实用户/每rank16；original/OFF输入、mask、embedding、序列、score、排序通过，历史序列/排序相同且最大score差0，target和16指标完全相同。S3/S4真实扰动、守恒、合法输出通过；实际lookup严格满足Token ID置换与向量置换相等。首次forward实IDs/当前位置/初始cache长度0通过；参数对象/storage/version不变为补充证据，不等于全权重重hash。未发现显式旧位置/cache旁路，但保留隐式层级身份和训练顺序先验，不支持匿名化/严格置换不变性结论。

R0模型加载前CLI失败runner4.450581秒/8.901162device秒。第1轮关闭缩写歧义、补原生参照，R1 runner231.184575秒/462.36915device秒。全预算471.270312device秒=7.8545052分钟，低于60分钟，含失败/加载/清收，R1 SUCCEEDED/exit0/cleanup verified。此前A0 sampler疑点已由冻结上下文解除，不伪记为运行缺陷。

正式设计：A0S3→A0S4→A1S3→A1S4；双GPU锁、实际UUID/占用门；独立job/output；A0 4h、A1 3h、累计18h保守上限及终止清收余量；前置真实成功/exit0/cleanup才能续；真实runner query，异常停止；CPU CUDA空；已成功预测的CPU错误不自动重生成；accepted runner脱离SSH/WSL，原series存在只查询。GPU stage/device、CPU及保守runner wall分别报告。

评价预审：全真实输入独立回放、原作者四指标/16/CCE/SID诊断/逐用户/七组独立复算，5000PCG64/seed20261113/95%用户配对Bootstrap；各模型配自己的Normal。Normal→Normal CPU fixture的16差值/CI/三项推荐变化及独立复核均0，仅证明入口回归。Bootstrap只估固定模型/固定Shuffle实现下用户不确定性。

正式提交硬门：最终完整commit和必要GitHub文件实际回读PASS；全部生效源码/JSON/共享依赖服务器字节映射；冻结原入口/Allowed/author source-root/模型adapter及外部输入校验；两CPU audit/共同manifest/最终GPU回执SHA绑定；A1真实非作者最终报告；最终launch/控制语法CLI通过。主控随后实际compileall、launch --help exit0，core三suite PASS。已有request/gate/plan只准完全相同字节，不同身份拒绝覆盖。

runner支持名前缀的补充发现和第2轮CPU-only清收验证见 [修正记录](CORRECTIONS.md) 及相应真实回执，需非作者补充复核后才启动。其余预审未发现需要改变已批准研究方法的必要阻塞。最终四组仍须全用户独立回放/复算、真实结果终审及交付验证。

## 非作者第2轮补充复核

同一审核者收到实际两prefix probe的request/status/cleanup/身份及runner两处prefix分支后给出 **RUNNER DETACHED-SESSION SAFETY CHECK: PASS**。S3/S4仅后缀不同，共享相同startswith分支，无须重复四probe。信号前initial_observation的owned_live和空unsupported/unconfirmed、PID/starttime原件、pidfd身份复核、仅三个保存PID的TERM及最终verified-empty足够，不要求另一超时前query快照。真实覆盖为“CPU手工detached-session probe，验证已支持namespace下marked进程跨session识别、pidfd清收及无残留；未执行完整elastic超时或GPU故障注入”。真实终态保留TIMED_OUT/exit-15，另判PROBE_PASS。第2轮收口，剩余只更新后完整提交GitHub回读、服务器字节部署及request/plan/证据绑定。
