# 逐项验收矩阵

对应用户第0–15节，C/D最终发布回读与服务器终验在外部FINAL_DELIVERY_RECEIPT闭合；不预写未发生事实。

|要求ID/要求|实际动作|证据|状态|未完成/边界|
|---|---|---|---|---|
|0.1 一次性独立P0、继承旧成果，不重开全库审计|独立support/worktree/STATE，引用旧v6；仅3个相关batch|`STATE.json; OLD_AUDIT_INHERITANCE; OLD_V6_PACKAGE_RECEIPT`|PASS|无；运行/理解仍未授权|
|0.2 成功只到READY_FOR_EXPLANATION，两个授权布尔值false|阶段状态与停止约束明确，未自动进入下一阶段|`STATE.json; docs/README; REPORT`|PASS|无；运行/理解仍未授权|
|1.1 仅文件准备/传输/静态分析，禁止来源执行|隔离解释器/独立cwd；AST、JSON、字节hash；无模型/loader/import执行|`纯prep requests/helper; STATIC_DEPENDENCIES; A/B/C reviews`|PASS|无；运行/理解仍未授权|
|1.2 禁止模型加载/前反向/生成/训练/测试/注意力采集|未调用torch.load/pickle/instantiate/Trainer/Notebook/test_saved_checkpoint，仅读正文|`job argv/inputs; helper reviews; CODE_READING_GUIDE`|PASS|无；运行/理解仍未授权|
|1.3 禁止新embedding/SID/权重、划分/映射/指标重算|使用固定Qwen三域实体；历史结果标HISTORICAL_REPORTED_NOT_RERUN|`ASSET_MANIFEST; Baseline Results; experiments说明`|PASS|无；运行/理解仍未授权|
|1.4 禁止改研究/接口逻辑及Parallel填空|123原件字节保持；PROPOSED_NOT_IMPLEMENTED明确|`SOURCE_MANIFEST; B_ROUND2; KNOWN_ISSUES`|PASS|无；运行/理解仍未授权|
|1.5 不接A800/不重传78GB/不清理旧目录|只经WindowsSSH访问new-server；保留所有旧项目/失败证据|`remote-precheck; restore plan; OLD_AUDIT_INHERITANCE`|PASS|无；运行/理解仍未授权|
|1.6 保护共享分支/设置/凭据/环境|新sid-extensions分支；无force/merge/visibility/SSH代理环境变更|`C_PRECHECK; LOCAL_HOOKS_REVIEW; Git操作日志`|PASS|无；运行/理解仍未授权|
|2.1 读实际旧README/REPORT/STATE/矩阵/v6恢复及机制边界|主控+A/B分别读旧原文和选中ZIP成员；不改冻结结论|`A_INHERITANCE_READONLY; OLD_AUDIT_INHERITANCE`|PASS|无；运行/理解仍未授权|
|2.2 固定完整baseline SHA、当前refs/观察时间、有界变化核对|GitHub current main与锚点相同，无版本漂移；API/ls-remote实际观测|`C_PRECHECK.json; STATE baseline_sha`|PASS|无；运行/理解仍未授权|
|2.3 核身份/路径/权限/空间，保护已有工作树|bwwu/zwlab-01/uid1005，/data约3.6TB余量；已见旧未跟踪文件故隔离|`remote-precheck.json; C runner readback`|PASS|无；运行/理解仍未授权|
|3.1 先ls/rules/同阶段状态，不覆盖已有阶段|首命令ls -la；适用祖先无规则文件；本地/远端同阶段不存在|`initial logs; remote-precheck existing_jobs; STATE`|PASS|无；运行/理解仍未授权|
|3.2 STATE/PLAN/矩阵/改进账/阻碍/报告入口|支持根全创建；Git内实质说明/审核/矩阵按实际结果维护|`STATE.json; PLAN.md; IMPROVEMENT_LOG; REQUIREMENTS_MATRIX`|PASS|无；运行/理解仍未授权|
|3.3 角色/依赖/作业/修正轮次/布尔值可恢复|真实A/B/C，固定job/request/sidecar，round0/1/2日志|`STATE; PREPARATION_JOB_SUMMARY; review/ROUND*`|PASS|无；运行/理解仍未授权|
|4.1 独立本地/远端副本与无冲突新分支|本地独立clone固定锚点，目标sid-extensions；已有同学工作树保留|`DEPLOYMENT_MAP; C_PRECHECK`|PASS|无；运行/理解仍未授权|
|4.2 完整可读结构，不重复baseline，不复制.git/env/归档|baseline沿原路径；native保留src/config/入口相对树；未复制Notebook/大张量|`SOURCE_MANIFEST; C_PREPUBLICATION_FILES`|PASS|无；运行/理解仍未授权|
|4.3 原字节、换行/过滤属性、许可来源保护|新区域GRID/** -text -filter；基线属性不改；LICENSE+同SHA notices载荷继承|`C_ORIGINAL_SECRET_SCAN; C_LICENSE_NOTICES_FINDING; SOURCE_MANIFEST`|PASS|无；运行/理解仍未授权|
|5.1 三域全部数据/划分/映射/SID/best checkpoint真实实体|1902数据已全hash；55固定LFS对象按需取得并服务器wholeOID验证|`ASSET_MANIFEST; ASSET_SUMMARY; restore/assembly结果; C_ASSEMBLY_RESULT_REVIEW`|PASS|无工作必需payload缺口；MODEL_LOAD_NOT_RUN/SEMANTIC_BINDING_UNVERIFIED|
|5.2 按new-server同身份→归档匹配→固定LFS优先，不混T5|Beauty已有同SHA复用；Toys/Sports匹配旧数据归档；Qwen SID/权重只固定OID|`A_BASELINE_EXPECTED_IDENTITIES; A_DATA_AND_SOURCE; ASSET_MANIFEST`|PASS|无；运行/理解仍未授权|
|5.3 最佳checkpoint与原有效配置/overrides/结果关系|固定test_results+stage_result和原command，43小记录真实载荷普通Git发布|`A_RUN_RECORDS_STATIC; baseline-run-records; CODE_READING_GUIDE`|PASS|无；运行/理解仍未授权|
|5.4 区分字节/provenance/语义/未模型加载/未授权|逐资产五层标记，Qwen配对缺精确hash运行证明统一UNVERIFIED|`ASSET_MANIFEST semantic/model_load/experiment字段`|PASS|无；运行/理解仍未授权|
|5.5 LFS指针不当payload，普通路径/只读链接受控|外部资产root普通实体；Beauty symlink目标664只读使用政策，其他444独立inode|`ASSET_MANIFEST; A_REMOTE_SELECTED_READBACK; C_ASSEMBLY_RESULT_REVIEW`|PASS|无工作必需payload缺口；MODEL_LOAD_NOT_RUN/SEMANTIC_BINDING_UNVERIFIED|
|5.6 预算/固定分块/有界并发/已验不重复、不下载全Qwen|100GiB reserve；32MiB顺序Range/WindowsSCP回执恢复；固定SID无需上游完整权重|`C_RELAY*; PREPARATION_JOB_SUMMARY; ENVIRONMENT`|PASS|无；运行/理解仍未授权|
|6.1 从new-server归档按需恢复原生模型及本地依赖配置|b01提取123原件/历史上下文，服务器打包回读，静态模块路径均可追|`SOURCE_MANIFEST; STATIC_DEPENDENCIES; C_SERVER_ORIGINALS_REHASH`|PASS|无；运行/理解仍未授权|
|6.2 原件/工程派生区分，许可证/敏感信息、历史资料有界|不部署旧eval-v2，不伪装已适配；原件秘密模式无命中；不扩展图/tensor|`references/README; C_LICENSE/SECRET reports`|PASS|无；运行/理解仍未授权|
|7.shuffle 共享表/双表、三重排、hook、mask/随机/YAML问题实际重定位|按原文/hash/行号解释，普通main保持原参数，后续建议未实施|`experiments/shuffle.md; B_ROUND2`|PASS|无；运行/理解仍未授权|
|7.mlp HE→2E→E/ReLU/Dropout、SEP/mask/user、3类MLP、AR/参数|原文定位及静态形状/参数推导，训练初始化/冻结/预算留待讨论|`experiments/mlp.md; B_ROUND2`|PASS|无；运行/理解仍未授权|
|7.parallel 同字节AR；非AR NOT_RECOVERED/待讨论，不造实现|保存原件、4BOS历史上下文、已知/未知与设计问题清单|`experiments/parallel.md; B_ROUND2`|PASS|无；运行/理解仍未授权|
|7.attention cross轴/四步骤、真实采集绘图/beam/mask/33CRC边界|逐函数阅读；指出没有实际mask保存与最终祖先；引用旧528/33不重扫|`experiments/attention.md; B_ROUND2`|PASS|无；运行/理解仍未授权|
|7.history v6 CPU有界接口/历史配置不能唯一绑定Python/Qwen与T5区分|所有历史身份/限制分层标明，未重造指标|`OLD_AUDIT_INHERITANCE; 各实验历史入口`|PASS|无；运行/理解仍未授权|
|8.1 总图+四说明，各含10项讲解要求/性质标识|入口/配置/数据/变体/shape/参数/历史/提案/未来验证/用户问题齐备|`CODE_READING_GUIDE; 四experiments; B_ROUND2`|PASS|无；运行/理解仍未授权|
|8.2 源码事实绑定版本/path/symbol/行范围/完整hash；教学值非实测|65源码证据+完整AST索引+原CSV行；H4/E128/M30槽静态标注|`SOURCE_LINES; SOURCE_MANIFEST; B_ROUND2`|PASS|无；运行/理解仍未授权|
|9.1 SOURCE_PROVENANCE/DEPLOYMENT_MAP不能只比HEAD|baseline逐Git blob、native捕获/member/server hash；全部source集合hash|`SOURCE_MANIFEST; DEPLOYMENT_MAP`|PENDING|server最终Git树部署待核|
|9.2 有效配置/路径只映射不改写、环境只metadata、动态依赖诚实|43原配置文本；PATH_MAPPING/ENVIRONMENT，未导入/安装/instantiate|`baseline-run-records; STATIC_DEPENDENCIES; ENVIRONMENT_METADATA`|PASS|无；运行/理解仍未授权|
|9.3 忽略/外部本地模块识别，实际import身份留预跑|显式src模块路径全找到；动态target/导出符号/runtime身份明确未确认|`STATIC_DEPENDENCIES; C prepublication`|PASS|无；运行/理解仍未授权|
|10.1 真实multi-agent A/B/C、非作者复核实际原件/命令结果|A全部来源+远端抽核、B源码/文档、C工具/服务器全123重hash/发布|`review/A_* B_* C_*`|PENDING|最终LFS/发布审核待闭合|
|10.2 不并发改源码/index/STATE，Git由主控串行|代理只写自己的review，冲突按原证据解决|`review/README; Git操作记录`|PASS|无；运行/理解仍未授权|
|11.1 整个P0最多3轮，失败保留、自主修正、不重置|round1文档六项；round2网络超时新attempt与精度；当前使用2/3|`IMPROVEMENT_LOG; ROUND1/ROUND2_CHANGE_RECORD`|PASS|无；运行/理解仍未授权|
|11.2 外部硬阻碍停止依赖链，未知/源问题不能降门槛|源问题保留边界；当前恢复任务不标完成；认证不暴力重试|`BLOCKERS; REPORT; STATE`|PASS|无；运行/理解仍未授权|
|12.1 复用runner，耗时任务请求/日志/状态/ID持久化与查询恢复|旧runner同字节新sidecar；local relay+remote restore/assembly全独立job|`C_HELPER/REQUEST reports; PREPARATION_JOB_SUMMARY`|PASS|无；运行/理解仍未授权|
|12.2 回执丢失先查ID、不重跑/不覆盖、Windows路径/字节核验|transport STAGE新路径、SCP验证SHA，v1失败保留v2关联恢复|`C_RELAY_V2; 工具原文; job requests`|PASS|无；运行/理解仍未授权|
|12.3 容量预算/每批空间/无本阶段训练队列/必要job最终终态|当前restore终态；relay进行中，结束需无live/结果取回|`PREPARATION_JOB_SUMMARY; STATE`|PENDING|relay/assembly/最后部署作业尚未结束|
|13.1 发布前权限/具体文件/secret/自动化/hook审查|push权限，GitHub无workflow，hook无非sample，新增区无大载荷/secret|`C_PRECHECK; C_LOCAL_HOOKS; C_PREPUBLICATION_FILES`|PASS|无；运行/理解仍未授权|
|13.2 实际CONTENT_COMMIT正常push并从GitHub取固定tree/blob字节|提交/推送后必须独立API回读全部必需source/config|`review/C_CONTENT_GITHUB_READBACK`|PENDING|C尚未提交/回读|
|13.3 DELIVERY_COMMIT、C/D同source、最终D再回读与server对应|D仅元数据，source路径/blob/hash集合相等；D完整SHA/最终收据包外|`C/D比较; FINAL_DELIVERY_RECEIPT external`|PENDING|D尚未提交/回读|
|14.1 10项最少交付齐全，小报告/清单/实质审核实际上GitHub|阅读/完整source/config/来源/资产/known issues/deploy/矩阵/审查/报告|`docs+manifests+references+tools; REPORT`|PENDING|最终发布尚未完成|
|14.2 成功最低门槛：实体、原版、无实验、实际发布、独立复核、无在途job|不因Parallel缺失等约定边界降门槛；必需缺口始终PENDING|`ASSET_SUMMARY; review; STATE; GitHub回读`|PENDING|等待全部必做闭合|
|15.1 最终中文A–G，固定版本入口/位置/真实审查轮次/限制/接手并停止|Codex最终消息提供D完整SHA/最终实际回读；两个布尔值false|`FINAL_DELIVERY_RECEIPT; STATE; 最终回复`|PENDING|仅收尾时执行|
