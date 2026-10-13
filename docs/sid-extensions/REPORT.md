# sid-extensions-p0：准备与发布报告

本报告冻结于 `CONTENT_C_ACCEPTED_AND_SERVER_DEPLOYED`：必需工作资产、原生源码/配置、可讲解性资料及对应非作者审核已完成；受审 C 已正常发布并从 GitHub 实际回读、部署至 new-server。最终交付 D 仅补齐说明与审核元数据，D 的完整 SHA、最后一次 GitHub/服务器终验及阶段终态由外部 `FINAL_DELIVERY_RECEIPT.json` 和 Codex 最终回复给出，避免本文件自引用和预写尚未执行的终验。

固定 baseline：`563dc50b1c7e9e2e05cdfddd13e91a3d1888001e`；初查实际 main 等于该锚点，无无声替换。仓库 `woobowen/recommend`、实际分支 `sid-extensions`。原件捕获 `A800-grid-20261010-v1`；旧阶段 `wmh-a800-audit-20261010` 的 v6 为 139153346B / `9238fe81eaaa09c7a13be3be0490694741b6f0588d27e5bdebb58e24feb9d849`。旧结论与凭据只读继承。

未执行模型加载、前向/反向、生成、训练、验证、测试或注意力采集；未实施研究/接口适配。未重新生成 embedding、SID、权重、划分或物品映射；未重算/挑选科研指标；未连接 A800、重传全档案、清理旧科研目录或升级公共环境。`experiment_execution_authorized=false`、`user_understanding_confirmed=false`。

## 实际材料与内容身份

- 恢复作业 `sid-extensions-p0-restore-v1` 为 SUCCEEDED/0、无活动 worker/group、cleanup 核验通过；2025 项全哈希包括 123 原件、1902 数据。仅访问 b01/b11/b12 三个相关 TAR 的 30 个固定块，不重跑全部302块审计。
- Beauty 549 数据复用已有匹配实体的 symlink；目标权限0664，约定只读使用，未声称操作系统强制只读。Toys 480、Sports 873 数据及123原件按需提取，独立 inode/0444。原件从服务器打包实际取回，WSL发布副本再次核 SHA；未规范化换行。
- 固定 baseline 指定的55个LFS实体由67个固定分块完成服务器全 OID/SHA-256 组装，共482736854B；三域各有明确 best recommendation checkpoint、SID、keyed item/SID映射和原配置/结果。43份小运行记录331716B另以普通Git文本发布，baseline 原 LFS Git blobs 保持不变。
- [ASSET_SUMMARY](../../manifests/sid-extensions/ASSET_SUMMARY.json)：1957实体、1939必需项，共5234484687B；Beauty562/562、Toys492/492、Sports885/885必需载荷均实际存在且匹配。没有仅以LFS指针充当必需工作资产。上游Qwen完整权重与旧MLP权重仅保留归档身份/恢复索引，未扩展下载。
- 内容完整性为 `PAYLOAD_PRESENT_HASH_MATCHED`，运行来源为 `PROVENANCE_BOUND`；Qwen配对语义缺少绑定这些精确内容hash的既有运行验证，统一 `SEMANTIC_BINDING_UNVERIFIED`。全部 `MODEL_LOAD_NOT_RUN`、`EXPERIMENT_NOT_AUTHORIZED`；字节核对不冒充语义或模型运行验证。

## 原文阅读与明确局限

[阅读总入口](README.md)、[baseline 总图](CODE_READING_GUIDE.md) 与 [Shuffle](experiments/shuffle.md)、[MLP](experiments/mlp.md)、[Parallel](experiments/parallel.md)、[Attention](experiments/attention.md) 覆盖入口、原始配置、真实数据/参数/生成/损失链、变体逐处差异、历史局限、后续提案和用户问题。关键事实有性质标签、符号、实际行范围和完整源hash；51个原稿源码链接已实际绑定 C 的完整 SHA，baseline链接绑定完整固定锚点。B 又独立回读36个固定GitHub raw实体核对。

原生模型及 YAML 未改逻辑、路径、接口或格式。Shuffle 的双表/三种重排/hook/mask 与随机协议、MLP有序 HE→2E→E 聚合/无SEP/末码mask/自回归、注意力四步与head/beam/mask/绘图链均直接按正文解释。Parallel 当前文件和 shuffle 同字节且仍自回归；真实非AR正文 `NOT_RECOVERED`，重建 `RECONSTRUCTION_PENDING_USER_DISCUSSION`。所有接口/研究建议 `PROPOSED_NOT_IMPLEMENTED`。历史指标 `HISTORICAL_REPORTED_NOT_RERUN`；旧T5/v6普通模型有界CPU接口证明不移用为Qwen或Shuffle通过证据。

## 固定 C 发布、实际回读与服务器映射

CONTENT_COMMIT：`84d4533fbdd6b010f54d671b433a9952608911ca`；tree：`eaa300cf10c2793c676eff2f00fdd8dc4cfb3be8`。

C 已正常 push，无 force、共享分支覆盖或 main 合并。非作者 C 在 2026-10-14 02:27:57+08:00 完成 [GitHub实际回读](review/C_CONTENT_GITHUB_READBACK.json)：整个2468文件tree等本地固定C，377必需/新区域路径、287去重blob/10576470B均实际API取原字节核size、Git blob和SHA-256，涵盖全部285受审源码配置、123原件、43小运行记录。主控另有[独立路径回读收据](../../manifests/sid-extensions/GITHUB_C_PRIMARY_READBACK.json)。

服务器C部署 `sid-extensions-p0-deploy-C-v1` 已 SUCCEEDED/0、证据取回、无live/cleanup通过；2468文件逐项对应固定C Git blob/原始SHA，完整结果见 [部署结果](../../manifests/sid-extensions/jobs/sid-extensions-p0-deploy-C-v1/RESULT.json)。非作者 C 再读服务器 marker、精确文件集合及8份关键文件，确认无额外隐藏源码或symlink覆盖；这是无嵌套`.git`的独立Git源码export。

受审源码配置集合285项，`source_set_id_sha256=8b3149a98731ba7857e446e0f488489be0715181b841f8c7f9f169d3c4e16ac3`；其中273研究/配置项digest `ed312e07317b84ec7baaa5172777bd6f06e78c001e4aa2dbbbc89c6c1d8fed64`。D不得改变这些字节；C→D证明和最终D外部回执与本报告联合验收。

真实位置：代码 `/home/bwwu/recommend-worktrees/sid-extensions-p0`；原件恢复 `/home/bwwu/grid-audits/sid-extensions-p0/restored/GRID`；工作资产 `/data/bwwu/sid-extensions-assets/563dc50b1c7e/`；状态/作业 `/home/bwwu/grid-audits/sid-extensions-p0`。逐项关系见 [DEPLOYMENT_MAP](../../manifests/sid-extensions/DEPLOYMENT_MAP.json)、[PATH_MAPPING](PATH_MAPPING.md)。旧 `/home/bwwu/recommend`、WSL `grid-original/wang-grid` 未移动、重置、删除或覆盖。

## 真实独立审核、修正与作业终态

主控为材料/工具/说明作者，A/B/C仅审核并写各自报告，未并发改Git/index/STATE。A最终全部1957资产身份关系与服务器55个LFS实体新鲜全hash通过；B正文、形状/参数、历史边界及最终链接通过；C工具纯文件fixture、权限/secret/许可/attrs/自动化、真实组装、候选、GitHub C 和服务器 C 通过。见[审核入口](review/README.md)，含初审失败原始记录。

全P0已使用集中修正2/3轮：round1修正六处说明事实/行号并明确notices复用；round2处理真实HTTPS超时恢复、两处精确表达和部署纯文件工具恢复边界。failed relay-v1及24块和临时件证据保留，另建v2复用已核块，v2/组装成功。未为静态全绿改研究原件。所有已提交的材料准备及C部署作业在本报告冻结时均终态，无模型/训练队列；D部署与最终只读hash作业须结束后才可由外部回执宣布READY。详见[作业账](../../manifests/sid-extensions/PREPARATION_JOB_SUMMARY.json)。

## 接手与硬停止

GPT网页版应先读最终D的本入口、矩阵及审核，再从Beauty历史交互→SID→offset→共享embedding→encoder/decoder→loss/beam/evaluator逐段讲解。只说明未来检查目标，不自动执行。Parallel正文缺失、原件已知接口问题、运行语义/加载未验证、用户尚未理解是明确后续边界，不能靠本次材料发布自动获得研究授权。

新增系统包、语言包、工具链、公共/用户环境配置修改均0，无待清理安装。最终交付后停止；不启动预跑、Shuffle/MLP训练、Parallel实现或attention采集，等待用户完成讲解并明确授权下一阶段。
