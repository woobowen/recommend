# 清单入口与证据含义

|文件|核查内容|
|---|---|
|SOURCE_MANIFEST.json|固定baseline逐Git blob/原字节；123原件捕获/member/hash；43份必要LFS小配置原字节阅读副本；纯准备工具身份|
|SOURCE_LINES.json|冻结Python AST符号及真实行范围，逐项绑定源SHA-256|
|STATIC_DEPENDENCIES.json|显式src模块路径、第三方import调用点、YAML target声明；动态解析/实际import未运行|
|ASSET_MANIFEST.jsonl / ASSET_SUMMARY.json|三域每个数据、SID、mapping、best ckpt、运行记录的真实服务器实体与身份；未就绪不填PASS|
|DEPLOYMENT_MAP.json|WSL、new-server源码树/原件/外部资产、GitHub分支的逐文件对应；最终D身份在包外回执|
|ENVIRONMENT_METADATA.json|已有解释器符号链接与dist-info元数据，只读，不导入研究包|
|OLD_AUDIT_INHERITANCE.json / OLD_V6_*|旧冻结证据引用与原包身份；没有改写旧结论|
|baseline-run-records/|从固定baseline LFS选择性取得的原config、overrides、hydra、job、stage_result、hparams、CSV等普通Git文本副本|

本次核验分层：`PAYLOAD_PRESENT_HASH_MATCHED` 仅说明实体字节；`PROVENANCE_BOUND` 说明配置/结果/路径与内容身份对应；`SEMANTIC_BINDING_UNVERIFIED` 明确没有可用的精确hash绑定Qwen配对运行证据；`MODEL_LOAD_NOT_RUN` 和 `EXPERIMENT_NOT_AUTHORIZED` 始终保留。旧T5语义绑定不移用到新Qwen。

数据划分按原配置：training=训练，evaluation=验证，testing=测试，items=物品记录。item/user索引保存在原TFRecord字段中，SID列轴约定见map_sparse_id_to_semantic_id；keyed SID生成参照另外保存，但不反序列化、不重建索引。

源内容集合SHA只由`source_config_audit_set=true`的路径/内容SHA组成，排除清单本身和报告，避免自引用。C/D是否同源要比这组路径/字节/Git blob，不能仅比较HEAD。

`ARCHIVED_REFERENCE_ASSETS.json`给出不需要部署的上游/历史权重具体归档身份与恢复位置；`OLD_A800_MECHANISMS.md`是未改写的旧冻结机制报告副本，不能当本次运行。
