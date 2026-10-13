# A 最终来源与资产独立审核

状态 `PASS_NONAUTHOR_FINAL_REQUIRED_ASSET_BYTES_AND_PROVENANCE`。复核者 `/root/audit_assets`；观察时间 2026-10-14T02:22:31.349748+08:00。固定baseline `563dc50b1c7e9e2e05cdfddd13e91a3d1888001e`；ASSET_MANIFEST SHA-256 `491211cad78ae638ce4d34160ee491fb9ff471d07f4e90c2f726e77e21ed44b1`。

全1957项与冻结Git LFS OID/size、恢复或组装实际结果逐条对应；其中1939必需项、18来源参考项全部实体实际存在且内容身份匹配，无必需指针冒充载荷。对全部55 LFS实体进行第二次独立远端完整SHA读取，共482,736,854字节，全部稳定/匹配/普通文件/0444/nlink1。既有数据1902项的生产全量hash与先前独立24数据抽核证据结合，不冒称独立第二次读取全部数据。

|域|数据分片|必需实体|最佳checkpoint步数|SID SHA-256|
|---|---:|---:|---:|---|
|beauty|549|562|12500|`d41adcfb22a9a47d5e0424c068f4c74f33cc9fdbaf5db08540fbfa0e3d720e18`|
|toys|480|492|6200|`c6d516be6b65708137e8eea078736b24f88fd50b45c132ac376947dd71c132ed`|
|sports|873|885|9500|`89b2c2bbb94eda26ef0e147404fbff29e6ec8246fbdc754c5d885f313f2de293`|

逐域检查了published result→best path、stage_result→同best与相同历史结果、有效config/overrides→本域数据和新SID、固定LFS/server_manifest→实体hash。没有自行挑选checkpoint或重算指标。123学长原件manifest与先前实际发布字节再次匹配。

- This PASS covers actual selected server payload bytes, three-domain required coverage and declared historical provenance. It is not model/checkpoint compatibility or semantic execution validation.
- All1902 data content hashes were read by the frozen restore worker; reviewer independently rehashed24 data files plus13 originals in earlier remote sample. All55 LFS payloads were independently rehashed completely again by this reviewer,482736854bytes, stable during read, regular,0444,nlink1,not pointers.
- SEMANTIC_BINDING_UNVERIFIED is intentionally retained for all new Qwen SID/data/item/checkpoint pairs: no exact content-hash-bound semantic producer/runtime pairing proof available. Old T5 SID axis and bounded CPU evidence are not reused.
- item/user identities are embedded in fixed TFRecord payloads; no new standalone item/user map or split/remap is created. Keyed SID capture PKL and original generated tensor are preserved as provenance without deserialization.
- Beauty data symlink targets remain0664; read-only use is policy, not OS enforced immutable permission. New Sports/Toys data and assembled payloads are0444/nlink1.
- Frozen baseline recommender uses T5 architecture over new Qwen/RQ-KMeans SIDs. T5EncoderModel in recommender config is not evidence of substitution by old T5-generated SIDs.
- Original historical absolute YAML paths are preserved. PATH_MAPPING is a proposal; no path adapter or research interface implementation was executed.
- GitHub content/delivery commits, fixed-version readback, final server source export and all-stage job summary are separately reviewed by roleC. This asset PASS alone does not authorize READY_FOR_EXPLANATION or any experiment.

详细逐域路径、大小、hash、配置声明、真实请求/终态原文身份及检查项见同名JSON。未执行torch.load/pickle、Hydra instantiate、model或loader。未实施研究/接口适配。
