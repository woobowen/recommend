# A 数据与源码按需恢复独立审核

复核者 `/root/audit_assets`；观察时间 2026-10-14T01:29:31.603308+08:00。状态 `PASS_NONAUTHOR_SELECTED_SOURCE_AND_1902_DATA_IDENTITY; LFS_SID_CHECKPOINT_REVIEW_PENDING`。

从冻结请求、accepted 与真实终态核对 `sid-extensions-p0-restore-v1` 成功、退出0、cleanup已核、任务成员为空。独立全读2025行输出与冻结worker，逐项和plan/旧capture/固定baseline清单比对；服务器源码导出TAR的123个安全普通成员、WSL发布123原件逐字节SHA一致。仅3个必要batch的捕获manifest和生产chunk/whole-TAR hash对应；未重跑全库审计。

独立通过本阶段transport在远端实际重哈希37文件（24数据、13关键源码，共61,771,232字节），全部匹配。生产job已实际读完1902数据文件的SHA，本审核没有第二次重哈希全数据。

|域|实际文件数|字节|准备方式|
|---|---:|---:|---|
|beauty|549|1385691766|已有同SHA实体symlink，只读使用政策|
|toys|480|1263466388|按需独立提取，0444/nlink1|
|sports|873|2102589679|按需独立提取，0444/nlink1|

- All 1902 file hashes were actually read by frozen production worker; independent reviewer rehashed 24 data files and 13 key original source files (61771232 bytes) remotely, not all 1902 a second time.
- Beauty target files are symlinks to existing /data/bwwu/amazon_data/beauty entities with actual mode0664; read-only use is a recorded policy, not OS-enforced immutability. No writable hardlinks connect originals to work assets. Sports/Toys selected newly extracted files have0444 and nlink1.
- Identity equality of TFRecord files does not establish current loader execution or new Qwen SID item-axis pairing. Old T5 SID semantic/CPU evidence is not used for Qwen.
- SID, keyed SID mapping, checkpoint and small effective run configuration LFS payloads are pending final assembled whole-OID review. No current complete three-domain asset PASS or READY_FOR_EXPLANATION claim.
- TAR reader traversed only required three original batches. Selected body hashes and whole TAR/chunk hashes were checked; no 302-chunk/full-library audit rerun.
- No research source, model, loader, torch/pickle constructor was imported or executed; no runtime experiment or interface changes.

详细身份、请求和结果原文hash、逐源码核对见同名JSON。

审计工具更正：首次误将TAR实际的 `GRID/` 成员前缀视为多余。原失败报告已保留；按真实前缀重建比对后，123个成员全部实际读取并匹配来源/恢复hash。未修改材料，不是准备任务的实质失败。详见 `A_EXPORT_MEMBER_PREFIX_CORRECTION.json`。
