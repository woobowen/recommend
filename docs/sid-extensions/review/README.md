# 真实非作者审核入口

主控与阅读文档作者为 `/root`。三位真实子代理仅审核/写各自报告，不修改研究源码、主控STATE或Git index：

|角色|实际代理|非作者范围|
|---|---|---|
|A 来源/资产|`/root/audit_assets`|旧v6、固定baseline、全部数据来源关系、原件/配置字节、服务器抽核及最终资产|
|B 源码/可讲解性|`/root/audit_code`|直接原文/配置追踪，源码符号与hash/行号，四说明十项要求、历史边界|
|C 部署/发布|`/root/audit_publish`|runner/请求/helper静态与纯文件fixture，秘密/属性/许可，服务器原件重哈希，候选和GitHub固定内容回读|

实际报告文件将随最终交付保存在本目录；原始详细记录位于本阶段支持目录 `review/`。初审失败与已关闭问题保留，不以最终PASS覆盖历史。`ROUND1_CHANGE_RECORD.json`、`ROUND2_CHANGE_RECORD.json`和`IMPROVEMENT_LOG.md`记录全阶段集中修正，最多3轮，跨任务/会话不重置。

旧审计包的非作者结论只用于旧阶段继承；本次A/B/C报告必须有实际新原文/字节/命令结果。重复主控摘要不算独立核验。Parallel原正文缺失、原件静态接口问题、未来运行/用户理解未完成属于明示边界；必需载荷/发布/最终服务器映射或本次非作者审核未闭合时不得称READY。

最终CONTENT_COMMIT读回记录在本目录C发布审核中；DELIVERY_COMMIT及最终回读/服务器对应的完整SHA和收据保存在外部支持目录 `FINAL_DELIVERY_RECEIPT.json`，并由Codex最终回复给出。这避免报告自引用其自身commit，也避免预写未来读回成功。
