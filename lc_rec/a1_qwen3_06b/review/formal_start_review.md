# A1 正式启动非作者只读复核

**结论：PASS_FORMAL_INITIALIZATION_AND_EARLY_UPDATES，正式任务继续RUNNING；不是完整训练或实验PASS。** 审核者 `/root/qwen_prelaunch_review` 未参与实现、未SSH/GPU重跑，仅读取主控桥接的原始request/status/pipeline、两rank报告/事件及最新监控。

唯一job：`letter-a1-q06-20261113-normal-seed42-r0`。接收于2026-11-13 06:57:04 America/New_York（11:57:04 UTC）；最新受审监控为06:59:54（11:59:54 UTC）。runner、supervisor和worker均实际存活，workgroup live；退出码为空，符合运行中状态。

- 正式GPU RUN：`0e3e2c6f5c1a76152ea0228390d94ffc1ff75ba5`；独立CPU评价E：`445f1475cd1dd70d19881dbc1639bb311f7cccf7`，request明确固定两者与各自release。train/checkpoint/compare/evaluate/config的实际request输入SHA与固定R的Git字节逐项一致，且沿用已预验收的精确源码和配置。
- 两rank均`TRAINING`、`probe=false`、initial step0；实际argv无`--probe-stop`或`--preflight-interfaces`。固定源码的探针hooks受probe_stop条件保护，正式事件没有probe初始化或acceptance-state采集。未把12步探针当成正式训练。
- 官方Qwen模型revision、config/manifest/权重大小及model class/source SHA与预审一致；strict官方加载missing/unexpected/mismatched/errors均空。作者数据131837/49544/24772，9897完整SID；两rank固定validation文本SHA一致。
- active tied权重同Parameter/storage；196对应投影、392 LoRA张量、393独立trainable对象/161295360元素，optimizer membership exact。实际DS BF16、ZeRO2、micro8×acc4×2=global64；Int8主干和native SDPA/gradient checkpointing由相同已验收源码及运行时断言确认。loader每epoch8240 microbatches/2060 updates，四轮max8240；未提前缩短训练。
- 独立temperature CE检查继续通过。两rank实际step10→20增长，loss/LR/grad norm均有限且两rank日志一致，详见下表。

| Step | 日志loss | 日志learning_rate | Grad norm |
|---:|---:|---:|---:|
|10|15.9133|2.168674698795181e-05|64.94112396240234|
|20|13.4698|4.578313253012048e-05|34.4680290222168|

optimizer-step事件记录的更新后LR分别为2.4096385542168677e-05、4.8192771084337354e-05；上表是原Trainer日志口径，二者未混用。

最新监控两卡UUID与预审一致，显存约16580/17182MiB，温度41/39℃；MemAvailable252113178624 bytes、磁盘可用2957465305088 bytes，日志持续增长。尚无epoch边界Validation/checkpoint，不能据本次启动复核宣称它们完成。

证据入口：`evidence/jobs/letter-a1-q06-20261113-normal-seed42-r0/`；监控 `evidence/observations/letter-a1-q06-20261113-normal-seed42-r0/20261113T115951Z-monitor.json`。request SHA-256 `de414ced1828b92b75db55e2be9e1b23c2e6cb14066f5809fadaf25a8767628b`；两rank报告SHA分别`3dad4ec1bcf8930615407174d5983a52bbb79ae28f65febced86b432fa85fbd0`、`c5669e93870135c03847b92bff35f9aaecd15e0d94bc53a456ca604b3f03b074`。主控按原job继续监控；此审核不授权新增任务或额外修正轮次。
