# Instruments A0 第一组收尾记录

原实验和原交付的网页二审状态为 `FIRST_GROUP_COMPLETE_WEB_REVIEW_PASS`。本轮说明、离线导航和新版 ZIP 待网页差异复核。本轮使用零正式 GPU 预算，研究代码、有效配置、模型与数据、预测、CSV、NPZ 和 bootstrap 来自原固定交付。 [第一组结果](../README.md)

## 观察授权

原要求为观察 15 分钟、每 5 分钟检查，依据是用户转交的审后回执。后续授权已直接从同一任务的用户消息中取得。Codex 会话 `01a1f9b2-a7d9-7ba2-90d0-5fdccc23578e` 的本地记录 `~/.codex/sessions/2026/11/20/rollout-2026-11-20T11-03-17-01a1f9b2-a7d9-7ba2-90d0-5fdccc23578e.jsonl` 第 812 行是 `role=user`，时间 `2026-11-20T17:07:45.063Z`，消息以“现在调整本轮 Codex 在线工作的返回时点”开头。

第二节授权“对已经实际启动的正式任务，从本次查询时刻 T0 起，观察30分钟，在以下七个时间点检查”，分列 T0、T0＋5、＋10、＋15、＋20、＋25、＋30。实际七次观测是另有回执的执行事实，网页二审已核验。 [七点原始回执](../../launch-observation-20261120/README.md)

## 版本和包内导航

- 正式 RUN `e2f54ebd2d7709821de1071693f91644f5ed9a30` 与串行控制 `4f5758ffb578f3a8d3d06a54262f59d3f63db370` 仍对应当时的服务器部署。
- 网页二审对应结果交付 `457c1c60853582693633cdae741e883f2bf2d643` 和原 ZIP。原包仍位于 Windows Downloads，大小 34,894,063 字节，SHA-256 为 `3dc6a752625137d25fe18ebac9685acbca2fa00706d30c8815bb327cd8257f48`。
- 新收尾版本仅发布说明、导航和打包校验程序。新版根导航链接实际 ZIP 成员。原审 README、清单、部署映射及历史说明按来源保留，包内身份文件记录收尾发布、原交付和两套运行版本。
- 本轮检查针对文档差异、成员保真、来源及离线入口。此前网页独立研究复算和原内部终审继续沿用。最终归档 CRC、解压、导航及命令检查由新版打包程序执行，GitHub 实际回读与版本对应收录在新版包内。

非作者 `/root/closeout_diff_review` 直接检查了原授权消息、冻结 CSV 的点估计顺序和跨零区间、旧 ZIP、拟交付根导航与清理保留边界。草案复核未发现阻塞。离线入口检查针对路径与 CLI 可用性，沿用已保存的研究复算结果。

## 临时件清理

本轮定向复查了第一组支持目录、Downloads 和服务器原审核导出传输位置。没有发现新增的可确认残留。本轮未删除既有文件。早期以 commit 命名的部署 tar.gz 属于运行阶段传输物，未纳入本次删除。服务器正式审核导出仍含 19 个文件，原导出作业 `SUCCEEDED` 且无活动 worker。新版构建的单次解压使用自动回收的临时目录。

前次收尾已删除下列可重建的传输或检查副本。以下为前次执行回执，空间按文件系统前后实测记录。

| 位置和用途 | 具体路径 | 逻辑字节 |
| --- | --- | ---: |
| WSL 冗余传输包 | `/home/addaswsw/recommend-support/letter-shuffle-v2-20261120/cpu-review-export-r0.transfer.tar.gz` | 30,823,885 |
| WSL 临时打包目录 | `/home/addaswsw/recommend-support/letter-shuffle-v2-20261120/export-offline/` | 113,044,100 |
| WSL 重复解压检查副本 | `/home/addaswsw/recommend-support/letter-shuffle-v2-20261120/final-zip-check-457c1c6/` | 124,098,994 |
| 服务器冗余传输包 | `/data/bwwu/lc-rec/letter-shuffle-v2-20261120/cpu-review-export-r0.transfer.tar.gz` | 30,823,885 |

前次实测释放 WSL 269,156,352 字节，服务器 30,826,496 字节。本轮复查确认上述目标已不存在，原 ZIP 和正式运行原件仍是保留副本。
