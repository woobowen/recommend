Toys A 双候选／Sports A1-A2：启动准入与固定执行材料

当前阶段：正式请求已编译和非作者审核，尚未提交；最终状态以本目录后续健康观察回执和服务器原 job 查询为准。研究结果未预签。

运行源码为 `b44399926c98369191c2ff5c2425377c8c07d3c8`，服务器只读 export：
`/home/bwwu/grid-audits/sid-extensions-p0/grid-a-launch-20261017/snapshots/b6203414ee58a400`。
244 文件经 manifest SHA-256 回读。固定快照与后续发布证据的 Git 提交分别表达。

四条工作流各自保留原身份和预算；34 个固定依赖阶段见 `compiled/queue-config.json`，每个阶段的实际 argv、输入哈希、物理 GPU UUID、超时、依赖与门均在请求中。最终 outer request 为 `compiled/queue-supervisor-request-final.json`，固定 ID `ga-20261017-queue-v1`，纯 CPU supervisor 最长48小时，子任务继续使用原 accepted runner。GPU请求总上限39.75小时，编译阶段预留141300秒，先期真实GPU准入已消耗58.650573秒；剩余未分配额度不用于新增研究。A2原量化为两卡，3600秒worker wall按7200 GPU秒计费。

Sports 两条生成器均从零完整双表初始化，seed42、32-true、Adam0.001、wd0.0001、无scheduler、raw16、cap512、累积16、每1600 loader batches完整验证、validation Recall@5选best、patience10、max_steps320000/max_epochs10。原Beauty A训练日志确证world_size1。Toys两历史checkpoint不训练、不重建SID、不按测试指标筛除；Qwen/3584声明与512维历史上游producer绑定未唯一确认，作为本次新适配使用，不冒称唯一历史执行。

准入证据在 `reviews-and-identities/` 与 `preparation-jobs/`。保留失败 `ga-20261017-sportsa1-cpu-gate-v1`；CPU标签分组修正后v2及真实GPU检查通过。任务级集中修正2/3，详情见 `CORRECTIONS.json`。前期C1固定metadata及两个Toys GPU门具有独立旧source身份；只有三份metadata通过明确capture映射复用，未来SID/best/embedding不在例外表。`prior-preparation-configs/` 仅为旧准入实际输入；其中C1 Sports GPU请求未执行，实际A1 GPU准入来自C2。

初始阶段：Sports A1单卡训练和A2单卡T5编码分别安排GPU0/GPU1；Toys两候选在GPU1可用后分别评测/注意力。A2量化必须同时取得两卡，因此可以等待A1释放GPU0，然后按SID身份门、A2真实GPU门、训练best门进入后续三条件测试、fixed32注意力与既定CPU统计。T5编码只使用原text和原权重/tokenizer/pooling，额外纯调度20% duty cycle及空闲时间均计入3小时worker wall。只有编码明确允许受控共享（至少8GiB空闲、peer不超过2GiB、启动利用率不超过30%、两次采样）；训练与双卡量化保守等待资源，不抢占其他用户。

失败、身份冲突、空间不足或超时停止相应链，不自动改参数、不无限重跑；48小时队列期限到达后停止新增提交，已接收的独立子作业继续，未接收尾链需未来用户开启Codex后判断接续。机器队列不作研究决策，也不替代未来结果审核。

接续先只读，不重复提交：

```bash
python3 /home/addaswsw/recommend-support/sid-extensions-p0/scripts/gridctl.py --target new-server query ga-20261017-queue-v1
python3 -B /home/addaswsw/recommend-support/grid-a-launch-20261017/health_query.py --queue-deployment /home/addaswsw/recommend-support/grid-a-launch-20261017/compiled-b6203414ee58a400/queue-deployment.json --label resume
```

服务器 queue 状态、每30秒机器健康日志及requests在 `compiled/queue-deployment.json` 所指 `queue-state`；每个实际job原件在 `/home/bwwu/grid-audits/sid-extensions-p0/jobs/<job_id>/`。`RUNNING`、`QUEUED`、阶段成功与研究验收分别表示。没有安装系统包、语言包、工具链或改全局配置，见 `INSTALLATION_LOG.json`。
