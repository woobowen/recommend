# G3 Beauty 补充 CPU 收尾

本实现沿用原 G3 Beauty 正式预测、选模和统计协议，对应固定运行提交
`21d495b0be5b606853ec7cec68a3eaa1b58acbd7`。原 `paired-a0` 的超时、原账本和
未提交的 `collect` 作为历史保留。该 CPU 任务在原控制器之外单独计费。 [原失败与已完成阶段](../formal_launch/observations/20261120T053043Z/README.md)

- `closure_config.json` 固定原任务、输入、CPU 授权、资源上限、runner 身份和数值协议。
- `deploy.sh` 将 Git 提交中的文件按完整 SHA 放入独立服务器目录。
- `launch.py` 以既有 CPU runner 启动作业，在同一稳定目录审计累计用量。它确认原作业已清收，
  按原预算和追加预算较小的余额扣除清收预留，按作业 ID 与请求哈希识别已有作业。
- `closure.py` 对原训练产物执行一次冻结验证，流式复核三份预测并独立计算 16 项指标，
  检查原文件身份在消费前后保持一致。随后按冻结 PCG64 协议计算两组配对及独立列式对照。
- `review_raw.py` 由非作者实现。它直接读取三份服务器原始预测、原评价和新产物，
  在同一 CPU 预算账本内作为独立作业运行。
- `test_guards.py` 在临时夹具检查输入变化、写占用、防覆盖、失败计费和未账本作业。

原生产调用链每个配对条件重复五次训练产物验证。本实现把一次真实验证的结果和输入身份
绑定到本次 CPU 进程，并在完成配对、结果收集前后检查原始输入。原训练、生成、评价源码
位于冻结提交，CPU 配对仍调用固定版本的 bootstrap 与独立列式算法。
[冻结调度](https://github.com/woobowen/recommend/blob/21d495b0be5b606853ec7cec68a3eaa1b58acbd7/lc_rec/a0_beauty/production_pipeline.py) · [冻结配对统计](https://github.com/woobowen/recommend/blob/21d495b0be5b606853ec7cec68a3eaa1b58acbd7/lc_rec/a0_beauty/paired_beauty.py)

[正式结果及审核证据](results/README.md)
