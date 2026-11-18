# G3 Beauty 正式启动与初期监视

本轮获得授权的范围为一次正式单卡训练，以及其后 Normal/OFF、S3、S4 生成评价。
训练使用固定 G3 Beauty 四位 SID 和 LETTER-LC-Rec 下游，从原始 LLaMA-7B 与规定初始化开始。
正式计算源码提交为 `21d495b0be5b606853ec7cec68a3eaa1b58acbd7`，
上一轮预跑交付为 `6db8ab4ec98f5692014eeac02b03f3f55015d1eb`，
最终准备交付为 `4075cf138fcbc2fd2e115001fd9fb5e8b63f6409`，
计划收口检查脚本为 `780040a4a3743d124004811495f9c45e6a3273ef`。
交付提交仅包含本轮回执、观察、核验与文档。固定运行导出和服务器输入沿用原身份。

- [本轮用户授权](authorization.json)、[文件 SHA-256](authorization.sha256)、
  [网页最终增量二审依据](web_final_incremental_review.json)。
  授权按用户委托由 Codex 序列化，时效为实际生成起 96 小时。
- [固定正式计划](../single_gpu/final_closure/final_plan.json)为 215,484 字节，
  含 541 项输入。原始文件 SHA-256 为
  `a56076cf07d67bc088442dc35f863134081b455f551fc45d2c6bfb2bfd246407`，
  生产 canonical digest 为
  `526987fa87f0feb565a7a9f42bc58f42e3121b9d9a13cea4613e47f842c394b6`。
  配置 SHA-256 为 `8e4c68c45b200d81624f25dd1c9bc204b5324f5da98e46ede6c68bc82a834853`。
  G3 tensor SHA-256 为 `d41adcfb22a9a47d5e0424c068f4c74f33cc9fdbaf5db08540fbfa0e3d720e18`，
  bridge.csv SHA-256 为 `10a7ca66dba43b69232a7c96b48d3287df9314a70a9ec9083ff00acfcf2a8c4f`。
  授权的 `asset_sha256` 按生产 schema 固定为 index、inter、item。
- [预存启动脚本](launch.sh)、[启动前检查脚本](precheck.py)、
  [实际 SSH argv](submission.json)、[本地 SSH 返回值](submission_result.json)、
  [stdout](submission.stdout)、[stderr 与调用跟踪](submission.stderr)。
  初次调用前逐项比对远端传输文件的 SHA。现存正式计划和 preparation-plan 保持各自身份。
  独立 prepare 输出见 [production_prepare.json](production_prepare.json)。
- 初次提交仅一次。正式封存前检查登记、目录、原 job 命名空间和本实验进程，
  确认 GPU1 UUID `GPU-59cac3bd-3645-668b-79c8-839e0d878893` 无已有计算进程。
  脚本先调用 `authorize` 并成功，再调用 `production_control.py start`。
  本地调用封装的一次参数计数检查在发出 SSH 前失败，校正后调用已预存的 argv。
  启动脚本的 Python 断言实际处于启用状态，远端解释器记录为
  [optimize=0](launch_python_optimize.stdout)。运行中的进程和账本未经历恢复或重试。
- [观察摘要](monitoring_summary.json)、[非作者直接复核](non_author_review.json)、
  [新连接启动确认](observations/00-reconnect.json)和[训练 rank 确认](observations/01-worker.json)。
  各次采样通过已有 Windows OpenSSH 通道和新的 SSH 连接查询 `new-server`。
  末次采样前另执行一次已封存计划的生产 [query 调用](controller_query_invocation.json)，
  保存了[实际返回值和原账本](controller_query_result.json)。

## 运行中证据

每份 [observations](observations/) 记录服务器采集 UTC、runner 查询 argv 和真实输出、
`/proc` 启动标识与批准范围内的环境变量、GPU1、内存、空间及账本快照。
文件的字节范围从打开时的大小确定，活动日志取有界后缀。
每份证据均保留独立的采集时间、来源 mtime、偏移、字节数和 SHA-256。
采集跨多个文件和查询，记录各自的时间与身份。末次采样的关键文件
可直接在 [latest](latest/) 阅读。原始字节也保存在观察 JSON 的 base64 字段中。
[verify_observations.py](verify_observations.py) 验证实际样本、稳定身份和更新进展。

观察锚点采用实际 `formal_train.py` rank-0 进程启动时间，来自
`/proc/stat` 的 boot epoch 和进程 start ticks，换算的 UTC 存在亚秒级误差。
采用超过 30 分钟的保守采样。runner 接受时刻和阶段 worker 的 `started_at`
另行保留在原始回执中。观察中的 optimizer update、loss、LR 和梯度范数来自
生产事件。账本消耗基于 `state.json` 快照，按该文件的 mtime 报告。
启动前准备和窗口后发布用时均未计入训练观察时长。

## 下次查询同一作业

服务器正式根目录为 `/data/bwwu/lc-rec/g3-formal-20261118/21d495b0be5b`，
其中保留 `plan.json`、`plan.identity.json`、`state.json`、`controller.log`、
`launch-20261118/authorization.json`、`requests/` 和 `training/`。
正式恢复入口由原计划和原账本约束。下一轮先核实 `state.json`、
实际 controller 进程和 runner 查询，再决定是否需要另一步操作。

在 `new-server` 上使用以下监视入口。生产控制器 `query` 会验证计划输入并刷新账本。
日常轻量检查可先读原 `state.json` 和 runner 状态，避免重复进行计划重校验。

```bash
root=/data/bwwu/lc-rec/g3-formal-20261118/21d495b0be5b
release=/home/bwwu/grid-audits/sid-extensions-p0/g3-production-20261118/21d495b0be5b/release
python=/data/bwwu/envs/letter-20261110/bin/python
runner_home=/data/bwwu/lc-rec/a0-beauty-20261115/production-closure-20261115/formal-candidate-5b2917d
cat "$root/state.json"

# 初次观察的原始训练 attempt，后续阶段使用 state.json 中相应的 job_id。
job_id=letter-a0-beauty-formal-g3-beauty-seed42-train-a0
/usr/bin/python3 -I -S -B "$runner_home/runner-beauty.py" query "$runner_home/jobs/$job_id"

# 已封存后，用生产入口重校验并刷新原账本。
(
  cd "$release/lc_rec/a0_beauty" &&
  PYTHONDONTWRITEBYTECODE=1 "$python" production_control.py query --plan "$root/plan.json"
)
```

控制器按固定依赖推进
`train → verify_best_adapter → normal → normal_eval → s3 → s3_eval → s4 → s4_eval → paired → collect`。
后续生成各条件共享本次正式模型的 `training/best_adapter`。
本轮只负责完成初期监视，服务器控制器接续完整流水线。
安装记录为空，沿用既有解释器、依赖、SSH、驱动和配置。
