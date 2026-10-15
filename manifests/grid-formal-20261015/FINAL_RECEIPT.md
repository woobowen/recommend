# GRID 正式阶段回执（进行中）

## A. 总体状态

PARTIAL：Beauty A/B六组正式全量评测及独立结果验收已经完成；Toys/Sports正式训练与新正式注意力图尚未完成。本文件会在本轮获授权工作终验后更新，当前不声称完整交付。

## B. 本轮完成摘要

继承93567预跑与Beauty MLP正式训练/best2600/fulltest/fixed32，不重跑。接管时wbw干净、无未推送提交或活动旧worker，只有原有未跟踪Python缓存；服务器代码是无.git export。

新正式编排首次CPU身份检查在未构造模型时失败。用户在原3/3之外明确批准一次限定例外，仅修复whole-cfg保存resolve=False；四组真实CPU保存/重载/实际使用子树解析exact与A/B入口identity均成功，非作者已复核。原失败保留，没有重置预算。

六正式评测固定源码提交 `2cf4c793c2e0b2e9dc7e7a9269f5e54d5c667896`、不可变export `3870cb40eb8d040a`，每组22363个目标，全部原始预测已fetch。模型/Shuffle sampler/beam/evaluator源与预跑未变。[完整指标与逐项输入](EXECUTION_SUMMARY.json)、[六条件非作者总验收](results-review/beauty-six-formal-results-independent-summary.json)、[方法与指标表](../../docs/sid-extensions/experiments/shuffle.md)。不从相近图像或小指标差异推断普遍顺序不变性。

## C. 当前逐项状态

|任务|真实结果|剩余事项|
|---|---|---|
|Beauty Shuffle A|Normal/Shuffle3/Shuffle4正式全量与独立复算PASS|对应正式attention待采|
|Beauty Shuffle B|Normal/Shuffle3/Shuffle4正式全量与独立复算PASS|主baseline B Normal及对应Shuffle attention待采|
|Toys MLP|用户批准作者RQ-VAE+STE整套、scratch全参数、raw32/acc48/val4800、12h总额度；真实CPU全val/test19412索引及fixed32生成PASS|精确新plan准入绑定、正式训练/选best/完整test|
|Sports MLP|用户批准本域Qwen/RK+作者逐物品MLP新结构实验、scratch全参数、raw16/acc16/val1600、12h总额度；真实CPU全val/test35598索引及fixed32生成PASS|精确新plan准入绑定、正式训练/选best/完整test|
|普通/Shuffle注意力|六份独立捕获计划绑定各自已审核正式eval RESULT/status/review及既有fixed32|单独准入与正式采集/绘图；B Normal同时作为主baseline，不重复采集|
|MLP注意力|Beauty正式fixed32继承，不重采|Toys/Sports正式训练成功且剩余额度足够后再采|

## D. 作业与资源

完整job_id、输入commit/snapshot、status、exit及fetch位置见 [EXECUTION_SUMMARY.json](EXECUTION_SUMMARY.json)。六次GPU请求作业worker wall合计1230.993362秒，即0.341942600556 GPU小时；包含CPU准备/复算，不是GPU忙碌计时。原预跑0.254719263611小时单列继承，不重置。

首失败 `gf-20261015-a-identity-v1` FAILED/1，无模型/GPU；其完整九成员原始export保留。配置回归 `gf-20261015-config-roundtrip-r4`、A/B两次CPU identity以及Toys/Sports两次full metadata均SUCCEEDED/0，cleanup已确认，不算正式模型评价。

正式前资源证据见 [domain-prelaunch-resources.json](domain-prelaunch-resources.json)。[来源/成本review](source-resource-review.json)仅以旧Beauty实际3600更新/36完整验证作规划情景，保守约Toys7.381h/Sports9.092h；不把3600设为停止步数，不保证12h早停，原max_steps320000保持。每域12h包含完整测试，耗尽须保存INCOMPLETE且不延长。

## E. 独立审核与版本

来源与训练源码：`/root/source_conditions_review`；root编排/CPU配置回归：`/root/implementation_review`；六正式结果原始预测/指标/身份：`/root/results_publication_review`。本人编写train.py的代理未对其自签。每份review保存真实范围与未运行边界：[source](source-formal-admission.json)、[CPU真实回归](config-roundtrip-runtime-review.json)、[六正式结果](results-review/beauty-six-formal-results-independent-summary.json)。后续attention和MLP正式结果须另审，未预签PASS。

## F. 未决事项

Toys/Sports的研究条件与12h预算已正式批准，不需再次决定。当前运行准备仅剩新增完整输入的精确准入绑定；未发现新的研究决策阻塞。原3/3与单轮例外事实保留，不能继续扩大返工。正式训练和注意力尚未启动，不将准备进度冒称实验完成。

## G. 交付材料

必要原始全预测与逐样本指标当前保留于WSL `.../sid-extensions-p0/jobs/new-server/<job>/retrieved` 和服务器原job目录；Git发布小回执/配置/指标/源码与身份。最终统一轻量ZIP将在本轮终验时生成、实际校验并给出大小/SHA/路径，当前尚未生成新ZIP。原已审核预跑ZIP及SHA不变并继续保留。

`GPT_UPLOAD_REQUIRED: YES`（最终包待本轮终验；本文件是进行中的真实状态，尚非完整交付）。没有新增安装/环境配置、清理正式资产、Parallel、额外训练或测试集调参。
