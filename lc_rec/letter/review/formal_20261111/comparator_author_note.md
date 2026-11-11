# 恢复比较器作者说明：验收 FAIL 与实现边界

说明者：Codex 子代理 `/root/a0_evaluation`，`formal_compare.py` 的作者。本说明属于作者对实际失败记录的解释，不是独立结果审核，也不取代其他审核者的结论。本次只读取现有证据并撰写本说明；没有修改比较代码、容差或训练配置，没有执行 GPU、远端作业或 Git 操作。

## 依据与结论

已实际读取完整 [acceptance_comparison.json](../../evidence/formal_20261111/acceptance_comparison.json)，其 SHA-256 为 `1819c4b076e4e88414534fa32f450370ceb6f04aa3f17c876613ee4df785b4bf`。报告环境为 Torch `2.6.0+cu124`、NumPy `1.24.4`，总状态与两个 rank 的状态均为 **FAIL**。

被比较的两个实际 GPU 验收作业为 `letter-a0-20261111-acceptance-cont-r3`、`letter-a0-20261111-acceptance-resume-r3`。两份 [continuous pipeline](../../evidence/formal_20261111/jobs/letter-a0-20261111-acceptance-cont-r3/job/output/pipeline.json) / [resume pipeline](../../evidence/formal_20261111/jobs/letter-a0-20261111-acceptance-resume-r3/job/output/pipeline.json) 都绑定 `87ba6556ce7237f8ef32c690165449d0d6a28294`，分别连续运行到 step 4、从 checkpoint-2 在新进程恢复到 step 4。它们的进程退出 0、`ACCEPTANCE_PATH_COMPLETE` 只说明各自有界路径结束，不代表连续/恢复一致性通过。

预先固定的逐元素判据一直为：

```text
abs(resumed - continuous) <= 1e-6 + 1e-3 * abs(continuous)
```

scheduler、global step、optimizer step/counters、RNG 和离散状态要求 exact。报告的相对差诊断使用 `max(abs(continuous), atol)` 作分母；该诊断不替代上述逐元素判据。`rtol=1e-3` 通常小于普通幅值处的单个 BF16 ULP，因此保守门槛可能拒绝存在非确定性数值差异的运行。不能在观察本次结果后放宽门槛。

## 枚举类型支持缺口

两个 rank 都有一个实现层面的失败：

```text
state['optimizer']['zero_stage'].__dict__['__objclass__']
Unsupported state value type: EnumMeta
```

比较器先对具有 `__dict__` 的 DeepSpeed 对象递归展开，后处理整数等标量。此次 `zero_stage` 枚举被展开到了其类元数据，遇到没有支持的 `EnumMeta`。这是比较器的类型分派缺口，不能据此认定保存与恢复的 `zero_stage` 语义值不同；该字段的 exact 验证没有完成。

这个缺口没有中止其他张量的比较。比较器把该问题追加到 `state_failures` 后继续遍历，两个 rank 均完整记录了另外 **453 个浮点张量超容差项**。因此，即使以后正确处理枚举，当前记录也不能变为通过。枚举分派发生在 CPU 读取既有状态之后，不能解释或造成已保存模型、optimizer、loss 或梯度范数的差异。

## 已记录的数值失败

每个 rank 都覆盖 450 个 trainable tensors、289,652,736 个 trainable elements；FP32 master partition 与两组 Adam moments 各含 144,826,368 elements。模型及 optimizer 张量的 finite 检查均通过。每 rank 的 454 项 `state_failures` 由 450 个 trainable 超容差、3 个 optimizer 超容差及上述 1 个枚举类型缺口组成。

| 检查项 | rank 0 | rank 1 |
|---|---:|---:|
| trainable 超容差 tensor 数 | 450/450 | 450/450 |
| trainable 最大绝对差 | 1.52587890625e-5 | 1.52587890625e-5 |
| `exp_avg` 超容差元素数 | 710,242 | 153,828 |
| `exp_avg_sq` 超容差元素数 | 28 | 2 |
| FP32 master partition 超容差元素数 | 144,681 | 46,414 |
| `exp_avg` 最大绝对差 | 0.005054392968304455 | 0.0008477792143821716 |
| FP32 master partition 最大绝对差 | 1.4796853065490723e-5 | 1.1045485734939575e-5 |

已支持的 optimizer step、global step、scheduler 没有不一致项。保存的 Python、NumPy、Torch CPU、两张 CUDA 设备和 loader generator RNG 快照 exact；报告中 4 个 RNG tensors 与 NumPy RNG 数组均 exact。快照相同不能单独证明之前所有随机数消费、优化和 GPU 运算轨迹相同。

事件检查在两个 rank 都通过输入/label hash、step、shape、loader position 与 LR 序列检查后，于 micro loss 检查失败。因此原 JSON 没有写入成功的 `events` 字段，也没有到达梯度范数比较。为解释这一短路，本作者另以 Python 标准库直接读取两份原始 events JSONL，按完全相同的预定公式作只读算术核对，结果如下；这不是新的独立审核：

| 恢复区间事件 | rank 0 | rank 1 |
|---|---:|---:|
| step≥2 micro loss 数量 | 8 对 8 | 8 对 8 |
| micro loss 超容差数量 | 3 | 1 |
| micro loss 最大绝对差 | 0.004361867904663086 | 0.004642486572265625 |
| step 3/4 grad norm 数量 | 2 对 2 | 2 对 2 |
| grad norm 超容差数量 | 1 | 1 |
| grad norm 最大绝对差 | 3.1667556762695312 | 3.1667556762695312 |

micro loss 是 `Trainer.training_step` 已除以 `gradient_accumulation_steps=4` 的返回值。两个 rank 的 step 2 四个 micro losses 都 exact；差异出现在后续 step 3 的 micro losses。step 3 的 grad norm 差为 `0.0069427490234375`，在原门槛内；step 4 的值为连续 `67.86307525634766`、恢复 `64.69631958007812`，超过同一门槛。这些观测定位了差异已经出现的区间，不能唯一确定原因。

原始事件入口：

- [continuous rank 0](../../evidence/formal_20261111/jobs/letter-a0-20261111-acceptance-cont-r3/run/training/events.rank0.jsonl)、[continuous rank 1](../../evidence/formal_20261111/jobs/letter-a0-20261111-acceptance-cont-r3/run/training/events.rank1.jsonl)
- [resume rank 0](../../evidence/formal_20261111/jobs/letter-a0-20261111-acceptance-resume-r3/run/training/events.rank0.jsonl)、[resume rank 1](../../evidence/formal_20261111/jobs/letter-a0-20261111-acceptance-resume-r3/run/training/events.rank1.jsonl)

## 测试范围与未确认根因

作者此前实际完成了语法/CLI 检查、标准库合成事件与离散状态比较，以及超容差、NaN/Inf、缺事件和乱序等故障注入；当时本地没有 Torch，未在本地执行真实全量 optimizer/张量比较，也未覆盖实际 DeepSpeed 枚举对象。此次服务器 CPU 的真实完整状态比较暴露了枚举支持缺口和独立数值违例。`formal_evaluate.py` 的 6 个 CPU 用例验证的是 evaluator/聚合逻辑，不是实际模型生成或正式推荐指标。

目前没有证据能够唯一判定数值违例来自 GPU 非确定性、恢复后的有效精度/参数/optimizer 行为差异，或其他尚未核实因素。不能将“RNG exact”“同 batch hash”“有限数值”或“进程退出 0”升级为恢复等价；也不能把全部数值差异归因于枚举比较缺口或直接称为允许的 BF16 舍入误差。完整 checkpoint 恢复验收仍未通过。

## 正式训练与 full test 未执行

[最终资源与任务回执](../../evidence/formal_20261111/final_resource_and_ledger.json) 记录 `formal_ledger_exists=false`、`formal_dir_exists=false`。本轮已保存的 pipeline 只有 `acceptance-continuous` / `acceptance-resume` 的有界 train 阶段；没有正式四 epoch 作业或 full-test 阶段，也没有本轮 `metrics` / `predictions` 产物。故本轮 full test 从未运行，没有 Recall/HR@5、Recall/HR@10、NDCG@5、NDCG@10 指标。不得把历史 beam20 技术验证、CPU 合成用例或本次四步验收当成正式结果。
