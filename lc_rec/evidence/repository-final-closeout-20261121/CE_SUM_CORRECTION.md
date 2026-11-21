# 六任务 CE 权重末位差异更正

2026-11-21 对上轮仓库整理报告及非作者记录作定向复核。原话和原值保留。上轮将 14 项末位差与 NumPy 1.24.4、1.26.4 关联，版本值来自环境记录。本轮使用同一批保存权重在三套实际 CPython 解释器上复核，14 项差异均可由 Python 内置浮点 `sum` 的实现变化解释。

固定运行源码 `c4a8aef801074f1579d113e041ebf3a24380746c` 对应 `analyze_grouping.py` 的 SHA-256 为 `aaeb799cb5888ff9a54b13d0bf0f77e0de20d37ed6fe179a9cba91f4869a7874`。其 `ce_grouping` 第 123 行按任务 `sum(row['sample_weight'] for row in rows)`，第 128 行 `sum(sample_weights)`。保存的 `per_sample` 在各 mixed global128 计划中按原 rank 和微批顺序取值，任务成员从同一份样本清单读取。正向累加和 `math.fsum` 比较采用二进制 64 位十六进制身份。Python 3.12 官方文档记载内置 `sum` 的浮点算法在该版本改变。[冻结源码](https://github.com/woobowen/recommend/blob/c4a8aef801074f1579d113e041ebf3a24380746c/lc_rec/six_task_lora/scripts/analyze_grouping.py#L90-L137) · [Python 3.12 文档](https://docs.python.org/3.12/library/functions.html#sum)

| 实际核对 | 环境 | 14 项对应 |
|---|---|---|
| 内置 `sum` | `new-server` 原任务解释器路径，CPython 3.10.22 | 14 个原历史值 |
| 内置 `sum` | 本地 WSL 已有 CPython 3.11.14 | 14 个原历史值 |
| 内置 `sum` | 本地 WSL CPython 3.12.3 | 14 个上一轮整理值 |
| 逐项相加及 `math.fsum` | 三套解释器，以同序列、同浮点值复算 | 前者 14 项等于原历史值，后者 14 项等于整理值 |

三个进程均未导入 NumPy。14 项最大绝对差 `2.220446049250313e-16`。这种归因只覆盖两处浮点归约的已观察差异。原整套历史试验、原历史 NumPy 环境和前轮输出各保留自己的身份。本轮未改变训练损失、原始记录、任一实际运行结果或统计阈值。[历史差异记录](prior-ce-differences.json) · [服务器 3.10.22](ce-sum-server-cpython-3.10.json) · [WSL 3.11.14](ce-sum-wsl-cpython-3.11.json) · [WSL 3.12.3](ce-sum-wsl-cpython-3.12.json)

在实际 `wbw-lc-rec` 根目录运行，使用 Python 3.12 可直接复核上一轮整理值。已存在的 Python 3.10/3.11 用 `--expect historical` 核对历史值。输出为新建临时目录中的 JSON。

```bash
out=$(mktemp -d)
ev=lc_rec/evidence/repository-final-closeout-20261121
six=lc_rec/six_task_lora
python3 -B "$ev/audit_ce_sum.py" \
  --source "$six/scripts/analyze_grouping.py" \
  --grouping "$six/evidence/jobs/lc-six-20261110-grouping-r2/output/grouping/grouping.json" \
  --samples "$six/evidence/jobs/lc-six-20261110-cpu-r0/output/workload/samples.json" \
  --differences "$ev/prior-ce-differences.json" --expect replay > "$out/ce-sum.json"
```
