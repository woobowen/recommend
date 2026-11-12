# A0 Checkpoint 恢复专项：独立限范围审核

审核者：真实非作者代理 `/root/history_review`。以下为其返回报告，主控按原文保存。

## 审核结论

- **A0 RESUME DIAGNOSIS：`ROOT_CAUSE_NARROWED`**
- **A0 RESUME GATE：`RESUME_GATE_FAIL`**
- **正式训练判断：`FORMAL_NO_GO`**
- **A0 FORMAL NORMAL：`NOT_STARTED`**

证据支持将分歧定位到第一次恢复更新的本地反向计算，并进一步缩小到所观测的 layer28 attention q 分支。尚未唯一确认具体运算或必要运行时状态，也没有完成能够解除恢复阻塞的修复。

本审核接受上述调查及失败结论，不构成完整源码 PASS、唯一根因确认或正式训练放行。

## 审核方式与范围

审核者是非实现作者，但没有直接可用的本地 shell 或 SSH 工具。原始文件由主控通过执行桥接读取；审核者独立编写检查脚本、审查桥接提供的实际源码，并独立计算比较结果。

检查脚本没有调用主控的 `resume_analyze` 或 `formal_compare` 分析函数。最终脚本通过 AST 仅加载审核者自己编写的 imports/functions，没有执行旧报告入口。

可复查的独立材料包括：

- `independent_check.py`、`independent_check_result.json`
- `independent_extended_check.py`、`independent_extended_result.json`
- `independent_final_result.json`
- 保留的 `independent_final_reader_error.json`

本审核独立比较了状态描述符及字节 SHA、事件、标量、attention isolate 记录和预算回执；没有亲自重新读取服务器全部大型张量或执行直接 SSH 核验。服务器实时状态与最终发布回读不在本报告的直接核验范围内。

## 已独立核验的主要证据

### 持久化状态与控制条件

历史 checkpoint S1 与严格加载后的 S2，在两 rank 上均满足：

- 模型 1,221 项 tensor descriptor 完全一致，包含 224 个 SCB。
- optimizer 的 FP32 master、Adam 两个 moments 和 step tensor 共四项 descriptor 完全一致。
- 比较覆盖 key、类型、shape、dtype、字节 SHA 及标量。

在两次独立恢复、default 连续/恢复和严格原生 deterministic 配对中，检查了下一次 forward 前的模型、optimizer、scheduler、RNG、参数清单及梯度记录。相应比较未发现差异；有效 CB/SCB 亦一致。

这证明已覆盖的序列化状态及控制条件一致，不能扩称“全部运行时状态一致”。参数 inventory 本身不证明共享别名；主控提供的实际 membership 断言补充了 optimizer 与 450 个 trainable Parameter 的集合关系，仍不等价于全部别名及 partition offset 映射审核。

各配对的输入、标签及 LR 事件每 rank 匹配 10 项，均 exact，无缺失或额外项。

### 最早采集到的计算分歧

两次独立恢复的首更新采样 forward/recompute 输出一致，而本地梯度已经不同。

最终 fulltrace 将证据进一步收窄：

- 每 rank 共 449 项 projection 输入/输出，全部 exact。
- rank0 的 output-gradient 为 28 项 exact、197 项不同。
- 两份原始事件的第 351 行首次出现差异：layer28 `self_attn.q_proj` output-gradient。
- 第 352 行出现该 q_proj LoRA B 参数梯度差异。
- rank0 共 59 项参数梯度 exact、391 项不同。
- rank1 首 microbatch 的 225 项 output-gradient 和 450 项参数梯度全部 exact。

“首次”指采集点及参考执行顺序中的首次，不是对全部底层操作的穷尽定位。q_proj 的 output-gradient 已不同，说明差异从其下游计算传入，不能据此认定 q_proj Linear 自身有缺陷。

差异早于该参数梯度归约及 Adam 更新。ZeRO overlap 可能同时处理其他参数，因此不能笼统宣称整个 backward 尚未发生任何通信。

### 原固定阈值仍失败

采用原门槛：

`abs(resume-reference) <= 1e-6 + 1e-3*abs(reference)`

独立计算的 Step 4 grad norm 结果：

| 配对 | 相对差 | 原阈值 |
|---|---:|---|
| 两次独立恢复 | 1.5251878% | FAIL |
| default 连续/恢复 | 0.6938837% | FAIL |
| fulltrace 两次恢复 | 0.3784086% | FAIL |
| 严格原生 deterministic 连续/恢复 | 7.0270911% | FAIL |

default 和 fulltrace 的 Step 3 grad norm 通过，但存在非零差异。通过某个标量容差不证明计算完全相同。

default 与 fulltrace 的两组恢复后 micro-loss，每 rank 均为 4/8 exact；超阈值数分别为 rank0 三项、rank1 两项。没有修改阈值或减少正式比较对象。

### Attention 隔离与未决解释

两 rank 捕获的真实 attention case，False/True 各重复五次，output、dq、dkv 在组内及组间均 exact。

这是所捕获 case 的有效阴性结果，不能确认 FlashAttention 是根因。捕获记录没有标注所属 layer，也不能将该阴性结果推广到 layer28 的具体调用。

严格原生 deterministic 配对仍失败，证明该组合不足以消除故障；Flash 显式 deterministic flag 仍为 False，不能据此排除自定义 backward。

当前仍未直接区分：

- attention q 内部 backward；
- unpad、rotary 及相关反向路径；
- 未捕获的非持久 rotary 缓存及其他必要运行时状态；
- 执行时序或 allocator 条件对相关计算的影响。

不能把这些未决项概括为已经证实的“GPU 非确定性”。

## 源码限定审核

固定源码比较基准：

- 历史 RUN：`87ba6556ce7237f8ef32c690165449d0d6a28294`
- fulltrace RUN：`be15ad0a7c8b3cb8f1adac2c4668e7e6973b7164`

独立 Git 字节检查确认 `formal_pipeline.py`、`formal_runner.py`、`formal_control.py`、`formal_ledger.py` 和 `compat.py` 未改变。

正式配置亦未改变，SHA-256：

`9e058cecb7151ebee3ab714ab8328af0008537156cd96684c988d89e7ee10600`

实际最小 diff 与桥接审查原文一致：

- `formal_train.py` 增加 opt-in diagnosis、受限 checkpoint 身份允许项和观测调用。
- 正式 pipeline 不启用该参数。
- hook 返回原 Tensor/梯度；CPU hash 会同步 CUDA，因此保留执行时序影响这一限制。
- `formal_compare.py` 增加 Enum class/member 的 exact 比较，没有修改浮点门槛。
- 没有实际恢复算法修复。

Enum 覆盖修复可以消除此前的不支持类型缺口，不能解释或消除浮点差异。

## 完整比较与保留的失败

主控提供的固定比较器执行回执显示，五套完整比较均以 exit 2 真实 FAIL：

| 比较 | rank0 浮点失败项 | rank1 浮点失败项 |
|---|---:|---:|
| 历史连续/恢复 | 453 | 453 |
| default | 453 | 452 |
| 两次恢复 | 453 | 452 |
| fulltrace | 453 | 452 |
| 原生 deterministic | 453 | 453 |

这些是比较失败项目数，不是同等数量的独立训练错误。EnumUnsupported 已不再解释这些浮点失败。审核者没有独立逐元素重算服务器所有大型张量；本表属于固定比较器执行回执，独立标量及事件检查另行支持 FAIL 判断。

以下失败必须保留：

- CPU-suite 将 Enum name/value 两项差异加一项浮点差异错误地预期为两项，实际三项，原运行 FAILED。
- loader replay 原运行因输出父目录不存在而 FileNotFound；后续相同源码及输入在已建立 output 目录完成复核，不改写原失败。
- 审核 reader 曾按扩展名遗漏三个文件；ERROR 已保留。后续明确逐条比较全部 20 项后成功，包含 `README.md`、`tokenizer.model` 和 `zero_to_fp32.py`。

辅助复核成功不能覆盖完整恢复比较的 FAIL。

## 预算与原件保护

独立读取 10 个 GPU request 的原始 request/status/pipeline 回执，按：

`max(2 × outer_wall_seconds, pipeline_gpu_device_seconds)`

逐项重算，结果为：

**92.95776905852041 GPU-device-minutes**

低于 120 设备分钟硬上限。预启动锁拒绝请求亦保守计费，没有因未启动模型而隐去。

原 checkpoint 的初次及最终回执已独立比较：

- 文件数均为 20；
- 文件大小全部一致；
- 20 项 SHA-256 全部一致。

这是对初末哈希回执的独立核对，不声称审核者亲自重新读取服务器大型原件。

## 最终停止判断

本轮已将笼统恢复不一致收窄为有原始事件支持的本地反向分歧，但尚未识别并修复唯一原因。

最有区分力的后续测试是绑定 layer28 的实际调用，分别比较 attention 输入、`dout`、原始 `dq`，再分离 unpad/rotary 边界；这属于下一次专项授权的候选，不是本轮继续运行的要求。

**本轮停止新增 GPU 作业。保持 `ROOT_CAUSE_NARROWED / RESUME_GATE_FAIL / FORMAL_NO_GO`，不启动正式 A0、A1、Shuffle、额外 seed 或数据集。**
