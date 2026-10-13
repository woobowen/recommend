# Parallel：真实非自回归正文尚未恢复

## 当前完整交付的含义

`SOURCE_FACT`：保全的 `tiger_generation_model_parallel.py` 与 shuffle 文件均为61418B，完整 SHA-256 均为 `e03082ad87bfc842c05467580fc479eff4345c6a0d18c5b2f7faf320d4f96c80`。`generate` 952–1061 仍循环 H 次，并把之前生成的码位作为 decoder前缀；它不是已经恢复的非自回归实现。

真实非自回归算法状态：`NOT_RECOVERED`。后续设计状态：`RECONSTRUCTION_PENDING_USER_DISCUSSION`。本阶段没有创建 Parallel 类、空壳或论文移植，也没有将文件改名后冒充新实现。

## 已知链与缺失链

`SOURCE_FACT`：已知普通 baseline 的入口/config/item→SID/offset/共享embedding/encoder/AR decoder/loss/beam/evaluator见 [总图](../CODE_READING_GUIDE.md)。现存 parallel 文件只能沿 shuffle 的双表、hook、encoder向量重排、AR生成和attention路径解释。其输入输出仍可按该文件的 `[B,L]`→encoder→逐码位生成 `[B,K,H]` 静态追踪，不能作为非AR流程图。

|问题|现存证据|仍未知的非AR正文|
|---|---|---|
|decoder query|历史上下文出现多个BOS/输出头|如何排布query、是否带位置/码位标识|
|attention mask|权重形状不能给出mask|四个目标query之间允许看到什么，是否泄露真值|
|forward/loss|配置与checkpoint键可提供结构线索|目标如何构造，各位置loss如何对齐/汇总|
|生成候选|多个输出头不证明生成协议|各位Top-K如何组合、如何过滤SID合法性、如何定义分数|
|参数共享|历史有双SID表线索|具体共享、冻结与初始化策略|
|历史身份|保留4BOS运行config/overrides|哪些Python、数据、权重、候选协议实际共同生产指标|

上表右列均 `UNKNOWN`。左/中列中的源码事实与历史结构线索分别标 `SOURCE_FACT` / `HISTORICAL_OBSERVATION`；不以权重或日志重建出不存在的mask/forward正文。

## 什么算研究目标，什么尚未选择

`PROPOSED_NOT_IMPLEMENTED`：可讨论的研究目标是固定SID与数据条件下，同时预测目标各码位，考察目标码依赖与候选组合影响。是否使用多个BOS、独立query或其他形式尚未选择；本阶段不开展新的论文综述或实现选型。四个输出预测头与Transformer attention head是不同概念。

`STATIC_INFERENCE`：若未来真正去掉目标前缀的串行依赖，decoder query和mask、loss、生成分数与候选组合都需要新定义；不能把现存 `for hierarchy` 删除就认为研究口径保持不变。尺寸只能写待设计符号，例如并行query `[B,H,E]` 是可能设计示例，**不是当前源码或checkpoint观察**。

`UNKNOWN`：真实非AR参数量、梯度路径、训练初始化、冻结、训练预算、推理复杂度与Top-K组合均待算法定义后推导；本阶段不编造具体数字，也不执行性能计时。普通main/Qwen历史指标与旧T5/4BOS历史线索分开，统一历史未重跑，不声称非AR改进。

## 后续验证清单与用户决策

`PROPOSED_NOT_IMPLEMENTED`：讨论并定义完整算法后，未来应核query与目标对齐、不读取目标真值、mask语义、loss逐位置汇总、共享参数与梯度、SID合法候选与分数、候选预算和AR基线公平性；新增实现/运行须另行授权。这里没有可自动开跑的框架。

用户需要先决定：希望独立各码位还是允许query交互；合法SID过滤在哪一层；候选数怎样与AR的K比较；训练/推理预算怎样控制；是否接受明确的新重建版本而非历史复现。推荐先读两个同SHA文件的 generate，再读4BOS保存配置；这能解释为什么“材料准备齐备”不意味着“原算法已存在”。


## 冻结源码证据

本表的行号由实际发布字节核对；源码集合身份 `8b3149a98731ba7857e446e0f488489be0715181b841f8c7f9f169d3c4e16ac3`。baseline链接固定完整SHA；原件链接在CONTENT_COMMIT发布后由交付元数据绑定该完整SHA，原始捕获版本和每文件完整SHA始终保留。

|文件/符号/实际行范围|版本|源SHA-256|
|---|---|---|
|[references/wmh-original/GRID/src/models/modules/semantic_id/tiger_generation_model_parallel.py::SemanticIDEncoderDecoder.__init__ L642–776](https://github.com/woobowen/recommend/blob/84d4533fbdd6b010f54d671b433a9952608911ca/references/wmh-original/GRID/src/models/modules/semantic_id/tiger_generation_model_parallel.py#L642-L776)|`A800-grid-20261010-v1`|`e03082ad87bfc842c05467580fc479eff4345c6a0d18c5b2f7faf320d4f96c80`|
|[references/wmh-original/GRID/src/models/modules/semantic_id/tiger_generation_model_parallel.py::SemanticIDEncoderDecoder.generate L952–1061](https://github.com/woobowen/recommend/blob/84d4533fbdd6b010f54d671b433a9952608911ca/references/wmh-original/GRID/src/models/modules/semantic_id/tiger_generation_model_parallel.py#L952-L1061)|`A800-grid-20261010-v1`|`e03082ad87bfc842c05467580fc479eff4345c6a0d18c5b2f7faf320d4f96c80`|
|[references/wmh-original/GRID/src/models/modules/semantic_id/tiger_generation_model_parallel.py::SemanticIDEncoderDecoder.model_step L1214–1267](https://github.com/woobowen/recommend/blob/84d4533fbdd6b010f54d671b433a9952608911ca/references/wmh-original/GRID/src/models/modules/semantic_id/tiger_generation_model_parallel.py#L1214-L1267)|`A800-grid-20261010-v1`|`e03082ad87bfc842c05467580fc479eff4345c6a0d18c5b2f7faf320d4f96c80`|
|[references/wmh-original/GRID/src/models/modules/semantic_id/tiger_generation_model_shuffle.py::SemanticIDEncoderDecoder.generate L952–1061](https://github.com/woobowen/recommend/blob/84d4533fbdd6b010f54d671b433a9952608911ca/references/wmh-original/GRID/src/models/modules/semantic_id/tiger_generation_model_shuffle.py#L952-L1061)|`A800-grid-20261010-v1`|`e03082ad87bfc842c05467580fc479eff4345c6a0d18c5b2f7faf320d4f96c80`|


## 已保留的历史记录入口

下列全部 `HISTORICAL_OBSERVATION`，仅记录原文件和test行位置，未选择/重算指标，未证明当前Python唯一生成它们；T5历史上下文不替换Qwen工作资产。

- [logs/train/runs/2025-12-02/00-47-41 原config](https://github.com/woobowen/recommend/blob/84d4533fbdd6b010f54d671b433a9952608911ca/references/wmh-original/GRID/logs/train/runs/2025-12-02/00-47-41/.hydra/config.yaml)，SHA-256 `a548943c6f1fca2a726b7daacaa8734a173f450cfc59c9dd526ef5dff0e538a3`；同目录overrides/hydra原文保留。 [完整CSV](https://github.com/woobowen/recommend/blob/84d4533fbdd6b010f54d671b433a9952608911ca/references/wmh-original/GRID/logs/train/runs/2025-12-02/00-47-41/csv/version_0/metrics.csv) 的原test行：64，SHA-256 `c4f05515c558b50640a8558229ff29c1d28c06a73948488dfac950aa1cd06ebc`。
