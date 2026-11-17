# 实际修改和结论边界

## 上游保持的协议

沿用 [GRID 基线](shared-code.md)：Qwen2.5-7B-Instruct、3584维、原WMH池化、右填充、128 token；RQ-KMeans 3×256、30步、seed42、残差归一化、第四位去重。Beauty/Toys使用已有SID，Instruments仅换输入数据并调用原量化流程。本实验没有用LLM2Vec替换Qwen生成这些SID。

归一化核查发现，Qwen编码输出的原始商品向量并非单位向量：Beauty范数均值约170.074，Toys约169.952。`normalize_residuals=true`发生在量化流程内部，不能把原始商品向量或最终码本向量一概描述成单位向量。证据位于 `audits/qwen_normalization/`。

## 自动分类 C1/C2

- 增加lm-format-enforcer JSON Schema约束，禁止重复JSON键，检查商品ID完整和类别ID合法。
- 分类输出失败时递归拆分批次；最小单件仍失败则停止，不静默漏商品或编造标签。
- Toys全局合并候选类别不收敛，因此改为每次12个候选、以源ID为必填键的逐项功能类型归一；固定两遍，每批最多两次结构重试。复用原1,500件、43批类别发现结果。
- Toys归一过程中384个候选名称→286个暂定类型→254个暂定细类、143个父类；这是中间草案。之后仍执行全量草分类、修订和冻结，**最终实际结果为183细类、122大类**，不要用草案数量替代最终数量。
- v7将完整类别词典由逐项重复JSON键改成“列名一次＋行值数组”，保留全部类别ID、名称、父类、定义及非空边界字段，仅省略全空的include/exclude列；没有为装入上下文截断类别或商品。
- Beauty保留v1冻结类别和标签，没有按Toys新协议重跑。v2–v5曾尝试全局提纲、字母顺序、已有类复用等方案，因漏类或串条等问题被拒绝；这些输出不属于正式标签。v6方法＋v7表格输入为Toys最终执行协议。
- 类别质量未经完整独立人工标注验证。pilot主要检查结构与完整性；按当时授权继续全量，不能将通过pilot写成人工准确率验证。
- 两数据集分类均读取原始Categories字段。Beauty类别表存在层级混杂/类别重叠，Toys仍有粒度碎片化/近义类别，因此两个数据集的纯度绝对值不适合直接作模型优劣比较。
- 分类器读取完整现有文本，而Qwen上游最多128 token：原Qwen tokenizer核查Beauty115件、Toys8件超过上限，存在小部分分类器读取信息更多的差异。不是重新编造description。

`archive/classification_beauty_v1/` 和 `archive/classification_toys_v7/` 保留原代码和配置；原服务器控制器在 `archive/classification_control/`。本次没有为了发布重新跑分类或调整类别。

## LLM2Vec两卡适配 E1–E6

- 严格使用Llama3底座，不把现有Llama3.1权重与Llama3适配器混用。
- 底座32层前16层放逻辑GPU0，后16层放GPU1；embedding在0，最终norm在1。用CUDA_VISIBLE_DEVICES指定物理卡对。
- 合并MNTP，再加载supervised；unsup-SimCSE是替代的第三阶段适配器，并未叠加或用于本轮正式结果。
- 保留LLM2Vec官方输入包装、双向注意力和正文mask；直接forward，避免官方多可见GPU `encode()` 路径复制模型或把跨卡模型移到单卡。
- 2080Ti使用FP16和SDPA，未使用BF16、FlashAttention2、bitsandbytes或低比特权重。禁用PEFT的可选bitsandbytes派发，避免环境中的无关CUDA兼容性问题。
- batch=2；仅按文本长度排序编码，不按SID筛选；每128件持久化检查点，恢复时核对输入/代码/配置哈希。
- 先转float32再L2归一化，统计聚合为float64；有限/非零/单位范数检查通过。不对LLM2Vec输出做RQ或离散化。
- 置换只缓存第一层父组内Gram矩阵，无需构造整个数据集的N×N矩阵。精确统计仍来自原 `semantic_metrics.py`。

## Instruments数据适配

固定Amazon2018官方5-core评论及元数据。原始5-core文件包含27,530用户、10,620商品、231,392条记录；匹配后10,611商品、231,312条交互，9件缺元数据排除。按ASIN排序分配连续ID，重复元数据按标题/文本丰富度和固定字典序裁决。保留官方文件商品范围，不再做迭代5-core。元数据中的description/features被保留，但本轮Qwen和LLM2Vec均没有额外编码这些字段。

## 本次发布的整理修改

- 历史代码按实验分目录归档；其内容没有修改，LLM2Vec两次launch收据中列出的源码哈希全部匹配。
- 既有GRID代码、数据、Qwen权重、SID和checkpoint只给出固定GitHub路径，不重复新增。
- 新增通用CPU评估入口及结果重汇总/核验工具；它们是发布用工具，不是历史实验运行入口，也不代表重新跑了GPU实验。
- 不复制模型传输中继、登录信息、临时PID、旧版重复Beauty导出、渲染缓存、压缩包或未执行的旧准备方案。

## 没有完成、不能写进结论的项目

本轮LLM2Vec没有运行更高商品数门槛敏感性、同文本去重重跑、unsup-SimCSE或Qwen连续向量语义对照。标题重复商品仍按不同ID保留（Beauty34、Toys121），标题本身可能带品牌，不能声称彻底排除品牌效应。第三层大多数节点只有一件，报告必须伴随约19%–22%的覆盖率。结果支持该编码空间下的统计细化，不能直接证明普适的人类三级分类树。
