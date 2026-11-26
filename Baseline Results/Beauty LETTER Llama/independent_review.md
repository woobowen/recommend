# Beauty LETTER-LC-Rec 独立审核

审核者：`/root/beauty_publication_review`，非训练/发布脚本作者。2026-11-26，北京时间。依据本工作区 AGENTS.md、PROJECT_RULES 与上传说明，对本次 Beauty 已完成实验定向检查。此文件为实际独立审读记录，不代替主控发布回读。

## 服务器结果审核：通过

直接读取服务器 `experiments/letter_beauty_seqrec_batch64_20261122_155523` 的真实 `launch.json`、`runner/common.py`、`train.py`、`response_collator.py`、`shuffle_grid_protocol.py`、训练配置/结果与 checkpoint trainer_state；读取作者 `LETTER-LC-Rec/data.py`、`evaluate.py` 以及原测试分片。独立脚本 [independent_verify.py](independent_verify.py) 在服务器 CPU 执行；完整校验输出见 [independent_runtime_verification.json](independent_runtime_verification.json)。没有加载模型、占用 GPU、重训或重新生成候选。

- 原启动清单的 9 项源码 hash 与原件一致；最优 adapter hash 与训练记录及三条件测试记录相同，均为 `19cc9d5267049718c8b9f20ea860989a3922f6ca8188beab647a6ab8ed1b529a`。
- 训练配置确为 seqrec 单任务、micro8×GAS8=有效 batch64、4 epochs、seed42、temperature1.0。INT8 基座加 LoRA 训练，embed_tokens 与 lm_head 同时训练并保存；测试加载未量化 FP16 基座。最优 checkpoint6160 的保存日志验证损失最小为 1.6796109676361084，最终训练8212步。
- 输入为 LETTER 官方 Beauty 12,101 个商品、22,363 用户；SID 共有12,088个唯一完整值。评价保留官方SID碰撞，有46名用户的目标在碰撞组中。这里是完整SID匹配，不等同唯一商品身份匹配。
- 作者训练只读取每用户倒数两件之前的前缀目标，验证倒数第二件、测试最后一件，历史最多20件，测试 prompt0。测试标签不参与输入打乱。
- 三条件各22,363行，共67,089行、672个分片全部遍历。按官方评价函数从每行20个候选和score重算 Recall@5/10、NDCG@5/10，逐用户值及汇总和原记录一致，误差门槛1e-14。全部用户顺序、target/SID输入身份、历史数通过；无重复候选。
- shuffle 协议按`seed2026|beauty|真实user_id|可见历史位置|shuffle_n`确定排列，允许恒等排列；S3固定第四位，S4排列四位。所有保存排列、变动计数及输入hash相等/不等关系均通过。n=0/3/4的变动物品数分别0/133544/153217，总历史159966件。

## 必须准确表达的评价边界

`get_prefix_allowed_tokens_fn`按码位允许该位出现过的token，**并非完整SID树的逐前缀合法约束**；因此会产生不在商品表中的组合，原评价没有将其过滤或补齐。完整有效候选比例由原结果明确给出。官方 NDCG 实现累加所有命中位置且不去重；本次三条件实际无重复候选。Hit在每用户单目标条件下写作Recall。

独立CPU复算支持原候选到表格的一致性，不证明从模型可逐位重复生成相同候选。未保存完整prompt token数组，本轮校验其hash一致性关系，没有重建全量prompt。单模型/单训练种子不支持跨种子显著性结论。

## 发布准备阶段

已读 `export_server.py`，范围合理：原源码、配置、SID/交互、tokenizer、最优adapter元数据与全部候选均保留；adapter按1GiB分片。省略optimizer/scheduler/RNG和最终checkpoint需明确意味着发布支持固定checkpoint重评及从头训练，**不支持精确优化器续训**。原基座不再托管，需固定HF revision与逐文件hash、取得/许可证条件；不能把只有服务器路径称作外部可取得。

已独立检查导出包的全部成员：metadata 82项、predictions 672项，ZIP CRC、无重复成员、逐成员长度/SHA-256均通过，详见`independent_export_verification.json`。尚不能由此推断GitHub下载已校验。

复现入口应处理一个具体依赖：原`common.verify_assets`遍历原`assets_audit.official_files`的全部路径，其中还包括LETTER-TIGER/RQ-VAE/Instruments/Yelp。只发布Beauty-LC-Rec必要文件时，不可直接沿用此全量校验而缺文件；应在独立派生环境中将运行audit限定到已发布且必要的文件，保留历史原audit及派生记录，或取得清单全部输入。此问题只影响移植复现，不影响原服务器运行。

## 发布前定向复核：通过

已进一步直接读拟发布的 README、reproduction.md、assets装配设计，以及`fetch_assets.py`、`prepare_runtime.py`、`verify_saved_results.py`和`validate_publication.py`。README准确区分官方SID+Llama下游与自有TIGER、单种子边界、LoRA+embed/lm_head训练、INT8训练/FP16测试、官方SID碰撞、无效候选及仅CPU复算。

源码审读发现上游链接误写为`HonghuiBao/LETTER`，向主控指出后已修正为原下载来源`HonghuiBao2000/LETTER`固定commit。资产audit多域问题也已修正：`prepare_runtime`保留原audit，仅在新runtime创建筛选为已发布Beauty/LC-Rec文件的派生audit。本轮直接核对19项实际发布作者源码/Beauty输入与原audit哈希，全部一致。

另直接查看服务器`codex/beauty_llama_publication_20261126/relocation_check`生成的`relocation.json`、`fixed_training_record/training_result.json`及原/派生`common.py`完整diff：仅替换ROOT/PREP/AUTHOR/MODEL及派生audit路径，没有更改研究参数；固定训练记录仅重定位最优checkpoint。该目录由主控运行prepare_runtime生成，本审核者没有再重复基座大文件hash或启动模型。新机器GPU训练/重评兼容性仍未实跑，文档已正确说明。

候选fetch在已下载的Git可读目录基础上取得672分片；完整fetch再取得4个adapter分片并按清单顺序装配及核验合并hash。具体GitHub载荷可取得性须由主控完成上传后实际逐字节下载核对。本独立审核通过的范围是服务器结果、导出包内容、拟发布文档/源码及实际派生差异；**尚不替代远端ref/正文/payload实际回读**。
