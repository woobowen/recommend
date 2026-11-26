# Beauty：LETTER 公开 SID＋Llama 推荐与 shuffle

维护：Jingchen Yin（Justin-hub-coder）。本目录发布原GPU2/3完成的 `letter_beauty_seqrec_batch64_20261122_155523`。LETTER官方SID固定不变，下游为LETTER-LC-Rec方式的Llama7B LoRA推荐，**仅seqrec任务**，同一个最优模型测试正常顺序、shuffle_3、shuffle_4。与[自有小型TIGER](https://github.com/woobowen/recommend/tree/74911e3ce668e265a8c3e7bc9a3929ae9247f11a/LETTER%20SID%20%2B%20TIGER)是不同下游。

## 结果

| 模式 | Recall@5 | Recall@10 | NDCG@5 | NDCG@10 |
|---|---:|---:|---:|---:|
| unshuffle | 0.048652 | 0.072307 | 0.033322 | 0.040907 |
| shuffle_3 | 0.030765 (-36.76%) | 0.048875 (-32.41%) | 0.020030 (-39.89%) | 0.025874 (-36.75%) |
| shuffle_4 | 0.020570 (-57.72%) | 0.034298 (-52.57%) | 0.013227 (-60.31%) | 0.017603 (-56.97%) |

括号为相对unshuffle的百分比，精确值见[metrics.csv](metrics.csv)。[原始结果](run/Beauty/testing/evaluation_result.json)与[训练结果](run/Beauty/training/training_result.json)保持原文。每条件22,363用户、单个最后商品目标，Hit等同Recall。单训练种子42，不声称统计显著性。

全部67,089条用户×条件候选记录按作者原评价函数CPU复算通过，逐行target、用户顺序和shuffle排列通过，见[cpu_verification.json](cpu_verification.json)、[verify_saved_results.py](verify_saved_results.py)及[非作者审读](independent_review.md)。这是保存输出复算，本轮未重新加载模型或生成候选。

## 实验与代码

- 官方代码来源：LETTER commit `8d0154e28de37dbb6e24871c508ad8ddb1921cda`；[实际vendor](vendor/LETTER/LETTER-LC-Rec)和[来源清单](preparation/assets_audit.json)。服务器为无Git导出；[launch.json](run/launch.json)逐文件hash绑定实际runner，本次Git提交是发布版本。
- [common.py](run/runner/common.py)配置和构造数据/模型 → [train.py](run/runner/train.py)训练 → [test_recommendation.py](run/runner/test_recommendation.py)三条件生成 → [作者evaluate.py](vendor/LETTER/LETTER-LC-Rec/evaluate.py)评价。[response_collator.py](run/runner/response_collator.py)仅修补超长prompt无监督目标的情况。
- 原模型`huggyllama/llama-7b`，固定revision `4782ad278652c7c71b72204d462d6d01eaaf7549`。INT8基座＋FP16 LoRA训练；测试为未量化FP16基座＋同一adapter。r8/alpha16/dropout0.05，7种线性投影LoRA，同时训练并保存embed_tokens/lm_head，故adapter约3.32GB。
- [有效配置](run/Beauty/training/config.json)：seqrec131413训练样本，验证44726条，4 epochs/8212次更新；micro8×累积8=64，两GPU单进程模型并行，不能再乘2。lr0.0002、wd0.01、cosine、warmup0.01、训练seed42、最长2048 token、历史20。训练temperature1.0用于损失logits缩放；测试do_sample=false的beam生成。
- 按epoch验证loss选最优，checkpoint6160、eval_loss1.6796109676361084。旧三任务Beauty在2950步由用户停止，本次从基座重新训练，未复用旧adapter/optimizer。变更原记录见[change_history](change_history)，旧停止上下文见[固定历史发布](https://github.com/woobowen/recommend/tree/74911e3ce668e265a8c3e7bc9a3929ae9247f11a/LETTER%20Llama%20Recommendation)。

## 输入、打乱与评价口径

[Beauty.index.json](vendor/LETTER/data/Beauty/Beauty.index.json)按原item_id映射四位SID，[Beauty.inter.json](vendor/LETTER/data/Beauty/Beauty.inter.json)保持作者序列顺序。12,101商品、22,363用户、12,088唯一SID、965个新增token；不重新筛选/映射官方SID。训练仅使用每用户倒数两件之前的前缀目标，验证倒数第二件、测试最后一件，历史20件。

仅测试输入历史商品SID打乱，目标和prompt其他部分不变：S3排列前三位固定第四位，S4排列四位，允许原顺序；每商品独立、按seed2026/beauty/user_id/可见历史位置确定排列，详见[shuffle_grid_protocol.py](run/runner/shuffle_grid_protocol.py)。三条件共用同一checkpoint、prompt0、beam20、最多生成5token，test batch1。

作者生成约束只限制各码位可用token，**不是完整SID树的逐前缀约束**，仍有无效完整SID组合；按原评价不滤掉、不补齐、不额外去重。有效候选比例分别99.7939%、99.9797%、99.9944%。全部实际候选无重复。官方SID碰撞保留，46个测试目标在碰撞组内，完整SID匹配不能区分同组商品。官方NDCG累加命中位置，完整实现随包保留。

## 资产、复现与验证边界

[assets.json](assets.json)给出Release实际载荷、每文件SHA256及adapter分片顺序；[复现说明](reproduction.md)给出输出复算、固定模型重评、从头重训和上游起点。原基座按固定HF版本单独取得，不将服务器路径当下载入口；遵守基座与原数据/作者代码的访问和使用条件。原vendor未附独立LICENSE时不自行声明其开源许可证。

最优adapter/tokenizer、全部候选/score/target/逐用户指标和原配置均交付。optimizer/scheduler/RNG/末步checkpoint保留服务器，本次未发布，因此不支持精确优化器续训。完整prompt token数组未保存；可核原输入hash关系、重放保存排列，未宣称全量prompt逐token重建。硬件/随机性影响重新生成，不能承诺逐位相同。

## 三阶段记录

实验前：[request](run/request.json)、[launch](run/launch.json)、[容量及集成预跑](run/Beauty/capacity_micro8)、[原准备核验](preparation)。实验中：[训练events](run/Beauty/training/events.jsonl)、[pipeline事件](run/Beauty/events.jsonl)、[supervisor](run/supervisor.log)。实验后：[完成状态](run/status.json)、[best trainer_state](checkpoint/trainer_state.json)、[全量测试结果](run/Beauty/testing/evaluation_result.json)、CPU复算和独立审核。

本轮仅CPU整理/验证和GitHub发布，没有重新训练、模型重评或修改服务器原件。历史原始绝对路径保留为证据；移植只在新目录生成派生runner。适配后的移植入口尚未实际重新执行GPU训练/测试。
