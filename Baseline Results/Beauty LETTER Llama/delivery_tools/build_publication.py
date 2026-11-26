from pathlib import Path
import json,shutil,subprocess
HERE=Path(__file__).resolve().parent
REPO=HERE.parent/'instruments_llama_publication_20261125/repo'
DEST=REPO/'Baseline Results/Beauty LETTER Llama'
TAG='beauty-letter-llama-seqrec-20261126'
def main():
 DEST.mkdir(parents=True,exist_ok=True)
 shutil.copytree(HERE/'metadata',DEST,dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__','*.pyc'));(DEST/'MANIFEST.json').unlink(missing_ok=True)
 for n in ['verify_saved_results.py','fetch_assets.py','prepare_runtime.py','metrics.csv','metrics.md','cpu_verification.json']:
  shutil.copy2(HERE/n,DEST/n)
 tools=DEST/'delivery_tools';tools.mkdir(exist_ok=True)
 for n in ['export_server.py','build_publication.py']:shutil.copy2(HERE/n,tools/n)
 for n in ['independent_review.md','independent_verify.py','independent_runtime_verification.json','independent_export_verification.json','export_receipt.json']:
  if (HERE/n).exists():shutil.copy2(HERE/n,DEST/n)
 specs=[json.loads((HERE/(n+'.json')).read_text()) for n in ['beauty-llama-metadata.zip','beauty-llama-predictions.zip']]
 parts=json.loads((HERE/'adapter_parts.json').read_text());assets=[]
 for s in specs+parts['parts']:
  assets.append({k:s[k] for k in ['name','bytes','sha256']}|dict(url=f'https://github.com/woobowen/recommend/releases/download/{TAG}/'+s['name']))
 (DEST/'assets.json').write_text(json.dumps(dict(assets=assets,adapter=parts),indent=2))
 (DEST/'README.md').write_text('''# Beauty：LETTER 公开 SID＋Llama 推荐与 shuffle

维护：Jingchen Yin（Justin-hub-coder）。本目录发布原GPU2/3完成的 `letter_beauty_seqrec_batch64_20261122_155523`。LETTER官方SID固定不变，下游为LETTER-LC-Rec方式的Llama7B LoRA推荐，**仅seqrec任务**，同一个最优模型测试正常顺序、shuffle_3、shuffle_4。与[自有小型TIGER](https://github.com/woobowen/recommend/tree/74911e3ce668e265a8c3e7bc9a3929ae9247f11a/LETTER%20SID%20%2B%20TIGER)是不同下游。

## 结果

'''+(HERE/'metrics.md').read_text(encoding='utf8')+'''
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
''',encoding='utf8')
 (DEST/'reproduction.md').write_text('''# 复现起点

先进入本目录，Python依赖关键版本见[preparation/runtime_provenance.json](preparation/runtime_provenance.json)，torch2.6.0+cu124、transformers4.44.2、peft0.12、accelerate0.33、bitsandbytes0.46、safetensors0.5.3、sentencepiece0.2。CUDA驱动/包应匹配目标Linux设备，不重装公共环境。本轮静态检查与CPU复算不能替代新机器的GPU兼容验证。

1. **输出复算（已实际完成）**：`python fetch_assets.py --predictions-only`；`python verify_saved_results.py`。下载全部672个候选分片，CRC及hash后复算全部67089行，不加载模型。
2. **固定checkpoint重评（材料具备，本轮未实跑）**：`python fetch_assets.py`下载4个adapter分片并按顺序合并，合并SHA256必须匹配assets.json；额外约7GB空间供分片及完整adapter，候选另约200MB。基座原始约13.5GB，不计环境/新输出。取得`huggyllama/llama-7b` revision `4782ad278652c7c71b72204d462d6d01eaaf7549`，可用`huggingface-cli download huggyllama/llama-7b --revision 4782ad278652c7c71b72204d462d6d01eaaf7549 --local-dir /ABS/MODEL`；若服务要求账户/授权，须自行具备权限，逐文件hash在原audit中。`python prepare_runtime.py --base-model /ABS/MODEL --output /ABS/NEW_RUNTIME`验证基座并生成只改变路径和资产审计范围的派生runner；原件保持不变。确认两张空闲2080Ti，按实验室资源安排设置`CUDA_VISIBLE_DEVICES=2,3`。在新runtime执行`python test_recommendation.py --dataset Beauty --training /ABS/NEW_RUNTIME/fixed_training_record --out /ABS/NEW_RUNTIME/new_testing`，自动三条件全测试。模型脚本不提供跨用户GPU调度，使用前自行确认资源空闲并按服务器机制取得锁。不要直接执行归档中的旧pipeline/launch绝对路径。
3. **从头训练（材料具备，本轮未重训）**：准备相同基座和派生runner，执行`python train.py --dataset Beauty --micro 8 --out /ABS/NEW_RUNTIME/new_training`；完成后测试入口的`--training`改为该训练目录。应使用两张11GiB设备或等价足够显存，保存新argv/有效配置/源码hash，4epochs原训练约58.6小时；新输出至少预留15GB checkpoint空间。INT8训练、FP16全精度基座测试及相同选模口径不可悄悄替换。
4. **上游重建**：本实验起点是LETTER作者公开SID表，不在本轮重新学习上游。官方完整流程https://github.com/HonghuiBao2000/LETTER/tree/8d0154e28de37dbb6e24871c508ad8ddb1921cda；SID/交互/文本输入已随本目录提供。原RQ-VAE协同向量与预训练权重不属于本次下游重跑必需输入，未声称逐位重建作者上游SID。

移植脚本生成新runtime对common.py的路径及非本域资产检查作显式派生，记录relocation.json；研究参数保持原文。最优checkpoint只交付推理必需adapter/tokenizer及元数据，省略optimizer/RNG，不能精确接续到历史8212步状态。
''',encoding='utf8')
 rootread=REPO/'README.md';link='- [Beauty：LETTER SID＋Llama推荐与三种shuffle](Baseline%20Results/Beauty%20LETTER%20Llama/README.md)：GPU2/3正式训练及三组完整测试、原始候选、最优adapter与复现材料。\n'
 rootread.write_bytes(subprocess.check_output(['git','show','HEAD:README.md'],cwd=REPO)+'\n'.encode()+link.encode())
 b=REPO/'Baseline Results/README.md'
 original=subprocess.check_output(['git','show','HEAD:Baseline Results/README.md'],cwd=REPO)
 b.write_bytes(original+'\n## LETTER-LC-Rec 下游实验\n\n[Beauty完整结果与复现](Beauty%20LETTER%20Llama/README.md)：单独列示，未并入上方Qwen＋TIGER基线表。\n'.encode())
 print(DEST)
if __name__=='__main__':main()
