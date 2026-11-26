# 复现起点

先进入本目录，Python依赖关键版本见[preparation/runtime_provenance.json](preparation/runtime_provenance.json)，torch2.6.0+cu124、transformers4.44.2、peft0.12、accelerate0.33、bitsandbytes0.46、safetensors0.5.3、sentencepiece0.2。CUDA驱动/包应匹配目标Linux设备，不重装公共环境。本轮静态检查与CPU复算不能替代新机器的GPU兼容验证。

1. **输出复算（已实际完成）**：`python fetch_assets.py --predictions-only`；`python verify_saved_results.py`。下载全部672个候选分片，CRC及hash后复算全部67089行，不加载模型。
2. **固定checkpoint重评（材料具备，本轮未实跑）**：`python fetch_assets.py`下载4个adapter分片并按顺序合并，合并SHA256必须匹配assets.json；额外约7GB空间供分片及完整adapter，候选另约200MB。基座原始约13.5GB，不计环境/新输出。取得`huggyllama/llama-7b` revision `4782ad278652c7c71b72204d462d6d01eaaf7549`，可用`huggingface-cli download huggyllama/llama-7b --revision 4782ad278652c7c71b72204d462d6d01eaaf7549 --local-dir /ABS/MODEL`；若服务要求账户/授权，须自行具备权限，逐文件hash在原audit中。`python prepare_runtime.py --base-model /ABS/MODEL --output /ABS/NEW_RUNTIME`验证基座并生成只改变路径和资产审计范围的派生runner；原件保持不变。确认两张空闲2080Ti，按实验室资源安排设置`CUDA_VISIBLE_DEVICES=2,3`。在新runtime执行`python test_recommendation.py --dataset Beauty --training /ABS/NEW_RUNTIME/fixed_training_record --out /ABS/NEW_RUNTIME/new_testing`，自动三条件全测试。模型脚本不提供跨用户GPU调度，使用前自行确认资源空闲并按服务器机制取得锁。不要直接执行归档中的旧pipeline/launch绝对路径。
3. **从头训练（材料具备，本轮未重训）**：准备相同基座和派生runner，执行`python train.py --dataset Beauty --micro 8 --out /ABS/NEW_RUNTIME/new_training`；完成后测试入口的`--training`改为该训练目录。应使用两张11GiB设备或等价足够显存，保存新argv/有效配置/源码hash，4epochs原训练约58.6小时；新输出至少预留15GB checkpoint空间。INT8训练、FP16全精度基座测试及相同选模口径不可悄悄替换。
4. **上游重建**：本实验起点是LETTER作者公开SID表，不在本轮重新学习上游。官方完整流程https://github.com/HonghuiBao2000/LETTER/tree/8d0154e28de37dbb6e24871c508ad8ddb1921cda；SID/交互/文本输入已随本目录提供。原RQ-VAE协同向量与预训练权重不属于本次下游重跑必需输入，未声称逐位重建作者上游SID。

移植脚本生成新runtime对common.py的路径及非本域资产检查作显式派生，记录relocation.json；研究参数保持原文。最优checkpoint只交付推理必需adapter/tokenizer及元数据，省略optimizer/RNG，不能精确接续到历史8212步状态。
