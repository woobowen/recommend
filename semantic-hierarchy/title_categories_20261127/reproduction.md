# 复现与资产取得

## 1. 保存输出复算（本轮已实际完成）

在本目录运行`python fetch_assets.py --output /ABS/RESTORED`。下载3个ZIP：metadata约185MB、Beauty向量约679MB、Instruments约559MB；包含原字段/raw/归一化/缓存/掩码/norm，ZIP展开后约2.1GB。完整缓存＋展开＋独立runtime建议预留7GB。每ZIP和每成员的长度/SHA256、CRC和安全路径均核验，拒绝覆盖不同原件。

`python verify_outputs.py --run /ABS/RESTORED/run --output /ABS/check.json`，CPU重新计算12组所有节点/边/路径/Gap、覆盖、2000次簇bootstrap区间，核保存的7992条置换统计与p/Holm。此脚本不重新加载大模型、不重新编码，不重新产生置换随机分配。依赖numpy2.0.2兼容环境。

`python prepare_runtime.py --archive /ABS/RESTORED --output /ABS/NEW_STATS`创建独立源/输入/向量副本；`python /ABS/NEW_STATS/source/analyze.py --run /ABS/NEW_STATS`会从原向量重新执行999次条件置换及全部分析，输出新analysis目录。该全置换重跑入口静态/原方法已验证，本次发布没有为凑验证重复运行同样全部置换；原实验已实际执行过。本轮CPU核验边界见verification/cpu_verification.json。

## 2. 固定模型重新编码（材料具备，本轮未重评）

原基座`meta-llama/Meta-Llama-3-8B-Instruct`，原兼容镜像`LLM-Research/Meta-Llama-3-8B-Instruct`的ModelScope revision `ccdc34064d2ac9f2d99d1a6885333773a6f12c09`；HF镜像`NousResearch/Meta-Llama-3-8B-Instruct` revision `53346005fb0ef11d3b6a83b12c895cca40156b6c`。原辅助配置存在不同镜像提交，必须以run/model_verified.json逐文件身份为准，不能只按模型同名视为字节相同。基座4个权重分片约16GB，需来源访问/许可；如需HF token只在自己环境提供，不写入仓库。实际使用的非权重配置及tokenizer已放metadata ZIP的encoder_assets/models，取得权重后按清单组合核验。固定公开模型来源可读不代表所有地区/账户都无权限限制。

LLM2Vec MNTP：`McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp@31474e395ada192e8ed1586db6be79fb3b70c9c0`；supervised：`McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised@baa8ebf04a1c2500e61288e7dad65e8ae42601a7`。两adapter实际bytes均在metadata ZIP，不必另下载；hash见model_verified。

关键包实际版本见run/environment.json：Python3.10.20、torch2.6.0+cu124、transformers4.44.2、peft0.12、llm2vec0.2.3、numpy2.0.2、accelerate1.8.1。CUDA/硬件组件按目标Linux安装；不要直接装旧本地路径freeze。FP16需两张足够空闲的11GiB设备，不能将8B模型塞到单张2080Ti。

`python prepare_runtime.py --archive /ABS/RESTORED --output /ABS/NEW_ENCODER --base-model /ABS/BASE`全量核验基座，复制实际adapter、文本和原配置，并只在新source修改ROOT/基座路径，记录relocation.json，原运行不改。选两张空闲卡（按服务器规约持锁），例如`CUDA_VISIBLE_DEVICES=0,1 python /ABS/NEW_ENCODER/source/encode_fields.py --run /ABS/NEW_ENCODER --dataset beauty`，Instruments同理。脚本会先最长文本/一致性预跑再编码。两域完成后`python /ABS/NEW_ENCODER/source/analyze.py --run /ABS/NEW_ENCODER`。不得直接运行旧prepare_experiment/launch_fusion的服务器绝对路径。

本轮移植源只做路径/配置验证，不启动新的GPU编码；不承诺跨硬件浮点逐位相同。原两域并行编码各约10分钟，CPU分析约2分钟，属本次观测，不能当别的设备保证。

## 3. 重跑实验

固定SID＋新编码＋原统计构成完整本实验；没有推荐训练/分类器微调阶段。按第2节从保存的实际文本开始，配置与输入hash足以复现本次起点。原始商品文本重建脚本保留，但依赖原Beauty TFRecords、映射及旧上游目录，不能直接在其他机器执行旧绝对路径。

## 4. 上游重建边界

本次未重新学习Qwen/LETTER SID；精确原SID表已随run/inputs提供。Qwen原上游代码和权重/TFRecord可从本分支既有GRID baseline与固定yjc-grid@6200d5875e264b5e6ec9841cf2cdb3b4a12d71a4读取；LETTER固定公开SID来自HonghuiBao2000/LETTER@8d0154e28de37dbb6e24871c508ad8ddb1921cda。不能把对齐后Qwen SID说成在9922商品上重训结果。

Instruments严格映射所用作者原item2id/user2id、下载来源与全图核验随alignment提供；不按标题猜配。Beauty全量sid-column/grid-item/P5-item/ASIN映射在run/source/beauty_g3_item_mapping.csv，历史完整交互轴核验依据在identity/beauty_mapping_audit.json。本轮从保存向量复算与商品轴检查，不冒称重新训练上游或完整重跑历史推荐测试。
