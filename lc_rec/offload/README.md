# 双 RTX 4090 CPU Offload 技术可行性核查

一次性任务，2026-11-08；继承已审核交付 `2e0e26633b58d44cc4595417f4a054f6bb6b7e77`。
本目录记录新的技术核查，不覆盖旧审核结论。正式复现仍受作者辅助数据错配、SID碰撞、逐位置生成约束及历史资产绑定未知项阻塞。

授权：Instruments、`huggyllama/llama-7b` revision `4782ad278652c7c71b72204d462d6d01eaaf7549`，官方代码 `e6f1b3b3f8b4fee55a49a7b3dfeba29dbfcf0669`；全参数、六任务联合定义、BF16、AdamW、global batch128、20件历史和2048tokens不变。首选ZeRO3+CPU optimizer offload；只有第一方案确实显存不足才进入CPU parameter offload第二候选。不得采用顺序六次微调、LoRA、量化、NVMe、改数据/标签/约束或新增设备。

本轮仅资源/计算技术探针：优先真实训练期seqrec样本，其他任务最多纯技术前向；不把错配辅助数据用于正式训练，不计算推荐性能。候选正式配方保留4epochs，但本轮不运行epoch、正式checkpoint、单seed正式训练或五seed。

硬预算：seed42，两张已确认空闲4090；每类候选一次主尝试、最多2 optimizer updates（修正共享额度），单GPU作业90分钟、累计120分钟；新增模型下载20GiB、磁盘100GiB；主机始终保留40GiB可用内存，禁止swap失控。本地不暂存整套权重。固定脚本/配置/commit后部署无.git源码export，复用已验收runner，持久job/log/exit/resource记录，断线先查原ID。

技术栈候选：Python3.10.22 / torch2.6.0+cu124（只读复用GRID组件）、Transformers4.31.0 / Accelerate0.21.0 / DeepSpeed0.16.9；独立新环境 `/data/bwwu/envs/lc-rec-offload-20261108`，旧环境不修改。最终构建复用原DeepSpeed源码，配合任务目录内官方NVIDIA CUDA12.4开发子集（nvcc/cudart-dev/CCCL）及只读链接的现有CUDA运行库。没有更改驱动、公共框架或CPUAdam计算源码；不伪造CUDA版本、不用DS_SKIP_CUDA_CHECK。早期无toolkit的可选CuPy探测patch未通过导入检查，r0/r1失败保留，不是最终生效方案。

有效配置与实际范围以本目录config、作业request和输出为准。最小计算路径、batch128累积布局、全部多任务资源可行、保存恢复、完整复现分别验收。数值实现/初始化差异必须保留，不能以算法相同冒称逐位一致。

round0为首次实现/初查，后最多三轮集中修正；不扩大上述GPU预算。主控串行Git，独立非作者直接检查原件、diff、配置和运行证据。交付提交推送wbw-lc-rec并回读，轻量ZIP实际解包/hash验证，随后停止。其他分支及旧证据保持不变。

进展和最终结论将放在本目录 `REVIEW_ENTRY.md`，本文件是任务范围，不是常驻看板。
