# 实际环境与安装边界

新环境 `/data/bwwu/envs/lc-rec-offload-20261108` 为独立用户级 venv overlay，使用 `--system-site-packages` **只读复用** `/data/bwwu/envs/grid-20261009-e61baf3` 的Python3.10.22、torch2.6.0+cu124及已有CUDA运行库。它依赖该基础环境继续存在，不声称复制了一套完全自含Torch。旧GRID、旧LC环境未安装或升级包。

本轮overlay固定：Transformers4.31.0、Accelerate0.21.0、DeepSpeed0.16.9、numpy1.24.4、tokenizers0.13.3、huggingface_hub0.16.4、safetensors0.3.3、sentencepiece0.1.99、ninja1.11.1.3、hjson3.1.0、py-cpuinfo9.0.0、einops0.8.1、msgpack1.1.0、nvidia-ml-py12.570.86。最终138项freeze及stderr见 `evidence/preparation/final_pip_freeze.json`；精确import来源、实际原生扩展及dtype见 `evidence/preparation/cpu-adam.json`。freeze含只读继承的包，不能把其中每一项称为本轮新增安装。

本轮没有系统级包安装、sudo或全局配置修改。任务级CUDA开发子集放在 `/data/bwwu/lc-rec/offload-20261108/cuda-12.4`：官方NVIDIA cuda-nvcc12.4.131、cuda-cudart-dev12.4.127、cuda-cccl12.4.127；下载源、字节和SHA见 `evidence/packages/manifest-r2-complete.json`，解包vendor info/license留服务器原始r2作业。cudart/cublas/curand只读链接到已有GRID运行库。CUDA_HOME、PATH、LD_LIBRARY_PATH、OpenMP等仅为作业局部环境，没有写.bashrc、系统驱动或公共CUDA。

CPUAdam实际预编译扩展13,239,928 bytes，SHA256 `d1d162b256c145ccda04c08d5c669ef16e130c4edacfa7cbdf1d3b532294239a`，记录accelerator=cuda、Torch2.6、CUDA12.4、NCCL2.21。只编译CPUAdam；最终DeepSpeed原始compute源码和setup不改。g++15.2只编译C++ CPU源，nvcc为真实版本探测及链接所需，不宣称用该GCC成功编译了CUDA kernel。早期缺toolkit的optional-CuPy打包patch只属于失败的r0/r1，不是最终环境方案。

| 任务级轮次／环境job后缀 | 实际源码版本 | helper耗时(s) | 原始状态／退出码 | 原因及保留证据 |
|---|---|---:|---|---|
| round0 / environment-r0 | 99f21ed35a8a9fae2f25d108922cd8813a3c4fd6 | 16.160 | FAILED / 1 | 隐藏GPU与Triton能力查询冲突；保留完整日志 |
| round1 / environment-r1 | fca3adc081277f97ab344ec0f9f2aa0cde38dec9 | 5.204 | FAILED / 1 | GPU可见后的DeepSpeed import扫描需要真实CUDA toolkit |
| round2 / environment-r2 | 3471b55486c2ce48c488d282b2f6a36e48072f17 | 146.475 | FAILED / 1 | 原生CPUAdam构建成功；import暴露einops等必需依赖缺失 |
| round3 / environment-r3 | 02f28235d38c15428c93f08d225c2ca5aaaaf2b7 | 33.196 | WORKERS_TERMINATED / 1；helper FAILED | 必需依赖和CPUAdam小张量两step通过；最后独立ldd命令缺Torch库搜索路径，断言失败 |

以上不是四次LLaMA GPU训练尝试。request、accepted、stdout、status、真实exit与cleanup原件见 `evidence/jobs/`，未重写失败结果。其后的**只读**加载路径核对证实实际Torch/native导入、进程maps和提供正确torch/lib路径的ldd均解析成功，见 `native_loader_linkage_readback.json`；没有重装或重跑r3。`training-imports.json`原件是import诊断前缀加JSON尾部，解析摘要另存 `parsed_training_imports.json`，不把混合日志误称纯JSON。

CPUAdam小张量两step是2048个FP32 CPU元素的规则/链接检查，对torch AdamW最大绝对差分别2.384e-7、4.768e-7；不是LLaMA更新，也不证明逐位一致、吞吐或全模型资源可行。原机memlock为8MiB，1MiB CUDA pin实测成功；候选仍显式pin_memory=False，不推断大规模锁页可用。

任务共享修正额度已到round3；静态审核收口不重置轮次，不扩展GPU时间、更新次数或模型下载上限。新环境、工具链、wheel、失败日志及已有模型片段均保留，未清理旧成果。
