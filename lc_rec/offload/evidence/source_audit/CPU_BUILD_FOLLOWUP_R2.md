# Round2 用户级CUDA子集与CPUAdam编译核查

本补充对应主控提供的真实r0/r1失败情况：GPU隐藏时Triton初始化失败；GPU可见时DS全量compatible_ops扫描中的FP quantizer提前探测nvcc失败。本子代理只读取其固定源码及本地现有包，不登录服务器、不运行安装/编译。先前“仅setup可选extras捕获缺nvcc”的候选不足以覆盖其他import-time探测，故不再推荐逐个patch optional op；按主控统一round2采用原DS源码不patch的用户级CUDA开发子集。

## 建议结论

- **支持主控方案继续有界验证**：原DS0.16.9源码、CUDA12.4.1用户级工具/headers、已有torch CUDA12.4 runtime库只读复用，CPUAdam仍只有两个.cpp，没有CUDA kernel编译。
- **建议同时补cuda-cccl12.4.127**：它是cudart-dev正式声明的header依赖，只有1,460,617 bytes。CPUAdam的直接include链不使用CCCL，但其提供的cuda_fp16/bf16和cooperative_groups头存在CCCL依赖；补全这项可减少剩余轮次受到间接include影响的概率。无需安装完整toolkit、驱动或无关数学开发包。
- **GCC15不应仅因NVCC12.4支持表止于GCC13而提前判失败**：这两个.cpp由g++编译。Torch2.6 BuildExtension只在sources中见到.cu才调用CUDA host compiler版本检查；NVIDIA host_config.h的GCC>13错误又在__CUDACC__条件内，纯g++未定义它。这里仍需实际编译/导入证明CPU C++/ABI可用，不能静态保证。

## 真实包身份与成员

`nvidia_cuda-nvcc_12.4.131_release.json`、`nvidia_cuda-cudart-dev_12.4.127_release.json`、`nvidia_cuda_cccl_12.4.127_release.json`来自NVIDIA Anaconda官方release API。已有两包直接读取 `/home/addaswsw/lc-rec-offload-20261108/control/packages/`，成员和info/index见`existing_cuda_packages_contents_audit.json`。另有一次422,630-byte cudart-dev的只读内存下载用于逐header依赖核查，hash已对官方API，没有落盘/解包安装该包。

| 官方包 | 精确字节数 | SHA-256 | info/index依赖 |
|---|---:|---|---|
| cuda-nvcc-12.4.131-0.tar.bz2 | 65,674,958 | 8342cc8ca1d82923bf46ee42f6f61e53b7b0f16cc5189573259ab9a655b78997 | [] |
| cuda-cudart-dev-12.4.127-0.tar.bz2 | 422,630 | 1cd8b5e46d89bef1c618fc58e7540dd113eee2e5d409554795d808b4fe6e2e2c | cuda-cccl, cuda-cudart>=12.4.127 |
| cuda-cccl-12.4.127-0.tar.bz2 | 1,460,617 | 0818bede93df28b2ab7993b591f73f6e00a680145011a4a4ad8c3fb10f0c9df2 | [] |

第三包原始下载入口为 `https://conda.anaconda.org/nvidia/linux-64/cuda-cccl-12.4.127-0.tar.bz2`。这份补充未下载它。

nvcc包实际有include/crt/host_config.h及CRT内部头；cudart-dev包含cuda_runtime.h/api、builtin_types、driver_types、vector_types等。逐文件读取发现：cuda_fp16.h / cuda_bf16.h / cuda_awbarrier_helpers.h依赖nv/target；cooperative_groups依赖cuda/atomic和cuda/std。只解两包而复用runtime时应称“选择性开发子集”，不要称Conda solver完整安装环境。cuda-cccl仅补header闭包，runtime仍按主控已确认的torch nvidia runtime实体复用。

## 编译器、include与链接的精确边界

固定官方`csrc/adam/cpu_adam.cpp`只include cpu_adam.h，另一cpp和该头只include torch/extension.h、标准库、simd.h；SIMD头使用标准/AVX intrinsics，无cuda_runtime/thrust/cub/cuda/std直接include。新增保存的`ds_cpu_adam.cpp`、`torch_v2.6.0_cpp_extension.py`和`ds_fp_quantizer_builder.py`均逐官方Git blob核验。

DS TorchCPUOpBuilder普通GPU可见路径：

| 阶段 | 相关实际flags/行为 |
|---|---|
| C++编译 | -O3 -std=c++17 -march=native -fopenmp、AVX宏、PyTorch ABI宏；DS还把-LCUDA/lib、-lcudart/-lcublas放cxx flags |
| 链接 | DS extra_ldflags有-lcurand及-LCUDA/lib；PyTorch CUDAExtension/load补torch/c10相关库与-lcudart。-lcublas主要还由torch_cuda的既有依赖链涉及，勿把cxx flags误称全部link flags |
| CUDA探测 | 多个import/setup/OpBuilder路径可执行nvcc -V；真实CUDA_HOME/bin/nvcc及12.4版本必须可读 |
| kernel编译 | CPUAdam没有.cu源；不应出现CPUAdam的nvcc编译命令，只应C++编译和共享库链接 |

CUDA_HOME下lib64若存在会被优先选中，否则才用lib。需保证实际被选目录里有可读的unversioned linker别名，如libcudart.so、libcurand.so；保持libcublas及cublasLt等DT_NEEDED可经已有runtime路径解析。全部symlink只指向已核实的现有torch nvidia库，不覆盖/复制/修改原库。最终以生成的编译日志、ldd/readelf、GPU可见环境import和CPUAdam实际小步检查为准。

GCC15真正剩余风险是C++17模板/AVX实现与运行时libstdc++/libgomp/GLIBCXX符号解析；Torch2.6会带自身_GLIBCXX_USE_CXX11_ABI值，不能手改ABI宏或禁用检查掩盖问题。NVCC的官方GCC上限适用于其处理CUDA源；[CUDA12.4.1官方文档](https://docs.nvidia.com/cuda/archive/12.4.1/cuda-installation-guide-linux/)与纯.cpp编译须分开解释。若日志意外出现.cu编译，不能继续沿“nvcc只版本探测”的结论运行，应重新定位是哪个额外op被启用。

## 本补充验证状态

源码/包元数据及header依赖核查完成；未安装、未编译、未运行GPU或优化器。实际round2/round3的成败仍以主控保存作业回执为准。本补充不增加修正轮次或计算预算，也不把GPU预跑/研究数据配对标为通过。
