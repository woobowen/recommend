# LLaMA-7B / 双4090：ZeRO-3 CPU optimizer offload 源码审核

审核者：真实子代理 `/root/asset_audit`。2026-11-08，新增 offload 可行性任务；只做本地证据读取及官方网络只读核查，无服务器操作、安装、模型下载、训练、Git写入或共享源码修改。正式规则三文件更新后的相应章节已显式重读。原先数据配对阻塞不由本次引擎可行性审核消除。

结论：**推荐候选为既有 torch2.6.0/cu124＋transformers4.31.0＋accelerate0.21.0＋DeepSpeed0.16.9；是可核查的实现候选，尚非已通过的运行栈。** CPU optimizer offload 在模型状态层面有合理容纳空间；仍需主控验证编译、双rank加载/resize、真实CPUAdam状态及完整长路径峰值。不得把本源码审核或静态估算写成GPU预跑PASS。

## 版本与证据身份

既有 `environment.json` 真实记录 torch2.6.0、CUDA build12.4、transformers4.31.0、numpy1.24.4；accelerate/deepspeed均未安装。原始证据快照在 `prior_environment_snapshot.json`，其来源/hash及“不是本次实时服务器检查”见 `prior_environment_identity.json`。

DeepSpeed候选固定 `v0.16.9` → **bdba8231bc8fc17980a5941437e6363dac69418d**，发布于2025-05-22；`ds_v0.16.9_tag.json` 和 release JSON 是真实 GitHub API回读。实际保存36份官方源码/CI/测试/文档，每一份本地Git blob均与GitHub回读相同，`source_identities.json` 包含URL、Git blob、SHA-256和字节数；没有改写官方源码。GitHub tag树只用于定位真实CI路径，首次猜测的 `.github/workflows/unit-tests.yml` 返回404，后由真实tree定位，不虚构CI文件。

TF4.31的依赖表仅声明 accelerate>=0.20.3 / deepspeed>=0.9.3；DS0.16.9运行依赖不强制升级transformers，但其dev依赖为transformers>=4.39，CI会取较新的HF。因此上述四者组合有可追踪的接口实现依据，**没有发现该精确组合的官方CI结果可据以直接宣称兼容通过**。不要装 `[dev]` 或全部latest覆盖现有HF/tokenizer行为。DS要求pydantic>=2.0，实际选定小依赖版本及lock仍由主控核定；本子任务没有安装任何版本。现有WSL Python无pydantic，未伪称已运行Pydantic2验证。

关键原始来源：[DS0.16.9 CPUAdam](https://github.com/deepspeedai/DeepSpeed/blob/bdba8231bc8fc17980a5941437e6363dac69418d/deepspeed/ops/adam/cpu_adam.py)、[DS builder](https://github.com/deepspeedai/DeepSpeed/blob/bdba8231bc8fc17980a5941437e6363dac69418d/op_builder/builder.py)、[HF4.31 DeepSpeed文档](https://huggingface.co/docs/transformers/v4.31.0/en/main_classes/deepspeed)。最新DS文档已为0.19.x，不能用其新增功能/defaults倒填0.16.9；这里以固定tag代码为准。

## CPUAdam真实精度、优化器语义与cosine

DS0.16.9 `DeepSpeedCPUAdam` 默认 `adamw_mode=True`、`bias_correction=True`、`fp32_optimizer_states=True`，betas=(.9,.999)、eps=1e-8、weight_decay=0。后者不是本任务最终decay：Trainer按参数名/LayerNorm生成原有两个group，decay组0.01、非decay组0，再由优化器接收。应核实际组和参数数量，不把所有参数都改成0.01。

`stage3._create_fp32_partitions` 将每rank的BF16参数分片 `.clone().float()` 放CPU，作为FP32 master；`initialize_optimizer_states` 为其建立FP32 gradient buffer；CPUAdam首次真正step时惰性建立CPU FP32 exp_avg / exp_avg_sq。CPUAdam要求参数在CPU；BF16在GPU执行模型计算，不意味着CPUAdam在BF16 master/state上运算。

精度需分层：DS0.16.9 engine.get_data_types在BF16＋ZeRO且未显式grad_accum_dtype时，gradient_accumulation_dtype默认仍BF16，所以stage3的CPU grad_partitions_flat_buffer可以是BF16；交给FP32 master的optimizer-facing `.grad` 才是FP32。不得笼统声称所有梯度都是FP32，也不要为CPUAdam静默新增FP32 accumulation配置。

CPUAdam C++ 的AdamW分支把decoupled decay施加到参数，再执行带bias correction、eps在平方根外的Adam更新；非AdamW分支把L2项加到梯度。应明确保留AdamW模式，不能仅看优化器类名。AVX/FMA与PyTorch实现可能有舍入顺序差异，本审核不承诺逐位同轨迹。固定tag官方 `test_cpu_adam.py` 有对torch AdamW的十步比较，但主要比较参数norm且容差较宽；本次环境仍应以小FP32张量、同初值/梯度/超参和两类decay组做有界实际数值检查，不能把官方测试文件存在视为本次已通过。

**最小集成首选：保留DS JSON中没有optimizer/scheduler节点的原结构，由HF创建adamw_torch及原cosine调度。** TF4.31 `deepspeed_optim_sched` 使用Trainer.create_optimizer/create_scheduler；Accelerate0.21 `_prepare_deepspeed` 遇 `offload_optimizer.device!=none` 且默认 `zero_force_ds_cpu_optimizer=True`，自动改建 `DeepSpeedCPUAdam(optimizer.param_groups, defaults仅lr/weight_decay)`。原group携带betas/eps/decay，CPUAdam补bias_correction=True。PyTorch2.6 `Optimizer.add_param_group` 直接保留传入group字典，故原LambdaLR和新CPUAdam共享group对象；应在实际日志确认group身份、CPUAdam class、每次真实optimizer update的LR和scheduler步数。

如果反而在DS JSON加入optimizer但不加scheduler，TF4.31明确报错；擅自加DS WarmupLR会失去原cosine语义。除非主控有明确替代实现与验证，不推荐这条不必要改动。

`gradient_clipping` 是DeepSpeed顶层参数，HF将其与TrainingArguments.max_grad_norm=1.0匹配；CPUAdam本身不实现模型全局梯度裁剪。DS stage3先汇总各分片norm，再用 `max(1, (global_norm + 1e-6)/clip)` 缩放FP32 gradient并step。不能把max_grad_norm写进CPUAdam optimizer.params；DS engine会拒绝。

## from_pretrained前的受支持顺序与resize

1. 先固定分布式rank/device与完整TrainingArguments，保留该对象活着。TF4.31 TrainingArguments创建HfTrainerDeepSpeedConfig并处理batch/BF16等auto；这是官方源码明确要求的“model created前”顺序。非Trainer路线才直接用活着的HfDeepSpeedConfig，且需要自行处理全部auto；只有一个暂时构造后被回收的对象不够，因为HF全局只持weakref。
2. base config仍为原vocab32000；模型严格从基础权重读取，再做有意resize到32930。提前将expanded config传给基础from_pretrained会制造embedding/head维度不匹配，不能用ignore_mismatched_sizes或strict=False遮盖。
3. HF4.31 from_pretrained检测活着的ZeRO3 config，进入`deepspeed.zero.Init`再构造模型，逐权重加载。**不要传原finetune的device_map，也不要low_cpu_mem_usage=True**：HF4.31明确拒绝这两者与ZeRO3并用。device_map省略/None与low_cpu_mem_usage默认False是该版本正确入口。
4. HF4.31 `_get_resized_embeddings` 已用GatheredParameters读取原embedding真实shape、复制旧行；untied `_get_resized_lm_head` 同样支持GatheredParameters。新增行仍走模型normal initializer，没有现代mean_resizing。
5. 新embedding/head在from_pretrained的zero.Init上下文退出后建立，可能暂时没有ds_id。DS parameter_offload._convert_to_zero_parameters会把这些参数纳管；已有zero参数的convert_to_zero_parameters最终先broadcast rank0、再partition。**所有rank都需执行同样resize/engine初始化；不要只有rank0调用GatheredParameters。** 实际要验证新增参数ds_id/分片、旧行未变、所有rank一致和input/head保持untied，不能只看API调用退出0。

新增行精度必须记录：原LC代码先加载FP32模型、resize，再进入Trainer；提前ZeRO3/BF16初始化可能让新行normal_直接发生在BF16张量上，随机数消费/舍入与原FP32后转换不同。保留normal(0,initializer_range)不自动证明精度与seed轨迹完全相同；若主控要声称官方初始化行为严格一致，需要专门核对/适配并记录，而非静默略过。模型权重和新行尚未在本子任务加载或验证。

## Pydantic2与auto bucket真实问题

HF4.31 `trainer_config_finalize` 和Accelerate0.21 `_prepare_deepspeed` 均把auto prefetch填成 `0.9 * hidden_size²`；h=4096时为 **15099494.4**。DS0.16.9把stage3_prefetch_bucket_size声明为int，依赖pydantic>=2；非整数float不是可接受的int值。

首选在本次派生DS配置中显式写整型 **15099494**（向下取整少于一个元素），并记录来源，而非等待auto产生非法float；reduce_bucket_size=16777216、stage3_param_persistence_threshold=40960。其余1e9类数值也可直接序列化为整数，避免版本间隐式转换。DS config_utils只过滤字符串"auto"，不会修补已经展开成15099494.4的float。显式值在HF/Accelerate的fill_only/auto检查中不会被该float覆盖；仍需实际将最终生效字典交DS配置解析器验一次。

## CPUAdam编译：JIT与官方预编译fallback须分开

CPUAdamBuilder只有两个.cpp源，但继承CUDAOpBuilder。**正常GPU可见JIT**会令build_for_cpu=False，调用nvcc_args→installed_cuda_version→CUDA_HOME/bin/nvcc -V，并使用CUDA库。这一路不能简单说“CPU kernel所以不需要toolkit”。缺nvcc也不能据此把所有官方路线判BLOCKED。

**官方预编译路线存在：** `CUDAOpBuilder.builder()` 捕获MissingCUDAException，置build_for_cpu=True，并使用CppExtension；TorchCPUOpBuilder随之只加CPU arch/SIMD/OpenMP，不链接CUDA库。构建开关可为DS_BUILD_OPS=0、DS_BUILD_CPU_ADAM=1。real_accelerator.py明确支持DS_ACCELERATOR=cuda；安装元数据保留accelerator_name=cuda及torch_info2.6/cu12.4时，运行中的CPUAdamBuilder.load可直接校验版本并导入已装原生CPU扩展，不进入JIT。

**但tag的setup前置探测有额外障碍，必须看实际sdist：** setup.py在torch可导入且accelerator=cuda时，为可选1bit/CuPy extras无条件调用installed_cuda_version，早于builder且未捕获MissingCUDAException。若主控选定PyPI sdist有同段，单设上面的构建开关不足以走到fallback。可首选一个明确的最小打包适配：仅对可选1bit/CuPy探测捕获MissingCUDAException、缺toolkit时不添加未请求的可选CuPy extras；不要伪造CUDA版本、不要设置DS_SKIP_CUDA_CHECK骗过检测。CPUAdam C++原件及运行配置保持不变，后续builder自然走其官方CppExtension fallback。主控应保留sdist hash、该最小diff、构建日志、wheel内installed_ops/accelerator_name/torch_info及GPU可见环境实际import证据。此处仅给方案，未修改setup或构建任何wheel。

直接DS_ACCELERATOR=cpu构建虽然可跳过该cuda metadata段，却会把installed accelerator_name记为cpu；CUDA训练进程的loader检测不匹配后重新JIT，因此不能以CPU构建成功代替CUDA可见时的预编译import检查。

CPUAdam FP32路径支持AVX512、AVX2及scalar；**AVX512不是FP32 master/state的硬门槛**。其BF16向量load/store有AVX512限制，但本候选CPUAdam实际应是FP32参数/梯度/状态；不应新增BF16 optimizer state。Gold6133通常有AVX512，但现场flags和编译结果由主控核实。无需AVX512_BF16原生指令来执行这里FP32 CPUAdam。

编译需可用C++17编译器、PyTorch headers/ABI匹配、OpenMP编译/运行时；源码明确使用-fopenmp和omp parallel for。JIT另需ninja；预编译是否用ninja由其build选项决定。不要安装所有DeepSpeed ops或NVMe AIO依赖；本任务不使用NVMe。partial offload ratio须保持1.0；ratio<1会额外建立GPU FusedAdam备用路径，带来未请求GPU kernel依赖。

## 分片、host pinning与资源边界

候选只offload_optimizer.cpu，不启用offload_param，因此BF16参数仍按ZeRO3分片驻GPU，FP32 master/gradient/Adam状态按rank分片放CPU。用本任务新增930token后的N=6,746,034,176、largest direct module=32930×4096=134,881,280，严格复写DS0.16.9静态公式可得：

| 静态模型状态估计 | 数值 |
|---|---:|
| GPU每rank，4×largest_layer＋2N/2 | 7,285,559,296 bytes ≈6.785GiB |
| CPU每节点，16N×buffer_factor1.5 | 161,904,820,224 bytes ≈150.785GiB |

原式与计算入参在`ds_stage3.py`、`static_model_states_estimate.json`。这与此前无offload超显存结论不矛盾：状态位置已获新方案改变。**GPU值不含激活/attention/logits/通信额外buffer/碎片/保存重载/新增行临时聚合；CPU亦仍需考虑shard加载副本、pinned缓冲和其他进程。** 不能据6.785GiB就宣称24GB一定可跑2048token全路径。

DS0.16.9 `offload_optimizer.pin_memory` 默认False；这与某些最新版文档的default不同。设置True时stage3会对grad_partitions_flat_buffer和各subgroup_gradient_buffer使用CUDA accelerator的tensor.pin_memory；可提升传输但增加host page-locked占用。FP32 master由clone.float产生，不应不加辨别地说所有CPU状态均pinned。需记录实际pin_memory值、CPU峰值、pinned分配成功性与权限；ulimit-l只是环境线索，不能以它单独证明CUDA host allocator的全部实际分配能力。不得修改系统memlock或公共设置来掩盖失败。OpenMP线程与NUMA亲和须按现场可用CPU/共享任务设置，避免两rank各默认占满全机线程。

## 主控最小验收边界

1. 固定且隔离的版本/import/编译产物：CPUAdam GPU可见环境实际走preinstalled导入，记录ds_report/installed_ops/accelerator_name/torch_info；小FP32 optimizer和torch AdamW有界比对。
2. resolved DS config不含非法auto/float；batch1×acc64×2=128、clip1、betas/eps、decay分组、AdamW与原cosine/warmup仍对应。记录真正CPUAdam状态dtype/device及更新LR，不能只报告TrainingArguments.optim字段。
3. 在from_pretrained前活着的HF ZeRO3配置，移除device_map/low_cpu路径，严格基础权重加载；resize和engine纳管跨rank证据，包括新增行初始化精度差异与原行保存。
4. 正式模型GPU检查必须另以保存脚本/固定commit/预算运行，覆盖真实长路径和明确停止；本审核不替代任何运行证据，不清除作者数据/SID/生成约束的旧阻塞。

本子任务状态：官方源码/静态核查完成；编译与实际运行PENDING。新增安装0、配置/共享代码改动0、权重下载0、服务器操作0。不是网页GPT二次审核PASS。
