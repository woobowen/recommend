# ZeRO-3 checkpoint：持久化载荷与保存/转换峰值的静态区分

日期：2026-11-09。审核者：真实子代理 `/root/asset_audit`。本追加只读已保存的DS0.16.9、Torch2.6、HF4.31、Accelerate0.21及官方LC源码；DS checkpoint engine / zero_to_fp32两文件从已存在的0.16.9 sdist取出，未联网取新实现、未运行模型或保存checkpoint，未操作Git/服务器，未修改共享文件。源码身份见`checkpoint_source_identity.json`。

**结论：当前gather16bit=false候选下，14N不是实际ZeRO-3 checkpoint的固定持久化格式。正常全参ZeRO-3＋CPUAdam原生checkpoint的主要tensor载荷是FP32 master4N＋Adam m/v各4N，即约12N。14N只对应额外保留一份2N的BF16完整权重导出的规划口径；它不是所有保存/转换路径的严格上界。** 本文所有字节数是静态tensor载荷算术，不是本轮文件大小、磁盘峰值或RSS实测。

## 1. 原生ZeRO-3实际序列化什么

固定DS提交`bdba8231bc8fc17980a5941437e6363dac69418d`，具体代码链：

| 已保存源码 | 静态执行语义 |
|---|---|
| `ds_stage3.py:2635` `_rigid_state_dict` | 保存optimizer.state_dict和FP32_FLAT_GROUPS；没有fp16/bf16参数分片字段，也没有gradient字段。 |
| `ds_v0.16.9_cpu_adam.py:143`附近 | 真正step后，state只有step、CPU FP32 exp_avg、CPU FP32 exp_avg_sq。 |
| `torch_v2.6.0_optimizer.py:704` | optimizer.param_groups中的参数改存数值ID，不再复制参数tensor；master不会在这里再存一遍。 |
| `ds_engine.py:3274` `save_checkpoint` | 首先调用optimizer.checkpoint_event_prologue，再保存module与zero optimizer文件。 |
| `ds_stage3.py:2929`＋`ds_parameter_offload.py:188` | prologue将全部参数分片/释放，检查param状态NOT_AVAILABLE。 |
| `ds_partition_parameters.py:281` `free_param` | partitioned状态的param.data为empty(0)，真实分片在ds_tensor，普通module.state_dict不会自动把ds_tensor写进去。 |
| `ds_engine.py:3509,3693` | model-state文件存参数placeholder、buffers、形状、共享关系和训练metadata；每rank zero optimizer文件保存上述FP32 master/m/v分片。 |
| `ds_torch_checkpoint_engine.py:18` | 直接torch.save(state_dict,path)，没有另存一套BF16分片的额外步骤。 |

因此两rank合计主载荷约`12N`，每rank约`6N`。BF16分片不是这条原生ZeRO-3保存格式的额外`2N`文件，FP32 gradients也不作为额外`4N`持久化。恢复时可从FP32 master重建BF16训练分片。普通model-state文件中的buffer、空tensor序列化、param_shapes、step、config、scheduler、RNG及tokenizer等仍要另计；不是文件总大小精确等于12N。

边界：假定全参均正确纳入ZeRO-3，无PP/TP额外副本、无frozen fragment和NVMe offload副本；DP分片padding/底层storage余量需另计。当前模型各参数维度对world_size2可整除，静态上没有必要再为每参数加一整份DP副本。实际checkpoint必须检查真实成员和tensor dtype/numel，不能用本推导替代未来验收。

## 2. 14N是条件性规划量，CPU瞬时内存也不能按磁盘量倒填

当前`lc_rec/offload/config/ds_optimizer_only.json`明确stage3_gather_16bit_weights_on_model_save=false。Accelerate0.21 get_state_dict在此条件下抛ValueError，HF4.31 save_model捕获后改为engine.save_checkpoint。因此当前候选预期是原生约12N，不能写成“实际BF16＋master＋m/v共14N”。

如果将来明确选择一份独立BF16完整导出，则另占2N，原生checkpoint＋这份文件才约14N。若先落盘FP32合并文件4N，再另写BF16文件2N，并保留二者，单套合计约18N。所有这些还要加metadata/容器/文件系统开销，故14N不是无条件峰值上界。

**CPU RAM与磁盘分开：** 原生保存时optimizer.state_dict和FP32_FLAT_GROUPS主要引用已经驻CPU的tensor；TorchCheckpointEngine没有deepcopy整套状态，不能自动声称保存瞬间又新增12N CPU副本。文件系统page cache、序列化缓冲和buffer搬运仍可能增加RAM，精确峰值未测。

DS `_zero3_consolidated_16bit_state_dict` 则逐module聚合GPU参数，并在rank0保留`.detach().cpu()`后的完整16bit state_dict：这是约2N的额外CPU持有量，与原有训练CPU状态并存；同时有逐layer GPU gather临时量。它与最后2N磁盘输出是两个不同维度，不能互相替代。

DS0.16.9 zero_to_fp32使用mmap=True读取optimizer文件、立即丢弃m/v引用，并支持lazy/sharded输出；当前HF hub旧版本是否具备其默认sharding API仍需未来实际核查。LC vendor中的旧zero_to_fp32是另一版本，会先torch.load收集所有optimizer文件，parse阶段可能同时装入约12N CPU载荷，然后才释放不需要的m/v。不要把新工具的内存特性套在旧脚本上。离线FP32合并/模型重载的RAM需求同样不能从4N磁盘输出推出固定峰值。

## 3. 单套与四套保留的有界静态载荷口径

令`N=6,746,034,176`为930新增SID token后的trainable参数数，`B=13,476,876,661 bytes`为两片基础safetensors的已知大小（约12.551319GiB）。表中是两rank合计，GiB按2^30，**还需加非负开销H**：padding/storage slack、buffers/metadata、tokenizer/config/RNG、序列化/文件系统及本任务其他资产。它们是方案比较范围，不是物理占用的严格上界。

| 口径 | 主载荷bytes | GiB |
|---|---:|---:|
| 单套原生ZeRO-3，12N | 80,952,410,112 | 75.392807 |
| 单套＋一份BF16导出，14N | 94,444,478,464 | 87.958275 |
| 单套＋FP32中间文件＋BF16导出，18N | 121,428,615,168 | 113.089211 |
| 4套原生＋base，48N+B | 337,286,517,109 | 314.122547 |
| 上一行＋仅一份最终BF16导出，50N+B | 350,778,585,461 | 326.688015 |
| 4套＋base＋一份FP32合并文件，52N+B | 364,270,653,813 | 339.253483 |
| 4套＋base＋同一选模的FP32中间/BF16输出并存，54N+B | 377,762,722,165 | 351.818951 |
| 每套都另保留一份BF16导出，56N+B | 391,254,790,517 | 364.384419 |

所以按“保留4套原生、仅对选中一套做一次受控合并”口径：不合并为314.122547GiB＋H；独立BF16导出为326.688015GiB＋H；FP32中间和BF16最终都保留则351.818951GiB＋H。不是“原生四套必然4×14N”。4套原生本身已超过本次准备/技术探针200GiB额度，正式存储预算必须另行确认，本次不启动任何正式保存任务。

## 4. 两条原始源码路径可能进一步增加占用，不能漏掉

**正式结尾可能第5套：** save_total_limit=5不意味着4epochs会凭空产5个checkpoint-*；按epoch策略正常是4个。但官方finetune.py结尾还调用trainer.save_model(output_dir=root)。在当前HF4.31/gather=false组合，这会再次触发root/global_step*的原生DS checkpoint，位于四个checkpoint-*之外；Trainer按checkpoint-*的轮转不负责清理这个root副本。若正式launcher原样保留此结尾行为，5套原生＋base主载荷为 **418,238,927,221 bytes≈389.515355GiB＋H**；再保留一份FP32中间和BF16输出则427.211758GiB＋H。

**官方旧convert.sh并不把最终文件转换成BF16：** 它遍历每一个checkpoint，先写tmp-checkpoint-*/pytorch_model.bin（旧zero_to_fp32，4N），然后convert.py以默认from_pretrained加载FP32文件、save_pretrained，不传torch_dtype，最终仍FP32（再4N）。清理行全部注释。对四套都原样转换且保留tmp/final时，四套原生＋两套FP32格式×4＋base= **80N+B=553,159,610,741 bytes≈515.170033GiB＋H**。若再算上述root第五套原生，则92N+B≈590.562840GiB＋H。

这些是不同路径，不能在没有确定未来转换入口的情况下拼成唯一“真实峰值”。本次没有修改任何保存/转换实现，不改save_total_limit5，不自动删checkpoint、不自动改save/选模策略，也不以静态规划替代重载/恢复验证。

## 5. 本追加的验收边界

已实际读取并静态核对保存链、序列化字段、原生/合并路径；字节算术与可引用身份清单已经本地验证，详见`checkpoint_static_payload_arithmetic.json`。没有产生checkpoint，没有实测保存时间、磁盘/RAM峰值或恢复成功；当前GPU主尝试/下载进度由主控作业回执记录，不由本静态报告改写。
