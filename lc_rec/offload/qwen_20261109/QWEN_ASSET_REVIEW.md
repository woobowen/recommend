# Qwen3-8B 有界资产调查（2026-11-09）

**总体 PARTIAL；完整权重 BLOCKED，等待用户明确 Base／后训练身份。** 本轮没有 Qwen 模型加载、训练、评价、GPU 操作或权重下载；下载的两个 `tokenizer.json` 只是 tokenizer JSON，并非模型参数。此结论不阻塞 LLaMA 链。推荐候选不是用户选择，也没有冻结为正式模型。

## 授权、检索与来源

本子任务显式读取 `/home/addaswsw/recommend/{AGENTS.md,PROJECT_RULES.md,REMOTE_EXECUTION.md}`，实际输出目录位于 `/home/addaswsw/recommend-worktrees/lc-rec/lc_rec/offload/qwen_20261109/`。第一个 shell 命令为规则入口的 `ls -la`；没有 Git 操作、SSH、依赖安装或共享配置修改。

在当前工作树只读检索项目 Markdown／TXT／YAML 和 LC-Rec 候选 JSON，未将 GRID 的旧 Qwen embedding 当成本项目模型选择。现有 `lc_rec/config/formal_five_seed_candidate.json:24`、`lc_rec/offload/config/formal_candidate.json:88`、`lc_rec/REVIEW_ENTRY.md:72` 均只记录后续“Qwen3-8B”。上一轮固定交付可定位 [候选配置](https://github.com/woobowen/recommend/blob/2e0e26633b58d44cc4595417f4a054f6bb6b7e77/lc_rec/config/formal_five_seed_candidate.json)；本轮任务书还明确说明 Base／后训练身份尚未冻结。没有找到适用于本项目的明确二选一确认，因此没有完整权重下载授权。此检索结论限于上述材料，不声称已读取用户所有历史会话。

官方依据：两仓库的固定 revision 模型卡、配置、tokenizer 配置、index 和 HF API `?blobs=true` 原始 JSON。API 的实时 main 仅用于取得以下完整 SHA；具体文件均通过该 SHA 的 resolve URL取得。原始响应、下载事件及逐文件校验见本目录。网页工具可读官方模型卡，对两个 API URL 返回 Internal Error；普通 HTTPS 成功取得 API，未改变任何网络设置。

## 两个候选的真实差异

| 项目 | `Qwen/Qwen3-8B` | `Qwen/Qwen3-8B-Base` |
|---|---|---|
| 固定候选 revision | `b968826d9c46dd6066d109eabc6255188de91218` | `49e3418fbbbca6ecbdf9608b4d22e5a407081db4` |
| 官方模型卡训练阶段 | 预训练＋后训练 | 预训练 |
| 架构 | Qwen3ForCausalLM，36层，hidden4096，FFN12288，Q32／KV8，head_dim128，Q/K norm | 相同结构声明 |
| 权重 API | BF16，报告 8,190,735,360 个参数；5片，16,381,516,776 bytes（15.25647638 GiB） | 相同报告数量、片数和文件总大小；五片 SHA-256 全部不同 |
| embedding／head | `tie_word_embeddings=false`；config vocab151936 | 相同 |
| config上下文 | max_position_embeddings40960，rope_theta1,000,000，无 rope_scaling | max_position_embeddings32768，其余相同 |
| 模型卡上下文 | 原生32768，说明另用YaRN扩展 | 32768 |
| tokenizer配置 | Qwen2Tokenizer；bos=None，eos=`<\|im_end\|>`，pad=`<\|endoftext\|>`，unk=None | Qwen2Tokenizer；bos=None，eos=pad=`<\|endoftext\|>`，unk=None |
| tokenizer原件 | JSON 11,422,654 bytes，LFS SHA256 `aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4` | JSON 7,031,645 bytes，普通 Git blob `443909a61d429dff23010e5bddd28ff530edda00` |
| generation_config | do_sample=true，temperature.6，top_k20，top_p.95，EOS含151645/151643 | do_sample=false，EOS151643，max_new_tokens2048 |
| 许可证／Hub访问 | 模型卡 apache-2.0，有 LICENSE 文件；API gated=false/private=false | 模型卡 apache-2.0；该固定 revision 未列单独 LICENSE；API gated=false/private=false |
| 兼容性 | 原生支持要求 Transformers>=4.51.0 | 同左；generation_config中的4.37.0是文件保存元数据，不代表该版本认识qwen3 |

上述是**metadata／文件身份核验**，不是 Qwen 权重载荷或架构实例化实测。API、两份 config、两模型卡不一致之处保留原状：后训练模型的 card/config/tokenizer 上下文分别为32768/40960/131072；不由此自动提高本项目2048长度。Base index `metadata.total_size=16,383,567,872`，但 API BF16参数数×2为16,381,470,720，实际LFS文件总大小为16,381,516,776；index标量与两者均不符，未通过修改index消除。下载预算按五片可信LFS长度之和，最终完整性按逐片SHA256，不按这个index标量。

两个 tokenizer 的模型词汇均151643项、merges151387项；配置均列26个added token至151668，但 Base 的 tokenizer.json 本体只有22项added token至151664。merges.txt／vocab.json 的API Git blob在两仓库相同，两个tokenizer.json的blob不同。尚未实例化AutoTokenizer，因此不把JSON字段推导当实际 `len(tokenizer)` 回读。

固定官方引用：[后训练模型卡](https://huggingface.co/Qwen/Qwen3-8B/blob/b968826d9c46dd6066d109eabc6255188de91218/README.md)、[后训练配置](https://huggingface.co/Qwen/Qwen3-8B/blob/b968826d9c46dd6066d109eabc6255188de91218/config.json)、[Base模型卡](https://huggingface.co/Qwen/Qwen3-8B-Base/blob/49e3418fbbbca6ecbdf9608b4d22e5a407081db4/README.md)、[Base配置](https://huggingface.co/Qwen/Qwen3-8B-Base/blob/49e3418fbbbca6ecbdf9608b4d22e5a407081db4/config.json)。

## 与原 LC-Rec 替换实验的关系及首选建议

**建议将 Base 作为待批准首选**：原 LC-Rec 初始化来自LLaMA语言预训练权重，Base保留“由语言预训练模型开始推荐微调”这一比较条件，避免额外引入指令／推理后训练。依据是官方训练阶段声明与控制变量推导，不是王学长方案，也不是模型效果实测。如果用户研究目标是后训练Qwen，则应明确选择后训练仓库，并将额外训练阶段作为主干差异报告。

无论选哪一个，Qwen都是另一个待批准主干实验，不能称为LLaMA的1:1复现。需要独立隔离环境；本轮不更新LLaMA的Transformers4.31栈。后续至少核对以下差异，当前均未实施：

- 原代码显式使用 `LlamaForCausalLM/LlamaTokenizer/LlamaConfig`，Qwen需匹配的Auto类／Qwen3类；不能仅换目录。
- Qwen的36层、GQA/QK norm、RoPE、151936行embedding/head、预训练语料和8.19B参数均与LLaMA不同，内存和CPU offload成本需另估。
- 原collator把 `labels==pad_token_id` 全部mask。Base默认pad=EOS，机械迁移会连回答末尾EOS监督一起去掉；原LLaMA的pad0/EOS2不能照搬。未来须验证tokenizer实际长度、attention_mask与response-only loss，保持既定EOS监督含义；不在本轮决定新的token/padding策略。
- 根据JSON推导，若loader实际词表长151669，则930个新增SID会占151669–152598，其中前267行已在151936行原权重矩阵内；简单resize只新增663行，不能宣称930行均沿原LC-Rec流程新初始化。此为静态风险，需未来实测并审批SID初始化方案。现代resize默认的均值初始化也不能静默替换4.31的Normal初始化。
- 后训练版chat template／thinking开关和采样generation_config会改变提示与解码。不能自动加入chat模板、think token、CoT或默认采样；须保留获批准的LC-Rec prompt、beam20、逐位置约束和evaluator，并实际回读生成参数。Base同样不能把其默认max_new_tokens2048替换官方10而不说明。

## 已做的小文件准备与验证边界

`asset_checks.json` 对已经取得的13份官方载荷逐一校验大小及身份：后训练7份（含LICENSE与tokenizer JSON）、Base6份；全部匹配固定revision的Git blob或LFS SHA256。另有两份API响应及 `candidate_summary.json`，逐片列权重官方URL、长度和可信LFS OID。没有权重载荷，因此权重全文件校验为NOT TESTED。

两份tokenizer JSON仅作WSL本地缓存，合计18,454,299 bytes；不需要加入Git或审核ZIP。独立merges.txt／vocab.json尚未下载，NOT TESTED；所需信息已包含在取得的tokenizer JSON及API身份中。本目录新增逻辑文件总量、每文件SHA和发布范围见 `delivery_file_manifest.json`。无新增系统包／语言包／工具链，无.bashrc或网络配置修改。

主控通过既有SSH只读检查的限定缓存路径：`/home/bwwu/.cache/huggingface/hub/models--Qwen--Qwen3-8B`、相应`-Base`和`/data/bwwu/.cache/huggingface/hub`均MISSING；`/data/bwwu/lc-rec/assets`根下无Qwen目录。原始回执为上一级 `evidence/preparation/source_hardware_qwen_cache_readback_20261109.json`。该小范围结果不能证明整个服务器没有Qwen权重，也不触发重复下载。

## 收集失败、中断及统一修正预算

`collect_metadata.py`首次metadata收集退出0；可选tokenizer收集真实退出1，因为脚本错误地将后训练tokenizer的LFS pointer blobId当载荷Git SHA1核对。已下载11.42MB文件的SHA256确与API的LFS OID匹配，直接只读核验见 `asset_checks.json`；不把原失败改写为PASS。

子代理未及时获知此前环境已用完任务级round1–round3，在收到该信息的同一工具边界前，已经修改collector的LFS分支并发出一次重跑。这次行为超过已用尽的修正额度，**如实记录为执行偏离**；主控消息到达后立即Ctrl-C，真实退出130，没有再改／重跑collector，没有继续网络下载。修复后的collector未完成整体复核，不能标为通过。两份已取得tokenizer只做只读hash/JSON检查，余下可选文件停做。原始错误摘录、已改内容及中断见 `metadata_failure_and_repair.json`、`collection_attempts.json`。该记录不会给子任务另开三轮额度，也不改变GPU预算。

## 后续下载命令、预算与完整性方案（本轮未执行）

需要用户确认“仓库名＋Base/后训练身份＋上表完整revision”，再将所选项置入独立下载request。下列只是准备的命令模板，不含默认选择；未来执行须按项目runner固化脚本、累计deadline和status，不依赖本地SSH会话。

```bash
# 仅在模型身份获用户明确确认后填入；两项必须与candidate_summary.json同一候选一致。
QWEN_REPO='<用户确认的官方仓库>'
QWEN_REVISION='<对应完整40位revision>'
QWEN_STAGE='/data/bwwu/lc-rec/assets/qwen3/<确认身份>/<完整revision>.incomplete'

# 在未来独立资产环境验证hf --help及其版本，再以同一job_id断点接续。
# 单一累计3小时deadline须由runner保留，重试不得重置为新的3小时。
hf download "$QWEN_REPO" --revision "$QWEN_REVISION" \
  --include '*.safetensors' 'model.safetensors.index.json' 'config.json' \
  'generation_config.json' 'tokenizer*' 'vocab.json' 'merges.txt' 'README.md' 'LICENSE' \
  --local-dir "$QWEN_STAGE"
```

官方接口依据为 [HF下载文档](https://huggingface.co/docs/huggingface_hub/guides/download) 的完整revision、文件过滤和local_dir缓存；该模板尚未对服务器网络或当前CLI实测。当前LC环境的hub0.16.4未必提供 `hf` 命令，不能原样声称已可执行；未来应在独立资产环境固定并核验CLI版本，或基于已验收分块receiver作有审查的Qwen身份参数化。本轮没有安装新Hub或改LLaMA relay脚本。已有relay硬编码LLaMA仓库、两片文件和接收路径，不能直接拿来下载Qwen。

未来保存5片safetensors16,381,516,776 bytes，仅一种格式、不下载bin/GGUF/量化副本；小文件约23MB以内，总权重低于20GiB。若local_dir或HF缓存引入重复载荷，先预算并只保留一个真实载荷位置，不能未经核实承诺15.26GiB就是所有临时峰值；保守预留至少40GiB工作空间，另保留文件系统max(30GiB,10%)安全余量。若服务器无出网，复用有界256MiB分块／已有Windows SCP与receiver，WSL staging维持远低于5GiB，先核已有同身份缓存；不改变网络设置。

每片完成后先核精确长度，再全文件SHA256匹配 `candidate_summary.json` 的官方LFS SHA；核index引用恰为这5片且无缺片，保留Base index元数据差异。断线先读原job和`.incomplete`／缓存状态，不能盲重下或把文件长度当完整性。全部片通过后再原子发布固定revision目录，输出server路径、逐片hash、下载累计字节/时间、退出状态和恢复入口。没有通过前不得加载模型；本轮也没有Qwen加载／训练授权。

## 发布范围与停止状态

Git/ZIP收录范围由 `delivery_file_manifest.json` 的 `publish=true` 精确给出：本报告、API/配置/模型卡/license、tokenizer_config、小型index、candidate_summary、asset_checks、实际收集程序、失败/中断回执、下载事件及文件清单。两份大型tokenizer JSON `publish=false`，留WSL本地cache；不上传整个仓库或其他项目文件。

Qwen子链终态：身份调查与候选推荐PASS（静态范围）；13份已取得官方文件身份PASS；collector整体PARTIAL（exit1、后续exit130均保留）；独立vocab/merges下载NOT TESTED；全权重BLOCKED等待明确身份；模型加载／训练／评价NOT TESTED且本轮不授权。无活动或排队Qwen作业。主控还需纳入非作者审核、固定Git发布和最终ZIP。此调查不单独声称网页GPT二审或整轮交付完成。
