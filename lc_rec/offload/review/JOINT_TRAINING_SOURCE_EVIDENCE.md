# LC-Rec 六任务联合训练：源码证据与数据流优化边界

2026-11-08，真实子代理 `/root/official_audit` 定向复核。读取对象是当前实际 `/home/addaswsw/recommend-worktrees/lc-rec/lc_rec/vendor/original/` 内的 finetune.py、utils.py、data.py、collator.py、run.sh；逐文件字节数/SHA-256见同目录 `source_identity.json`。本次未跑旧数据审计、未读模型/数据、未操作Git或服务器、未修改上述原件。另显式重读规则入口三份已更新规范相关章节。

## 明确结论

官方是**六个任务Dataset合成一个ConcatDataset，交给一个模型和一个Trainer，执行一次联合训练**。六任务不是六份7B模型/Adam状态；把任务改成六次依序训练不等价，也不能因此声称解决原联合配方的显存需求。CPU offload所需处理的是这个共享模型/优化器的状态；Dataset数量主要影响CPU数据驻留，不会乘六或消除7B参数和Adam状态。

| 证据位置 | 实际逻辑 |
|---|---|
| `utils.py:140–168` | 按tasks顺序建立seqrec、两个ItemFeat、fusionseqrec、itemsearch、preferenceobtain六个Dataset，加入同一个list。这里的for仅构建数据，没有train/model/optimizer调用。 |
| `utils.py:170–174` | `train_data = ConcatDataset(train_datasets)`，另建SeqRec验证集，返回这一对dataset。Concat本身不决定shuffle或task-balanced采样。 |
| `finetune.py:39–48` | 一次load_datasets；tokenizer统一新增第一个SeqRecDataset提取的SID token，构建一个Collator。 |
| `finetune.py:51–56` | 一个LlamaForCausalLM.from_pretrained并resize一次。没有按任务各建一个模型。 |
| `finetune.py:64–97,101–106` | 一个Trainer收全部train_data；一个全局optim/learning_rate/scheduler配置；一次trainer.train、一次最终save。优化器由该Trainer/DeepSpeed内部管理，源文件没有六个task专属优化器或权重。DDP/ZeRO的进程副本/分片仍对应同一逻辑模型。 |
| `collator.py:25–44` | 各任务返回相同的input_ids/labels文本接口，共用full-text＋EOS、2048上限、longest动态padding、response-only mask。无按task加权分支或task专属head。 |

训练样本的实际采样器、跨rank的索引序列及优化器类，应由本轮锁定Trainer/DeepSpeed运行回读；仅这五份源码不能证明某个具体batch里一定同时出现六任务。尤其Concat的注册顺序不能解释成六阶段课程顺序。源码没有显式task loss权重；若Trainer在concat索引上等概率采样，任务曝光量由各任务实际长度决定，不能擅自改成六任务各1/6。

## 官方默认与 Instruments 覆盖

| 配置 | parser默认（utils.py） | Instruments脚本实际覆盖（run.sh:71–93） |
|---|---|---|
| model / dataset | `./llama-7b/`（18–20），Games（33） | 预期huggyllama/llama-7b、Instruments；72行`BASE_MODEL= huggyllama/llama-7b`存在shell空格错误，实际入口应修拼写并记录diff，不能原样执行三域脚本。 |
| tasks | 六任务，顺序同上（31） | 六任务同顺序（90） |
| prompt/data sample | `1,1,1,1,1,1` / `0,0,0,100000,0,0`（45–48） | prompt全1；data改为`0,0,0,20000,0,0`（91–92） |
| lr/epoch/wd | 2e-5 / 4 / .01（62–68） | 5e-5 / 4 / .01（83–85） |
| micro/accum/world | 8 / 2（64–65）；world来自环境 | 8 / 2 / torchrun8进程（76,81–82），有效batch128；两卡offload若采用micro1则accum64才能保留128。具体本轮配置由主控保存并回读，本报告不执行。 |
| optimizer/scheduler/warmup | adamw_torch / cosine / .01（61,80–81） | 未覆盖，继续取默认；offload的实际optimizer实现与AdamW等价范围需单独验证。 |
| history/token上限 | 20件 / 2048 token（37,67） | 未覆盖；二者不是相同限制。 |
| response-only/BF16 | 都为False（42–43,85） | 显式开启（88–89） |
| validation/save | epoch（82）、valid sample=True且每用户2模板（50–55） | epoch（86），finetune.py:88–93另固定save_total_limit5、load_best=True和epoch eval_delay1。技术预跑的验证/保存预算需要独立限制，不能从正式默认自动继承全量。 |

数据采样0不表示零条：ItemFeat `data.py:513–519`、Fusion `341–346`、ItemSearch `612–618`、Preference `722–728`均仅sample_num>0时抽样；SeqRec训练 `146–163`直接生成全部训练prefix，根本不按sample_num截取。Fusion20k为构造时无放回抽一次，不是每次取样重抽。

动态随机必须保留：SeqRec `data.py:252–272`、Fusion `457–475`、ItemFeat `535–546`每次getitem随机模板；ItemSearch `645–663`另随机偏好/query；Preference `747–760`另随机偏好。不能预先固定一种模板然后把缓存文件称作同一联合训练定义。

## 六次顺序训练的差异

1. 若六次分别调用官方train(args)，每次会重新seed、加载base model、resize SID并创建Trainer；这是六个独立训练，未实现联合对齐。
2. 若上一任务权重接到下一任务，但每次新建Trainer/optimizer/scheduler，Adam矩、step数、warmup/余弦进度与选模范围会重置，变成阶段微调。
3. 即使刻意保留同一optimizer并顺序遍历六任务，也改变了task的时间次序、各update混合、随机模板流及epoch/验证边界。AdamW与非线性模型通常依赖更新顺序；总样本数和总epoch相同不足以证明等价。
4. 逐任务训练也仍然需要同一7B模型/梯度/Adam状态；除非删除任务、冻结或变精度等另改方法，不能靠去掉Concat释放模型状态。以上替代不属于可冒称无语义影响的数据优化。

## 可以保持联合定义的优化候选

本轮首选保留官方loader/Concat/Collator，不为CPU offload再引入数据流重写。确有CPU内存/吞吐瓶颈时，才考虑下面的小范围候选并用定向等值回归验收；本次未实现、未运行这些候选。

- **共享只读资产解析或SID字符串缓存。** 当前SeqRec/Fusion/ItemFeat/Search/Preference重复json.load同一index、item或user文件，可使底层资产只读共享，避免重复parse/驻留。需留意ItemFeat `data.py:506–510`直接修改feat，Search/Preference getter也回写样本dict；因此必须隔离每样本/每任务可变字段，不能盲目把原dict共享后并发写。不会节省模型/Adam状态显存。
- **惰性prefix记录。** SeqRec/Fusion目前构造完整历史文本list（146–161,315–339）。可只保存保持原顺序的(uid,target_position)等轻索引，访问时按同规则拼接；保持样本总数、concat全局索引→任务/局部索引映射、20件截断、title清理、Fusion原20k选中索引完全一致。不得重新抽一套20k子集或改变np.random调用顺序。
- **只缓存确定性转换。** 可缓存item→canonical SID、清理后的title和确定性token片段；完整样本若含随机模板/偏好/query，则不能只缓存第一次输出。多worker/prefetch、缓存命中引起的RNG调用次序变化也要记录；保持分布不等于逐步数值轨迹相同。
- **CPU准备与GPU传输重叠。** 可优化加载/prefetch而不变样本/标签，但要与DeepSpeed CPU offload共同预算CPU RAM、锁页内存、线程和PCIe。collator已经longest动态padding（32行），改成packing、按长度重排或固定padding并非这里已证明等价的替代。

最低等值回归应对同一全局索引与相同Python/NumPy RNG状态比较原实现和派生实现的输入/回答及tokenized input_ids、labels、attention_mask；同时比对各任务长度、Fusion20k索引与concat映射，验证集保持原固定模板集合。这里提出验证要求，没有虚构测试结果。

**本次结论边界：** 已以实际五文件完成静态单模型/六任务联合定义复核。CPU offload是否在本次双4090、CPU内存/带宽和预算中可运行，须由主控独立资源估算及有界真实更新证明；本报告不替代该结论。没有新增安装、GPU作业或旧任务重跑。
