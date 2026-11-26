# 同一 LETTER Instruments 目录上的 GRID 上游与下游

本实验固定 LETTER Instruments 的 9,922 件商品与 24,772 个用户的完整有序交互。每件商品按 ASIN 找到历史 GRID Qwen2.5-7B-Instruct 的编码文本与 FP32 向量。量化器从原初始化流程重新拟合，生成新的三位 RQ-KMeans 前缀与一位碰撞编号。用这一目录从随机初始化训练一次 Normal，在原验证集选择一个 best，再用该 best 依次评测 Normal、Shuffle3a 和 Shuffle4。

## 固定来源

| 材料 | 采用内容 |
| --- | --- |
| [A · LETTER 数据](https://github.com/HonghuiBao2000/LETTER/tree/8d0154e28de37dbb6e24871c508ad8ddb1921cda/data/Instruments) | 原始完整交互、商品和旧 index 校验。ASIN 由 LC-Rec 作者归档 item2id 提供，其交互与商品文件逐字节匹配 A |
| [B · GRID Instruments Qwen](https://github.com/woobowen/recommend/tree/bc6176dc144b827b1b29f73ad3b216e28b39f48f/Baseline%20Results/Instruments%20GRID%20Qwen) | 冻结向量和逐商品文本、原 RQ-KMeans、去重、配置与初始化 |
| [C · GRID Normal](https://github.com/woobowen/recommend/tree/6be303ef7e05968c6a41400702732d954ea1eea1/experiments/letter_instruments_grid_normal) | 有效 Hydra 配方、原生模型、训练数据读取／截断／collator、选模、自回归 Beam10 和评价 |
| [D · Shuffle V2](https://github.com/woobowen/recommend/tree/af360a158280165f07249b80d2cfa0e871902498/lc_rec/shuffle_v2) | 规范 JSON、SHA-256 无偏拒绝采样和独立 factoradic reference。沿用历史 A0／A1 的绝对用户与历史位置 |

原 Qwen 编码对每件商品使用既有文本、右 padding 到 128、float16 模型和按本行 mask 的 last hidden state 均值。保存为 float32，未额外单位归一化。normalize_residuals=true 生效于量化器内部。旧档案缺少证明全部历史权重分片来自同一 Hugging Face revision 的完整记录。

## 调用与参数

| 阶段 | 实际入口和契约 |
| --- | --- |
| 共同资产 | assets.py build/verify 校验 A、LC-Rec 作者归档及 B 的固定字节，构建 LETTER ID ↔ ASIN ↔ 旧 GRID 行 ↔ item_id。本轮内部编号沿用 LETTER ID。用户轴沿用 V2 绝对编号 |
| 量化 | upstream.py fit 调 upstream/runtime.py 和冻结 rkmeans_train_flat。三层 256、seed 42、输入维度 3584，原 KMeans++、初始化 buffer 3072、源 batch 2048、SGD 0.5，正式 trainer max_steps=30。初始化计算与 Lightning 更新分别记录。详见 [上游说明](upstream/README.md) |
| SID | upstream.py sid/verify 由同一次拟合严格取得前三码。原 deduplicate_rows_in_tensor 为单件前缀组分配 0，为重复前缀组的全体成员按 item_id 升序分配 1 到组大小。自动检查 [4,9922]、词表容量、整数、各层输出与来源、唯一性、双向商品和 SID 对应 |
| 派生数据 | downstream.py prepare 重新产出训练 S[:-2]、验证 S[:-1][-21:]、测试 S[-21:]。读回两种 decoder 的所有 TFRecord 与 9,922 项目录，当前 SID 与哈希绑定到数据、lookup 和下游 |
| 训练 | 复用 C 的 run_normal.configure 校验原有效配置。共享 SID embedding，四层 encoder／decoder，d_model 128、6 heads、d_kv 64、d_ff 1024、两层隐藏 MLP、dropout 0.15。seed 42、随机初始化、单卡 FP32、compile=false、matmul medium、沿用确定性配置。Adam lr 0.001、weight decay 0.0001、无 scheduler |
| 数据与选模 | 源 batch 16，窗口最多 120 SID token（最多 29 个历史商品与目标），原 causal duplicate collator 保留有放回采样和上限 128。累积 16，实际扩展量逐 microbatch 可变。eval batch 8。每 1,600 microbatch 验证一次，按验证 Recall@5 最大值保存单个 best，patience 10，max_steps 320000，max_epochs 10 |
| 三种评价 | 普通 eval seed 2026。每种条件都严格载入同一 best 文件和 state hash。四步自回归 Beam10，自由生成与原 softmax 概率乘积、缓存、原生排序全部沿用。每人记录原候选顺序、原 FP32 分数位模式、四组 native top-k、目标、历史、mask 和实际层级 ID |
| 汇总 | 复用 C 的原生 Recall／NDCG 与冻结统一 SID／CCE。保留目录外 SID、重复候选、并列顺序及原分母。逐条件确认同用户顺序，按本轮 Normal 计算绝对差与相对差。Normal 零分母的相对值存为 JSON null |

原 TFRecord 训练流无显式 TensorFlow shuffle seed。有限预跑可验证真实路径和参数更新，跨环境训练权重不具备逐位复现的已知保证。正式四码若超出既定词表，SID 关卡终止依赖阶段。

## V2 的 GRID 输入边界

原 GRID 在每个裸码上叠加层级 offset，随后读取共享 embedding 表，再插入 SEP。本实现由 EncoderV2 在测试 encoder 调用的局部作用域内移动 offset 后的整数 token。真实历史由已核验原序列、collator 后的裸码和 mask 三方匹配确认。目标标签、padding、SEP、decoder BOS、码头及 T5 原生相对位置规则沿原路径。teacher-forcing 与 generate 各自进入 encoder，使用相同的确定性计划并严格比较两次实际 lookup 记录。

shuffle.py 对 ["GRID-LETTER-SHUFFLE-V2",42,m,global_row_idx,visible_item_idx,rejection_counter] 进行紧凑 JSON 编码、SHA-256 与无偏拒绝取模。Shuffle3a 取前三层的 6 种排列，第四层保持原位置。Shuffle4 取全部四层的 24 种排列。两者都包含恒等。独立 CPU 回放核对全用户与逆序分片。历史 A0／A1 与本轮测试段的用户和可见历史商品位置逐项一致。GRID 的 token 值来自本次拟合。实验覆盖 Normal、Shuffle3a、Shuffle4。

## 运行、复核与接续

- [正式完成结果与全量复算](evidence/formal/grid-letter-rerun-20261125-formal-r0/completion/)记录正式 r0 终态、最终 best、三条件指标、证据与复核。
- [正式启动与 30 分钟观察](evidence/formal/grid-letter-rerun-20261125-formal-r0/)记录固定作业、进展、实测验证和接续入口。准备阶段的正式模板及预跑回执见[运行证据](evidence/)。
- pipeline.py request 组装已有 P0 runner 的 immutable request。正式和预跑分别固定 formal.json、preview.json，要求不同的 job ID。它核对冻结提交和文件输入哈希、当前 GPU 对应与空间。用户另行批准正式计算后才提交已固定的正式模板。
- pipeline.py execute 仅在 runner 的独立作业根运行。量化、SID、SID 准入、数据重建、Normal 训练、三项同 best 评测和离线汇总按依赖顺序执行。关卡捕获源码版本、资产、状态和产物 SHA-256。
- 先查询原 job ID。需要新 attempt 时，--reuse-job 仅接续同模式、同源快照和同输入、真实终态且阶段产物按哈希核验成功的已完成阶段。失败训练不自动重跑。P0 runner 的单请求墙钟限额为 86,400 秒，和 trainer 上限分别计。断线时服务器 runner 继续独立管理原作业。
- 单包含审核原件、只读证据和独立审核。源码入口见本目录，历史配方及评价原件见同仓库的 [C 实验目录](../letter_instruments_grid_normal/)，历史大资产保留其原 Release 身份。审核包内提供本轮实际使用的可直接审读副本。
