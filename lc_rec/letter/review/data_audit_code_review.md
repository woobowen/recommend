# LETTER A/B CPU 数据审计代码说明

本文件是脚本作者的实现说明和自检记录，**不是非作者独立 PASS**。运行产物由主控在 new-server 使用实际固定解释器获得；本文不预写服务器结果。

## 源码依据与边界

实际读取官方固定提交 `8d0154e28de37dbb6e24871c508ad8ddb1921cda` 导出的 `LETTER-LC-Rec/data.py`、`prompt.py`、`collator.py`、`utils.py`、`lora_finetune.py`、`test_ddp.py` 和 `evaluate.py`，以及作者补充的 Instruments 配置原文。三规则来自显式读取 `/home/addaswsw/recommend/`，不声称在相邻工作树中自动发现。

`audit_data.py` 不加载模型、不用 GPU、不生成推荐指标、不读取辅助 intention 文件、不更改 A/B 资产或作者源码。只生成新的隔离输出目录。运行前和分批 tokenize 时检查 40 GiB 主机可用内存下限；默认 CPU 审计时限 1,200 秒，超时保存已有资产证据后返回 `PARTIAL`，不将缺项写成通过。

## 执行与产物

```bash
python audit_data.py --source /fixed/source/LETTER-LC-Rec \
  --a-data /fixed/LETTER/data --b-data /fixed/LC-Rec/data \
  --model /data/bwwu/lc-rec/assets/llama-7b/4782ad278652c7c71b72204d462d6d01eaaf7549 \
  --output /new/isolated/audit-directory
```

`--a-data` 和 `--b-data` 都接受数据根目录或直接的 `Instruments` 目录。`--output` 必须不存在，防止覆盖既有回执。只需当前依赖中的 `torch/numpy/transformers/sentencepiece`，没有安装动作。

- `audit.json`：源码和资产字节 hash、完整 A/B 比对、各轴和碰撞、实际执行原 `SeqRecDataset` 的全部行一致性、tokenizer 身份、label/mask 检查、长度分布、有效计数、状态/异常/耗时/RSS。
- `tokenizer_a_ids.json`、`tokenizer_b_ids.json`：各自扩容的全部 token → ID，不把两套 SID 顺序混用。
- `tokenizer_a/`：A 的小型扩容 tokenizer，保存重载后核对全部词表与 padding ID；不含模型。
- `tokenizer_implementation_*.py.txt`：实际 tokenizer 的 added-token 分割实现，支持有限类别长度推导审核。
- `length_rows.npz`：`train/valid/test_{input,full}_lengths`，shape 为 `[真实行数,17]`；`full` 包含 EOS。另含历史长度、训练用户 ordinal、target position、seed42 验证 prompt IDs。
- `probe_samples.json`：q10/q50/q90/max 的真实 `row_idx + prompt_id`；128 个固定混合训练样本；test prompt0 与全部模板的代表；实际 item/history/user 和长度；资产/源码引用。

## 完整比对与实测/推导区别

全部 inter/item/index JSON 都做字节 SHA-256、规范化 JSON SHA-256 和完整 per-key 值比较；交互按用户与 target position 完整比较；原始 ID 标签与 SID 字符串身份分别检查。映射结论限于实际提供的 metadata 和交互轴，不虚构未提供的历史 ASIN、user remapping 或 checkpoint 绑定。

实际导入作者 `SeqRecDataset` 并构建 train、valid、test，全部 `inter_data` 与显式重建的 leave-two-out、history20 逐行一致后才继续。训练 record `i` 的源 target 位置不含尾部两个保留项，valid target 倒二，test target 倒一。训练真实访问随机抽 17 模板；`sample_num` 在 SeqRec 训练分支没有采样作用。验证使用 NumPy seed42、每用户两个无放回 prompt，并逐条与原 valid text 比较。

train/valid/test 的 **prompt0 全体输入和带 EOS 全文都实际 tokenize**。其他模板使用有限 history-length 类别：所有 SID 必须为新增 atomic `<...>` token、item SID 长度相同；遍历所有真实历史长度与全部 17 模板，实际 tokenize 类别文本；每个真实 source row 的全部非 SID 片段都与其类别逐字核对。然后在每种历史长度、每种 split、每个非零模板上 tokenize 固定真实样本核对。运行记录给出实际检查数和条件性穷举最大值，不把它描述成逐条穷举 tokenize 了全部 `N×17` 文本。该分类推导依赖实际保存的 LlamaTokenizer added-token trie 分割规则；独立审核可直接复核这项条件。

新增 SID token、全部 item SID 拼接都实际 encode；训练/评价共用同一 index 的候选集、position allowed sets、真实 `Response:` 起点约束、全部 SID decode/evaluator normalize 均核对。原 constraint 是逐位置集合，允许不存在于 item 集合中的交叉组合，脚本记录这一边界。`test_ddp.py` 的原 sampler `drop_last=True` 与潜在尾用户丢失单独记录；不擅自修改。

mask 检查真实 q10/q50/q90/max 混合 batch，直接执行作者原 `Collator`，独立期望为 input prefix 与 padding `-100`，SID 与 EOS 保留；同时核对 input token IDs 确为 full prefix、每个目标 SID、attention mask 和 5 个监督 token（以实际 SID 长度为准）。

## 需主控和非作者审核核验

本地仅可做语法和标准库逻辑检查；服务器依赖加载、全部 tokenizer/原 Dataset/Collator 运行、时限、文件内容与实际 exit code 需由机器产物证明。本文没有将论文指标、历史训练命令或 checkpoint 的身份作任何补全。更新次数 `ceil(N/64)` 是保守预算值；正式 Trainer/DeepSpeed 对最后累积批的取整须绑定实际版本另核。

作者本地实际验证：`compile()` 与 `--help` 通过；真实官方三 JSON 的标准库全体重建检查通过，train=131,837、valid=test=24,772，全部 row 的 target position 回查一致。官方 index 为 9,922 items、每项 4 级 SID、9,897 唯一 SID；23 个碰撞组涉及 48 items，最大组 3 items。碰撞原件保留，未进行任何去重或重编码。A/B 服务器比较和 tokenizer 检查待实际运行，不在本说明中预先判定。
