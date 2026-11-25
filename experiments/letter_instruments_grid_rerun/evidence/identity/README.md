# 共同 Instruments 输入与商品身份核验

本轮以 LETTER `8d0154e28de37dbb6e24871c508ad8ddb1921cda` 的三份原件为数据依据。这里保存的 GitHub 原件从该 commit 的 raw URL 取得并与 new-server 历史工作副本逐文件核对。LC-Rec 作者公开的 `Instruments.zip` 与既有服务器副本的 SHA-256 一致，解压成员全部通过 CRC。该归档的 `Instruments.inter.json` 和 `Instruments.item.json` 与 LETTER 固定版本字节相同，因此同归档的 `Instruments.item2id` 可桥接 LETTER 商品编号和原始 ASIN。两来源的 `Instruments.index.json` 哈希不同。本实验的数据轴核验使用 LETTER 固定版本的 index。

GRID 原件来自 `bc6176dc144b827b1b29f73ad3b216e28b39f48f` 记录的 `instruments-grid-baseline-20261125` Release。`instruments-grid-original-assets.zip` 实际下载 516,896,970 bytes，SHA-256 为 `8801dfe5d3fcc75f99101dea1d50cf5abc742d44ce3771746ffd04ff9277929a`，同发布摘要及 `grid/grid_assets.json` 相符。从中读取的 mapping、文本及 embedding 成员均单独校验 SHA-256，成员内容保存在 `grid/`。完整历史归档保留在用户控制的本地暂存目录。

逐 ASIN 等值连接得到 9,922 条一对一记录，覆盖全部 LETTER 商品。旧 GRID 共 10,611 个 ASIN，其中 689 个只在旧目录。连接不依赖标题或文本。 `common/item_identity.jsonl` 逐行保留 LETTER ID、内部 ID、ASIN、旧 GRID ID、原 embedding 行号及文本和行值的 SHA-256。旧 GRID 的文本、ASIN 文件按 item ID 0 到 10,610 排列，原 `qwen_worker.py` 以同 ID 顺序将向量写回输出行。共同目录中内部 ID 沿用 LETTER ID。`common/items_qwen.jsonl` 按共同目录重排原始编码文本，严格保持原字符串。向量复用保持 FP32 和原始 3,584 维数值。9,922 行保存后重新加载，比对全体行的字节 SHA-256。共同向量文件在 new-server 为 142,242,972 bytes，SHA-256 `4ee521af070598bc91333f2d1cb0664193ad2d6f93401254641e73d494f2e6af`。审核材料保留一份完整 `[10611,3584]` 原张量。全量派生向量可通过读取桥表后按 `old_grid_embedding_row` 依序索引构造，每行的位级预期均在桥表中。

旧模型为 Qwen2.5-7B-Instruct，max_length 128、右侧 padding、模型权重 float16，按单行 token mask 对 last_hidden_state 做 WMH MeanAggregation 并存为 FP32。原产物未增加单位向量归一化。原归档有权重路径、模型 config 摘要和编码记录，尚缺证明全部权重分片来自同一 HF revision 的记录。旧 Qwen 模型只用于历史已冻结向量，本轮复用没有重新编码。量化器的 `normalize_residuals` 属于后续单独阶段。

全量用户、序列及历史边界统计如下。

| 检查项 | 实测 |
| --- | ---: |
| 共同用户 | 24,772 |
| 共同商品 | 9,922 |
| 全序列交互数 | 206,153 |
| user–item 重复行 | 0 |
| 用户最少交互数 | 5 |
| 商品最少交互数 | 5 |
| 经 ASIN 往返后逐项不变的完整序列 | 24,772 |
| 训练保存的商品出现次数 `S[:-2]` | 156,609 |
| 验证最多 20 个历史商品的总出现次数 | 148,927 |
| 测试最多 20 个历史商品的总出现次数 | 172,929 |
| 验证及测试目标次数 | 各 24,772 |

原 `Instruments.inter.json` 的用户插入顺序为数字 0 到 24,771，与既有 A0、A1 Instruments Shuffle V2 原始回放的 row_idx 一致。`prior-v2/prior_v2_replay_projection.jsonl` 是两份历史回放及 LETTER 原始交互的全量结构化提取，保留每个用户的编号、目标、实际可见商品 `S[:-1][-20:]` 及 Shuffle3a 和 Shuffle4 的排列编号。两个路线的每个目标、可见长度和逐出现位置的排列编号均相同。历史回放源码按该有序交互和同一截断规则构建可见历史，验证与历史 tokenizer 实际输入一致。这为本轮逐出现位置比较提供原始序列和原排列编号。历史恒等次数分别为 28,935 和 7,147。文件移除了两个历史路线的重复排列字段、与本轮排列无关的输出和本轮范围之外的历史条件。原始回放文件的路径和完整 SHA-256 均在 `common/assets_manifest.json`，提取实现为 `assets.py` 中的 `replay_projection`。

`identity-build.log` 记录实际执行终点。new-server 的 `/data/bwwu/grid-letter-rerun-20261125/inputs` 是本轮共同输入的工作副本。其 `assets_manifest.json` SHA-256 为 `6ce5d859e060f916c5e865efb524fc7efdc57b2b7139fe4dcb22b0f34d700536`，实际执行资产脚本 SHA-256 为 `573b9750fbcf4c2cefa602a395a7788d87e63218ab71dfee21c90125590734bd`。通过该固定 manifest 摘要及 `assets.py verify --root ... --manifest-sha256 ...` 核对共同输入实际字节。源摘要、输入路径、构建参数及逐文件映射在同一 manifest 内。

[LETTER 固定 Instruments 原件](https://github.com/HonghuiBao2000/LETTER/tree/8d0154e28de37dbb6e24871c508ad8ddb1921cda/data/Instruments) · [LC-Rec 作者 Instruments 归档](https://drive.google.com/file/d/19tsIxV_LbErh5FfYhT732M82y0V-3bxR/view) · [GRID 既有上游原件和脚本](https://github.com/woobowen/recommend/tree/bc6176dc144b827b1b29f73ad3b216e28b39f48f/Baseline%20Results/Instruments%20GRID%20Qwen) · [原始向量 Release](https://github.com/woobowen/recommend/releases/tag/instruments-grid-baseline-20261125) · [Shuffle V2 固定协议与调用入口](https://github.com/woobowen/recommend/tree/af360a158280165f07249b80d2cfa0e871902498/lc_rec/shuffle_v2)
