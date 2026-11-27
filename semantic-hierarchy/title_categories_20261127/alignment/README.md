# Instruments 共同商品轴：Qwen 筛选到 LETTER 的9,922商品

用户本轮要求：不重训LETTER上游，在现有Instruments-Qwen的10,611商品中剔除多余商品，与LETTER公开9,922商品一一对齐。只生成独立派生副本，不覆盖旧训练/评估输入或旧指标，不启动编码、训练、模型重评，不上传GitHub。此旁支没有调用子代理，以下为脚本全量核验，不冒称独立非作者审核。

## 完成状态

| 商品轴 | 商品数 | 非空categories | 编号 |
|---|---:|---:|---|
| 原Qwen（保留原件） | 10,611 | 10,330 | 原Qwen编号 |
| Qwen对齐副本 | 9,922 | 9,661 | LETTER编号0–9921 |
| LETTER原件 | 9,922 | 9,661 | LETTER编号0–9921 |

剔除689件，明确清单见[removed_qwen_items.jsonl](aligned/removed_qwen_items.jsonl)。Qwen SID只按已核ASIN筛选、重排，码值差异为0；没有重新学习码本或重新量化，不能称“Qwen在9,922商品上重新训练的SID”。LETTER SID保持原件字节。两套商品行均一一对应，不要求不同商品SID互异：Qwen完整SID唯一9,922，LETTER完整SID唯一9,897，官方碰撞保持原状。

服务器派生目录：`/home/jcy/GRID_wmh_Qwen_TIGER_20261009/codex/instruments_letter_alignment_20261127/aligned/`。

## 差异原因

原Amazon2018 Instruments_5文件共27,530用户、10,620商品、231,392条评论。原Qwen语义实验仅匹配元数据，得到10,611商品、231,312条交互；没有重新去重和迭代5-core。文件名含5并不保证在去重/元数据关联后仍满足两侧5-core。

按作者规则保留最早用户—商品交互，得到219,156条，关联元数据后219,091条；迭代双侧5-core后剩24,772用户、9,922商品、206,153条。商品差异来自这些筛选，而不是不同模型自动漏生成SID。

## 身份依据与全量核验

1. 从LC-Rec固定README链接的Google Drive公开数据包取得原`Instruments.item2id`、`Instruments.user2id`。[下载脚本](fetch_author_mapping.py)、[来源与ZIP校验](author_download_receipt.json)、[原映射](author_mapping/Instruments.item2id)。公开数据包ID为`19tsIxV_LbErh5FfYhT732M82y0V-3bxR`。
2. 包内item.json/inter.json与LETTER固定commit `8d0154e28de37dbb6e24871c508ad8ddb1921cda`商品和全部用户序列逐值相同，并且字节SHA256相同。因此能把原item2id绑定到LETTER原数字ID，不是仅凭商品数或标题相似判断。
3. 全部9,922个原映射ASIN的清洗后title/description/brand/categories四字段，与LETTER商品逐条完全一致；映射ASIN集合等于原评论经元数据关联和迭代5-core后的集合。
4. 原user2id/item2id转换后的全部206,153个用户–ASIN交互对，与清洗后的原评论图精确相同。全部用户序列按原时间非递减；同日时间戳平局下原始行顺序与作者顺序可能不同，直接保留LETTER原序列，不重新排序它。未把仅5,082用户的原行平局顺序相同冒称全用户逐位重建。
5. 作者编号 → ASIN → Qwen原编号，检查9,922个不同ASIN、不同旧编号和不同新编号一一对应。原Qwen item_mapping的固定hash为`262c3bc796ece5826f2e171f585c1664750e9d5b3ef336b5d13899f9715246e8`。
6. 服务器从原Qwen tensor `[4,10611]`按旧编号取列，全量与筛选后的JSON SID逐值比较一致。原tensor hash为`0f12825d1de428e8cba9d9827537be229446f0f076effbf6bd695efa42c83015`。两套输出均为`[4,9922]`，CPU生成，不占GPU。

探索期只靠四字段和交互邻域时有16件完全同信息商品不能唯一解歧，脚本当时拒绝输出猜测映射，见[历史诊断](unresolved_diagnostic.json)。最终映射完全改用作者原始item2id，该诊断不代表最终仍有未对齐商品。

## 产物入口

- [item_mapping.jsonl](aligned/item_mapping.jsonl)：`item_id=letter_item_id`，并含ASIN、`qwen_original_item_id`和身份依据。
- [qwen_semantic_ids.pt](aligned/qwen_semantic_ids.pt)、[letter_semantic_ids.pt](aligned/letter_semantic_ids.pt)：同一LETTER商品轴，四行依次为SID四个码位。
- [Qwen重排SID JSONL](aligned/qwen_item_sids.jsonl)、[Qwen重排index](aligned/Instruments.qwen.index.json)；LETTER原index仍用`<a_...>...<d_...>`字符串格式，不能误当同一种文件序列化。
- [共用title/categories](aligned/items_title_categories.jsonl)：两树后续语义评价读同一份商品文本，每行含ASIN。[LETTER原文本](aligned/Instruments.item.json)、[原完整交互](aligned/Instruments.inter.json)均原样复制。
- `qwen_original_item_ids.npy`是新行到旧Qwen行/列的索引；原LLM2Vec向量没有在本轮重新编码或替换。若后续更换为共用文本，不能静默沿用不相同文本生成的旧向量。
- 服务器`qwen_original_metadata_aligned.jsonl`额外保存重排后的原Qwen完整元数据，约265MB，供追溯，不与共用LETTER评价文本混淆。
- [本地完整身份核验](aligned/alignment_verification.json)、[服务器全量核验](server_verification.json)。各文件来源、长度、SHA256均记录。

## 后续使用边界

用于两SID的语义层级对照时，以本目录的共同9,922商品和同一文本/有效掩码重新统计节点人数、覆盖率和各指标。旧10,611商品的均值与覆盖不能直接当对齐后的结果。

本轮未重训推荐器，不能把已有Qwen的27,530用户测试指标和LETTER的24,772用户测试指标称作同测试集结果。若以后比较推荐指标，需要另行按本目录的LETTER序列训练/测试，原结果保留历史范围。

核心生成脚本[align_instruments.py](align_instruments.py)在本工作区运行，仅读既有原件，默认输出新`aligned`目录且拒绝覆盖；[materialize_server.py](materialize_server.py)在服务器同名独立目录校验上传产物并生成CPU tensor，拒绝覆盖已有tensor。直接调用这两个入口时需按真实工作区调整来源路径，不直接覆盖原实验目录。
