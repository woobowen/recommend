# H1–H4：原向量、逐商品标签与历史运行补件

维护人：Jingchen Yin（Justin-hub-coder）。本次为2026-11-21事后原件恢复/发布；保存初次发布版本e8f4603，历史运行原件不覆盖。未重编码、分类、训练或改变统计协议。

| 条目 | 已补内容与实查 |
|---|---|
| H1 | [六组vector_index](vector_index.json)、[资产清单](assets.json)、同序ID/SID/输入文本、编码完成收据。原float32 N×4096向量全量核归一化和唯一ID，从原向量重算全部合格节点，最大余弦误差约1.4e-14。 |
| H2 | results/classification/beauty与toys补入原final JSONL、taxonomy和manifest。逐标签复算12行主要weighted-purity/pair-agreement与原统计一致，UNKNOWN为405/219；全部现存原始请求与失败重试在metadata.zip。 |
| H3 | [classification_preparation](classification_preparation/)保存真实readiness/schema/版本/数据审计；[classification_control](classification_control/)保存阶段控制器和失败修正记录。完整启动/原始请求/阶段产物在metadata.zip。 |
| H4 | 原模型来源、tokenizer和文件SHA256收据，编码器MNTP/supervised固定revision，历史源码/launch和环境一一对应。公开Meta权重受访问/许可限制，不能把仅名称可见说成获准下载。 |

## 原件取得

```bash
python semantic-hierarchy/delivery/fetch_assets.py --manifest semantic-hierarchy/delivery/assets.json --asset hierarchy-embeddings-originals.zip --output ./retrieved
python semantic-hierarchy/delivery/fetch_assets.py --manifest semantic-hierarchy/delivery/assets.json --asset metadata.zip --output ./retrieved
```

向量包约870MB，展开约1.14GB，保留float32无有损量化；每组目录含embeddings_normalized.npy、item_ids.npy、norms_before_normalization.npy。向量行必须按同序ID关联SID，不能仅凭文件名或行数。Beauty/Toys为12101/11924，Instruments为10611；这里的Instruments不是LETTER官方9922商品轴。

metadata.zip解包后：`hierarchy/classification/<dataset>/raw/`为system/user/response原文及耗时/截断记录，`classification_history/`为原launch、失败和控制日志；`hierarchy/control/.../inputs`为文本与SID输入；`classification_preparation`为readiness原件。MANIFEST给源路径、范围、字节和hash。它保存全部此次找到的相关原件，不补造原先没有记录的人工审核或更早失败。

## 分类阶段、依赖和复现边界

Beauty固定入口archive/classification_beauty_v1，Toys最终分类入口archive/classification_toys_v7。原阶段为prepare/verify_readiness→discover→draft→revise→pilot→final→evaluate。阶段输入输出的实物与manifest在分类preparation、metadata中的classification及history中。原控制器先复用已经完成Beauty，然后修订Toys taxonomy，不能当作全新机器一条空目录命令；使用docs/reproduction.md中的路径替换，准备model/data和真实readiness，禁止复制旧hash来跳过检查。保留原config的历史状态字符串，不把它当当前状态。

分类使用Llama3.1-8B-Instruct、Title/Brand/Categories/Price；编码评估使用Llama3-8B-Instruct→合并MNTP→加载supervised，双向注意力正文mean pooling，再L2归一化；两种模型职责和底座不同。固定适配器revision见docs/shared-code.md和原模型manifest，模型权重不重传。基座文件hash和许可/镜像来源见preparation，原站Meta访问需复现者自己的授权；这是一项外部访问条件，不声称本轮下载验证了所有授权对象。

## 四种起点

| 起点 | 材料和本轮实际执行 |
|---|---|
| 保存统计复算 | 原节点/边/路径/Gap/置换明细与标准校验器；本轮再次运行 |
| 原向量/逐标签复算 | 原连续向量与ID/SID、逐商品final、冻结taxonomy均可取得；本轮全节点余弦和12行主要分类指标实际复算 |
| 固定模型重新编码/分类 | 原输入、配置、来源权重/适配器身份、真实readiness和阶段代码已定位；本轮不重新推理，Meta访问取决于复现者授权 |
| 重建上游 | Instruments原下载/ID映射及Qwen量化程序可复核；Beauty/Toys沿固定GRID资产，最早转换限制见yjc-grid |

[cpu_verification.json](cpu_verification.json)记录逐组形状、ID、norm误差、全部合格节点比较、UNKNOWN与主要指标；[verify_hierarchy_delivery.py](verify_hierarchy_delivery.py)可从本次原件重验。条件置换与bootstrap未另跑一遍，仍使用原11988条置换记录；不得把节点余弦复算说成重新编码或重跑完整统计。未声称新的非作者审核或网页GPT终审已通过。
