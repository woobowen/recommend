# B round 0：可讲解性非作者审核

审查者 `/root/audit_code`；观察时间 `2026-10-13T17:30:36.713490+00:00`；状态 `CHANGES_REQUIRED`。

已直接读取全部六份候选说明，并与固定 baseline、三域有效配置及本次发布原件核对。123 个新原件与独立保全副本逐字节相同；未运行来源代码、模型或绘图脚本。

|问题|级别|需修正的说明|直接证据|
|---|---|---|---|
|B-R0-01|HIGH|仅改说明中的划分名称，保持原配置与数据不变，并指向有效config的对应位置。|GRID baseline/configs/experiment/tiger_train_flat.yaml [134, 168]|
|B-R0-02|HIGH|删除不存在的mask文件SOURCE_FACT，并把没有稳定样本/SID/mask/rank共同manifest作为当前生产链缺口；未来伴随记录仅标PROPOSED_NOT_IMPLEMENTED。|references/wmh-original/GRID/src/models/modules/semantic_id/tiger_generation_model_shuffle.py [1121, 1212]|
|B-R0-03|MEDIUM|改为精确的649–651或645–651，另将用户条件分支范围包含实际if在665。|references/wmh-original/GRID/src/models/modules/semantic_id/tiger_generation_model_mlp.py [645, 658]|
|B-R0-04|MEDIUM|忠实写为传全部model.parameters；requires_grad/冻结和实际更新范围另列后续核实，不运行证明。|GRID baseline/src/models/modules/base_module.py [168, 189]|
|B-R0-05|MEDIUM|明确这是启发式裁剪、未读取真实mask，不能当padding身份已核对。|references/wmh-original/GRID/visualize_encoder_attention.py [16, 67]|
|B-R0-06|MEDIUM|将full形状写为堆叠成功时，说明失败/文件可缺与原tuple另存；不将squeeze描述成末query选择。|references/wmh-original/GRID/src/models/modules/semantic_id/tiger_generation_model_shuffle.py [488, 519]|

原问题、候选说明哈希、逐项源哈希/行范围、限制见同名 JSON。本角色没有改作者文档或原件；由主控合并到集中修正轮次。

最终复核还需核完整源码证据表、SOURCE_MANIFEST/SOURCE_LINES及历史结果直接入口；目前不授 PASS。
