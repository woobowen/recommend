# 已知问题与后续待决定的改动

本页所有建议均 `PROPOSED_NOT_IMPLEMENTED`。原生代码按字节保留；静态发现不被伪装成运行确认，也不为使检查全绿而修模型。

|事项|实际证据/性质|本阶段处置|后续待确认|
|---|---|---|---|
|Shuffle 整类替换改变参数共享|SOURCE_FACT：原件双SID表，main普通共享表|保留原件，说明差异|是否仅向main加无参数查表后重排|
|shuffle YAML target与参数冲突|CONFIG_DECLARATION / STATIC_INFERENCE：普通target与shuffle专参，父类闭合签名不接|未实例化；登记静态不兼容|选择入口、target、参数传递与hook后单独适配|
|test/predict/eval hook不同|SOURCE_FACT：is_testing由on_test_start/end控制；inference正文predict|没有以手动test/eval“验证”|真实Trainer生命周期与干预触发协议|
|不整除H、部分mask、恒等排列|SOURCE_FACT：返回原输入、整组判断、每次随机argsort|原协议保留|N范围/整组约束/seed/共用排列|
|MLP聚合不是天然不变/并行|SOURCE_FACT / STATIC_INFERENCE：有序HE拼接，末位mask，无SEP，decoder AR|解释三种MLP与参数公式|初始化/冻结/预算及mask协议|
|原BaseModule与普通/MLP测试接口|SOURCE_FACT / STATIC_INFERENCE：原test传batch_idx/save_attentions，原eval签名不接；main调用已不同|不把旧问题套到main，不执行确认报错|原件若未来运行需显式接口适配|
|连续子序列抽样计数疑点|STATIC_INFERENCE：collate计数表达式与物品边界枚举不同|不静默修正、不评价旧指标|未来核真实采样分布与公平训练口径|
|Parallel真实非AR源码缺失|UNKNOWN：现存parallel和shuffle同SHA且AR|NOT_RECOVERED|RECONSTRUCTION_PENDING_USER_DISCUSSION；禁止填空实现|
|Attention缺最终beam祖先与稳定样本绑定|SOURCE_FACT / HISTORICAL_OBSERVATION：当步首beam与batch名不足|没有新采集/路径回放/修图|观测接口和producer/sample/mask/beam元数据设计|
|原绘图注释与正文相反|SOURCE_FACT：complete拼接实际cross在前、自注意力在后|以正文解释并保留注释|未来是否重写图协议，需另行授权|
|no_pos高H返回协议|STATIC_INFERENCE：chunk两项解包与attention五项返回，beam临时限3|保留接口问题|不据当前代码反推历史H8矩阵producer|
|旧33/528 CRC冲突|HISTORICAL_OBSERVATION：旧冻结审核绑定|不重扫、不修张量、不把495可读当共同链确定|仅将健康矩阵/旧图用作受限参考|
|原件rootutils历史根标记与绝对路径|SOURCE_FACT：原源码/脚本引用历史路径，原嵌套.git未发布|只给PATH_MAPPING，不改YAML/脚本|未来独立适配根发现和运行路径|
|第三方依赖与动态target|STATIC_INFERENCE：模块路径索引、环境metadata可读，动态符号/版本接口未运行|记录版本与静态缺口|真正import身份、依赖兼容需预跑阶段|
|Qwen资产语义绑定|HISTORICAL_OBSERVATION：固定manifest+LFS OID+config/results provenance；旧语义证据属T5|PAYLOAD内容核验与SEMANTIC_BINDING_UNVERIFIED分开|未来授权后核item/SID/checkpoint配对；本阶段不torch.load|
|历史Python/配置同名同内容|UNKNOWN：旧checkpoint结构可不同|不以相同配置自动绑定producer|新增历史材料或明确新版本研究|

没有新增科研结果。CPU合成接口旧验证、三域历史结果、原件静态问题分别保留身份；未来运行验证尚未做不妨碍本阶段文件交付，但必需载荷/发布/非作者审核未闭合时状态必须为 PARTIAL。
