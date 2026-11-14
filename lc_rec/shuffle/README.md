# LETTER A0/A1 test-time SID Shuffle3/Shuffle4

四组正式全量测试及真实非作者方法/统计终审均已通过。请从[最终审核入口](FINAL_REVIEW_ENTRY.md)开始；[完整结果表](results/RESULTS.md)含全部16项、差值/相对差/95%区间、SID诊断、first-hit、分组、推荐变化及成本。GPU/CPU实际RUN均固定为`f0242b9f90ae23a4979d688a287734ca9f88cea1`，后续交付文件没有改变服务器冻结源码。

用户于2026-11-14明确授权的一次性任务。仅在冻结 Normal 模型测试输入中，完整 Prompt 经原 tokenizer 后，对每个可见历史物品的四个 SID Token ID 做组内置换。Token 身份不变；不改训练、目标生成顺序、Beam20、SID、数据或评价协议。

正式顺序：A0-S3 → A0-S4 → A1-S3 → A1-S4，双 rank 严格串行。各条件24,772用户、20候选。A1须先完成原授权最终非作者审核。正式 GPU 门槛：全用户 CPU 输入校验、32真实用户 OFF/ON GPU 预验收、非作者预审均通过。

随机协议 `GRID-LETTER-SHUFFLE-V1`，seed42；compact JSON UTF-8 `[protocol,42,n,row_idx,visible_item_idx,counter]` 的 SHA256 按 unsigned big-endian 解释，拒绝采样避免模偏，索引字典序排除恒等的5/23种排列。方向 `output[j]=original[perm[j]]`。同用户/历史出现位置两模型共用排列。每条件仅一个固定随机实现。

预验收最多60 GPU-device-minutes，含加载、失败及清收。正式各A0最多4h、各A1最多3h，正式累计启动/运行/清收最多18h。任务 round0；随后最多三轮集中修正。原始失败保留。模型加载和生成只用服务器既有环境，无重新训练授权。

代码复用固定A0/A1真实原推理入口。`entry.py`先验证原字节SHA，做唯一锚点最小派生补丁，执行前保存实际派生源码和diff。`runtime.py`仅在传入原生generate之前处理当前历史Token IDs；A1保留原Allowed对原始Prompt及Response边界的验证，再验证置换并绑定当前前缀。正式输入不显式传入旧坐标或旧cache。原Token可能隐式暴露SID层级，不声称层级匿名化或严格置换不变性。

`cpu_inputs.py`全轴校验、排列manifest和固定分层样本；`test_core.py`是作者定向测试，不能替代非作者审核。统计复用冻结16项/CCE/分组/5000次PCG64用户Bootstrap定义。Bootstrap只刻画固定模型、固定置换实现下用户不确定性。

全用户CPU校验、每模型32真实用户GPU OFF/ON验收和非作者技术/统计预审通过。预跑含round0失败及runner清收累计7.8545052 GPU-device-minutes。三轮任务修正全部使用，四组正式GPU均首试成功。A1 Normal最终科研非作者审核已于正式启动前通过并发布至RUN固定提交，见[A1入口](../a1_qwen3_06b/A1_NORMAL_FINAL_REVIEW_ENTRY.md)。四组GPU阶段合计25222.072秒，四runner实际墙钟合计25638.797秒；所有终态/清收及96个export文件最终字节复核通过。最终发布回读和ZIP身份属于本提交冻结后形成的外部交付回执，随实际交付提供。

[验收回执](evidence/preflight_final_receipt.json)、[统计预审](review/STATISTICS_PRELAUNCH.md)、[修正记录](review/CORRECTIONS.md)。`launch.py prepare`只构造固定request，`launch.py submit`提交服务器自主管理的串行作业；已有job只查原身份，失败门槛停止后续依赖链。每条件先生成、清收，再进行独立CPU输入回放/评价，不重新生成已成功预测来修复CPU问题。
