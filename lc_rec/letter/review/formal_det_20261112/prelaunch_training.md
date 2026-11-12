# 训练 / FA / checkpoint 非作者正式预审

PRELAUNCH_REVIEW: PASS

审核者：真实子代理 `/root/preaudit_training`。审核对象：GPU实测提交 `a1b1618e04c8be9bb62918c90b59418c8de34f52`、CPU回执提交 `d9d0a5337e23ce0ceb8666ebdbd7ea209403f9fc` 及核心文件逐字节映射。

审核范围：审核者本轮没有可调用工具；主控桥接实际源码原文、依赖源码、双rank GPU事件和CPU原始回执。审核者独立质询并形成结论，未独立SSH或重跑GPU，不能将本审核描述为独立服务器执行。

最终结论原文：

> 训练／FA／checkpoint分项最终正式预审：PASS。可按既有授权开启唯一正式A0，无需再次GPU预跑或请求用户启动许可。

放行检查：正式FA deterministic覆盖真实native及extension backward，两rank调用计数一致；有效参数为4 Epoch、Global64、LoRA、Int8/BF16、ZeRO-2及原Loss，实际推导8240 updates，没有短max_steps。完整checkpoint保存、严格DS hot加载、scheduler恢复、450个adapter tensor核对及fresh BF16 Beam20接口通过。ReplayLoader直接提取正式源码AST，CPU两rank中途与1/2/3个完整Epoch边界数据、模板及相关RNG一致。实际保留数2、完整标记和必需成员验证满足已审存储适配。

Replay回执SHA-256：`bdb4e8da009d7312aee3d4b283907d78e24c25ef8cfef4fcc34e5da614c4bbab`。新增预验收累计22.0989842/60 GPU-device-minutes。

固定最终RUN后只需确认训练、FA、bnb helper及配置与验收字节一致，request关闭preflight/diagnosis/probe-stop；不要求新增GPU测试。hot Int8加载后不再执行forward的路径边界明确：正式best加载后导出adapter，再fresh BF16加载测试；没有冒称新增hot Int8 forward实测。

本PASS仅代表允许正式运行。真实4 Epoch、完整validation、较早best权重实际恢复、全量24,772用户测试及最终指标仍待结果审核；CPU跨Epoch回归不冒称GPU完整Epoch恢复实测。
