# 正式全量生成非作者审核 A

审核者为 Codex 独立代理 /root/review_generation。独立的标准库程序 audit_formal.py 直接在 new-server 读取四个正式预测原件、冻结 Normal 和 CPU 输入证据。CPU 作业 g3-beauty-v2-cpu-final-audit-a-r0 退出码为 0，清收确认为空，worker wall time 为 122.674151 秒。结果和自然恒等逐行回执经新的 SSH 连接按 SHA-256 回读。

- 逐行核查四个正式条件共 89,452 条预测、1,789,040 个原始候选及 639,864 次历史商品组内置换。用冻结 Beauty 商品索引、交互和全量 CPU manifest 独立重建 V2，核验全局顺序、目标、固定槽、组外 token、mask、S3a/S3b 相对置换共享和全量输入摘要。
- 四组运行源码为 86660be5a62c8325caf74cf2810581850ca12803，冻结模型训练提交为 21d495b0be5b606853ec7cec68a3eaa1b58acbd7。根据与冻结 Normal 一致的模型级身份回执和每条保存的实际 runtime witness，核对严格装载、eval 模式及权重未修改证明、20 beam embedding 和 prefill、原生 position IDs、空初始 cache、callback、EOS、稳定排序及候选原始 FP32 分数。四组 GPU 产物与 CPU 对应评价绑定的预测 SHA-256 一致。
- 自然恒等全输入案例为 S2 640、S3a 6、S3b 6、S4 0。652 个条件×用户案例均与同一用户的冻结 Normal 原件精确匹配，涵盖全部候选记录、生成顺序和排序顺序，最大保存分差为 0。涉及 646 名不同用户。
- S2、S3a、S3b、S4 的目录外候选分别为 317、90、210、58，四组重复数均为零。目录外候选保持原位置。
- 用原 runner 查询 10 个正式作业，确认父任务和九个阶段均退出 0、清收为空，子阶段时间顺序符合串行计划。四组正式 GPU 作业引用同一设备 UUID GPU-59cac3bd-3645-668b-79c8-839e0d878893。核对了实际 request 哈希与接受回执。

审核结论为 PASS_NON_AUTHOR_FULL_FORMAL_GENERATION。范围为完整保存的推理原件及实际运行回执。这次 CPU 审核复算输入置换和校验摘要，模型数值核对使用冻结装载回执和实际保存的边界证据。完整评价与配对统计由独立方向 B 核查。
