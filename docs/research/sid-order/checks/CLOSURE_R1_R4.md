# 2026-11-17 定向收尾依据

本记录对应对 `27808115d27ba5c0eb7c5eb5d75446eab10817e6` 的外审原件 [external-review-27808115.json](external-review-27808115.json)，不代表修订版已获网页外审。新增原件 SHA-256 为 `49acf329dcaff0be08dfc00187e97c76605633bf404535b43e9bcda49e2516c4`。

- **R1**：[sources.json](../sources.json) 与 [builder](build_sources.py) 将真实 `train` request 只放入训练角色；生成与评价读取 `12eb8b176b156f3be7d7533ee8f443db02441303` 的 `lc_rec/a0_beauty/evidence/paired-s4-supplement-20261117/evidence/original/control/state.json` 的对应 attempts、Plan 与 primary metrics。生成/评价原始 request 字节未在该归档中取得，记录 path/摘要不声称重算原文件。train request 字节 SHA `f930c136e44fce00cad5bf05573779144e43aa108184c1dfc603762be20002b8`；按排序键紧凑 JSON 的摘要 `abd09c34bc141f0f97772a54ebd85d467c48d3f2d00252b0c8e5de5e98ac1e86` 与 state 一致。S3 原配对和 S4 独立补算分开；原 paired 超时、原 collect 未运行。
- **R2**：[LETTER.md](../LETTER.md) 的 Beauty 阶段及训练适配小节绑定固定 `6e1b211d9c37f358c43b1540b40b17f6703150d2` 的 `formal_train.py`、`trainer_loop_patch.py`、`training_rows.py`、`zero2_partial_state.py` 及已有有界验收回执。记录四轮跨 epoch 累积、原 sampler 补齐、训练/验证 collator、恢复、局部 ZeRO-2 兼容和 checkpoint 选模，不把恢复预验收写成正式运行一定恢复。
- **R3**：[beauty_roles.py](beauty_roles.py) 用同一真实函数核验角色、stage/job/attempt、state selector、request 摘要和 argv、RUN/config、primary metrics、预测与资产；[test_beauty_roles.py](test_beauty_roles.py) 对内存 fixture 执行六个负例。[closure_checks_20261117.json](closure_checks_20261117.json) 保存实际正负例输出、21条件/84个表格值、137次固定位置检查与74个不同 commit/path、重生成字节一致性。旧 `validation.json` 与旧回执保持历史身份。
- **R4**：早期 GRID checkpoint 训练源码绑定未唯一恢复；大权重、LFS、服务器预测本次未现场重验。这些是明确历史/现场验证范围，不等于本轮文档错误。本地非作者代理此前无文件工具，本次未重复调用；新文档外审仍待网页GPT。

阅读核对：根 README → 主题 README → LETTER.md「Beauty 阶段来源与训练适配」→ sources.json 中 `letter-beauty-a0-normal`、`-s3`、`-s4`；该路径区分 train request、生成/评价 state、三组原结果、S3 原配对和 S4 补算。检查仅针对文档及固定小型 Git 证据，不执行模型、评价或服务器任务。
