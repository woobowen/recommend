# Toys/Sports Shuffle 资产及原选模审查

审查者 `/root/asset_audit`；只读原件、有效配置、CSV 和 payload SHA。本回执未运行模型或 GPU，不替代正式准入。

| 域/路线 | 判定 | 依据与边界 |
|---|---|---|
| Toys B | 原来源与载荷身份通过；真实严格配对门待核 | val/recall@5 最大值 0.05609932169318199，CSV step6199 对应 selected ckpt6200，训练终点7200 |
| Sports B | 原来源与载荷身份通过；真实严格配对门待核 | val/recall@5 最大值 0.0332883857190609，CSV step9499 对应 selected ckpt9500，训练终点10500 |
| Toys A | 两份完整候选，不能静默选一份 | 10-29/23-42-04 best5300；11-03/02-31-15 best5000。同SID、双独立1024×128表、178 state keys；之间的选择规则未批准 |
| Sports A | 捕获范围未发现配套链 | 既有345保存Hydra上下文无Sports；不宣称历史从未运行 |

B 两域实际服务器 SID、config、checkpoint 的新鲜 SHA 与 Git LFS OID 均相符。正式核验应使用 `candidate_specs.json`，训练验证记录与原来源字节在 `B-original-records/`。

Toys A 的原配置、覆盖、train.log、CSV 与哈希收于 `A-original-records/`；保存配置中的 Qwen/3584 与既有实际512载荷冲突必须继续披露。原历史 Python 与完整生成 producer 仍未知，不能据参数形状恢复历史。

本轮仅完成来源审查：待主控保存 runner 门验证 checkpoint 内 SID 数值、item轴和严格加载后可补充正式准入结论。详细逐条证据见 `asset-source-review.json`。
