# 路径映射：记录，不自动适配

|逻辑来源|保留原路径|本阶段真实目标|
|---|---|---|
|工作baseline项目|`/home/jcyin/GRID_wmh_Qwen_TIGER_20261009`|`/home/bwwu/recommend-worktrees/sid-extensions-p0/GRID baseline` 的固定Git源码树|
|Qwen baseline数据/SID/最佳checkpoint|上述项目下 `original_assets/data/amazon_data/`、`experiments/`|`/data/bwwu/sid-extensions-assets/563dc50b1c7e/` 下保持相应相对结构；详见ASSET_MANIFEST每行|
|学长原件|`A800:/data_a/mhwang/Rec/GRID`（仅历史标签）|长期档案 `/data/bwwu/archives/wmh-a800-audit-20261010`；本轮选中实体 `/home/bwwu/grid-audits/sid-extensions-p0/restored/GRID`|
|原生网页阅读副本|同捕获各成员|`references/wmh-original/GRID/`；服务器交付树同相对路径|
|本阶段小记录/作业|无历史实验路径替换|`/home/bwwu/grid-audits/sid-extensions-p0/{requests,inputs,jobs}`；WSL支持根 `/home/addaswsw/recommend-support/sid-extensions-p0`|

`SOURCE_FACT`：原 YAML、command、job、shell 和 Python 的历史绝对路径保持原字节；不能直接在新目录执行。Git树中的 LFS 指针保留固定blob，真实载荷放在外部资产根；它们的关系逐项写入 ASSET_MANIFEST/DEPLOYMENT_MAP。没有暗中通过可写硬链接连接档案与工作树。

`HISTORICAL_OBSERVATION`：Beauty的549个数据文件经本次实际SHA核对后，以symlink只读使用现有 `/data/bwwu/amazon_data/beauty/` 实体；目标原权限为0664，没有修改原目标权限，也没有声称操作系统强制只读。Toys/Sports从已验证归档提取为独立inode、0444文件。新的SID/checkpoint/mapping载荷在固定32MiB分块后由服务器组装、全SHA核验，工作实体独立保存。

`PROPOSED_NOT_IMPLEMENTED`：未来运行时需在明确的新适配配置或入口中把旧root/data/SID/checkpoint/output替换到上述路径，重新核实际import与模型/loader协议。本页映射不表示适配已经验证。本阶段没有改baseline、原件YAML或launcher。
