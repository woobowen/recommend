# 仅文件准备与静态核验工具

这些工具不导入/实例化研究模型，不反序列化checkpoint/SID/PKL，不执行来源项目或Notebook。原生代码即使有问题也只记录。

|文件|实际用途|
|---|---|
|restore_selected.py|仅选中归档成员；绑定manifest、块和整TAR SHA；已有数据按hash只读使用|
|relay_lfs_chunks.py / relay_lfs_chunks_v2.py|固定baseline的32MiB HTTPS Range，经一份当前完整块及有限失败partial证据的WSL暂存与既有WindowsSCP传输；逐块回执恢复|
|assemble_lfs_chunks.py|服务器独立拼接，逐块与最终LFS OID核验后原子发布普通载荷|
|transport.py / runner.py / gridctl.py|复用旧已审字节与隔离namespace；请求/日志/终态可靠落盘；transport只调整本阶段Windows暂存路径|
|build_asset_manifest.py|从实际恢复/组装结果写资产清单，未完成不标payload就绪|
|build_static_catalog.py|只读取Git/文件，生成源码hash、AST符号、imports与YAML target索引|
|static_verify.py|只读核对SOURCE_MANIFEST、Python AST和文档本地链接|

前六个工具保留本次支持目录/服务器固定目标约束，恢复执行必须使用支持目录的已绑定request/profile/sidecar与原job_id，不能在仓库目录随意启动或替换参数。runner本身是通用进程监督器；本阶段已审request只允许纯文件任务，不能据工具存在推断实验授权。

可独立重复的静态校验形状：`python3 -I -S -B tools/sid-extensions-prep/static_verify.py --repo <仓库副本> --output <独立报告路径>`。其输出只证明文件/AST/链接，不证明模型可运行。旧源程序任何validate/test入口均未执行。

relay v1 的真实HTTPS超时失败被保留；v2使用新不可变job/request，复用已核验远端块，未完成块最多三次网络尝试并保留各attempt partial。全阶段集中修正上限仍是三轮，不以单块重试重置。

`deploy_git_tree.py` 只向本阶段隔离服务器源码export根部署已绑定Git tree；C/D之间只允许元数据更新，受审source/config变化拒绝。`verify_server_delivery.py`对明确清单做fresh稳定性/SHA读取，不反序列化资产。
