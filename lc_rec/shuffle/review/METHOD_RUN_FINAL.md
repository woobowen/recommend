# 四组 Shuffle 方法与运行非作者终审

**IMPLEMENTATION：PASS**

**RUN：COMPLETE**

**METHOD／RUN NONAUTHOR FINAL REVIEW：PASS**

**最终发布与ZIP交付审核：PENDING（本报告形成时）**

审核者 `/root/shuffle_non_author_review`，不是被审核实现及运行控制的作者。本文由该真实子代理返回，主控转录落盘。审核者独立分析主控桥接的源码与真实回执，并独立编写输入重放程序；没有独立SSH、磁盘读取或GPU工具。不声称其亲自读取服务器原件或重跑GPU。

正式RUN `f0242b9f90ae23a4979d688a287734ca9f88cea1`；独立输入程序SHA-256 `9bfc1a37d1123943f6061b976ac6553d426fea55446b3e77f2c9d203b96bfc33`。

四组均实际覆盖24,772用户，两rank各12,386用户，每人20候选，共495,440条；无缺失、重复、跳过或失败用户。

| 条件 | GPU stage秒 | 硬上限秒 | 实施／运行 |
|---|---:|---:|---|
| A0-S3 | 7960.193 | 14400 | PASS／COMPLETE |
| A0-S4 | 8172.262 | 14400 | PASS／COMPLETE |
| A1-S3 | 4552.287 | 10800 | PASS／COMPLETE |
| A1-S4 | 4537.330 | 10800 | PASS／COMPLETE |

每组独立输入重放均通过172,929个真实历史SID组、全部原始Normal与实际Shuffle输入、全部495,440候选解码前缀。非恒等比例及用户输入变化率均100%；目标、物品顺序、组外Token、mask、SID身份守恒；S3第四位固定。A0/A1相应条件同用户使用相同确定性排列，依据逐行重放及共同manifest，频数一致只是摘要。S4覆盖全部23种非恒等排列，允许部分位置保持原位。

A1-S4两原始预测SHA：

- `5b4c5de6684f1ff59deeaf7a783a3af7e90582bdab782eed30cfffe03d45f77a`
- `da8053461db000e11562dfdbf94d1522f247c9341b6988e0a9fe8532110b2056`

四组正式GPU均首试成功，无正式生成重跑。每组真实退出0、清收通过后才启动下一组，series亦成功退出并清收。GPU阶段累计25222.072秒（7.006小时），GPU-device-time50444.144秒（14.012 device-hours）；四runner墙钟25638.797秒（7.122小时）。单组及18小时总预算均满足。预验收另计471.270312 device-seconds=7.8545052 device-minutes，CPU统计未冒充GPU计算。

最终collector重核96个export文件与冻结部署一致，runner未改，五个accepted作业真实终态及清收通过。13个模型/adapter等文件全量hash核验通过；最终GPU无compute进程。终态/部署回执SHA `286b1b4fccf8c742bd9ab86831681dd4ba73593c68a20ce194b351565a9b1a1f`，见[真实回执](../evidence/final_control/terminal_and_deployment.json)。

每组396,352个逐用户指标值独立复算差0；所提供Bootstrap、配对、分组误差低于预声明1e-13。完整统计判定以统计非作者报告为准。

四组下降是固定模型、数据与固定置换实现的观察，不能证明真正SID层级语义，不能排除RoPE、因果可见性及训练顺序先验的贡献。全部输入改变与个别用户推荐集合不变兼容。原Token身份保留，本实验不是层级匿名化，不估计Shuffle实现间方差。

R1参数解析/原始生成区块对照补证；R2受支持runner namespace及CPU detached-session清收probe；R3交付collector本地SSH参数过长→读取服务器冻结JSON并绑定canonical SHA→复核通过。三轮额度全部使用，失败记录保留，不重置额度；R3未执行远端命令或改变研究产物。[完整修正记录](CORRECTIONS.md)。

本报告只判方法及真实运行终审PASS，不预判随后最终发布、GitHub回读、ZIP成员/CRC/完整解压/秘密检查、Windows/WSL访问及保留备份。交付门实际通过后，才可判完整最终审计PASS。
