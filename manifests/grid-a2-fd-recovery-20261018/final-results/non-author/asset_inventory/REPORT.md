# 大资产目录映射核验

`ASSET_DIRECTORY_MAPPING_PASS_WITH_EXPLICIT_ENVIRONMENT_IMAGE_LIMIT`。这是事后交付清单；不把本文件 SHA 冒充过去执行时使用的 manifest。科研实现和结果验收由独立专项报告承担。

本轮聚合四路线的实际 checkpoint/SID、新 A2 两份 quantizer checkpoint、3 层原始量化码、复用 T5 编码输出与六个原 T5-small 成员，共 21 个文件。另收录 1293 个 Sports/Toys TFRecord 成员及各自路径、字节数、SHA、原 archive/member/manifest 行号恢复入口。Toys 两候选共享同一个 SID，均保留，不按指标择优。

| 数据 split | 成员数 | 逻辑字节数 |
|---|---:|---:|
| sports/evaluation | 279 | 671484481 |
| sports/testing | 279 | 769690722 |
| sports/training | 279 | 606958686 |
| toys/evaluation | 152 | 398471423 |
| toys/testing | 152 | 452901782 |
| toys/training | 152 | 376831140 |

只读探针对清单给出的 1325 个明确路径调用 stat，没有递归扫描。1314 个模型/SID/T5/TFRecord 文件均存在、为独立 regular file，大小与原记录相同。模型、T5 权重和数据内容没有重新散列或下载。它们的 SHA 明确继承真实旧加载/恢复回执或本轮 runner 终态 manifest；本轮先前独立 strict-load、SID 和量化审核另有原回执，不由这次 stat 代替。

8 个目录均不声明单一目录 SHA。6 个数据目录按逐成员清单恢复；T5 目录按六个已保存成员恢复。环境目录不递归测量容量，不宣称有完整逐文件镜像。环境版本源包含 256 条历史 distribution 元数据，9 个关键库通过本轮只读 METADATA 核对与修复前记录一致；历史记录中的 python3.1 路径原样保留，不当作已验证路径。

仅对两个明确的小文件做了本轮有界 hash：2290B 的 legacy installed-requirements.txt 与 17506296B 的 Python executable。它们不构成完整可重建环境；环境目录 SHA 和完整字节大小仍为 null。没有安装包、全局配置修改、GPU/model import、重训练、恢复写入或实验提交。

限制继续保留：Toys 上游 producer 不唯一；A2 是方法复现，不是找回历史 Sports 权重；原历史 hash 的继承与当前 payload 重新测量必须区分。各路线有效配置的三个 loader 目录均已直接对应本清单，但 Toys 评价配置中的 train loader 不证明其历史 checkpoint 的上游生产过程。

交付入口：`assets-not-copied.json`，2262018B，SHA-256 `47474f529c9cce39aa4144f2c525ad4f9834cd206b8d7d1f61ce2adb9a9ca746`。逐路径 stat 原件为 `targeted-remote-stat.json`，机器校验为 `validation.json`；环境版本与限制为 `environment-manifest.json`。可放入最终轻量 ZIP 的根目录；大文件留在 new-server 原路径。GPT_UPLOAD_REQUIRED: YES（由主控聚合一次最终 ZIP）。
