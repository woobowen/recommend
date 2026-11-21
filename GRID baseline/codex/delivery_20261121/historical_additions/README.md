# 本轮有界历史材料检索

执行者：独立子代理 `/root/historical_search`。这是本轮历史材料检索，不是对主控修改的独立验收，也不把检索脚本或两次自查称为非作者审核。时间与实际范围见 `search_receipt.json`。全部服务器操作只读、CPU，无模型加载、预测、编码、分类或训练；没有修改服务器实验目录。

## 本轮新增实际证据

1. 在恢复记录的明确线索下，读取 `/home/jcy/GRID_wmh_Qwen_TIGER_20261009/codex/restore_20261102/` 中 `core_restore.json`、`restoration_complete.json`、`historical_data_link.json` 和完整恢复说明。前两份原记录已经列出12份重配置材料在用户名由 `/home/jcyin/` 变为 `/home/jcy/` 时被迁移。它们是恢复过程证据，不能独自代替本轮两端内容比较。
2. 读取明确保管位置 `/home/jcy/.restore-archives/20261102/` 的备份说明、清单和共享输入记录；只打开其中约189KB的 `02_当前项目私有记录与配置_已剔除训练checkpoint.tar.gz`，枚举成员并取得两份历史 audit 脚本与 `runtime_audit.json` 原字节。未重读1.16GB历史输出归档本体，复用其已恢复目录和成员清单；未打开用户设置、SSH、工具环境归档。
3. 原 `runtime_audit.json` 的 `checked_at` 为 **2026-10-10T11:25:36.244630+08:00**，保存三个基线训练的真实PID、有效配置、step增长、Beauty恢复消息和当时GPU观察。保留其历史时间和原字节。两份 `grid_audit_*.py` 是程序检查，不包含可证明审读者独立身份的署名或对话；**不能称为新增历史非作者审核记录**。
4. 实际用 `torch.load(..., map_location='cpu', weights_only=True)` 读取五种子目录下全部 **40份 `full_test_reference_*.pt`**：每份均为100个用户键，记录含 `ids` 和 `scores`，候选形状为 `[10,4]`、分数为 `[10]`。这是固定样本候选参考，**不是全部22363/19412测试用户预测**，也不含保存的全测试目标。逐文件SHA、完整用户键和结构记录在receipt；包内一份原张量支持直接阅读结构，其他未变化的固定样本分析不在本轮重算。

## 查找范围与结果

`remote_bounded_search.py` 是实际执行脚本，`run_search.py` 保存SSH结果、原字节和SHA。`search_receipt.json` 列出目录、文件计数、文本检索范围、全部命中和取得来源，`originals/` 为只读取得的非敏感原文。不是根据文件名直接认定内容身份。

| 事项 | 实际检查 | 本轮结论及边界 |
|---|---|---|
| Sports/Toys早期ASIN和转换链 | 当前项目入口及复制说明、原Sports/Toys目录、历史辅助脚本；恢复目录 `Tiger_discarded_20261002` 排除环境/模型后的997份文件名与小型文本；两个原备份清单中的项目相关行；明确旧本地上传目录下名称线索 | 未取得 `ASIN↔GRID item_id` 原映射、原Amazon→TFRecord转换代码或该转换源版本/过滤排序参数。已找到恢复/传输身份记录，没有找到足以建立新的全量ASIN对应的独立键，因此未生成猜测映射。 |
| 原数据提供位置 | 当前ECNU检查 `/data_a/mhwang/Rec/GRID/amazon_data`、`/data/jcyin/sid_order_reproduction` 是否存在 | 两者在当前主机均不存在；这是“本机无法取得记录指向对象”，不能断言原机器或其他备份已删除。未扫描其他账户或尝试访问提供者主机。 |
| 全测试逐用户候选/目标/分数 | 三域既存Git/LFS记录沿用；当前完成的五种子输出、分析入口与原恢复目录沿名称/文本线索查询；实际读取40份固定样本参考 | 未取得覆盖完整测试用户且绑定正确checkpoint的历史候选/目标表。不能从聚合Recall/NDCG反推候选，也不能把样本参考冒充全测试逐用户复算。 |
| 历史审核 | 原恢复归档 `runtime_audit.json` 与两份程序审查脚本；当前计划内audit脚本和已有原过程记录 | 程序检查原件可取得并已补；未取得具备真实参与者、独立性、对象和结论的原非作者审读记录。未找到不等于确认从未发生。当前独立审核由主控另行安排。 |

### 仍需的最小原件

- **ASIN链：**原TFRecord数据提供方（历史源路径账户为 `mhwang`，实际保管人未知）提供当时Sports/Toys `ASIN→item_id`/逆映射、Amazon源档版本、过滤/排序/切分及TFRecord生成脚本/配置。映射应能绑定既存分片SHA及Sports18357、Toys11924商品轴。不能只提供同样商品数的另一个项目映射。
- **全测试候选：**实际运行保管方（是否另存未知）提供对应run/checkpoint SHA、分片、用户/目标、10个候选SID与分数、无效/重复处理原记录；若原未存，只能另行授权模型重评形成明确的新记录，本轮不执行。
- **历史非作者审核：**原审读者/保管方（身份未知）提供当时记录和对象版本。当前回溯审核必须用本轮真实时间，不倒签。

Sports/Toys目前仍可从已冻结TFRecord/SID开始复用下游，内部item ID、向量行、SID列及测试分母沿上轮已通过证据。早期外部身份链受限不会自动否定已保存结果；本轮没有重新生成数据或更换商品轴。

## 检索命令与证据层级

正式保存查询：在本地以Python执行 `run_search.py`，其通过既有 `ssh ECNU-Server` 将 `remote_bounded_search.py` 原文送给现有grid解释器，服务器仅输出结构化结果；没有在服务器新建脚本。前期发现使用只读 `find`、小型Python读取具体恢复JSON和目录名线索。旧本地位置为 `C:/Users/Justin Yin/Documents/Codex/2026-10-07/base-ps-c-users-justin-yin/work/recommend-upload`，实际存在；按 `rg --files` 对 ASIN/datamap/item-map/convert/preprocess 文件名检索未返回匹配（退出码1），未将普通TFRecord读取模块当成原始Amazon转换脚本。

备份清单仅是成员/路径证据，不是该路径载荷已取得的证明。原字节实际取得的20份材料由 `acquired_originals` 单列，含每份来源、大小、SHA256和包内路径；两份audit脚本在活动恢复目录和小归档中同字节，打包时可按SHA去重并保留对应关系。

发布路径说明：检索执行时的originals/为本轮取件目录；本仓库按SHA去重存入objects/，具体原路径、取件local别名与published_path见[sources.json](sources.json)。检索脚本保留其真实执行路径身份，不能当作任意位置的通用调度器。完整审核ZIP另含原执行上下文。
