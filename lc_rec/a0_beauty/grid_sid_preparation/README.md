# GRID SID → LETTER Beauty：部分核查，未就绪

状态 **PARTIAL**。本次没有选择正式 SID，没有启动正式实验；也没有完成下游适配、候选全量配对、GPU 预跑或非作者审核。本材料是可复查的中间证据，不是启动入口。

## 本轮已核实

- 从 GitHub 分页 API 和 `git ls-remote --heads` 得到一致的 9 个分支及 HEAD；`branch_inventory.json` 是固定 SHA 的目录树路径扫描，不是全历史提交搜索，也不是对所有内容的穷尽语义审核。共用 Git blob 不计作不同候选。
- 共同 Qwen SID 入口 `GRID baseline/experiments/beauty_rkmeans_qwen_20261010_012253/semantic_ids.pt` 在 SID-Order-Invariance、main、sid-extensions、sid-semantic-hierarchy、wbw-grid、wbw-lc-rec、wbw-sid-order 的固定树中具有同一 Git blob `9216a937011d552ff26f7c062e6cd1e81ed8796c`，其 LFS OID 为 G3 下列 SHA。wmh-grid 搜索到 RQ-VAE、RKMeans 配置及多种运行日志，未从本次路径扫描中找到此类 SID 载荷；yjc-grid 含 G4。细项及完整 SHA 见 `branch_inventory.json`，wmh-grid 和其余路径仍需按内容定向补查。
- 旧 Beauty A0 正式训练/生成 RUN 为 `6e1b211d9c37f358c43b1540b40b17f6703150d2`；配对 S4 补算及记录身份和外部审核 PENDING 见 [原审核](../evidence/paired-s4-supplement-20261117/FINAL_AUDIT.md)。旧资产不是本轮预跑。
- `LETTER-LC-Rec/data.py` 的 BaseDataset 从 index JSON 取 item→token 列表，逐层收集允许 token；`formal_train.py` 注册新增 token、resize embedding 并将 embedding/lm_head 与 LoRA 一同处理。`production_identity.py` 等处绑定旧数据、词表和 checkpoint 身份。适配需要逐项替换这些绑定并跨训练、保存、加载、推理、指标和回执验证；未实施此改动。

## 候选与边界

| 编号 | 来源及入口 | 本轮直接检查 | 未决事项 |
|---|---|---|---|
| G1 | 历史 T5/RKMeans Beauty A；服务器 `.../assets/188af0341d8c232de73acd9aa7fc3390db8d430c8c71a94ec9816962d3d94f68/merged_predictions_tensor.pt`；仓库 `experiments/formal_20261015/specs/beauty-a-3.json` 引用 | SHA `188af0341d8c232de73acd9aa7fc3390db8d430c8c71a94ec9816962d3d94f68`，388534 bytes；4×12101、唯一 12101，四位范围 0–255/0–255/0–255/0–16 | 上游历史身份部分可核，严格对 LETTER item 仍未证明；未做新组合 CPU/GPU 验证 |
| G2 | 历史 T5/RQ-VAE+STE Beauty MLP；`experiments/beauty_extensions/specs/mlp-formal.json` 引用 | SHA `2a9513a5d7c2a45ee8ccbd923647699350bffe41c804379d6c48e3ddd27645ba`，388534 bytes；4×12101、唯一 12101，四位范围 0–255/0–255/0–255/0–9 | 来源细节仍需核对，严格配对未证明；未做新组合 CPU/GPU 验证 |
| G3 | 本项目 Qwen/RKMeans，固定 GitHub LFS 入口见上；真实载荷 `/data/bwwu/sid-extensions-assets/563dc50b1c7e/experiments/beauty_rkmeans_qwen_20261010_012253/semantic_ids.pt` | SHA `d41adcfb22a9a47d5e0424c068f4c74f33cc9fdbaf5db08540fbfa0e3d720e18`，388405 bytes；4×12101、唯一 12101，四位范围 0–255/0–255/0–255/0–17 | 完整物品语义配对未证明；未做新组合 CPU/GPU 验证 |
| G4 | `yjc-grid` 固定提交 `c92e1576210b3daf2ce6f63f77b65f651f6f7127` 的 `GRID/artifacts/beauty/semantic_ids.pt`，Git 普通 blob，非 LFS 指针 | 实际读取 485366 bytes、SHA `5fb9642b4837683e3b52bf5c411f76ecaec26d67b5e86402d15528e047178713`；ZIP 中 PyTorch tensor 声明 5×12101、stride 1×5；标准库直接读取 int64 存储：12101 个不同五元组，范围 0–255/0–255/0–255/0–255/0–6 | 生成来源、item 轴及与其他候选关系未核完；五位不能直接沿用本轮四位 Shuffle，不自行截断 |

G1/G2/G3 服务器载荷两两 12101 列均有至少一位不同。张量第 1 维作为物品轴是本轮结构检查假设；不得把 SHA 或形状相同当作 item 配对证明。前三者四元组未发生碰撞；不代表前三位无碰撞或第四位来源已完全核验。G4 不能因为五位就直接认定其内容与其他候选完全独立。

`assets.json` 来自已保存的 `inspect_assets.py` 在服务器现有解释器的真实执行；`item_text_r1.json` 来自 `inspect_item_text.py`。LETTER 的 index 和 item 各 12101 项、inter 22363 用户；GRID TFRecord 数字 id 与 LETTER index 键均覆盖 12101 项。按相同数字 id，12094 条 GRID 文本包含 LETTER title；7 条 GRID `Title: nan` 对应 LETTER `title_N` 占位。LETTER 中有 27 组重复标题、55 项涉及重复；所以不能据此断定全部真实 item 对应，也不能自动按模糊标题映射。原始 ASIN/编号来源或同源预处理证据仍需追查。`item_text.json` 为原始较简单检查，服务器保留原脚本和输出；r1 是补充检查，非研究方法修改。

## 环境与未完成项

实际服务器 `zwlab-01`、账户 `bwwu`，GPU0 有 VLLM 进程占约 20 GiB，GPU1 查询时空闲；未占用 GPU。脚本及原始输出在 `/home/bwwu/grid-audits/sid-extensions-p0/grid-letter-prep-20261117/`。只读 CPU 脚本均 exit 0；本任务 GPU-device-minutes 为 0。未安装依赖或改系统配置。

真实非作者审核未完成：三个子代理均明确报告没有文件/终端工具，未读取原件，不能签发 PASS。项目已有旧任务中同类数值审核也不能替代本轮非作者审核。全量物品身份、数据转换、身份守卫、训练/推理/Shuffle/评价适配、真实 GPU 及保存重载、正式 dry-run 入口、服务器代码发布映射均未完成；四位候选均不能称为 ready。待用户选择与正式启动授权是独立的后续门槛，并非仅剩的缺口。本报告未选择候选。

本轮检查和补充取证为 round 0；尚无研究实现修正轮次。未更改旧结果、数据、模型或活动任务，未清理原有文件。
