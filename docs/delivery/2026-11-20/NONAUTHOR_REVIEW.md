# 2026-11-20 独立非作者复核

审核者为本轮并行代理 `/root/review`。我于 15:31–15:42 EST 独立查询原件、运行 CPU 核查并审读四个分支候选及两份学弟文档。发布候选正文由主控和其他作者编写。下列随包核验程序由我编写，主控另行核对后纳入交付。

## 分支范围和候选

- 直接读取 GitHub 所有 10 个 branch refs、206 条活动和 PR 清单。四条本人分支的研究发布在 v1.3 同步以前已发生。2026-11-20T19:38:39Z 的四次规则发布各只改三份规范与 `docs/CURRENT_WORKFLOW.md`。`yjc-grid` 由 `woobowen` 代建，随后由 `Justin-hub-coder` 发布三次研究内容。两条学弟专题分支由后者创建和上传。主分支为共享入口，`sid-extensions` 为冻结 P0，`wmh-grid` 为历史快照。候选写入范围与这些源事件对应。原始活动位于审核包 `review/github-activity-raw.json`。
- 直接审读 `wbw-grid` 的实验导航及源文件。实际核过 MLP best steps 2600/4400/4900、`AuthorShuffleModel`、`AuthorMLPModel`、`BaselineShuffleModel`、`GenerationAttentionObserver`、`records.finalize_paths` 与 A2 `collate_items_contiguous`。原件目录内的 `tiger_generation_model_shuffle.py` 与 `tiger_generation_model_parallel.py` 字节相同。早期 B 报告关于 A 受阻的说法只对应旧时点。固定 A2 正式计算使用 `85b77f5e2a5b497fdc1f620649ba488fc783a245`，独立从服务器读取同一 export 根仍存在。
- 直接比对 `wbw-sid-order` 补证表与固定 GRID 证据索引。三条早期 A 路线共九项执行 commit 对应 `b44399926c98369191c2ff5c2425377c8c07d3c8`，Sports A2 三项为 `85b77f5e2a5b497fdc1f620649ba488fc783a245`。原 21 条中的 12 条 GRID 与新增 12 条共同覆盖八条 GRID 历史路线。重新执行原 `checks/validate.py`，实际验证 21 条、137 处引用、74 对独立 commit/path 和 84 个数值，退出 0。终稿中新增链接需对提交后路径复核。

## 实际运行与原始结果复核

- GRID A 路线。读取原 `GRID_A2_FD_RECOVERY_FINAL_20261018T045438Z.zip` 的 330,060 行原预测及固定发布 CSV。12 份逐用户文件 SHA-256 与表一致，四指标最大差 1.93713e-9。8 组配对的用户名单、目标和全部计数对应，每条件步进保留原候选顺序。独立运行结果见 `review/grid-a-route-nonauthor-recompute.json`，程序为 `review/recompute_grid_a_route.py`。历史代码的 1e-7 指标算术误差阈值适用于该比较。
- Beauty MLP。经现有 Windows SSH 只读取回六个指定原件，全部与正式 `status.json` 和 `training_result.json` 的文件哈希和长度精确一致。从原预测核出 22,363 个唯一且准确的 testing key/target，四指标最大差 8.726e-10，重复候选数一致。固定 evaluation 和 testing 索引均有 22,363 键，每个共用键的 target 都不同。原件见 GRID 原始输出卷，独立结果及程序为 `review/beauty-mlp-nonauthor-verification.json` 和 `review/verify_beauty_mlp_originals.py`。
- GRID baseline。直接从固定 GitHub media 下载 Beauty/Sports/Toys 各一个原运行 metrics CSV，三份 SHA-256 与 LFS OID 相同，三个 test 行的全部五项值与原发布 JSON 精确相同。对三份最佳权重、三份 SID、每域各一个 evaluation TFRecord 和一个 Qwen 分片的 HEAD 返回 HTTP 200、长度与 ETag 均匹配各自指针。证据见 `review/grid-baseline-lfs-nonauthor.json`。后十项是可取得性抽查，未获取全部大文件，也未重测模型或独立核全量配对。
- GRID 汇总报告。对照主控产生的 27 条、706,641 行排名复核收据和 26 组、9,984 行注意力复算收据。我的独立重新算子覆盖上面列出的 12 条 A 路线和 Beauty MLP。注意力本轮独立审核范围为读生产链、报告口径和原件入口，现有历史非作者证据继续保留。
- LC-Rec。复核原 LETTER Beauty 和 G3 Beauty 各 9 份请求，原字节与本轮清单一致，规范 JSON 哈希、job、attempt、原状态与固定旧 controller 账本匹配。逐一解压 21 份预测，原始字节哈希与历史清单相符，共 282,810 行。读取保存矩阵与 5000 次抽样均差，重跑提交候选 CPU 核查，12 条点估计与 8 个配对比较通过。核查见 `review/lc-rec-nonauthor-checked-originals.json`、`review/check_lc_originals.py`、`review/lc-rec-nonauthor-verified-metrics.json`。
- LC-Rec 末次补证复核。读取最终候选 `verify_raw_predictions.py` 并重新执行，逐行对齐 282,810 位次用户和目标，读取并按原生成分数核对 5,656,200 个候选，重算逐用户 16 项，最大误差为零。该核查使用保存预测和固定数据，记录见 `review/lc-candidates-nonauthor-verification.json`。逐个检查 5 份补取源文件／输入／授权的字节，两个原 Beauty controller 的固定 plan 都绑定 runner SHA-256 `cc1a1e277faa9bc713a4d0d1b620248d47c398e8ed17d4ed3d7ab52805cc657e`。
- Shuffle V2。直接读 `core.py` 的活动槽映射和拒绝采样，读 `statistics.py` 的共享 PCG64 配对。原 V1/V2 单元测试七项通过。20:34:09Z 的五份 V2 观测成员哈希与收入文件一致，保留首条完整预测及 snapshot 时的父子状态。20:41:09Z 再独立查询，同一父 job 仍为 RUNNING。该观察不提供四条件全量结果。

## 学弟材料与修正

- 审读三条学弟分支的固定内容及两份交付全文，独立对照 875 份捕获文件的字节、SHA-256、Git blob 与对应树。确认五种子源清单涉及的 14 个 `codex/` 固定入口在该分支当前树中缺失。原 `collect_results.py` 复算两个 40/8 行结果表逐字段一致。复跑 `sid-semantic-hierarchy` 的 `verify_results.py` 成功，覆盖六组结果、十二条分类行与 11,988 条置换记录。已查看学弟自有六分片注意力 ZIP 的字节合并核验日志。该卷为可取得的展示数据，模型重评和服务器原张量仍受各事项所列限制。见 `review/junior-nonauthor-verification.json` 和学弟证据卷。
- 首轮发现两份正文引用了早期 v1.3 SHA。作者修正为候选交付目录中可解析的 v1.4 相对入口。两份最终正文分别覆盖当前逐分支整改和通用操作流程。整改计数、文件位置、所需原件及四个复现起点均已审读。GRID 候选说明已明确保存结果复算和重新生成的范围。候选差异检查通过，LC-Rec 保存的两份原始 Normal CSV 保留 CRLF，以临时 `git -c core.whitespace=cr-at-eol diff --cached --check` 核验。

在上述范围内同意发布候选内容。固定大资产重评、完整训练与上游重建、存在历史缺口的 Parallel、仍运行的 V2 和网页 GPT 二审均保留报告中的边界。完整审核包的组装后检查另行记录。
