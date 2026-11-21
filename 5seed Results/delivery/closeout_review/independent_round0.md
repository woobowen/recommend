# 本轮真实非作者审读：round 0

审读者：`/root/independent_review`；实际时间（北京时间）：2026-11-22T01:18:45.048802+08:00。未参与受审分支修改，仅写入review证据。

结论：**CHANGES_REQUESTED**。A/B/C实质修正方向正确，存在3项需集中修正的问题；尚未审最终发布和审核ZIP。

## 实际读取

- User JUNIOR_GITHUB_FINAL_CLOSEOUT_PROMPT.txt full text
- AGENTS.md full text and inherited project rules scope
- All A/B/C before-to-overlay diffs and modified complete README/code/reproduction files
- 31 actual frozen baseline payloads matched to baseline_records_verified.json, original bytes and SHA256
- 31 corresponding members directly read from unchanged metadata.zip; current previously published migration copies
- 12 newly included frozen_originals complete bytes
- Original vector evaluator main() and actual semantic_metrics.py group/tree implementation
- Historical Beauty/Toys and Instruments analysis_config.json seeds, SID paths, 999 permutations and 2000 bootstrap
- Current validation hierarchy_command, 2 negative group tests, saved-statistics check and evaluator --help records
- History search README, read-only search script, restore core JSON, original runtime_audit.json excerpt and acquired-original metadata

## 独立执行

直接读取31对冻结原件/metadata成员并核SHA256、LFS非指针及当前迁移副本；19份相同、12份仅根路径替换。12份新增frozen_originals与原字节一致。
独立组装隔离最小输入目录运行修改后的verify_hierarchy_delivery.py：12行标签指标与6个SID分组深度通过，direct_vector_cosines明确not_run。stdout/stderr与31对核对记录见independent_checks_round0.json及相邻文件。
实际GET核查一个迁移LFS原文URL（200）与一个普通Git非LFS原文URL（404）；没有重新下载大资产。

## 发现

- **R0-1 / medium** `semantic-hierarchy/delivery/README.md`：Four-start table still states this round recomputed all node cosines, while new 2026-11-22 section correctly says no vector recomputation. Historical section date is visible but this round is ambiguous. 修复条件：Qualify the table/header with explicit 2026-11-21 prior-round status and 2026-11-22 current execution.
- **R0-2 / low** `semantic-hierarchy/docs/reproduction.md`：New paragraph points to Section 2 for dependency-install command, but the actual pip requirements command is in Section 1. 修复条件：Correct section reference.
- **R0-3 / medium** `GRID baseline/codex/delivery_20261121/migration_provenance.json`：original_download_url uniformly uses media.githubusercontent.com; actual GET for non-LFS empty beauty val1600 launch log returns 404. Migrated LFS source tested returns 200. 修复条件：Choose raw.githubusercontent.com for ordinary Git originals and media domain for LFS payloads, then bounded link check.

## 边界

本轮没有运行模型/编码/完整向量分析/置换/bootstrap；旧网页二审仅是任务给定的继承证据。历史检索的不存在结论没有独立远端重扫，只直接审了已保存查询实现和回执。最终报告、打包覆盖、最终ref和下载回读待后续审查；本报告不能替代它们。
