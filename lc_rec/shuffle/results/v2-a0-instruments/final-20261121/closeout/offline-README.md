# Instruments A0 Shuffle V2 第一组收尾审核包

第一组研究和原结果交付已通过网页二审，状态为 `FIRST_GROUP_COMPLETE_WEB_REVIEW_PASS`。本包的说明、离线导航和清理差异等待网页复核。实验采用冻结 A0 LLaMA-7B、step8240 LoRA、LETTER Instruments SID 和 Shuffle seed=42，四种 Shuffle 各覆盖 24,772 名用户，Normal 复用冻结预测。

## 阅读顺序

1. [现行结果与网页审核状态](repo/lc_rec/shuffle/results/v2-a0-instruments/final-20261121/README.md) · [方案](repo/lc_rec/shuffle_v2/README.md) · [16 项指标](repo/lc_rec/shuffle/results/v2-a0-instruments/final-20261121/metrics.csv) · [五项配对结果](repo/lc_rec/shuffle/results/v2-a0-instruments/final-20261121/paired.csv) · [逐用户与 bootstrap 数组](repo/lc_rec/shuffle/results/v2-a0-instruments/final-20261121/arrays.npz)。
2. 正式 RUN：[生效配置](actual-source/formal/lc_rec/shuffle_v2/a0_instruments.json) · [置换核心](actual-source/formal/lc_rec/shuffle_v2/core.py) · [输入和模型边界](actual-source/formal/lc_rec/shuffle_v2/runtime.py) · [入口构建](actual-source/formal/lc_rec/shuffle_v2/entry.py) · [实际生成](actual-source/formal/lc_rec/shuffle_v2/generate.py) · [评价](actual-source/formal/lc_rec/shuffle_v2/evaluate.py) · [CCE](actual-source/formal/lc_rec/letter/review/final_audit_20261113/external/official_cce.py) · [统计](actual-source/formal/lc_rec/shuffle_v2/statistics.py)。
3. 实际派生与依赖：[服务器派生入口](original-evidence/gpu-formal-shuffle2-r0/test/derived.rank0.py) · [派生 diff](repo/lc_rec/shuffle/results/v2-a0-instruments/final-20261121/generation/shuffle2/test/derived.rank0.diff) · [冻结 LETTER 入口](frozen-normal-source/lc_rec/letter/vendor/LETTER/LETTER-LC-Rec/test_ddp.py) · [串行控制](actual-source/controller/lc_rec/shuffle_v2/series_control.py) · [runner](repo/lc_rec/a1_qwen3_06b/scripts/runner.py) · [transport](repo/lc_rec/evidence/execution_source/transport.py) · [gridctl](repo/lc_rec/evidence/execution_source/gridctl.py) · [部署映射](source-map.json)。
4. 终态与审核：[原运行终态与预算](repo/lc_rec/shuffle/results/v2-a0-instruments/final-20261121/terminal-and-budget.json) · [CPU replay r0 清收](repo/lc_rec/shuffle/results/v2-a0-instruments/final-20261121/jobs/letter-a0-20261112-v2-20261120-cpu-replay-r0/status.json) · [CPU replay r1](data/cpu-replay-audit.json) · [运行终审](repo/lc_rec/shuffle/results/v2-a0-instruments/final-20261121/final-run-review.md) · [统计终审](repo/lc_rec/shuffle/results/v2-a0-instruments/final-20261121/final-stats-review.md) · [七次观察](repo/lc_rec/shuffle/results/v2-a0-instruments/launch-observation-20261120/README.md)。
5. [结构化数据字段和原分片身份](data/review_export_manifest.json) · [保存的输入复核](repo/lc_rec/shuffle/results/v2-a0-instruments/final-20261121/offline-export-verification.json) · [保存的独立统计复核](repo/lc_rec/shuffle/results/v2-a0-instruments/final-20261121/offline-recompute.json)。`data/` 收录五条件全部 2,477,200 条候选、binary64 原精度分数及排序、用户与目标、SID–item 映射、逐行输入和模型边界。它是原始 JSONL 的结构化提取，省去了重复解码 Prompt 和常量字段。原始 JSONL、冻结模型和运行环境的取得条件见正式配置与导出清单。
6. 本轮差异：[授权与清理记录](repo/lc_rec/shuffle/results/v2-a0-instruments/final-20261121/closeout/README.md) · [变更 diff](closeout/changes.patch) · [版本和 GitHub 回读](closeout/github-readback.json) · [保真与导航检查依据](closeout/checks.json) · [成员、来源和别名](MANIFEST.json) · [收尾打包程序](closeout/package_closeout.py)。

## 离线验证

在 ZIP 解压根目录使用本地 Python 与 NumPy，无需 GPU 或联网。保存的两份验证回执见上面的数据项。

```bash
OPENBLAS_NUM_THREADS=1 python3 -B repo/lc_rec/shuffle/results/v2-a0-instruments/final-20261121/verify_offline_statistics.py --export data --statistics repo/lc_rec/shuffle/results/v2-a0-instruments/final-20261121
python3 -B repo/lc_rec/shuffle_v2/export_review_data.py verify data --matrices repo/lc_rec/shuffle/results/v2-a0-instruments/final-20261121/arrays.npz
```

根导航的链接均指向实际成员。历史原文保留原相对链接，可通过 `source-map.json` 的部署源码映射和 `MANIFEST.json` 的 `aliases`、`relocated_originals` 定位历史内容。 [原包根导航](history/reviewed/README.md) · [原包清单](history/reviewed/MANIFEST.json)
