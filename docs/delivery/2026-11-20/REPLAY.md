# 从单一审核 ZIP 复算保存结果

当前交付使用 [GRID-Repository-Closeout-v1.5-20261121.zip](https://github.com/woobowen/recommend/releases/download/grid-repository-closeout-2026-11-21/GRID-Repository-Closeout-v1.5-20261121.zip) 和 [SHA-256](https://github.com/woobowen/recommend/releases/download/grid-repository-closeout-2026-11-21/SHA256SUMS-closeout-v1.5.txt)。2026-11-20 的分卷和首次 v1.5 发布回执只说明当时的取证和交付。现行复算使用下面的单包。实验运行身份见原任务的 request、配置和版本记录。

## 解包、依赖和别名

在下载目录校验同名 ZIP 与校验文件，再在一个新的临时目录解包。命令适用于 Bash、`sha256sum`、`unzip`、Python 3.12 和 NumPy 1.26.4。GRID 排名与历史 JSONL 的字节恢复只使用 Python 标准库。attention、LETTER 和六任务全量分组另需 NumPy。整个流程在 CPU 运行，需为解包和 GRID 临时恢复预留磁盘。评分输入、仓库源码和固定依赖已随 ZIP 收录。使用已有满足要求的解释器。

```bash
sha256sum -c SHA256SUMS-closeout-v1.5.txt
ROOT=$(mktemp -d "${TMPDIR:-/tmp}/grid-closeout.XXXXXX")
BUNDLE="$ROOT/bundle"
OUT="$ROOT/output"
mkdir -p "$BUNDLE" "$OUT"
unzip -q GRID-Repository-Closeout-v1.5-20261121.zip -d "$BUNDLE"
cd "$BUNDLE"
PYTHON=python3
"$PYTHON" -B -c 'import sys, numpy; print(sys.version); print(numpy.__version__)'
"$PYTHON" -B tools/restore_aliases.py "$BUNDLE" repository/
"$PYTHON" -B tools/restore_aliases.py "$BUNDLE" evidence/grid/
```

若直接取得 ZIP 解压目录，从 `cd "$BUNDLE"` 开始，先设定实际绝对路径和一个独立的输出目录。`MANIFEST.json` 中 `files` 是实际成员及 SHA-256，`aliases` 指向同字节来源。按前缀恢复只写解包副本，遇到已有不同字节的文件会停止。运行其他历史命令时可以不传前缀恢复所有别名。

## GRID 逐用户预测和注意力

```bash
"$PYTHON" -B compact/grid/grid_prediction_archive.py replay --workers 2 > "$OUT/grid-ranking.json"
"$PYTHON" -B compact/grid/grid_prediction_archive.py extract \
  --job beauty-mlp-formal-v1 --output "$OUT/beauty-mlp-predictions.jsonl"
"$PYTHON" -B evidence/grid/recheck_attention.py \
  --repo "$BUNDLE/repository/wbw-grid" --raw "$BUNDLE/grid/attention" \
  > "$OUT/grid-attention.json"
```

`compact/grid/manifest.json` 为 27 个 job 提供原预测 JSONL SHA-256。恢复结果带原次序、所有候选和分数，能与历史 JSONL 原字节匹配。排名报告核对 706,641 行、真实目标、候选与 Recall/NDCG。注意力程序使用独立的原始 record/NPZ 和运行时汇总口径，覆盖 26 组 fixed32 样本。原工作 baseline 的三域 LFS CSV 在 `grid/baseline-lfs/`。最早 baseline 的逐用户预测仍是历史原件缺口。

## LETTER／G3 和原六任务

```bash
LC="$BUNDLE/repository/wbw-lc-rec"
"$PYTHON" -B compact/letter/audit_compact_predictions.py --repo "$LC" \
  > "$OUT/letter-g3.json"
SIX="$LC/lc_rec/six_task_lora"
"$PYTHON" -B "$SIX/scripts/analyze_grouping.py" \
  --records "$BUNDLE/compact/workload/records-numeric.jsonl.gz" \
  --samples "$SIX/evidence/jobs/lc-six-20261110-cpu-r0/output/workload/samples.json" \
  --output "$OUT/lc-six-task-grouping.json"
```

LETTER／G3 从 12 份结构化记录恢复 21 个原分片的全部已解析字段和值，并验证各分片规范化 JSON 哈希、候选、目标、掩码、指标和 paired 结果。运行脚本直接由 NPZ 和原 index/inter 计算，输出的分片身份、12 个条件和 8 组配对保留。原 gzip 和原 JSONL 文本序列化另有字节身份。需要原容器或原文本时请项目保管人从 `repository-delivery-20261120/drafts/lc/evidence/raw-predictions/`（WSL）交付对应 `.jsonl.gz`，按 `compact/letter/manifest.json` 或 `collection/` 的原冻结 SHA-256 校验后再解压。具备项目 SSH 权限时可按同一清单读取 new-server 记录的原路径。用结构化记录重编码得到的文本属于恢复式产物。

六任务保存的是 221,225 行声明字段的全行投影。完整原 gzip 有额外的逐行随机选择及文本摘要，固定原件是 `lc_rec/six_task_lora/evidence/jobs/lc-six-20261110-cpu-r0/output/workload/records.jsonl.gz`，按 [LC-Rec 原仓库入口](https://github.com/woobowen/recommend/blob/wbw-lc-rec/docs/research/LC-REC.md) 及包内 `compact/workload/verified.json` 的记录取得和核对。分组输出为保存工作负载的长度、mask、资格、padding 和权重统计。Python 的浮点求和末位差见[报告更正附注](REPORT.md#六任务-ce-权重归约更正)。本流程用于保存结果复核；固定 checkpoint 重评、完整流程重跑和上游重建各自依赖实际权重、环境、输入和缺件条件。
