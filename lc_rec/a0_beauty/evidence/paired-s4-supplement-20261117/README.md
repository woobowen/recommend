# Beauty A0: supplemental S4 pairing and final audit

This is `SUPPLEMENTAL_CPU_POSTPROCESS`, explicitly authorized by the user's current task. It adds no GPU computation, training seed, permutation, model or dataset. The original pipeline remains `PAIRED_TIMED_OUT_COLLECT_NOT_RUN`. Results here must not be represented as the original controller's collect output.

Original RUN: `6e1b211d9c37f358c43b1540b40b17f6703150d2`.
Original task/series: `beauty-a0-seed42-formal` / `beauty-a0-seed42-original`.
Original Plan canonical digest: `0597958bcf852be1537252507a6c20e153256d3135f990b7362e45e867f95206`.
Original failed job: `letter-a0-beauty-formal-beauty-a0-seed42-formal-paired-a0` (`TIMED_OUT`, exit -15, unchanged).
Auxiliary computation and internal audit source: `1934c01eee767b665ef1043995ffc741958e30e0`.
Supplemental job: `letter-a0-beauty-supp-s4-paired-20261117-a0`.

The fixed original `paired_beauty.py` CLI computes Normal versus S4 exactly once, including `verify_training`, prediction provenance, all-user matching, 5000 PCG64 user bootstrap replicates, and the original independent column-wise implementation. S3 production pairing is not rerun. Its original JSON SHA must remain `578b6e4a0249c9b808ce7942345e753e77874673f8232f0c3e11b2f33f572dbd`.

`prepare.py` binds inputs, preserved original files and immutable source to one request. `worker.py` invokes the original CLI, then audits original CSV/NPZ matrices, all 16 raw-prediction metrics, original CCE implementation, group means, coverage, stored bootstrap intervals, pairing points, hit transitions and list changes within the same bounded request. It does not rerun primary or S3 production bootstrap. The numerical matrices supplied for external review contain no original user IDs or recommendation strings; their row order is bound to the original axis SHA.

The new wall-clock ceiling is 1800 seconds including a 120-second cleanup reserve; Runner computation timeout is 1680 seconds. All attempts consume this single budget. The original ledger is read and preserved, never refreshed via the original controller. Runner request, accepted record, actual status and independent cleanup evidence govern execution conclusions.

Frozen statistical code: https://github.com/woobowen/recommend/blob/6e1b211d9c37f358c43b1540b40b17f6703150d2/lc_rec/a0_beauty/paired_beauty.py

Frozen independent numerical implementation: https://github.com/woobowen/recommend/blob/6e1b211d9c37f358c43b1540b40b17f6703150d2/lc_rec/letter/review/final_audit_20261113/statistics_verify.py

Original failure evidence: https://github.com/woobowen/recommend/blob/29d60cc3a77e90eb583297ecf0ded3cff7f1b6d9/lc_rec/a0_beauty/evidence/formal-launch-20261116/status-check-20261117T015412Z-summary.json

Read `FINAL_AUDIT.md`, `final-receipt.json` and `results/beauty_supplemental_final_summary.json` for actual acceptance, costs, limitations and source navigation. A separate model/human nonauthor review is PENDING because the previously available reviewer lacked file and command tools; no independent reviewer PASS is claimed. The distinct frozen numerical implementation is a numerical crosscheck, not a substitute for that external review.
