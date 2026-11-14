# A1 Normal final review entry

**A1 NORMAL FINAL AUDIT: PASS.** Original computation and nonauthor scientific audit completed; final publication/readback binds this entry's fixed commit.

GPU RUN `0e3e2c6f5c1a76152ea0228390d94ffc1ff75ba5`; CPU RUN `445f1475cd1dd70d19881dbc1639bb311f7cccf7`. Model Qwen3-0.6B-Base revision `da87bfb608c14b7cf20ba1ce41287e8de496c0cd`. Best checkpoint-8240. No training or full Beam20 was repeated for final audit.

[Actual nonauthor final report](review/final_20261114/FINAL_REVIEW.md), [reviewer-authored independent source](review/final_20261114/independent_points.py), [new full-prediction CPU result](review/final_20261114/independent_points.json). The reviewer used controller-bridged source/evidence and execution; no independent SSH/GPU claim. Source/config, original raw reports, all16 metrics, CCE, SID diagnostics, groups, bootstrap and paired evidence are in the unchanged [completed-results snapshot](evidence/status_checks/20261114T034030Z/README.md).

| Metric | A1 Normal |
|---|---:|
| hit@1 | 0.061359599548 |
| hit@5 | 0.087195220410 |
| hit@10 | 0.106935249475 |
| hit@20 | 0.130913935088 |
| ndcg@5 | 0.074377046646 |
| ndcg@10 | 0.080685962145 |
| ndcg@20 | 0.086790361942 |
| mrr@5 | 0.070180041983 |
| mrr@10 | 0.072740824894 |
| mrr@20 | 0.074437196883 |
| ItemHit@5 | 0.086710802519 |
| ItemHit@10 | 0.106437375532 |
| ItemHit@20 | 0.130671726142 |
| ItemNDCG@5 | 0.073551971414 |
| ItemNDCG@10 | 0.079863420618 |
| ItemNDCG@20 | 0.086027039611 |

Both ranks12386 users,20 candidates/user,24772 unique users/495440 candidates. Final audit checked396352 user-metric values and all seven fixed groups; bootstrap and paired independent checks passed1e-13. Conditional user uncertainty is not training-seed variance.

A1 may serve as the authorized Shuffle control after this fixed entry is published/read back. Shuffle implementation and final results need their own gates. Original Normal predictions/metrics and all history remain unchanged.
