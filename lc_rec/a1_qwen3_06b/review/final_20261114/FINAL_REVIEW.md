# A1 Normal nonauthor scientific final review

**A1 NORMAL NONAUTHOR SCIENTIFIC REVIEW: PASS.** Reviewer: `a1_non_author_audit`, a real nonauthor agent. The reviewer read controller-bridged frozen source and raw evidence, authored independent formulas, and examined actual execution results. It did not independently SSH, rerun GPU inference, or rehash all base weights. Main controller transcribed the review below; original predictions and metrics were not changed.

GPU RUN `0e3e2c6f5c1a76152ea0228390d94ffc1ff75ba5`; CPU RUN `445f1475cd1dd70d19881dbc1639bb311f7cccf7`. Both training ranks completed8240 updates/four epochs. Validation at2060/4120/6180/8240 covered49544 records each, with losses1.9656445980072021/1.6748563051223755/1.5704634189605713/1.5451041460037231. Minimum validation loss selected checkpoint-8240; actual best restore and adapter export agree.

Model Qwen/Qwen3-0.6B-Base revision `da87bfb608c14b7cf20ba1ce41287e8de496c0cd`,1192135096 bytes, SHA `cd2a512003e2f9f3cd3c32a9c3573f820bb28c940f73c57b1ddaa983d9223eba`. Adapter635225010 bytes, SHA `fe96ea74fc3c7253b0c3687239266b0f073e1a51ef9158e7d97bf30b4527bd03`. Actual strict reload checked394 keys/392 LoRA keys, shapes, finite values and every saved/loaded tensor. Active BF16 embedding/head share the same Parameter/storage, shape[152587,1024]; both saved aliases exact. Frozen tokenizer/reference vocab,918 symbols and EOS/PAD agree. Existing base-weight identity was inherited explicitly; this audit does not claim a fresh1.19GB hash.

Frozen generation source preserves Prompt0/batch1/two ranks,Beam20/20 returns/max10/non-sampling, original positional allowed tokens, complete prefix and Response delimiter checks, real EOS at position4, original stable score order and parser, no filtering/deduplication. Both prediction shards cover12386 users,24772 unique users/495440 candidates total, with frozen user/target axes.

| Rank | Bytes | SHA256 |
|---|---:|---|
|0|184629790|790506cf834bca27f4bb01d35d7d4048d494872f74ff4d9a8bcdac516c475b4d|
|1|184130524|028a74dbab9ba2afaa018994b1518c465d20d58ce05218720634e5c42a4a2a45|

New reviewer-authored [independent_points.py](independent_points.py) was executed by the main-controller bridge in accepted job `letter-shuffle-20261114-cpu-r0`, actual8.192918282 seconds/exit0. Actual source6599 bytes/SHA `67bcdb38f652fe9014f2890d5dd23342dd2b5619e73de0300417c9844a9be528`; full result in [independent_points.json](independent_points.json). It independently checked all396352 per-user metric values,16 overall means and seven groups, plus saved token constraints/parser/score order, with predeclared atol5e-12/rtol1e-10. Original three data SHA and both prediction SHA matched.

HR is any match; NDCG sums all matching discounts; MRR uses first match. Official CCE actual function was read and compared: first target SID, all prior groups expanded, invalid SID one position, repeated prior SID expanded repeatedly, uniform target-group credit and Top-K truncation. Official SHA `173c257a8711a885e8f6d5abcdb0ad2510ad4fab01c1700631b5ec261f50cd2d`. No ZCR or generation-side collision changes.

Training popularity uses only items[:-2],156609 interactions,9913 positive-frequency/9 unseen items,nearest terciles5/11. History groups12556/8652/3564; popularity35/3990/4132/16615. All seven complete16-metric groups independently matched.

Actual existing independent statistics compared80000 draw means,5000 users-with-replacement replicates,PCG64/seed20261113/95%linear percentile. Max draw delta6.661338147750939e-16,CI delta5.134781488891349e-16,point/group delta0. Independent paired check bound792704 CSV/NPZ values on the same user axis; max draw delta3.8163916471489756e-17,CI delta1.3877787807814457e-17,point delta0; all below1e-13. Protected inputs unchanged. These estimate user uncertainty conditional on fixed trained models, not training-seed variance.

Original runner2026-11-13T11:57:04.861057Z→2026-11-14T02:50:46.514479Z,53621.653422s,SUCCEEDED/exit0/cleanup verified/owned-live empty. Real-time controller bridge also found both GPUs empty. Original training/predictions/results were not rewritten, retrained or regenerated.

No necessary issue prevents A1 serving as this task's frozen Normal control. Fixed publication/readback closes delivery; A1 Shuffle still requires its own implementation/OFF/ON/position/cache/GPU and nonauthor gates. A1 PASS does not substitute for those gates.
