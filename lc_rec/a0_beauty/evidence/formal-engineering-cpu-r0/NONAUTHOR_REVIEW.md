# Non-author review receipts — bounded, not overall PASS

Two nonauthor collaboration agents had direct access to the local workspace,
CPU commands and the server over the existing Windows SSH connection. Neither
edited implementation files or ran GPU workloads.

`/root/review_capability` directly inspected the original and derived runner,
server runner and sidecar hashes, and the evaluation changes. It executed the
identity test, the existing small metric tests, and the isolated runner test.
After its findings, the author added per-user file identity checks, rejected
optimized Python, and updated the state simulation to reject unknown states;
the reviewer independently reran the updated simulation and runner probe.
The reviewer states that the 2-second sleeping-process test is not real
`torchrun`, 96-hour operation, lost-connection recovery, checkpoint recovery,
a cumulative budget, or an actual authorization gate.

`/root/stats_review` directly inspected the frozen implementations and actual
CPU outputs. The independent program `independent_statistics.py` derives the
full-axis ten HR/NDCG/MRR expectations and separately recomputes CCE for the
first 50 users in each rank and condition. Its actual output and precise
limitations are in `independent_statistics.json`. It did not run the formal
paired entry, independently recompute CCE for all users or independently rerun
bootstrap. It also identified the missing independent original LETTER stage-A,
missing 15-case full-chain failure injection, and the fact that fixture size
checks occur between phases rather than continuously.

These are real bounded nonauthor reviews. The missing formal controller,
full provenance and full evaluation chain have not received a nonauthor PASS;
`NONAUTHOR_REVIEW: PENDING` for the task as a whole.
