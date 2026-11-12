# 本轮作业原始终态清单

观测 UTC: 2026-11-12T07:14:35.794167+00:00。所有 job 均为服务端 accepted runner 独立执行；数值比较 exit2 的 FAILED 表示恢复门槛真实失败。

| 完整 job_id | 状态/exit | GPU device-min（保守） | 来源 |
|---|---|---:|---|
| `letter-resume-20261112-attention-r2` | SUCCEEDED/0 | 0.947396 | `af00e365eb150e3b9a4ee09fbe93a4a76f9a8a38` |
| `letter-resume-20261112-cpu-compare-default` | FAILED/2 | 0.000000 | `d84088222df1f00dffef02f3c97c43efb200ccdc` |
| `letter-resume-20261112-cpu-compare-historical` | FAILED/2 | 0.000000 | `d84088222df1f00dffef02f3c97c43efb200ccdc` |
| `letter-resume-20261112-cpu-compare-repeat` | FAILED/2 | 0.000000 | `d84088222df1f00dffef02f3c97c43efb200ccdc` |
| `letter-resume-20261112-cpu-compare-torchdet` | FAILED/2 | 0.000000 | `d84088222df1f00dffef02f3c97c43efb200ccdc` |
| `letter-resume-20261112-cpu-compare-trace` | FAILED/2 | 0.000000 | `d84088222df1f00dffef02f3c97c43efb200ccdc` |
| `letter-resume-20261112-cpu-final-receipt` | SUCCEEDED/0 | 0.000000 | `d84088222df1f00dffef02f3c97c43efb200ccdc` |
| `letter-resume-20261112-cpu-loader-replay` | FAILED/1 | 0.000000 | `d84088222df1f00dffef02f3c97c43efb200ccdc` |
| `letter-resume-20261112-cpu-r0` | SUCCEEDED/0 | 0.000000 | `822983bd50d3307cfddad6b50544d01f59e5be52` |
| `letter-resume-20261112-cpu-replay-final` | SUCCEEDED/0 | 0.000000 | `d84088222df1f00dffef02f3c97c43efb200ccdc` |
| `letter-resume-20261112-cpu-suite` | FAILED/1 | 0.000000 | `d84088222df1f00dffef02f3c97c43efb200ccdc` |
| `letter-resume-20261112-default-cont-r2` | SUCCEEDED/0 | 13.491100 | `af00e365eb150e3b9a4ee09fbe93a4a76f9a8a38` |
| `letter-resume-20261112-default-resume-r2` | SUCCEEDED/0 | 10.021288 | `af00e365eb150e3b9a4ee09fbe93a4a76f9a8a38` |
| `letter-resume-20261112-load-r0` | FAILED/1 | 0.008779 | `822983bd50d3307cfddad6b50544d01f59e5be52` |
| `letter-resume-20261112-load-r1` | SUCCEEDED/0 | 9.351471 | `d321dc5976d80c7b31a15814ccb2a1f4b9d56ad0` |
| `letter-resume-20261112-repeat-r2` | SUCCEEDED/0 | 9.310329 | `af00e365eb150e3b9a4ee09fbe93a4a76f9a8a38` |
| `letter-resume-20261112-torchdet-cont-r3` | SUCCEEDED/0 | 13.233943 | `be15ad0a7c8b3cb8f1adac2c4668e7e6973b7164` |
| `letter-resume-20261112-torchdet-resume-r3` | SUCCEEDED/0 | 10.193077 | `be15ad0a7c8b3cb8f1adac2c4668e7e6973b7164` |
| `letter-resume-20261112-trace1-r3` | SUCCEEDED/0 | 13.571538 | `be15ad0a7c8b3cb8f1adac2c4668e7e6973b7164` |
| `letter-resume-20261112-trace2-r3` | SUCCEEDED/0 | 12.828848 | `be15ad0a7c8b3cb8f1adac2c4668e7e6973b7164` |

总计 92.957769059 device-min；活动本任务作业 []。

request/status/cleanup/work.log 在每个同名目录 job/ 中；CPU-replay-final 结果在 job/output/loader_replay.json。
