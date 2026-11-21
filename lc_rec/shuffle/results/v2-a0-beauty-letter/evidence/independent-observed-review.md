# Beauty Shuffle V2 独立运行证据复核

审阅者 `review_frozen`。运行源码 `d9fbfadd92864e5432c55c998cf23ffbd24a68b3`。本记录补充源码、选模和资产审查，范围为已保存的 CPU replay、原 Normal 评价和实际 GPU 预跑失败。

## 全量 CPU 输入记录和 runner 终态

- `cpu-replay-r2` 的生成进程完成产物、退出码 0。runner 清理了一个在主进程退出时仍存活的受管子进程，终态 `WORKERS_TERMINATED`。在该清理结束后，独立程序读取 22,363 行输入记录和冻结 `Beauty.inter.json`、`Beauty.index.json`，核对全部用户、目标、可见历史和 639,864 次模式内历史置换。独立 SHA-256 oracle 与全部实际排列 ID 一致。S2／S3a／S3b／S4 的自然恒等用户数依次为 640／6／6／0，token 移动数依次为 160,058／319,567／319,567／479,920。预跑清单与原两 rank 的冻结清单顺序一致。
- `cpu-replay-r3` 采用已验收运行器 SHA-256 `1fefc09a15c1cb73878417f9eb50d0fe9e8fa1b2f60cc70b27f7cc4ed9cb0947`，按现场 `query` 确认 `SUCCEEDED`、退出码 0、`NORMAL_EXIT_CHECK`、cleanup verified，任务进程和进程组已清收。检查的六个覆盖及资产文件与 r2 输出逐字节相同。正式待用 r3 `inputs.jsonl` 20,555,834 字节，SHA-256 `2e69ea72e47eb978e3d7375c47aa62cd3d3588fa633bdf7b6d1cf7000adc23b2`，`audit.json` 为 `f5802e83be477126eccde54a1945ffacd900c9f4a9fa30999ac0e2543569a57a`，`single-shard.json` 为 `00f4d70d297611a822191e7f3fa4aba45d4d657a2207e6a67824eba4b73cf2cd`。r2 的独立内容复核适用于这组相同字节。参见独立 `CPU_REPLAY_CONTENT_R2.json` 和 `CPU_REPLAY_BINDING_R3.json`。

## 冻结 Normal 实测与独立复算

- 控制器派生的 `cpu-eval-normal-r0` 实际 `SUCCEEDED` 且退出码 0，`summary.json` SHA-256 `8f9cfbf9e59dbc3e3fd1d193a57375547f9df377a12ad5c0541efffa8be4c73c`，复算的逐用户矩阵、逐用户 CSV 和 16 项汇总均与原件严格一致。
- 独立 `cpu-independent-normal-r0` 实际 `SUCCEEDED`、退出码 0、进程已清收。验证程序 SHA-256 `afe81c376015aa5ab35cbc16c003471723d99cfdf0b441b51eab03c5d539fdde`，服务器回执 SHA-256 `3e20ef5053ab7719a8741ccd55a3c0586f974c3318b3600536f76068c8258659`。以独立候选排序、显式 item 槽 CCE 和逐用户计算，复算冻结双分片的 22,363 人、447,260 条候选。逐用户矩阵和汇总最大差均为 `0.0`。Top20 中有效完整 SID 445,425 次、无效 1,835 次、重复 0 次、碰撞 SID 1,351 次。

## GPU 门槛与余留验证

- `gpu-preflight-r0` 在 S2 开始时失败，退出码 1。原 `formal-run/requests/normal-a0.json` 将 `CUDA_HOME` 指向 `/data/bwwu/lc-rec/offload-20261108/cuda-12.4`，`LD_LIBRARY_PATH` 为 `/data/bwwu/lc-rec/offload-20261108/cuda-12.4/lib64:/data/bwwu/envs/grid-20261009-e61baf3/lib/python3.10/site-packages/torch/lib`。本轮 GPU 请求缺少两项，DeepSpeed 在 Transformers／PEFT 导入链检查 CUDA 时抛出 `MissingCUDAException`。本轮实际 GPU 装载、四模式模型边界、预跑耗时估计和正式启动门槛仍缺。补齐已冻结环境参数属于明确可定位的工程修复，需按本轮修正上限处理后续授权。
- 本轮结果评价证据支持复用原 Normal。四种 Shuffle 正式逐用户输出及五项配对缺失，Beauty 真数据的独立 bootstrap 对照仍待正式输出后运行。冻结模型与抽样实例的研究解释待四条件结果。
