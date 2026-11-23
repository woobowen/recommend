# LETTER-LC-Rec Llama推荐：运行中与停止记录

本目录仅发布已冻结的实际runner、配置与固定时点观察，和已做容量/4用户预跑的历史记录。当前正式测试未完成，不能与LETTER SID+小型TIGER结果混用。用户于2026-11-22停止Beauty三任务旧run（2950/19448），新run改seqrec、有效batch64（micro8×GAS8）、temperature仍1.0、从基座重新训练；Instruments继续seqrec/global64。两者LoRA r8/alpha16、Int8基座/FP16、原模型huggyllama/llama-7b，代码和资产audit在快照中。

observations由采集时src.stat().st_size固定读取，截止时间和范围见原server_capture及各事件时间。旧Beauty progress仍可能写training_running，但user_stop_20261122明确覆盖其状态；本轮没有改旧状态或续训，当前任务进度仅代表采集时点，最终结果需完成后另补。未发布活动checkpoint或推测指标。运行代码与预跑失败/容量配置按原文保留；不重新下载Meta模型，不声称所有已做实验都已完成。
