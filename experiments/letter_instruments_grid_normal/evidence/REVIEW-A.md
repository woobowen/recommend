# 非作者审核 A · 研究、数据、模型和训练

**PARTIAL / BLOCKED**。独立审核者直接从固定发布提交逐个读取 83 个源码文件并核对 manifest，确认两处观察适配。用 PyYAML 递归对照本轮正式有效配置与历史配置，差异限于路径、run ID 和一个未被实例化的占位字段。正式配置没有预跑批数上限，测试目录为 `data/testing`，预跑为 `preflight_testing`。

- 在 `assets-readback` 独立重放全量核验。三个划分各覆盖 24,772 人，9,922 件商品、206,153 条交互、9,897 个完整 SID、23 个碰撞组、48 件碰撞商品和 122 个碰撞测试目标均匹配。
- 直接读 CPU collator 和模型夹具，检查 4 至 20 件测试历史、8/8/4 尾批、训练窗口 30 件含目标、padding 和 mask、层级 offset 及 SEP/BOS。CPU 与 GPU 模型实例字段逐项相同，13,128,064 个 FP32 可训练参数、共享 1024 槽 SID 表和四个预测头。初始化摘要为 `12f42b216ea887553dbedfcf4efd46d4d52fd6149aa91bcd007dbb270ff7ebd3`。
- 独立核对 GPU r2 归档的 13 个成员字节和哈希、训练回调和实装 Lightning 2.5.0 的 `loops/optimization/automatic.py`、`loops/training_epoch_loop.py`。该版本在梯度累积中先执行训练前向和 backward，再调用 `on_train_batch_end`。回调在 detach 后通过有限性检查，随后触发 `AttributeError`。始终为 step 0，未取得梯度快照和参数更新。
- `SequenceDataModule.get_dataloader` 返回含一个 DataLoader 的 tuple。原 `TransformerBaseModule.training_step` 先解包一次再取模型输入。新增 `Recorder.on_train_batch_end` 直接读 `batch[0].mask`，实际收到 list。修复应对齐训练回调的解包，并独立核验测试仍遵循 `ProtocolMixin.eval_step` 的 `inputs, labels` 契约。实际验证事件、选模、严格重载和 Normal 生成仍待补测。
- GPU r2 的原始输出、请求和清收记录可在 ZIP 的 `runner-jobs/preflight-r2/` 审读。失败行在预跑观察分支，正式启动所需验证仍未闭合。非作者只执行只读检查和 CPU 离线复核。
