# 非作者审核 A · 研究、训练与 checkpoint

**结论 PASS。** 审核对象为执行提交 `390df9f508b7dc11eaa57b79a5a91fb5a7f99385` 和 GPU 作业 `grid-letter-instruments-normal-20261124-preflight-s1`。审核使用 Git 固定对象、服务器实装 Lightning 2.5.0 源码、原始请求、有效配置、运行事件和实际 checkpoint。独立检查仅用 CPU，未训练或生成新的推荐结果。

## 研究边界与运行身份

- 对实际 GPU 请求的 104 个源码路径，逐项从运行提交读取 Git blob 并比对 SHA-256。10 项数据资产 SHA 与旧 `preflight-r2` 请求一致。启动回执的依赖、全量数据覆盖和模型初始身份与上一版相同。复用上一版已核验的完整 Instruments 数据、SID 及划分。
- 对正式有效 YAML，迁移旧 runtime 路径后逐字段比较一致。GPU 预跑有效 YAML 与旧预跑配置在映射 runtime 及 job ID 后逐字段一致。正式维持单卡 FP32、seed 42、`ckpt_path=null`、eval seed 2026、100 次更新对应一次验证的 1,600/16 microbatch 配置、最大 320,000 steps 和 10 epochs，使用完整 `data/testing`。独立核验明细见 `review-a-bindings.json`。
- 训练 iterator 的 `__iter__` 在数据用尽后重新打开 TFRecord，按原配置进行行 shuffle。Lightning epoch 不等同于一次 TFRecord 全量遍历。当前校验覆盖一次验证与 patience 10 夹具，正式训练的实际最优点和自然早停仍需在正式阶段观测。

## 原生 batch 契约与实测

- 服务器实装 `fit_loop.py` 训练用 `CombinedLoader(max_size_cycle)`，`evaluation_loop.py` 验证和测试用 `CombinedLoader(sequential)`，后者直接交付单 loader 样本。原 `TransformerBaseModule.training_step` 在使用 inputs 和 labels 前解包一次。新增 `Recorder` 对训练同样解包一次，对验证和测试读取直接二元组，并核验对象类型、mask 和标签尺寸。
- 直读 CPU `boundary-s1` 真实事件，训练外层 tuple 含内层 tuple，验证和测试直接接收 inputs/labels。两次真实更新、两批验证、严格重载与两遍 8/8/4 测试通过。GPU `preflight-s1` 训练事件记录外层 tuple 含内层 list，内层仍是同样两个模型对象。GPU 验证和测试的二元组直接派发契约也通过。
- GPU 从 step 0 和空 Adam state 开始，共 1,600 个 microbatch，实际扩展大小 55–128，平均 105.853125。首末两次探针分别位于 microbatch 16 和 1,600。独立检查 SID embedding、BOS、SEP、一组 encoder 和 decoder 权重及四个预测头共九个代表参数的有限非零梯度、参数 SHA变化及 Adam 计数。首个探针更新前的九组参数 SHA 与重新构造的 seed 42 模型匹配，末次探针更新后的九组 SHA 与 checkpoint 张量匹配。
- 在 step 100、microbatch 1,600 后执行唯一一次验证，共 24 批。实际 `val/recall@5=0.0364583320915699`，对应 ModelCheckpoint 记录的 best score。预跑期间未调用 Shuffle。正式选模仍以正式 run 的验证指标为准。

## 独立 checkpoint 验证

用现有 GRID Python、Lightning 2.5.0、Torch 2.6.0+cu124 隐藏 GPU，重新实例化原生模型，将实际产物 `checkpoint_epoch=000_step=000100.ckpt` 严格载入。Checkpoint 159,007,470 字节，SHA-256 为 `08aac6a10e0ec8972338923a8d2615ee398bacc11cea117730cf9ee7f07de3ed`。逐项确认 177 个模型状态键和张量相等、完整 state SHA 与两次 GPU 严格加载回执一致、127 个 Adam 状态全部为真实 step 100、moment 有限、完整 loop 和监控回调状态存在。EarlyStopping checkpoint 内容记录最佳监控值、patience 10 和 wait_count 0。运行内另用相同回调逻辑测试 10 次相同分数及改善后 reset。

独立复核脚本为 `verify_checkpoint_a.py`，原始回执为 `independent-checkpoint.json`。审核时读取同一 job 的运行路径，通过服务器 SSH stdin 执行脚本，运行时间 7.212849 秒，单列为 CPU 离线审核。用能导入训练环境依赖的 Python 复验解包文件时，传入 `--runtime <解包实验源码根目录> --output <解包预跑目录> --sid <解包的 semantic_ids.pt>`。这个入口只读取本任务生成的可信 checkpoint，重新实例化并严格加载权重。
