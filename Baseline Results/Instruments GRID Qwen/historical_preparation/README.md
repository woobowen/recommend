# Instruments GRID 基线适配与启动

用户授权复用 2026-11-15 的 Qwen Instruments SID，执行自有 TIGER 推荐训练及完整测试。不是 LETTER SID 实验，不新增 shuffle。

参考 `woobowen/recommend` 的 `yjc-grid@6200d5875e264b5e6ec9841cf2cdb3b4a12d71a4`，直接取得 Toys 最终有效配置及模型、collator、dataloader 源文件；具体内容身份见 `reference_manifest.json`。服务器对应模型和 collator 与固定 GitHub 文件哈希一致。服务器是无 Git 导出，实际运行另存完整 source 快照及内容清单，不把历史 commit 冒称新适配已发布版本。

## 实际运行

- 服务器：ECNU-Server，用户 jcy。
- 工作根：`/home/jcy/GRID_wmh_Qwen_TIGER_20261009`。
- 当前 run：`experiments/instruments_grid_qwen_20261125_005542`。
- 监督进程 654321，正式训练初始进程 655019；进程身份需结合 run 路径与启动时间查询。
- GPU 0 用于新实验，GPU 1 未分配；GPU 2/3 的 Beauty Llama 活动作业保持原状。
- 双重取得既有 GRID 与 Llama 的 GPU 0 锁，并确认空闲后执行；后台 stdin 关闭、日志落盘、监督与 SSH 脱离。
- 预跑已完成 2 次 optimizer update、2 个验证 batch、2 个测试 batch，退出码 0。预跑指标不属于最终结果。
- 正式训练从头启动，2026-11-25 00:56:49 +0800 观察到 optimizer step 8 / batch 131、loss 1.101313591003418，有限值。

## 输入和适配

复用原 SID `[4,10611]`，完整 SID 唯一。哈希 `0f12825d1de428e8cba9d9827537be229446f0f076effbf6bd695efa42c83015`。ASIN 映射与原上游生成记录一致，全量检查交互 ASIN 和 item_id 对应。

原对齐交互 231,312 条、27,530 用户，不另做 k-core 或去重。每用户按时间排序，相同时间按原文件行号稳定排序。训练 `S[:-2]`，验证 `S[:-1][-21:]`，测试 `S[-21:]`，沿用最近 20 件历史及目标的评价口径。仅排除 5 名训练序列不足两件商品的用户，验证及测试仍各 27,530 用户。三个 TFRecord 全量回读与预期逐值一致。详见 `runtime/data_audit.json`。

直接继承 Toys 最终有效配置，仅替换源码根目录、数据目录和 SID 路径：seed42、FP32、源 batch16、causal 展开至多128、梯度累积16、序列长度120 token、4+4层/d128、Adam lr0.001/wd0.0001、每1600 batch验证、patience10、max_steps320000。不是固定10个有限数据epoch，原训练流无限循环。使用原解码/评价器、beam10，不把原模型描述为约束解码。

训练完成自动选择验证 Recall@5 最优 checkpoint，测试 Recall@5/10、NDCG@5/10。当前未完成正式测试，尚无最终指标。

## 修正与原件

独立静态审核指出并已修正评价历史截断、配置内容清单生成时机及超时退出码记录。首次运行 `instruments_grid_qwen_20261125_005511` 因环境记录代码按 `tensorflow` 分发名查询版本而失败；实际环境提供 TensorFlow 模块，修改为读取模块版本后新建尝试。没有安装或改变计算依赖，失败原件保留。

当前 run 保存 `launch.json`、`source_manifest.json`、`environment.json`、`effective_changes.json`、`data_audit.json`、每阶段 command/console/exit、Hydra 生效配置和进度。正式完成后才核对最终 checkpoint 与测试结果。当前适配尚未上传 GitHub；本轮完成的是读取基线、适配、预跑及正式启动。
