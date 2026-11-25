# 复现与验证边界

在本目录执行命令，使用独立输出目录。历史原件只读；下载资产需要网络，模型重评/重训还需兼容CUDA环境。原Python3.10.20、torch2.6.0+cu124、Lightning2.5.0、Transformers4.47.0、Hydra1.3.2、TensorFlow2.18.0，见run/environment.json。依赖freeze为较早参考，不保证任意新环境可直接运行。

## 1. 保存结果复算（本轮已执行）

```bash
python verify_saved_results.py
```

核正式result与stage_result、CSV、完整源码manifest、最佳验证step，重建metrics.csv/md；四个数均从原全精度值生成。不从逐用户候选重算，因为本次原运行未保存这些候选；不把此检查称模型重评。

## 2. 固定checkpoint重评（材料已交付，本轮未执行）

```bash
python fetch_assets.py --manifest assets.json --asset instruments-grid-original-assets.zip --output /new/retrieved
python assemble_runtime.py --assets /new/retrieved/instruments-grid-original-assets --output /new/instruments-runtime
cd /new/instruments-runtime
CUDA_VISIBLE_DEVICES=0 python evaluate_checkpoint.py ckpt_path=/new/instruments-runtime/best.ckpt hydra.run.dir=/new/instruments-eval
```

assemble_runtime先核所用checkpoint、SID、TFRecord及全部运行源码内容身份，复制到新目录；新增路径适配配置和test-only入口，原件不变，diff在assembly_receipt.json。评价入口沿原pipeline_launcher、模型、生成和指标，只调用trainer.test，不fit。历史grid_stage.py始终先fit，不可拿它冒充只测试入口。源工程的checkpoint含自定义配置类，加载前须将冻结源加入sys.path并按原入口先导入src.utils.custom_hydra_resolvers和launcher_utils；直接torch.load而缺此导入会报src缺失或循环导入，本轮CPU核验已按原导入顺序成功。

本轮验证了资产hash/CRC、checkpoint CPU加载及SID唯一性，并实际完成新runtime的文件装配/配置映射。新增test-only入口静态语法已核，未实际GPU运行，不承诺此命令已通过模型重评。

## 3. 从头重训（材料具备，本轮不重跑）

同上装配后：

```bash
CUDA_VISIBLE_DEVICES=0 python _运行/grid_stage.py --config-name=instruments_reproduction ckpt_path=null hydra.run.dir=/new/instruments-train
```

不使用best.ckpt初始化。输出选模/停止应按原验证Recall@5与patience10执行。参考本次GPU0从约00:55至10:40的全流程，约9小时45分钟；含准备和预跑，不作为其他机器保证。原始输入重建脚本run/runner/prepare_data.py已附，绝对路径需改为资产目录，不能覆盖原目录。直接复用已交付三份TFRecord是本次原条件的最稳妥起点。

## 4. 重建上游（复用原件起点明确；完整重新下载/编码未验证）

上游原控制脚本在upstream_control，冻结源、量化和SID阶段命令/配置在upstream_metadata。原ASIN映射、items_qwen、Qwen向量、量化器和SID都在资产ZIP，可从固定向量/量化器继续核对上游，无需再下载大型编码模型。

若从原始Amazon数据重建，来源见upstream_metadata/download_manifest.json和upstream_control/download_prepare.py；若重新编码，需要取得Qwen2.5-7B-Instruct权重并满足许可/空间。当前归档保留Qwen模型config哈希和编码记录，但没有可证明全部原始权重分片来自唯一HF revision的完整清单；因此不能声称已具备逐字重建历史编码权重的证据。本轮不重新下载原始Amazon大文件或Qwen权重，不重新编码、量化或训练。
