# 当前有效实验：新版 MLP 与 shuffle

2026-10-15 已停止旧版调度和训练进程，删除旧实验组、checkpoint、日志及全部检查用输出。基线向量、码本和 SID 清理前后 SHA-256 一致。旧实验清理记录.json 只保留操作和输入校验记录。

当前唯一有效实验组：

`/data/jcyin/sid_order_reproduction/qwen_rkmeans_shared_sid_attention_v2_20261015`

服务器：jcyin@172.23.148.121。源码根目录：`/home/jcy/GRID_wmh_Qwen_TIGER_20261009`。新增代码位于 codex/sid_order_launch。

## 输入与训练

MLP 和 shuffle 复用各数据集自己的基线 Qwen 文本向量、RQ-KMeans 码本和 SID，推荐阶段直接读取相同 SID。六项推荐训练均从第0步启动，ckpt_path=null，不恢复旧变体、不加载 GitHub 已有推荐 checkpoint。

基线目录为项目 experiments 下的 beauty_rkmeans_qwen_20261010_012253、sports_rkmeans_qwen_20261010_093035、toys_rkmeans_qwen_20261010_093036。

MLP：有序 SID embedding 拼接后经512→256→128聚合，每个物品一个encoder token，无SEP，自回归解码。shuffle：保留基线共享的encoder/decoder embedding表，仅测试时重排物品内部embedding，训练和验证不打乱。

公共参数沿用基线：seed=42，4位SID，sequence_length=120，T5 encoder/decoder各4层，d_model=128，6 heads，d_ff=1024，dropout=0.15，内部mlp_layers=2，Adam lr=0.001、weight_decay=0.0001，无调度器，FP32，梯度累积16，max_steps=320000，max_epochs=10，每1600训练batch验证，按val/recall@5保存最佳checkpoint，early stopping patience=10。训练loader原始batch=16，causal duplicate collate可扩展模型batch至128，验证/测试batch=8，beam top_k=10。

## 注意力和完整测试

正式任务均train=true,test=true，完整testing划分，无检查用batch限制。MLP用本次最佳checkpoint评估一次。shuffle用同一个本次最佳checkpoint依次测试unshuffled、shuffle_3、shuffle_4，每组seed=42。

新版test_step显式将save_attentions_on_test和真实batch_idx传给eval_step，直接eval_step默认读取配置。shuffle正式配置save_attentions_on_test=true，predict使用独立的save_attentions_on_predict=false。验证阶段不保存。

注意力保存在各shuffle实验的attention_weights/test/{unshuffled,shuffle_3,shuffle_4}下，按batch保存encoder、原始cross、完整cross矩阵和decoder self attention。先转CPU再无损gzip压缩为.pt.gz，保留完整张量数值和dtype。缓存多query时取与预测logits对应的最后query。完整cross矩阵形状为[batch,4,6,encoder_length]；沿用原实现每步第一个beam的选择，不代表最终最优候选的完整回溯路径。

读取：

```python
import gzip
import torch
with gzip.open('cross_attention_full_batch_0.pt.gz', 'rb') as stream:
    attention = torch.load(stream, map_location='cpu', weights_only=False)
```

测试结果为test_<mode>.json，包含指标、注意力文件数和shuffle调用数，汇总为stage_result.json。训练未结束时尚无正式测试注意力文件。

## 队列与查询

GPU0：Sports MLP；GPU1：Sports shuffle；GPU2：Beauty MLP然后Toys MLP；GPU3：Beauty shuffle然后Toys shuffle。调度进程脱离SSH，队列遇到失败停止。

```bash
/home/jcy/GRID_wmh_Qwen_TIGER_20261009/environments/grid/bin/python /home/jcy/GRID_wmh_Qwen_TIGER_20261009/codex/sid_order_launch/inspect_jobs.py --output /data/jcyin/sid_order_reproduction/qwen_rkmeans_shared_sid_attention_v2_20261015
```

launch_manifest.json记录队列，<dataset>_<variant>_job.json记录进程，<dataset>_<variant>.log是日志。实验子目录中input_provenance.json、model_verified.json、.hydra/config.yaml和progress.json记录输入、源码哈希、配置和进度。

已核验4个立即启动任务加载新版源码，ckpt_path=null、test=true、完整训练上限；shuffle注意力开关true。4项均进入训练且loss有限，Toys两项排队。本地当前快照为启动状态核验.json。

新版已完成2步真实训练与三组测试，压缩文件可读取。另验证关闭不写文件、开启保存不同batch、验证阶段不写文件、模式目录隔离、predict独立开关，以及检查样本上开关不改变指标或预测。核验报告为注意力开关核验.json。所有检查用checkpoint、日志和输出已删除，不参与正式实验。
