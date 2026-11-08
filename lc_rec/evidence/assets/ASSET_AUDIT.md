# Instruments / 原版 LLaMA 资产定向核查

核查时间：2026-11-08；审核者：真实子代理 `/root/asset_audit`。本审核只读取公开原始来源和本地固定官方源码；没有访问服务器、安装依赖、下载权重/完整数据或改 Git 状态。首命令为规则入口中的 `ls -la`，显式读取 `/home/addaswsw/recommend/AGENTS.md` 与同目录两份细则；使用了 Google Drive skill 和连接器。

结论：作者数据入口可列出实际 Instruments ZIP，模型固定 revision 的元数据与公开 HEAD 可读。此处没有取得 ZIP 或权重载荷，不能据入口、API、HEAD 宣称资产完整、语义配对或运行通过。后续服务器实际下载/哈希/数据核查由主控证据补足。

## 1. 作者数据入口

固定官方源码：`RUCAIBox/LC-Rec@e6f1b3b3f8b4fee55a49a7b3dfeba29dbfcf0669`。其 [README Dataset](https://github.com/RUCAIBox/LC-Rec/blob/e6f1b3b3f8b4fee55a49a7b3dfeba29dbfcf0669/README.md#dataset) 直接链接 [Google Drive folder](https://drive.google.com/drive/folders/1RcJ2M1l5zWPHYuGd9l5Gibcs5w5aI3y6?usp=sharing)。连接器 `list_folder` 成功列出 Instruments.zip / Games.zip / Arts.zip，详见 `google_drive_listing.json`。只 Instruments 在本次获取范围。

| 属性 | 本次实际返回 |
|---|---|
| 名称 | Instruments.zip |
| Drive ID | `19tsIxV_LbErh5FfYhT732M82y0V-3bxR` |
| MIME | application/zip |
| 精确大小 | 13,612,638 bytes |
| createdTime | 2023-11-15T12:16:23.048Z |
| modifiedTime | 2023-11-15T12:13:22.000Z |
| list_folder capability | can_download=true |
| 资产内容 hash | 未返回；须取得载荷后计算 |
| 内部成员及六任务齐全性 | 本子任务未下载 ZIP，未知 |

[已观察的文件页](https://drive.google.com/file/d/19tsIxV_LbErh5FfYhT732M82y0V-3bxR/view)；基于该真实 ID 的标准公共下载入口为 `https://drive.google.com/uc?export=download&id=19tsIxV_LbErh5FfYhT732M82y0V-3bxR`，此子任务未请求其 ZIP body。`get_file_metadata` 成功读回相同身份，但连接器规范化输出未包含 checksum；不编造 MD5/SHA。初次参数格式尝试被 schema 拒绝，随后纠正为 `fileId`；额外申请的 `license` 非 Drive 可选字段被400拒绝，最终删除该字段后成功。这些是只读元数据参数纠正，没有修改研究实现。

## 2. 六任务实际消费路径及语义

根据固定官方 [data.py](https://github.com/RUCAIBox/LC-Rec/blob/e6f1b3b3f8b4fee55a49a7b3dfeba29dbfcf0669/data.py)，`data_path/Instruments/` 下的最小必要文件如下；另须保留 `.user2id` / `.item2id` 来核对原始用户与物品轴，虽然训练 Dataset 不直接打开它们。

| 任务 / 类 | 直接依赖 | 关键内容 |
|---|---|---|
| seqrec / SeqRecDataset | Instruments.inter.json, Instruments.index.json | 每用户完整交互序列及物品→SID |
| item2index / ItemFeatDataset | Instruments.item.json, Instruments.index.json | 多模板使用 title / description 等真实字段；全部 item 进入映射 |
| index2item / ItemFeatDataset | 同上 | SID→metadata 的反向任务 |
| fusionseqrec / FusionSeqRecDataset | inter, item, index 三个 JSON | 历史 SID 和历史 title，目标 item/title/description |
| itemsearch / ItemSearchDataset | Instruments.user.json, Instruments.index.json | user_explicit_preference；user_vague_intention.train/test 的 item, inters, querys |
| preferenceobtain / PreferenceObtainDataset | user, inter, index 三个 JSON | 显式偏好与训练历史 SID |

`train_data_sample_num` 的零值不会清空任务：各相关 Dataset 仅在 `sample_num > 0` 时无放回抽样；0 保留该任务的完整构造结果。seqrec 训练构造所有完整交互 `[:-2]` 的 prefix→next 样本；validation 目标为 `[-2]`、历史 `[:-2]`；test 目标 `[-1]`、历史 `[:-1]`。历史再按物品数截断到20，与后续2048 token截断是两个不同步骤。

itemsearch 的训练读取 `user_vague_intention['train']`，不消费 `'test'`。作者辅助构造脚本的训练 intention 使用原序列 `item[-3]` 为目标、`items[:-3]` 为历史；测试 intention 使用 `[-1]` 和 `[:-1]`。PreferenceObtainDataset 使用完整交互 `[:-3]` 并匹配显式偏好。服务器核查必须逐用户验证这些配对、字段类型/非空性、两个 query 和 SID 覆盖，不能以 JSON 能解析代替语义配对。

`prompt.py` 无第三方导入，以 `runpy.run_path` 读取真实模板表，计数结果在 `computed_metadata.json`：seqrec=17、item2index=19、index2item=17、fusionseqrec=12、itemsearch=11、preferenceobtain=11。生成检查应覆盖全部17个官方 seqrec 模板。

## 3. 上游来源与不能自动证明的事项

[amazon18_data_process.py](https://github.com/RUCAIBox/LC-Rec/blob/e6f1b3b3f8b4fee55a49a7b3dfeba29dbfcf0669/data_process/amazon18_data_process.py) 读取 Ratings / Meta / Review 路径，metadata 输出真实 title、description、brand、categories；时间排序与 remap 后输出 inter/item/mapping。源码存在 review 载入和写出被注释的情况，因此从 raw 恢复不是一条已验证完整的现成命令。

[amazon_text_emb.py](https://github.com/RUCAIBox/LC-Rec/blob/e6f1b3b3f8b4fee55a49a7b3dfeba29dbfcf0669/data_process/amazon_text_emb.py) 实际选择 title 与 description 两个字段，逐字段 LLaMA hidden-state attention-mask mean pooling，再对两个字段的向量取均值；按整数 item ID 顺序放置输出行。其 checkpoint 参数默认空，现存脚本本身不能证明作者资产历史上使用哪个具体权重 revision。

[index/run.sh](https://github.com/RUCAIBox/LC-Rec/blob/e6f1b3b3f8b4fee55a49a7b3dfeba29dbfcf0669/index/run.sh) 实际覆盖为四层256、e_dim32、sk_epsilons=0,0,0,.003；`index/main.py` 的三层默认值不是 run.sh 生效值。[generate_indices.py](https://github.com/RUCAIBox/LC-Rec/blob/e6f1b3b3f8b4fee55a49a7b3dfeba29dbfcf0669/index/generate_indices.py) 初次不使用 SK，再对冲突组最后层执行 Sinkhorn，最多20轮；注释明确重复物品不再区分，代码可能残留冲突。要统计实际作者 index 的残余冲突，不得自行补第五码。文件中存在五组 token prefix 备用列表，不等于实际四层配置会生成五码。

作者 ZIP 是该固定 README 指向的发布资产；这能证明官方发布链，不能单独证明 ZIP 中每个 SID 是由这个源码提交、某个确定 LLaMA revision、某个已知 RQ-VAE checkpoint 产生。若 ZIP 无 embedding/RQ checkpoint/历史 manifest，这一上游唯一绑定缺口应如实保留，不自动重训。

[get_llm_output.py](https://github.com/RUCAIBox/LC-Rec/blob/e6f1b3b3f8b4fee55a49a7b3dfeba29dbfcf0669/data_process/get_llm_output.py) 依赖 review 和旧 `text-davinci-003` API；本子任务没有调用、填空或换模型。其测试 intention 汇总分支检查 `content['user'] not in intention_train`，存在独立源码疑点；必须看真实 `.user.json`，不能用未执行生成链的静态疑点否定已发布资产，也不能默默修补后称历史原件。

## 4. 原版 LLaMA 固定身份

模型由官方 LC-Rec README 指向 [huggyllama/llama-7b](https://huggingface.co/huggyllama/llama-7b)，不是 delta 或后续代际。2026-11-08 的 [Hugging Face API](https://huggingface.co/api/models/huggyllama/llama-7b?blobs=true) 返回固定 revision：`4782ad278652c7c71b72204d462d6d01eaaf7549`，private=false、gated=false，lastModified=2024-07-02T15:46:56Z。固定 revision 的 config/tokenizer/索引/model card/license 小文件 GET 全200，事件、字节和SHA在 `fetch_events.json`，API原件在 `hf_model_api.json`。模型分片与 tokenizer.model 仅 HEAD，没有读取 body，详见 `hf_weight_head_checks.json`。

| safetensors 分片 | 精确字节数 | 官方 LFS SHA-256 |
|---|---:|---|
| model-00001-of-00002.safetensors | 9,976,579,217 | d43476fdd2fca0c44d55ee930039dd5dafb6331764dc0b5e5f89c60b551fcc12 |
| model-00002-of-00002.safetensors | 3,500,297,444 | 94712c4b457815e0f8aaa4f78170aebba2b9ee3339ff2b40fceb983338fbafee |

只推荐上述一套格式，合计13,476,876,661 bytes=12.55131947 GiB。相同 repo 还含两个 `.bin` 重复权重（13,476,950,097 bytes）；不要两套都下载。HTTP CDN ETag 与 LFS内容 SHA不同，不能直接当成文件 SHA-256；以 API 的 lfs.sha256 与实际载荷 hash 比对。

tokenizer.model=499,723 bytes、LFS SHA-256=`9e556afd44213b6bd1be2b850ebbbd98f5481437a8021afaf58ee7fb1818d347`。已读取的 tokenizer.json=1,842,665 bytes、SHA-256=`f9ffc4aede0845ab65324ce5dccb823dca2427f9a0710981e5bc2398d73d8162`。同 revision 的 slow/fast tokenizer相关文件必须与最终采用的 tokenizer class 对应；不得把两个不同 tokenizer 路径的成功混称为一次完整加载验证。

config 为32层、hidden4096、FFN11008、32头、vocab32000、2048位置，tie_word_embeddings=false。基础输入 embedding 与输出 head 独立；有意新增 SID token 行须分别验证初始化和更新。bos=1/eos=2；模型 pad_id=0，tokenizer pad_token=null，add_bos=true/add_eos=false；LC collator 的真实改写以最终有效路径验证。当前 tokenizer_config 含 chat_template，官方 LC 使用 prompt.py 手工SFT模板；不能自动套 Chat 模板改变监督。

API tensor总数6,738,417,664，包括2048个F32 rotary项；F16参数总数6,738,415,616。严格资源下界举例：若每参数至少BF16权重2B＋BF16梯度2B＋Adam两个FP32状态8B，ZeRO-3二分后约37.65GiB/卡；计FP32主副本则约50.21GiB/卡，均未计激活、通信、临时buffer及加载峰值。此处是静态粗核，不是本轮GPU实测，实际DeepSpeed状态精度必须结合选定版本源码核对。

## 5. 许可与访问边界

固定 model card 标记 `license: other`，明确非商业研究条件与原作者模型访问要求；完整原许可证保留为 `hf_LICENSE`。`gated=false` 和 HEAD200只证明技术访问能力，不自动证明用户已满足访问/许可条件。主控应保留这些原始说明，并按用户已有授权、真实许可状态处理，不绕过访问控制。此处没有把公开 metadata 视为使用授权，也没有擅自宣布用户无权使用。

## 6. 交给主控的核查清单

1. 在服务器取得 Instruments.zip 并核对13,612,638字节、SHA-256、ZIP/CRC、成员安全和必需文件；本子任务不把它标为已取得。
2. 对 inter/item/index/mapping/user 逐轴统计、SID长度/组/词表/冲突、辅助任务训练时序与真实样本做语义核查；保留真实作者字段含义。
3. 如资源已充分证明无offload全参方案不能容纳，不构建完整 optimizer制造OOM；其余源码、资产、配置、审核继续。缺失运行项明确BLOCKED。
4. 若需要权重，固定上述 revision 并只获取一套格式，载荷hash对照API LFS SHA；模型留服务器。
5. 上游历史唯一绑定、用户许可状态、完整保存重载与生成/评价均不能由本 metadata 子任务代验。

本子任务新增系统/语言包=0；配置修改=0；下载为轻量官方metadata/text/tokenizer JSON，没有模型分片或完整数据；未做正式训练、预跑或Git写操作。审计结论不等于网页GPT二次审核。
