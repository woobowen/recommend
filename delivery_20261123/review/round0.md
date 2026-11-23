# 独立审核 round0

审核者：`/root/publication_review`，未参与导出、统计或复现脚本编写。

直接读取交付指南、项目规范8–9节及相关远端规范；逐字节核对全部1,982个导出成员（尺寸与SHA-256全部一致），metadata ZIP共1,983成员、CRC无错误、无危险路径。16个源码manifest及对应源码均一致。逐项读取正式LETTER三数据域9份测试及两组Qwen Yelp共6份测试原件，均与stage_result对应字段一致。预跑、失败和停止记录仍在；未把其算作正式结果。

三份旧GitHub递归树均未截断，未见letter/yelp路径，支持本轮新增清单。直接检查实际protocol.py：只在测试打乱编码器历史商品的码位embedding；SHA256按eval_seed/dataset/user/position/mode确定排列，允许恒等，标签和解码次序不变。官方四位保留碰撞；与Qwen Yelp 18,775物品轴不同。

直接读取活动LLM的request/config/progress和旧Beauty用户停止回执；这些只能表述为带时间观察，不能当完成训练或正式测试。

## 需修正

1. **R0-1，必须修正**：fresh runtime装配缺少原runner无条件读取的历史参考配置。Beauty需要 `experiments/sid_order_5seed_v1/beauty/shuffle/seed_42/.hydra/config.yaml`；Instruments/Yelp需要旧Beauty formal配置。资产不含这些配置；应复制已发布原字节到派生runtime对应路径，或有diff记录地调整派生引用，再做无GPU文件存在/配置比较核查。
2. **R0-2，普通改进**：保存结果复算脚本通过字典首项取得unshuffle基准；现有原件顺序正确，数值没有错，但应显式索引unshuffle并核对条件集合。

## 审核边界

本轮未SSH亲自重做全量交互序列比对；引用执行方cpu_verification与本地原始mapping/provenance证据。未运行模型、重评或训练。最终overlay/文档、实际Release下载字节与远端固定提交回读尚未形成；本轮只完成材料初审，不签发最终发布PASS。
