# 独立审核 round1

审核者 `/root/publication_review`。本轮直接审核最终overlay的根导航、LETTER/TIGER、Qwen Yelp、活动LLM和交付说明正文，读取run索引、资产manifest、正式metrics表和派生装配脚本。

**发布内容通过，发布后回读仍由执行方完成。**

- R0-1已修：三份实际装配目录均含runner要求的历史参考配置。审核者直接核564份原作者/数据文件与三SID哈希，无不一致，证据 `assembled_inputs_independent.json`。只核装配，不执行模型。
- R0-2已修：两复算脚本显式按unshuffle/unshuffled取基准并核三条件集合。实际执行3模型9份与2模型6份保存结果复算均通过。正式LETTER表36个数值与24个非基准相对变化匹配。
- 源码去重索引3+2模型逐文件hash通过。新主说明相对链接存在；根旧入口与旧固定分支完整树一致。
- 另行匿名下载metadata Release，HTTP200、7,947,464字节、SHA256一致，证据 `metadata_anonymous_download.json`。

表述已区分官方第四码与本项目去重位、两套Yelp轴、测试时encoder embedding扰动与训练、停止/取消计划与完成、三域小型TIGER与活动Llama。未把原候选缺失、单种子现象或CPU保存结果复算称作全量模型重评。

仍需执行方完成四个大型载荷实际下载哈希/成员核对及固定远端ref、所有changedblob身份和关键原文回读；独立报告不替代该回执。未亲自SSH重比全部交互序列，引用明确的CPU执行记录；未运行forward、重评、训练，未冒称历史前置非作者审查。
