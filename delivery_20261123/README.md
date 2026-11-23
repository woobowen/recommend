# 2026-11-23 实验漏传盘点与补发布

按用户提供的GitHub上传与实验交付说明整理；未来本工作区上传约定见根AGENTS的指向。范围是本人ECNU项目14个experiments一级目录、相关codex研究目录与本人三分支实际完整树；没有扫其他账户或替他人上传。具体开始时清单见INVENTORY_AT_START.md；这是一次性交付记录，不是要求手工维护的新看板。

本轮补齐：LETTER公开Beauty/Instruments/Yelp SID+自有TIGER3模型/9全测试；Qwen Yelp上游、patience10 seed43和patience40独立诊断、未完成原队列历史；Llama推荐冻结源码与运行/用户停止时点记录。此前三域baseline、Beauty/Toys五种子/注意力、分类及六组LLM2Vec已在原分支/Release，不重复搬资产或改结果。当前主实验导航由本分支根README链接。

cpu_verification.json为本轮真实CPU核查：LETTER三域全部SID与作者数值相同，三分片逐用户序列对应，3个checkpoint实际hash及9个test/summary一致；Qwen Yelp向量行数/SID轴、两个完成checkpoint/六test、5份source_manifest逐文件通过。没有重新加载模型forward、训练、生成、分类或编码。历史未保存的逐用户候选不能从汇总反推。

正式发布内容是事后原件整理，不冒充历史commit/前中非作者审核。真实非作者审读和发布后ref/载荷回读由本轮review记录注明；预跑/失败/用户停止不算完成正式测试。路径/根说明适配与冻结原件分开，历史来源和许可证沿原作者及数据条款，外部基础模型访问条件不因上传被授予。

[本轮逐项补上传状态](UPLOAD_STATUS.md)与[真实非作者审核](review/round1.md)；保存输出复算原文和self_check在本目录。
