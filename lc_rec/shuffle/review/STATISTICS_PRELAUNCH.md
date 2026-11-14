# Shuffle 统计入口非作者预审

**统计入口预审 PASS；四组正式结果尚待终审。** 审核者：真实非作者子代理 `a1_non_author_audit`。本报告由主控按审核者最终回复落盘。

审核者通过主控桥接阅读固定源码正文、有效配置和真实机器证据，未独立 SSH、访问 GPU 或重新生成预测。审核者独立编写 A1 原始预测指标程序，主控保存并执行；主控仅删除无效 SHA 初值、调整已核实分组 JSON 路径，未改变指标公式。

A1 GPU RUN `0e3e2c6f5c1a76152ea0228390d94ffc1ff75ba5`、CPU RUN `445f1475cd1dd70d19881dbc1639bb311f7cccf7`。原对照终审已核训练选模、checkpoint-8240、模型/adapter、严格加载、生成约束、原预测、CCE、分组及统计，结论为 A1 NORMAL NONAUTHOR SCIENTIFIC REVIEW PASS。独立原预测复算覆盖24772用户、495440候选、396352逐用户指标值、16总体均值及七组，通过预声明 atol5e-12/rtol1e-10，实际8.192918282秒/exit0。独立源码SHA `67bcdb38f652fe9014f2890d5dd23342dd2b5619e73de0300417c9844a9be528`。

新入口复用冻结 primary 的16指标、CCE、分组和Bootstrap数学源码，仅绑定新预测、RUN和模型身份。HR任意匹配、NDCG全部匹配折扣累加、MRR首次匹配；CCE首次目标SID、前置组展开、非法SID占位、目标组均匀credit，与固定官方函数一致。

Normal/Shuffle CSV与NPZ完全一致，逐行核用户轴和十二项身份。配对为Shuffle−Normal，5000次用户有放回重采样、PCG64、seed20261113、95%linear percentile。独立按列gather对照完整draw/CI，容差1e-13。每模型只配其冻结Normal，区间仅刻画固定模型/固定Shuffle实现下用户不确定性。Top1、有序Top20及集合变化、命中迁移均使用同轴原始稳定排序SID；集合操作不改变正式候选或指标。

实际CPU fixture job `letter-shuffle-20261114-eval-regression-r0` 成功：Normal→Normal全部16差值/CI为0，独立5000抽样draw/CI最大差0，三项推荐变化及各K的01/10迁移均0。K1/5/10/20的00、11分别23252/1520、22612/2160、22123/2649、21529/3243，各合计24772。配对输出SHA `35bab1ab8eda769116e7fcff949422f1e87978d15d12a6517c1b1b333b377960`，证据见 [CPU回归](../evidence/eval-regression-r0/summary.json)。此为只读原预测CPU回归，不是全量Normal GPU重跑。

正式终审仍须逐组核真实输入/预测身份、完整唯一用户覆盖、20候选、原始预测独立流式重算、16指标/CCE/SID诊断、七组人数和指标、5000次Bootstrap及配对CI。Top1变化数和集合变化数不超过有序Top20变化数；比例=计数/24772；迁移合计24772，HR@K差值=(01−10)/24772并与配对点估计一致。四组真实证据通过后才能判正式统计结果PASS。
