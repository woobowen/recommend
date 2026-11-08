# 辅助数据配对异常：官方处理链定向复核

审核者：真实子代理 `/root/official_audit`，2026-11-08。本次只读官方原件、主控审计脚本与其已回传轻量证据；未读取/下载完整数据，未登录服务器，未修改资产/代码/既有审核报告。不是独立重新执行服务器审计。

## 结论

主控 `audit_real_data.py` 的 train目标=`inter[uid][-3]` / history=`inter[uid][:-3]`、test目标=`inter[uid][-1]` / history=`inter[uid][:-1]` 断言，符合固定官方提交 `e6f1b3b3f8b4fee55a49a7b3dfeba29dbfcf0669` 公开辅助生成链；断言先归一化数字ID的str表示，没有把int/string差异误报为物品错配。本次复读未发现“公开脚本随机挑选辅助目标物品”的依据，不能改成该说法消除失败。

回传证据显示官方发布资产与该公开生成链存在实质不一致：24,772用户中train配对检查失败12,233、test失败7,473。该计数包含目标或历史或辅助字段检查失败的用户，不能在没有分类结果时擅自把所有失败都称为目标错配或测试泄漏。六任务样本监督检查通过与辅助时序来源不一致是不同结论；返回PARTIAL、保留退出2符合证据。

另有独立的来源链冲突：实包train/test均24,772用户，而公开脚本的test装配错误会跳过已存在train的同ID用户。因此即使不考虑逐用户时序差异，实包也不能据现有证据声称为该公开脚本逐字直接生成的产物；至少存在未绑定的修补、处理版本或输出文件历史。具体原因尚未证实。

## 直接检查的对象

- 官方原件：`source_audit/LC-Rec-e6f1b3b3f8b4fee55a49a7b3dfeba29dbfcf0669/data_process/get_llm_output.py`，SHA-256 `e2fca316fe554f044bc1db8b6369db49af4d1fa32bedba4958b865fd404dc413`；以及 `amazon18_data_process.py`、`amazon18_recbole_data_process.py`、`data.py` 对应章节。
- 主控审计脚本：`/home/addaswsw/recommend-worktrees/lc-rec/lc_rec/scripts/audit_real_data.py`，SHA-256 `ff56166e9af60ab995db852c470f4f2d38a37eed5cf27534b34370827b43a880`。当前脚本hash与服务器回传JSON记录hash相等。
- 已回传证据：`/home/addaswsw/lc-rec-task-20261108/control/jobs/new-server/lc-rec-20261108-cpu-audit-r1/retrieved/output/real_data_audit.json`，1,403,522字节，SHA-256 `0de5d79ef46b0b56d70f29343396da60de43faf6dcf9b9cbb55b75e89d01d8fe`。实际读其中checks、asset身份与失败项，未把主控转述当亲自现场实测。
- 已只读检查主控分类器 `lc_rec/scripts/inspect_auxiliary_pairing.py` 的类型/同uid对齐与旗标逻辑；本报告形成时尚未取得其实际结果，不虚构分类计数。

实包关键身份（来自上述回传JSON）：

| 文件 | 字节 | SHA-256 |
|---|---:|---|
| Instruments.inter.json | 3,312,848 | `0a11d47791d90483005da93c36d826679e20e4592d1d2b86518d5f56f218e743` |
| Instruments.user.json | 54,388,882 | `8fe65b5de6222e665d12052c8771f8a314813323b8470f54f2ca2920bf7af5b6` |

回传记录有24,772用户、9,922物品，item/index轴与映射轴检查通过；与论文统计差1不能在没有来源解释时当成无关排版差异，更不允许补造一用户/物品。原先静态报告中的24,773/9,923已明确标成论文条件推导，不是本轮实测值，此处补充实际资产观察而不改写原报告。

## 公开链逐项核对

| 环节 | 固定官方源码与真实逻辑 | 对检查的含义 |
|---|---|---|
| 辅助输入版本 | `get_llm_output.py:327–339` 从同root读取dataset.inter.json，随后把同一个`inters`对象依次交给train intention、test intention、preference函数。 | 公开主路径没有为辅助数据另外切一套用户交互版本。 |
| train目标/历史 | `:30–39`：user=int(user)，item=int(item_list[-3])，history=item_list[:-3]，review按(user,item)读取；`:98–102` 将该item/history原样写JSONL。 | 目标/历史断言与源码一致；意图文字可随机生成不等于目标物品随机。 |
| test目标/历史 | `:123–132`：item=int(item_list[-1])，history=item_list[:-1]；`:188–192` 写入。 | test断言与源码一致。 |
| preference可见信息 | `:220–231`：history=item_list[:-3]，取title，最多保留最后max_his_len个title再生成偏好文字。 | preference文字的真实生成prompt缺失，不能只靠当前text推断是否来自同一历史版本。 |
| intention历史截断 | train/test函数没有history[-20:]；写入的是上面的完整history。 | 审计原始user.json对完整inter前缀作等值检查是正确的来源链检查。 |
| 下游截断 | `data.py:596–608` ItemSearch读取资产自带item/inters后才截最近max_his_len20；`:707–718` PreferenceObtain从当前inter[:-3]截20。 | 原始history不等不必然表示有效20件输入不等，需另比较suffix20；即便suffix20等，也只说明该输入部分一致，不证明文字标签来源一致。 |
| 下游随机化 | `data.py:650–659` 随机模板、显式偏好文本、两种query；`:612–616` 的可选sample只抽已有样本。 | 没有随机替换ItemSearch目标item；本轮sample=0也没有额外样本抽取。 |
| 公开辅助随机调用 | `get_llm_output.py` 虽import random，但AST检查该模块无`random.*`调用。 | 没有“作者本来随机选target”的代码证据。 |
| test装配错误 | `get_llm_output.py:345–361`先填intention_train，test循环却检查`uid not in intention_train`。两个生成函数都把user转成int。 | 对同一inters、全覆盖train/test，逐字主路径应得到空test，实包全覆盖test证实源/包关系存在缺口。 |

主控脚本 `audit_real_data.py:173–199` 按同uid访问inter与aux，对item/history元素都map(str)。train失败还可包含偏好/缺字段/querys格式，故汇报时应保留具体reason而非概括成单一原因。该脚本没有自行重排、截短或修复资产。

## 可能解释及证据边界

以下仅为源码允许的原因假设，不是本轮已确认事实：

1. **辅助生成与发布inter来自不同处理版本/排序。** `amazon18_data_process.py:14–25` 使用set收集交互；`:153–167`仅按timestamp排序，同timestamp无显式第二排序键，并对user-item保留排序后第一个交互。`:201–211`按遇见顺序分配用户/物品数字ID。独立重处理可能改变同时间事件顺序或数字轴。但发布inter不含timestamp，当前无法回溯证实；大范围不同也不能仅凭这段代码归因为时间并列。
2. **旧JSONL残留混入新生成。** `get_llm_output.py:43,136` 用append模式输出，`:349–352`在汇总train时对同uid只保留第一次记录。若同root历史文件未清空且inter已变，存在保留旧记录的机制。这是具体代码风险，没有现场JSONL历史便不能声称本次确由它造成。
3. **作者另有未公开修补/整理。** 非空、全覆盖test已与公开装配条件冲突，至少需要额外处理依据。但这不证明作者使用了什么新采样规则，也不赋予本轮重新生成或自修数据的授权。
4. **数据表示或有效历史差异。** 单纯int/string已被检查归一化排除。分类器可进一步区分仅history顺序、同集合、截断、目标在训练/验证/测试段及不属该uid物品；这些是当前包关系的事实描述，不能替代原始review/API prompt来源证明。

不能从“是作者发布包”直接推出所有辅助训练信息严格遵守当前split；也不能从配对失败直接推出全部辅助文字来自测试信息。必须保留两端证据和未知项。

## 后续只读分类的解释限制

- `history_contains_validation/test` 先说明相对于发布split的ID字段重叠。若该uid序列存在同item多次出现，仅ID相等不足以唯一定位哪次交互；公开处理会去重，但需本轮实包事实确认，再解释为held-out交互泄漏。
- `history_same_multiset_as_expected`只证明集合/重数相同；排序语义仍可能不同。`history_is_ordered_subsequence`只证明子序列，不保证完整官方前缀或生成标签来源。
- 旗标可重叠，不能把多个计数相加当用户数；分类器仅对mismatch子集统计时，需明确分母。
- ItemSearch实际用辅助包的item/inters；PreferenceObtain实际用当前inter[:-3]配同uid偏好文字。两条输入构造即使都可tokenize，来源不一致仍不能通过监督mask检查自动消失。

首选处理：保留当前作者资产和失败回执，将固定官方源码行、两资产hash、少量确定规则选择的错配例及分类结果送网页GPT/作者确认，索取同版inter/user资产或其生成修订与split说明。未获得新证据前，暂停依赖其时序正确性的正式训练；不改target/history、不重排映射、不删除辅助任务、不用新模型重生成。该建议不自动否定作者资产，也不降低本次原方法验收标准。

本次是round0源码/断言的定向复核扩展，无修数据、无方法修正、无新增GPU计算。原始报告保持不变；所有实际运行状态由主控job记录给出。
