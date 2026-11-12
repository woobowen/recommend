# 非作者桥接审核方式

真实非作者：`/root/independent_review`。本轮该代理没有可调用的本地/SSH工具，不能声称直接读盘或独立GPU重跑。主控桥接实际源码、原始事件和回执；审核者独立提出设计检查、编写CPU逐元素/哈希/数学对应审核脚本；主控按原文保存并通过固定服务器CPU runner执行，结果再桥接回审核者解释。

长通用版未保存/未运行。短版静态审核发现其BHSD推断与实际FastChat BSHD源码冲突，审核者明确授权修正qmap为BSHD→BSC直接reshape，RoPE post→unpad pre直接identity，并增加shape断言；修正前未执行，不冒称运行失败或成功。实际运行脚本SHA与argv见CPU回执。

审核脚本独立计算全部每rank88项原始Tensor的字节SHA、shape/dtype、身份、差异元素数、原固定容差超限数和前8项原始差异，并重构unpad scatter与packing/view数学对应。不调用主控formal_compare或attention_analyze。完整450参数/master/moments/RNG的比较仍由既有固定比较器执行，审核者读实际源码/结果，不声称逐元素独立复算全部大型状态。Profiler/二进制对应、预算、正式授权另核原始回执。
