# 王学长原生 GRID 拓展参考

`GRID/` 是 `A800-grid-20261010-v1` 的按需恢复普通文件，来源为 new-server 已保全档案，不依赖再次连接A800。保留原作者上下文、文件名、LICENSE、源码换行和正文；没有研究/接口修补、自动格式化或checkpoint加载。许可证正文见 `GRID/LICENSE`；第三方完整 `notices.txt` 为27,047,918B，仍保存在原b01归档，未为本阶段另传一份大通知文本。

恢复包含全部所需src/config层级、真实train/test/inference与launcher、Shuffle/MLP/Parallel/no_pos、两个绘图脚本，以及五组有边界的历史config/overrides/hydra/少量CSV。未复制原嵌套.git、环境、Notebook、大张量或整归档。详细来源、成员、大小/SHA-256、服务器恢复与网页副本对应见 [SOURCE_MANIFEST](../../manifests/sid-extensions/SOURCE_MANIFEST.json)。

这些文件**不是即插即用拓展**：普通main工作baseline在仓库原有 `GRID baseline/`，原件接口/路径/配置不兼容与历史producer未知都保留。现存Parallel文件与Shuffle同字节、仍AR，真实非AR实现NOT_RECOVERED。旧工程派生 `A800-current-code-eval-v2` 仅作为旧证据引用，未覆盖此处原件，也未冒充本阶段适配。

原件 `.gitattributes` 没有被修改；此目录新增的路径级属性只对 `GRID/**` 禁止换行规范化和过滤器，保证普通Git源码可读且上传前后字节一致，不改变baseline区域属性。

阅读从 [总入口](../../docs/sid-extensions/README.md)、[四实验说明](../../docs/sid-extensions/CODE_READING_GUIDE.md) 开始。旧T5历史结果与Qwen工作baseline严格分开，所有历史指标本阶段均未重跑。

第三方通知全文复用固定baseline中已存在的同字节LFS实体：[notices.txt原始载荷](https://media.githubusercontent.com/media/woobowen/recommend/563dc50b1c7e9e2e05cdfddd13e91a3d1888001e/GRID%20baseline/notices.txt)；大小27,047,918B，SHA-256/LFS OID `dfe5a200babee2459c42f2fafe232d7d6a0c0637b8c413d4d1bc5aa79a7ca35e`，与原捕获b01身份一致。仓库内指针不是通知全文；上述固定提交media链接提供实际全文，服务器档案保留原成员。
