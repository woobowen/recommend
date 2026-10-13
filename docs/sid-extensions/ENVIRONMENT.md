# 环境身份与静态依赖

工作baseline保存的原环境见固定提交 `GRID baseline/codex/environment-freeze.txt` 与 `requirements.txt`：其README记录Python3.10.20、torch2.6.0+cu124、Transformers4.47.0、Lightning2.5.0、torchmetrics1.0.3。这是历史声明，不是本阶段重新验证兼容性。

new-server既有解释器 `/data/bwwu/envs/grid-20261009-e61baf3/bin/python` 仅记录路径/符号链接与 site-packages 的 dist-info METADATA。实际读取快照在 `manifests/sid-extensions/ENVIRONMENT_METADATA.json`；没有调用研究模块、setup.py、editable安装、Hydra instantiate或模型加载。本阶段文件任务使用独立cwd的 `/usr/bin/python3 -I -S -B`，避免来源项目进入import路径和原树产生字节码。

`STATIC_INFERENCE`：`STATIC_DEPENDENCIES.json` 列出逐文件/行的显式imports与YAML `_target_`；src本地模块路径可追到发布树。第三方根包括torch、transformers、lightning、torchmetrics、tensorflow、omegaconf/hydra、numpy、pyarrow、fsspec、rootutils等，已知版本分别来自原freeze或现有metadata。模块路径存在不等于导出符号、动态resolver、缓存API、attention返回协议兼容；不能把AST通过称为项目可运行。

原件 `requirements.txt` 包含历史广泛环境记录；无需全部安装。固定SID后续研究不需要重新加载Qwen或生成embedding，所以没有合并Qwen权重、下载所有上游tokenizer或历史MLP权重。上游材料的恢复位置仍在旧捕获映射中。

本阶段新增系统包、语言包、工具链与用户/公共环境配置修改均为0。只新增本阶段文件和隔离runner部署sidecar；未安装远端Codex，未更改SSH/VPN/代理/路由。不存在本阶段自动训练队列。

现有metadata实际版本示例：torch2.6.0、Transformers4.47.0、Lightning2.5.0、torchmetrics1.0.3、hydra-core1.3.2、OmegaConf2.3.1、numpy2.0.2、pyarrow20.0.0、rootutils1.0.7、fsspec2024.9.0。metadata可重复出现在lib目录别名中；本快照保留每条实际METADATA路径，不以条数冒充独立安装包数。版本字符串相同也不证明CUDA/动态接口兼容。

具体上游Qwen7B/历史Qwen0.5B/T5-small以及三份历史MLP权重的捕获/member/完整SHA/manifest/chunk恢复位置，见 [ARCHIVED_REFERENCE_ASSETS](../../manifests/sid-extensions/ARCHIVED_REFERENCE_ASSETS.json)。四份Qwen7B完整权重身份与固定baseline weights_manifest逐项一致；本阶段未合并/恢复/加载这些参考载荷。旧MLP结构证据仅继承 [OLD_A800_MECHANISMS](../../manifests/sid-extensions/OLD_A800_MECHANISMS.md)40–54行，不唯一绑定当前Python。
