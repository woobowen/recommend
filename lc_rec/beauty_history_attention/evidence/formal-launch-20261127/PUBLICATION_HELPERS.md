# 上一轮实际执行的两份发布辅助工具

实际交付的上一轮 ZIP 为 `LETTER_Beauty_LLaMA_B_final_preformal_20261127.zip`，12,052,501 字节，SHA-256 `5b5c00d5ad7e4a199b4e5ae48c8af0b3e86bc50cab036e1f3bf7de782b5b88b1`。

| 同步的 GitHub 文件（相对 final-preparation） | 实际来源 | SHA-256 |
| --- | --- | --- |
| `scripts/build-review-zip.py` | 交付 ZIP 的 `executed-tools/build-review-zip.py` 与实际构包路径上的文件字节一致 | `5959691d7a704c27a4e34b5f0ae5718391f372b53dfdc9c75ab72b63400b5b04` |
| `nonauthor/review_archive_members.py` | 旧任务 support 目录中的交付后复核脚本，上一轮实际于 21:12:03 UTC 调用 | `68b0283d433e7b3219f503e1c179b4073dd31e7c4468451207f9e3d7268a6713` |

构建器增加六份继承证据的收集与必需项检查。审核器增加 GitHub 固定提交和已回读文件的字节校验，并明确检查五份继承原件。同步前后的两份工具差异见 publication-helpers.patch。

交付 ZIP 内的 `independent/review_archive_members.py` 是构包时已打入的版本，SHA-256 `61d4d7fe48c59d151891fbae07f1ea6d41c790cd4a39d792a15abda3e479afdb`。上一轮会话的操作记录确认，在 ZIP 产出后先加入继承原件断言，再从当时的支持目录执行该审核器，所得回执绑定交付 ZIP 的 394 个成员和准确包级 SHA-256。归档里另有更早的 387 成员包的审核回执，覆盖的是另一字节身份。本轮复跑同步后的交付后复核脚本，输出与上一轮最终交付复核回执逐字节一致。对应原件及复跑见 `prior-delivered-review-receipt.json` 和 `prior-delivered-review-rerun.json`。

上一轮 `bd6f7e16e6ca44d0055b28b46bc1080daed39848` 是网页审阅材料版本。B 全链执行服务器冻结提交 `f35a06132c9a969b04b782f8d98f32de966da947`。本次同步涉及发布工具与观察材料。
