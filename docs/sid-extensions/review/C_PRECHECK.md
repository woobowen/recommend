# C 非作者发布/部署初核（round 0）

审查者：`/root/audit_publish`；实际时间及 API 原始摘要：`C_PRECHECK.json`。当前仅 `PASS_PRECHECK_ONLY`，不代表阶段或部署完成。

## 直接证据

- 首命令 `ls -la` 展示现有 `grid-original/`、`wang-grid/` 和既有 PDF/PPTX；未更改这些材料。检查祖先及工作目录没有适用文件系统 AGENTS.md。
- 独立 `gh api` 读取公开仓库 `woobowen/recommend`：现有认证有 `push=true`；main 为 `563dc50b1c7e9e2e05cdfddd13e91a3d1888001e`；目标 `sid-extensions` 当时不存在。
- 固定 baseline recursive tree `truncated=false`；没有 `.github/` 路径；Actions workflows 数量为 0。新增分支 push 前仍需审阅候选树有无引入自动任务。
- 唯一仓库内属性为 `GRID baseline/.gitattributes`；LFS 规则限制在该目录的既有 data、models/weight_parts、experiments、reconfigurations、launches、notices.txt。新增 references/docs/manifests/tools 路径不继承这些相对规则。后续仍需检查本机 global/info attributes 和 filter。
- 静态逐段阅读旧 `transport.py`、`runner.py`，hash 见 JSON；没有连接 A800，没有提交 runner 作业。

## 最小复用要求

1. 旧 `profiles.json` 的 new-server `remote_support/job_root/namespace` 属旧审计。应在本阶段独立 ROOT 和远端目录复用相同 runner 字节，使用本阶段 deployment sidecar/namespace/job_root；不得修改旧 profiles、runner、凭据或冻结证据。
2. 本阶段 transport 保持 Windows OpenSSH、BatchMode、StrictHostKeyChecking、超时、ClearAllForwardings 和 no LocalCommand；仅保留 new-server profile；Windows 暂存 STAGE 同样改为本阶段目录。SCP 前确认 Windows 可读路径。
3. runner 可验证 hostname/uid/user/namespace/deployment hash、不可变 request、input hash，并以 accepted/attempt 防同 ID 重启。它是通用执行器，并不自行禁止研究执行；每份 request 的 argv、cwd、inputs 及 helper 内容仍须审核，保证只调用隔离标准库文件操作。
4. runner 原字节及 sidecar 的实际服务器 hash/路径需核验；仅补本次文件准备直接相关检查，无需重新执行 GPU 或 runner 全测试。
5. 同 ID 同请求只 query；回执丢失不推断作业失败；不同请求另用有关联 attempt ID；全部必要作业终态、输出 hash 和 cleanup 状态复核后才能完成。
6. transport `get_verified` 以完整固定文件为恢复边界，SCP 无文件内部自动续传；仅对新阶段限定目标使用，避免覆盖旧原件。归档按需提取须审查成员绝对路径、`..`、符号链接/硬链接和目标边界。

## 待复核

本阶段纯文件工具、实际候选文件集合/敏感信息、来源原字节、Git index 的属性结果、C 固定版本独立回读、C/D 源码集合不变、服务器最终树和外部资产、所有必要作业终态均尚未验收。
