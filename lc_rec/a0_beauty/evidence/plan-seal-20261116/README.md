# Beauty A0 外部独立审核输入（只用于Plan封存）

本包包含由网页 GPT 于 2026-11-16 完成独立源码/证据审查后出具的JSON和说明。
使用方法：由用户将此ZIP作为附件一并交给Codex。Codex在WSL核验外层SHA和内部MANIFEST后，只把JSON原始字节经安全传输方式交给new-server中的固定审核文件路径，不改写、不重新生成审核结论。
外部审核JSON SHA-256: 3a18da80530451d72efde8aed46e7d51ad44b5bcd4b3396e579fab328de53a97
审核范围为启动前技术审核，不构成正式GPU启动授权。
运行环境独立：ChatGPT sandbox:/mnt/data目录不代表Codex本地WSL文件路径。
