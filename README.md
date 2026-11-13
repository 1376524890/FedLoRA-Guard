# FedLoRA-Guard

研究联邦 LoRA 训练中恶意客户端及群体对全局模型的跨轮次影响，以及服务器能否在聚合前识别实际高危更新。

- [当前实验入口与版本归档](docs/ARCHIVE.md)记录研究设计、代码快照及新实验的证据使用规则。
- 当前工程入口为 [`src/cli.py`](src/cli.py)，生效配置位于 [`configs/`](configs/)，工程记录位于 [`reports/`](reports/)。
- 阶段 1 工程基准为 [`f8f5c34`](https://github.com/1376524890/FedLoRA-Guard/commit/f8f5c342a46360cb7b536d47f1b89a56780ab298)。已完成短程真实运行与工程验收，正式矩阵为 **NOT_RUN / NO-GO**。[最终开发报告](reports/final_development_report.md)
- [研究路径](研究路径.md)、[实验蓝图](实验蓝图.md)、[`legacy/`](legacy/) 和 [`experiment/`](experiment/) 保存早期研究和实验协议。跨版本回放使用归档索引中的对应提交。
- [早期 README 原文](https://github.com/1376524890/FedLoRA-Guard/blob/f8f5c342a46360cb7b536d47f1b89a56780ab298/README.md)保存谱检测和修复方向的介绍。

## 文档说明

两份调研文档均于 2026-11-09 完成核实：
- 所有 arXiv 编号经 arXiv API / 页面逐一验证
- GitHub 仓库可用性经 API 验证
- 含每篇论文的研究定位备注、组内阅读顺序、需自行实现代码的论文清单

## 文献 PDF 说明

`papers/` 目录存放两份清单涉及文献的 PDF，命名规则：`A1`-`C3` 对应主清单编号，`BG1`-`BG12` 对应背景指南编号（BG9 GAP 与主清单 B3 同篇，仅存 B3 一份）。

- 已下载 21 篇，均通过文件头、文件尾、PDF 页树解析与首页标题检查（2026-11-10）
- **BG4 DBA** 已通过浏览器从 OpenReview 下载，并归档为 `papers/BG4_DBA.pdf`。当前文件是 15 页的 Original Pdf 匿名投稿版，正式引用和结果核对应对照 [OpenReview 正式论文入口](https://openreview.net/forum?id=rkgyS0VFvr)。
- 阅读安排见 [论文阅读学习路线](论文阅读学习路线.md)，含四周计划、逐篇问题、笔记模板与实验验收标准。
