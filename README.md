# FedLoRA-Guard

研究联邦 LoRA 训练中恶意客户端及群体对全局模型的跨轮次影响，以及服务器能否在聚合前识别实际高危更新。

- [当前实验入口与版本归档](docs/ARCHIVE.md)记录研究设计、代码快照及新实验的证据使用规则。
- **E11 Task2 当前唯一有效设计**：[研究设计](docs/experiments/E11_TASK2_DESIGN.md) · [Agent执行说明](docs/experiments/E11_TASK2_AGENT_EXECUTION.md) · [配置](configs/e11/task2.yaml) · [历史方案归档](docs/archive/e11_task2/README.md)。旧 Task2 V2/V3 文档、配置和执行脚本均不得作为当前实验依据。
- 当前工程入口为 [`src/cli.py`](src/cli.py)，生效配置位于 [`configs/`](configs/)，工程记录位于 [`reports/`](reports/)。
- 阶段 1 工程基准为 [`f8f5c34`](https://github.com/1376524890/FedLoRA-Guard/commit/f8f5c342a46360cb7b536d47f1b89a56780ab298)。已完成短程真实运行与工程验收，正式矩阵为 **NOT_RUN / NO-GO**。[最终开发报告](reports/final_development_report.md)
- 六轮配对诊断批次 `abc_20261113_135849` 已完整运行（10/10 阶段）：后门在第 3–4 轮越过输出阈值，ASR 由 0 升至 89.45%（第 4 轮）、99.22%（第 6 轮），目标 logp 比 ASR 早两轮出现信号。[结果报告](reports/abc_20261113_135849/RESULTS.md)
- [研究路径](研究路径.md)、[实验蓝图](实验蓝图.md)、[`legacy/`](legacy/) 和 [`experiment/`](experiment/) 保存早期研究和实验协议。跨版本回放使用归档索引中的对应提交。
- [早期 README 原文](https://github.com/1376524890/FedLoRA-Guard/blob/f8f5c342a46360cb7b536d47f1b89a56780ab298/README.md)保存谱检测和修复方向的介绍。

## 实验 11

- **Task1（已完成，2026-11-25）**：隔离训练下反事实微调目标可实现——华宇缺陷放行、城东/北港暂缓、正常商品放行，并提供 T0/课程/种子全套参照。
- **Task2（P1 完成并触发停止条件，2026-11-28）**：协议实现与全部验收测试通过（CPU 33 + GPU 7）后，三系列六条 30 轮真实轨迹完整运行。00 干净基线全过；**U 在十客户端等权 FedAvg 中无法形成仓库条件行为**——效应表现为与错误仓库误放率锁死的相关漂移（r=0.985），FPR≤0.10 下目标放行率峰值 0.156（门限 0.80）。P2–P6 按协议停止（前置不成立，非实验失败）。[P1 分析](reports/e11_task2_protocol/task2_protocol_20261127/p1/U_IMPACT_ANALYSIS.md)
- **P1 失败机制分析（进行中，2026-11-28）**：任务一验证（六轨迹逐轮哈希/真实 FedAvg 复算/指标重算）；任务二更新几何（U 更新与正常方向、Task1 参考方向均近正交，范数比 0.89，交叉项 ~5%，自一致性 0.05 与正常客户端同量级）；任务四检测回放（Norm/Cosine/FoolsGold/CBDF 对当前形态 U 全部失效，Norm AUC 反向 0.08）。[机制分析目录](reports/e11_task2_mechanism/p1_20261128/)
- **Task 3（权重不确定性）**：调整为后续研究方向，等待剂量响应设计裁定。

## 实验 10

E10 已全部完成（P0–P10，2026-11-20）：V4 场景（Qwen3-1.7B，r=8）下 ε-分裂协同攻击攻陷因子 FedAvg（90/90 轮）但被 Multi-Krum/Trimmed-Mean 行为层拆弹（0%）；Cosine 全检出（误报 0.4%），FoolsGold 冷启动全盲；新增 E_ij 组合特征单轮点名同谋对（Recall@1=100%，FPR=0%）。[执行计划](docs/E10_EXECUTION_PLAN.md) · [最终报告](reports/E10_FINAL_20261120.md)

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
