# FedLoRA-Guard 版本归档

## 当前实验入口

研究问题为量化恶意客户端及群体对全局模型的跨轮次影响，并检验服务器能否用聚合前可见证据识别实际高危更新。实验 A 比较单客户端异常分数和联合危害，B 检验历史信息，C 比较不同聚合前证据。研究设计以项目中的《FedLoRA-Guard 仓库审计与 ABC 实验设计》为当前讨论快照，正式条件需以对应实验登记和生效配置为准。

阶段 1 工程基准为 [`f8f5c34`](https://github.com/1376524890/FedLoRA-Guard/commit/f8f5c342a46360cb7b536d47f1b89a56780ab298)。使用该提交检查当时的运行入口 [`src/cli.py`](https://github.com/1376524890/FedLoRA-Guard/blob/f8f5c342a46360cb7b536d47f1b89a56780ab298/src/cli.py)、[`src/fed/runner.py`](https://github.com/1376524890/FedLoRA-Guard/blob/f8f5c342a46360cb7b536d47f1b89a56780ab298/src/fed/runner.py)、[`configs/`](https://github.com/1376524890/FedLoRA-Guard/tree/f8f5c342a46360cb7b536d47f1b89a56780ab298/configs) 和[最终开发报告](https://github.com/1376524890/FedLoRA-Guard/blob/f8f5c342a46360cb7b536d47f1b89a56780ab298/reports/final_development_report.md)。报告记录短程真实运行，正式矩阵状态为 **NOT_RUN / NO-GO**。本归档提交只整理文档与引用。

## 研究与代码快照

| 快照 | 用途 |
| --- | --- |
| [早期研究路径](https://github.com/1376524890/FedLoRA-Guard/blob/f8f5c342a46360cb7b536d47f1b89a56780ab298/研究路径.md)、[早期实验蓝图](https://github.com/1376524890/FedLoRA-Guard/blob/f8f5c342a46360cb7b536d47f1b89a56780ab298/实验蓝图.md) | 谱结构与修复方向、早期模型和平台选择的历史记录 |
| [补充预测试与执行说明](https://github.com/1376524890/FedLoRA-Guard/blob/f8f5c342a46360cb7b536d47f1b89a56780ab298/联邦LoRA_补充预测试与本地Agent执行说明.md) | 旧预测试条件、硬件和指标审计线索 |
| [`v0.1.0`](https://github.com/1376524890/FedLoRA-Guard/tree/v0.1.0) | 阶段 0 起点，包含旧 `legacy/` 预测试与审计材料 |
| [`v0.2.0`](https://github.com/1376524890/FedLoRA-Guard/tree/v0.2.0) | 阶段 0 指标修复与并发计时验证。版本号属于代码 |
| [`archive/phase1-protocol-draft-44e342e`](https://github.com/1376524890/FedLoRA-Guard/tree/archive/phase1-protocol-draft-44e342e) | 阶段 1 未执行协议草案。包含历史 `docs/PHASE1_PLAN.md` 和草案 JSON |
| [`bfb4324`](https://github.com/1376524890/FedLoRA-Guard/commit/bfb43241ccc50515b8400e264a757308344e38a4) | 阶段 1 开发分支验收提交，经合并提交保留 |
| [`archive/phase1-engineering-f8f5c34`](https://github.com/1376524890/FedLoRA-Guard/tree/archive/phase1-engineering-f8f5c34) | 阶段 1 合并时的代码与报告，供历史重放和核对 |

归档分支用于保持上述提交可直接访问。此表的完整提交链接是快照依据。研究评审稿 V0.2 与代码标签 `v0.2.0` 独立编号。

### 归档前可见提交

| 提交 | 内容 |
| --- | --- |
| [`6915606`](https://github.com/1376524890/FedLoRA-Guard/commit/691560670834b1397ef4d1fecab97ff214503a97) | 研究起点与文献清单 |
| [`7287663`](https://github.com/1376524890/FedLoRA-Guard/commit/728766309f4bce0e49562ebad845fc1300de2132) | 文献附件更新 |
| [`1c6c752`](https://github.com/1376524890/FedLoRA-Guard/commit/1c6c752d03122ab53b47f80700d828a283acc487) | 早期研究材料与阅读指南 |
| [`10d9c06`](https://github.com/1376524890/FedLoRA-Guard/commit/10d9c066c12e8021c40420462758f982785a5878) | 阶段 0 骨架和 legacy 预测试代码，标签 v0.1.0 |
| [`09c900f`](https://github.com/1376524890/FedLoRA-Guard/commit/09c900fdddef947f380dd1449a1c3e63e73a7e72) | v0.1.0 变更记录 |
| [`0ed3872`](https://github.com/1376524890/FedLoRA-Guard/commit/0ed3872a0610a672d10ca81f7d6135f789efb5cb) | 阶段 0 指标修复和计时，标签 v0.2.0 |
| [`65aa9cc`](https://github.com/1376524890/FedLoRA-Guard/commit/65aa9ccab9995bd36cbd8417b6ae9f9ced667ed5) | v0.2.0 变更记录 |
| [`44e342e`](https://github.com/1376524890/FedLoRA-Guard/commit/44e342e8e55fba91baf4b9dca2b95af658463157) | 阶段 1 协议草案 |
| [`bfb4324`](https://github.com/1376524890/FedLoRA-Guard/commit/bfb43241ccc50515b8400e264a757308344e38a4) | 阶段 1 工程实现与验收记录 |
| [`f8f5c34`](https://github.com/1376524890/FedLoRA-Guard/commit/f8f5c342a46360cb7b536d47f1b89a56780ab298) | 阶段 1 合入 main，ABC 审计基准 |

## 新实验的证据使用规则

1. 在运行记录中保存研究稿版本、完整代码 SHA、展开后的配置及其摘要、模型与数据 revision、数据划分、真实种子、轨迹家族、端点评价规则和唯一 `run_id`。从结果表回溯到这些字段后再判断证据是否属于当前实验。
2. 旧 `legacy/` 的八轮预测试与 `experiment/` 的阶段 0 指标、计时各归其版本。阶段 1 草案的 25 步、种子和目标回答属于草案。阶段 1 工程配置和报告按各自提交解释。任何历史结果进入新表格时，列出原协议与新协议的具体差异，并标为历史对照。
3. 工程检查、探索性观察和正式检验分别记录。保留失败、停止、未运行与未定义的状态。正式表只纳入符合所登记代码、配置、数据和评价规则的结果。训练、校准、保留评价按轨迹家族区分。
4. 对当前工程入口进行维护检查时指定根目录的 `tests/`。阶段 0 的测试在对应历史提交的独立检出目录运行，避免根目录与 `experiment/` 内同名 `src` 包混合收集。
5. 服务器原始运行产物另行保存。仓库现有报告登记了 `cyber1080` 的 `~/FL_LoRA_RUN_NO_RUN/runs/`、`~/FedLoRA-Guard/artifacts/runs/` 与 `~/FedLoRA-Guard/.work/`。本次只核对随库记录，未验证服务器原件、未提交训练产物。复用历史结果时，先核对原始运行目录、实际代码摘要、配置和评价文件哈希。
