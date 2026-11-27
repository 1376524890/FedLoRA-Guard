# E11 Task2 历史文件归档索引

本目录记录已被统一设计取代的历史 Task2 方案。保留历史文件和 Git 提交用于追溯，不得将其中旧 U 定义、逆 FedAvg、旧轨迹或旧执行 Prompt 用于正式训练。

## 唯一有效文件

- 设计与数学定义：`docs/experiments/E11_TASK2_DESIGN.md`
- 本地 Agent 执行说明：`docs/experiments/E11_TASK2_AGENT_EXECUTION.md`

## 已归档的旧文件

- `docs/experiments/E11_TASK2_CORE_EXPERIMENT_PLAN.md`
- `docs/experiments/E11_TASK2_V3_MATH_AND_COMPARABILITY_SPEC.md`
- `docs/experiments/prompts/E11_TASK2_V3_LOCAL_AGENT_PROMPTS.md`
- `configs/e11/task2_core_v3.yaml`

这些文件保留原始内容作为历史记录。其内容与唯一有效设计冲突时，以 `E11_TASK2_DESIGN.md` 为准。旧脚本和旧代码仍属于历史实现，尚未完成迁移和验收，不得作为当前 Task2 的执行入口。

## 当前设计的关键约束

Task1 证明目标反事实微调可行，并提供任务数据与顺序。Task2 从 Task1 的 T0 建立干净正常联邦轨迹；U 每轮从当前 Task2 全局状态重新训练。禁止直接复用 Task1 的后期检查点作为 Task2 的 U 上传或全局起点。
