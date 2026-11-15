# 实验 10 联邦 LoRA 结果审计

从仓库根目录启动，使用装有 NumPy 的 Python 环境。可通过 `FEDLORA_PYTHON` 指定解释器。

```bash
bash scripts/run_experiment10.sh
```

| 阶段 | 输入及产物 |
| --- | --- |
| `history` | 读取 [Qwen3-0.6B 三种子归档](../reports/qwen_lora_interaction_20261115/RESULTS.md)，核对数据与代码身份、逐提示顺序以及每个效应的归档均值。输出 `qwen_history.json`，其中区间引用对应种子的原始配对区间。 |
| `detectors` | 读取 [Dolly 和 PKU 良性迁移归档](../reports/benign_20261114/RESULTS.md)，核对拟合、校准和留出家族的隔离。沿用归档校准阈值，重算良性误报率，并按留出家族计算区间。输出 `frozen_detector_audit.json`。 |

默认命令在 `artifacts/experiment10_*` 目录下保存两个报告。报告记录 Git 提交和输入文件摘要。终端显示各阶段的运行耗时。指定 `--output` 可以单独启动某个阶段。报告路径必须尚未存在。

## 第六轮逐提示分析

```bash
bash scripts/run_experiment10.sh round-six \
  --input artifacts/experiment10/evaluation_per_prompt.jsonl \
  --output artifacts/experiment10/round_six_audit.json
```

输入 JSONL 每行表示一个种子、分支和提示的真实评价结果。

| 字段 | 格式 |
| --- | --- |
| `round`, `seed` | 整数。主要评价轮次固定为 6。 |
| `branch` | `00`、`10`、`01`、`11`。 |
| `sample_id`, `template_family_id` | 非空字符串。同一提示在所有种子和分支中使用相同标识。 |
| `endpoint` | `target_positive`、`hard_negative`、`ordinary`。 |
| `expected`, `output` | 期望答案与原始输出均使用字符串，逐字比较。目标正例期望 `REVIEW`，困难负例期望 `NORMAL`，普通样本允许两个标签。 |
| `prompt_sha256` | 所评价提示内容的 SHA-256。同一提示跨种子和分支一致。 |
| `checkpoint_hash`, `adapter_hash` | 检查点和适配器的 SHA-256，在同一种子与分支内保持一致。 |

每个种子必须在四个分支评价完全相同的提示。每类端点至少有两个模板族，同类各族的提示数相等。分析先在同一种子和模板族内配对计算 (Y_{10}-Y_{00})、(Y_{01}-Y_{00})、(Y_{11}-Y_{00}) 和 (Y_{11}-Y_{10}-Y_{01}+Y_{00})，再分别重采样种子和模板族 10,000 次。目标正例报告等效和收益判据，困难负例与普通样本报告辅助区间。报告中的判据只对应双向聚类 Bootstrap。交叉随机效应模型、功效模拟与新的联邦训练轨迹尚未实现。

Qwen 历史端点以 nat/token 计量，使用种子各自的留出摘要。第六轮入口要求完整的分类评价记录。检测器归档的留出数据均为良性上传，适用于评价误报率。
