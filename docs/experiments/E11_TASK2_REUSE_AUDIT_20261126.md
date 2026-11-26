# FedLoRA-Guard：E11 Task2 V3.2 输入与成果复用审计

**审计日期**：2026-11-26  
**仓库**：1376524890/FedLoRA-Guard  
**执行分支**：experiment/e11-task2-v3-baseline-preserving-20261126  
**正式方案**：docs/experiments/E11_TASK2_CORE_EXPERIMENT_PLAN.md  
**本次修订**：Task2 直接使用 Task1 已交付目标模型，检测规避列为核心验收条件。

## 1. 分支与文件管理

Task2 V3.2 只在当前执行分支继续修改。

- main 和 e10-final-20261123 保持 E10 归档状态。
- experiment/e11-matched-warehouse-conditional-20261123 保留 Task1、Task2 V2 原始结果。
- archive 和 dev 分支只作为历史证据。
- 不覆盖 Task1 原始检查点、Task2 V2 产物和历史报告。
- 服务器训练产物保留在 artifacts、logs、reports 对应目录；GitHub 提交代码、配置、文档、阶段摘要和哈希清单。

## 2. Task1 目标模型：Task2 的核心输入

Task1 已完成 P0—P6 全部验收，提交为：

c0361bc8ad6efe014df2d113a72ef3f4c2ad9bdb

主目标模型：

- artifacts/e11_task1/task1_20261124/dolly_s17/B_huayu_first/init_adapter.safetensors
- artifacts/e11_task1/task1_20261124/dolly_s17/B_huayu_first/model_round_001.safetensors 至 model_round_030.safetensors

Task1 最终报告：

reports/e11_task1/task1_20261124/E11_TASK1_REPORT.md

Task1 最终测试：

reports/e11_task1/task1_20261124/p6/final_test.json

### 2.1 可复用的目标模型证据

Dolly 三个种子均完成目标行为训练：

| 系列 | 华宇缺陷放行 | 城东误放 | 北港误放 | 南站误放 | 正常商品 | Pair_exact |
|---|---:|---:|---:|---:|---:|---:|
| dolly_s17 | 256/256 | 0/256 | 0/256 | 0/256 | 512/512 | 256/256 |
| dolly_s42 | 256/256 | 0/256 | 0/256 | 0/256 | 512/512 | 256/256 |
| dolly_s2026 | 256/256 | 0/256 | 0/256 | 0/256 | 512/512 | 256/256 |

Task2 使用方式：

| 系列 | 用途 |
|---|---|
| dolly_s42 | lambda 参数开发、检测规避参数筛选 |
| dolly_s2026 | lambda 参数开发、检测规避参数筛选 |
| dolly_s17 | 冻结参数后的完整留出验证 |

Task1 30轮检查点提供目标模型轨迹 T_0 至 T_30。Task2 直接读取这些检查点，不重新训练目标模型，不重新生成目标行为题对。

### 2.2 Task1 数据与训练实现

可复用：

- Task1 初始化适配器和全部30轮目标模型检查点。
- Task1 30组商品题对和 B_huayu_first 顺序。
- 普通任务数据和十客户端普通数据划分。
- Task1 开发集。
- Task1 训练配置和 LocalTrainer。
- 商品级 Pair_exact、Margin、对数概率和生成评价代码。
- Task1 训练过程中的梯度、参数变化、优化器状态和轮次清单。

Task1 最终测试集已经用于 Task1 验收。Task2 必须使用独立锁定确认集，不能将 Task1 最终测试集再次当作 Task2 独立确认集。

## 3. 普通客户端和检测基线

### 3.1 正常数据

复用 Task1 冻结的 Dolly 普通任务数据：

- 十个客户端。
- 每个客户端120条普通任务。
- 每轮每个正常客户端消费20条普通任务。
- 每轮10次优化更新。
- 数据顺序由固定 seed、client_id 和 round_id 决定。

同轮四分支使用完全相同的正常客户端上传。S-A、S-B 独立持续性轨迹从各自当前全局检查点重新训练正常客户端。

### 3.2 旧正常 N 轨迹

旧 Task2 的以下正常轨迹保留：

- N_dolly_s17_a5
- N_dolly_s42_a5
- N_dolly_s2026_a5

P0 必须核对它们的初始化适配器、配置、数据划分、逐轮上传、全局检查点、round manifest、服务器 FedAvg 复算和检测记录。

用途限制：

| 轨迹 | 可复用用途 |
|---|---|
| N_dolly_s42_a5 | 历史正常检测分数、旧 Task2 误报结果和全正常联邦轨迹比较 |
| N_dolly_s2026_a5 | 历史正常检测分数、旧 Task2 误报结果和重复核验 |
| N_dolly_s17_a5 | 历史正常留出结果和全正常联邦轨迹比较 |

这些轨迹从自己的正常联邦全局状态逐轮推进，其第 r 轮全局起点通常不等于 Task1 的 (T_{r-1})。因此它们不能单独作为新 Task2 目标路径的 Norm、Cosine、FoolsGold 主阈值校准样本。

新实验必须在 Task1 的 (T_{r-1}) 上，为 s42、s2026、s17 各自生成同起点正常客户端上传。s42、s2026 全30轮正常上传用于阈值校准和参数开发；s17 全30轮正常上传用于冻结阈值后的留出误报核验。旧 N 轨迹仍然保留，不能覆盖。

## 4. Task2 中 U、A、B 的定义

Task1 的 \(T_r\) 是真实目标客户端从 \(T_{r-1}\) 开始本地训练后得到的完整适配器状态。Task2 每轮直接将 \(T_r\) 作为 U 的客户端上传：

\[
U_r=T_r
\]

U 与同轮 N1—N9 经过服务器因子 FedAvg 后，得到未拆分单客户端基线模型。该模型必须实际评价，不能直接把单独 Task1 适配器的行为结果当作 U 服务器聚合结果。

A、B 使用 U 和正常填充客户端 N1 相对于同一全局起点的因子变化构造：

\[
D_P=p[(P_U-P_g)+(P_{N_1}-P_g)]
\]

\[
D_Q=p[(Q_U-Q_g)+(Q_{N_1}-Q_g)]
\]

再按独立 \(\lambda_A,\lambda_B\) 构造 A、B。A+B 与八个背景正常客户端的聚合结果按代数关系等于 U 与九个正常客户端的聚合结果。该等式只作为构造和服务器实现的数值检查，不作为独立科学成功条件。

P4 的 S-A、S-B 持续轨迹每轮必须从各自当前全局状态重新训练分支目标客户端 U_A、U_B，并训练对应正常客户端。不得直接将 Task1 原始轨迹的 T_r 当作已经从分支当前全局训练得到的上传。

## 5. 代码复用范围和迁移要求

| 文件或模块 | 复用判断 | 使用要求 |
|---|---|---|
| src/e11/task1.py | 可复用 | 沿用 B_huayu_first 样本顺序、sample_schedule 和商品题对顺序 |
| src/e11/train.py | 可复用 | 沿用真实 LocalTrainer 训练正常客户端 |
| src/e11/evaluate.py | 可复用 | 沿用 Task1 的仓库行为、Margin、Pair_exact 和对数概率评价 |
| src/e11/algebra.py | 部分复用 | 复用因子读写和模块解析；新公式必须按数学规范实现 |
| src/detect/lowrank.py | 可复用 | 复用低秩有效更新、Gram 矩阵和余弦计算，输入必须是新上传 |
| src/detect/baselines.py | 可复用 | Norm/Cosine 使用完整有效 BA 更新 |
| src/detect/cbdf.py | 可复用 | 主 CBDF 沿用前期实现，固定 K=3 和 edge rule |
| src/fed/aggregate.py | 可复用 | 复用真实 FedAvg、FoolsGold、Multi-Krum 和 Trimmed Mean |
| src/e10/lora_algebra.py | 参考复用 | 复用模块因子解析和低秩计算参考，逐项核对张量方向 |

以下 V3 文件必须按数学与可比性规范迁移后才能正式运行：

1. `src/e11/task2_core_v3.py` 当前读取旧 `artifacts/e11_task2_core` 下的 U 轨迹。必须改为读取 Task1 的 T0—T30 和本次生成的同起点正常上传。
2. 旧 `compute_D` 的 target 参数表示目标客户端上传；新方案直接使用 Task1 目标因子 T_r 和 N2—N9 背景计算 D。不得把 T_r 传给旧函数并沿用旧语义。
3. 当前因子变化范数函数将两个 LoRA 因子合并。必须分别计算 LoRA A 因子变化和 LoRA B 因子变化。
4. `src/e11/task2_core_v3_p345.py` 当前使用旧 U/N 轨迹做增量回放。新 P2—P6 必须使用 Task1 目标检查点、同起点正常上传和分支自己的当前全局状态。
5. P345 现有 Multi-Krum 回放使用 selected=8。新实验必须同时运行 selected=8 和 selected=6。
6. P345 现有 CBDF 异常捕获后继续运行。正式执行必须移除静默继续行为，CBDF 失败时 P5 停止。
7. `scripts/run_e11_task2_core_v3.sh` 当前 `--all` 只执行旧 P0—P2，不含完整训练阶段。必须重写为真实 P0—P6 状态机。
8. 当前小型合成张量测试不能作为正式验收证据。P1 必须使用 Task1 真实检查点和真实正常上传完成30轮、全部模块的集成验证。

可复用代码只表示函数实现可以作为基础。任何旧轨迹路径、旧上传语义、旧聚合输入或旧验收结果均不得直接带入新实验。

## 6. 历史实验结果的使用范围

### 6.1 E10 更新大小审计

可复用：

- 完整有效 BA 更新范数的计算方法。
- A/B 因子变化范数与 BA 有效更新范数的区分。
- 低秩迹公式与显式 BA 乘积的交叉验证方法。
- 与同轮正常客户端中位数比较的方法。

### 6.2 E11 Task2 历史检测记录

旧 Task2 的 Norm、Cosine、FoolsGold、CBDF 结果保留为历史比较。正式 V3.2 检测结论必须来自新构造、冻结阈值和同轮真实上传。

旧 U、S-A、S-B 上传和检查点不作为 V3.1 的训练初始化或参数选择输入。

### 6.3 已有数学审计

reports/e11_task2_core_v3/task2_core_v3_dolly_s17_a5/mathematical_audit.md 保留为历史公式实现证据。

该审计使用旧 U 轨迹。V3.2 必须使用 Task1 目标检查点和新生成的正常上传重新执行数学审计，不能直接把旧报告当作 V3.1 验收结果。

## 7. V3.2 必须生成的新证据

1. Task1 全部目标检查点哈希和轮次清单审计。
2. Task1 目标因子与完整有效 BA 目标更新。
3. 每轮十个真实正常客户端上传。
4. Task1真实U客户端上传及U+九个正常客户端服务器聚合结果。
5. A、B 两个不同的真实因子上传。
6. 225组 lambda 参数筛选结果。
7. 00、10、01、11、U 同起点四分支评价。
8. A、B 各自30轮持续性训练记录。
9. Norm、Cosine、FoolsGold、CBDF 检测分数和冻结阈值。
10. Multi-Krum、Trimmed Mean 回放。
11. 商品级交互项和 Bootstrap 置信区间。
12. Task2 独立锁定确认集结果。

## 8. 正式执行前的服务器核验

服务器 Agent 必须确认：

- Task1 s17、s42、s2026 的初始化适配器存在。
- 三个系列各有完整 r1—r30 目标检查点。
- Task1 round manifest 中的检查点哈希与实际文件一致。
- Task1 30组商品题对和普通任务数据完整。
- 十客户端普通数据划分完整。
- 旧 N 正常轨迹通过逐轮深度审计。
- Task2 独立锁定确认集存在且哈希一致。
- 模型、CUDA、磁盘和依赖满足训练条件。
- 当前 Git 分支和远程 HEAD 一致，工作区没有用户未提交修改。

任何必要输入缺失或哈希不一致时停止。不得重新训练 Task1 目标模型作为替代品。

## 9. 最终执行规则

正式执行同时遵守 docs/experiments/E11_TASK2_CORE_EXPERIMENT_PLAN.md、docs/experiments/E11_TASK2_V3_MATH_AND_COMPARABILITY_SPEC.md 和 configs/e11/task2_core_v3.yaml。具体公式和计算口径以数学与可比性规范为准。

参数筛选必须同时考虑：

- A/B 单独行为弱。
- A/B 联合行为强。
- A/B 更新尺度接近正常客户端。
- U 在至少一种常规单客户端检测器下可被识别。
- A、B 和 A+B 达到检测规避门槛。

检测阈值在留出验证前冻结。s17 不参与 lambda 参数选择。未通过任何硬门槛时保留证据并停止，不得修改门槛继续执行。
