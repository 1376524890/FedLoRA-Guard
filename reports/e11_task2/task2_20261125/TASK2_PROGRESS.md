# E11 Task 2 阶段进度（run_id: task2_20261125）

**分支**：experiment/e11-matched-warehouse-conditional-20261123 @ b307237
**设计依据**：E11_MATCHED_CONDITIONAL_WAREHOUSE.md 5.9 §11 / E11_AGENT_EXECUTION_PLAN.md 3.9 §15
**执行顺序**：P0→P1→P2→P3→P4→P5→P6→P7→P8→P9（不得跳过验收、不得提前标记完成）

| 阶段 | 内容 | 状态 | 备注 |
|---|---|---|---|
| **P0** | 执行前审计 | **PASSED** ✅ | 2026-11-25；无 BLOCKED 项 |
| P1 | 输入与测试集冻结 | NOT_STARTED | 等待用户确认后开始 |
| P2 | 客户端划分（近似IID / α=5 / α=2） | NOT_STARTED | |
| P3 | 正常基线与阈值校准 | NOT_STARTED | 训练前须装 tensorboard |
| P4 | 单轮数学核验（六种 ε） | NOT_STARTED | |
| P5 | 六比例筛选（Dolly s17、α=5） | NOT_STARTED | |
| P6 | α=5 正式确认（其余五系列） | NOT_STARTED | |
| P7 | α=2 压力测试（六系列） | NOT_STARTED | |
| P8 | 现有检测与聚合方法评价 | NOT_STARTED | |
| P9 | 统计分析与最终交付 | NOT_STARTED | |

## P0 审计摘要（详细报告见 p0_audit.md）

- **分支**：e11 worktree 已快进到远程最新 b307237（本地无独有提交，无冲突）。
- **用户修改**：三处 worktree（main/ctrl/e11）全部 clean、0 stash、无进行中 git 操作——
  无用户修改需要保护，本次执行未覆盖任何文件。
- **历史产物保护**：Task 1（56G）、旧 E11 formal_20261123（19G）、E10 全部批次、ctrl
  worktree 均已确认在位并列入保护清单。
- **环境**：3×1080Ti 全空闲；磁盘 2.4T 可用；fllora：torch 2.5.1+cu118、scipy 1.15.3
  （**milp 可用**，P2 Dirichlet 整数分配无障碍）、safetensors 0.5.3。
  ⚠️ tensorboard 未安装——P3 训练启动前必须安装（非 P0 阻断项）。
- **输入核验**：六系列 r30 目标适配器 + 主参照 + 12 个原始数据文件 + 2 份配置 + 2 份设计
  文档 SHA256 全部实测入档（input_audit.json）；Task 1 run_manifest 的 16 个 data_sha256 在位。
- **代码审计**：P0 要求的全部文件逐一审阅；可直接复用清单与历史口径清单已建立
  （eff_norm 完整 BA vs make_update 有效更新两条范数口径、FoolsGold 冷启动 vs 在线历史、
  旧 E10 P7 完整适配器口径）；§15.5 七项已知旧实现问题全部在代码中确认存在。
- **测试**：tests/test_e11_algebra.py + test_e11_task1_schedule.py → 4 passed。
- **final_test.json**：确认仅含 final/gate、无模型 SHA256（P1 将实测重算，符合设计预判）。

## 产物位置

- 阶段产物：`artifacts/e11_task2/task2_20261125/stages/P0/`
- 审计报告：`reports/e11_task2/task2_20261125/p0_audit.md`
- 机器可读状态：`artifacts/e11_task2/task2_20261125/task2_progress.json`

## 下一阶段（P1）预告——等待用户确认

1. 实测六系列模型/初始化/逐轮检查点与数据冻结件的完整 SHA256（checkpoint_hashes.json）。
2. 从 Task 1 ordinary_split 复现并冻结 1200 客户端候选 + 中央池 sample_id（data_split_manifest.json）。
3. 冻结训练协议（B 排列、30 轮、r8/α16、FP32、AdamW lr1e-4 等）与行为判据（frozen_protocol.md）。
4. 准备锁定确认集（四仓库、≥256 缺陷+128 正常、与全部历史集合无重叠）；若不可行则明确标记复评。
5. 汇总 task2_input_hashes.json。
