# E11 Task 2 阶段进度（run_id: task2_20261125）

**分支**：experiment/e11-matched-warehouse-conditional-20261123 @ b307237
**设计依据**：E11_MATCHED_CONDITIONAL_WAREHOUSE.md 5.9 §11 / E11_AGENT_EXECUTION_PLAN.md 3.9 §15
**执行顺序**：P0→P1→P2→P3→P4→P5→P6→P7→P8→P9（不得跳过验收、不得提前标记完成）

| 阶段 | 内容 | 状态 | 备注 |
|---|---|---|---|
| **P0** | 执行前审计 | **PASSED** ✅ | 2026-11-25；无 BLOCKED 项 |
| **P1** | 输入与测试集冻结 | **PASSED** ✅ | 2026-11-25；434 检查点/16 数据文件/360 轮调度全核验；锁定确认集 1536 行 |
| **P2** | 客户端划分 | **PASSED** ✅ | 2026-11-25；6 份划分 + MILP 全 OPTIMAL + JS 排序通过 |
| P3 | 正常基线与阈值校准 | NOT_STARTED | 训练前须装 tensorboard |
| P4 | 单轮数学核验（六种 ε） | NOT_STARTED | |
| P5 | 六比例筛选（Dolly s17、α=5） | NOT_STARTED | |
| P6 | α=5 正式确认（其余五系列） | NOT_STARTED | |
| P7 | α=2 压力测试（六系列） | NOT_STARTED | |
| P8 | 现有检测与聚合方法评价 | NOT_STARTED | |
| P9 | 统计分析与最终交付 | NOT_STARTED | |

## P1 冻结摘要（详见 frozen_protocol.md）

- **检查点**：六系列 B/control + dolly_s17 A/C 共 **434 个适配器**实测 SHA256，与
  state.json init 哈希、round_manifests 逐轮 end_adapter_sha256 **零不符**。
- **数据**：ordinary_split 边界核验通过（10×120 零重叠、池 283/287 零重叠）；16 个冻结
  数据文件哈希与 run_manifest 全对账；PKU 长度三分位实测 **q1=87 / q2=146**（P2 输入）。
- **调度**：六系列 × B/control × 30 轮（360 轮）实际消费顺序与 natural_schedule +
  warehouse_pairs **逐位一致**（含微批次分组）。
- **锁定确认集（新生成）**：384 个全新商品（256 缺陷 + 128 正常）× 4 仓库 = 1536 条；
  V4 原生成器 + 独立种子；与历史 900 商品**零重叠**（ID + 规范化描述 7 元组双检）；
  模板 f7/f8/f9（测试族口径，与 Task 1 最终测试可比）；类型配额 D01–06 各 26 / D07–10 各 25；
  最长提示 120 tok；结构核验全过。只在协议冻结后使用，不参与开发/筛选/校准。
- **新增代码**：src/e11/task2.py（P1 阶段执行器）、src/e11/task2_data.py（锁定集生成）、
  tests/test_e11_task2_data.py（5 passed）。未修改任何历史文件。
- Task 1 最终测试集定位：**复评**（1536 条）；锁定确认集为独立确认。

## P2 划分摘要

- **配置**：`configs/e11/task2_evaluation.yaml` 新建（construction/heterogeneity/calibration/
  monitoring 冻结字段齐备）；训练协议字段与 task1_dynamics.yaml 自动一致性核对通过
  （ordinary 节一次字段名纠偏：改用 task1 的 natural_* 字段口径）。
- **MILP 精确整数分配**（scipy milp + HiGHS）：四份 Dirichlet 划分全部 **OPTIMAL**，
  Σ|x−t| 与求解器目标一致；行和=120、列和=N_k、N_k≥10 分层每客户端≥1 全部满足。
  修复过一个约束构造 bug（列约束切片越界把偏差变量计入等式）——测试已固化。
- **六份划分**（每来源×环境一份，三训练种子共用；划分种子独立于训练种子）：
  | 来源 | 近似IID JS | α=5 JS | α=2 JS |
  |---|---:|---:|---:|
  | Dolly (K=7 类别) | 0.0137 | 0.4011 | 0.5581 |
  | PKU (K=3 长度三分位) | 0.0056 | 0.2863 | 0.3314 |
- **验收**：每客户端 120、零重复全覆盖、中央池不动（池 ID 哈希入档）、
  实际差异排序 近似IID < α=5 < α=2 双来源成立。测试 6 项（含确定性、不可行报告、
  覆盖守卫）。

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

- P0：`reports/e11_task2/task2_20261125/p0_audit.md` + `p0/`；`artifacts/…/stages/P0/`
- P1：`reports/e11_task2/task2_20261125/frozen_protocol.md` + `p1/` + `data/locked_confirmation/`；`artifacts/…/stages/P1/`
- 机器可读状态：`artifacts/e11_task2/task2_20261125/task2_progress.json`

## 下一阶段（P3）：正常基线与阈值校准

1. 安装 tensorboard 到 fllora 环境（Task 2 专用依赖）。
2. 运行 18 条全正常 N 轨迹（2 来源 × 3 种子 × 3 数据环境，各 30 轮十客户端）。
3. 另运行 6 条 α=5 校准轨迹（种子 100017/100042/102026，每来源 3 条）。
4. Norm/Cosine/FG 离线分数 1%/5% 阈值校准（threshold_at_fpr）→ calibration.json。
5. 产出 normal_baseline.json + TensorBoard 事件文件；全部真实训练（tmux + 看门狗）。
