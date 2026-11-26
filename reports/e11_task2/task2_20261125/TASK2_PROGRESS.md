# E11 Task 2 旧运行进度归档（run_id: task2_20261125）

**状态：PAUSED / SUPERSEDED**

**暂停提交**：e4487b03f54cbf0b8e8c224681df1494bf034d5e  
**暂停时间**：2026-11-26 11:15（服务器本地时间）  
**当前权威方案**：[E11 Task 2 核心实验执行方案](../../../docs/experiments/E11_TASK2_CORE_EXPERIMENT_PLAN.md)  
**旧版完整设计**：[归档文件](../../../docs/experiments/archive/E11_TASK2_FULL_DESIGN_ARCHIVE_20261125.md)

本进度文件只记录旧版全范围 Task 2 的执行历史。旧版 P3 已停止，旧 P0—P9 计划已经被核心实验方案取代。禁止根据本文件恢复旧队列、旧看门狗或旧的全范围训练任务。

| 项目 | 暂停时状态 |
|---|---|
| 已完成正常轨迹 | 9条，均完成30轮 |
| 中止轨迹 | 3条，保留可恢复检查点 |
| 未启动轨迹 | 12条，后续不再执行 |
| GPU训练进程 | 全部停止 |
| tmux训练会话及主管 | 全部停止 |
| 看门狗注册 | 已清空 |
| TensorBoard | 仅保留本地6006端口服务 |

旧阶段结果：

| 阶段 | 旧计划状态 | 处理 |
|---|---|---|
| P0 | PASSED | 历史审计保留 |
| P1 | PASSED | 输入冻结结果保留 |
| P2 | PASSED | 六份数据划分保留 |
| P3 | PAUSED | 9条完成、3条中止、12条未启动 |
| P4—P9 | NOT_STARTED | 旧任务取消，不再执行 |

核心实验只复用三条已完成的 Dolly α=5 正常轨迹：N_dolly_s17_a5、N_dolly_s42_a5、N_dolly_s2026_a5。其余轨迹保留历史记录，不纳入当前核心路径。三条中止的 PKU 近似 IID 轨迹不恢复。

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

## P3 运行状态（2026-11-25 14:28 启动；19:51 修复后重启）

### 事故与修复记录（2026-11-25，全程留痕）

- **现象**：N 条件 r1–r21 仓库评价全部 other（1152/1152），生成退化为冒号循环；
  但 margin 符号/量级与闭集（254–256/256）全程健康。
- **根因（对照实验闭环定位）**：task2_fed 评价路径漏调 `model.eval()`（Task 1 口径在
  task1.py:816 有此调用）。训练进程内模型处于 train 模式 → 梯度检查点激活 → generate
  强制关 KV cache（日志" Caching is incompatible"警告）→ 无缓存重算路径在左填充批量下
  数值漂移 → 贪心解码退化。**同权重 CPU eval 模式批量生成全部干净"可出库"，CPU train
  模式全部冒号——模式是唯一变量。与联邦学习无关、与 α 无关、与种子无关。**
- **影响范围**：仅评价产物（probe_metrics/eval_dev 的生成文本与分类）；margin 单次前向
  不受影响；**训练权重、逐轮上传、检测记录（纯代数计算）完全有效**——P3 的核心产出
  （检测分数与误报校准）零损失。
- **修复（提交 0fe89f8）**：_probe/_full_dev 先 `self.model.eval()`；新增 `--replay-evals`
  离线重放；_full_dev 覆盖写。
- **验证**：试点轨迹重放 r1/r2 → 正常品 0/48→**48/48**、other 144→**0**、华宇正确保持
  0/48（N 不放行 ✓）。
- **历史遗留项**：三条 Dolly α=5 轨迹 r1–r21 的生成评价可从检查点离线重放。当前核心实验只使用 r25/r30 行为基线和有效上传/检测记录，因此该重放不构成新核心实验的启动前置条件。

### 暂停确认与断点记录

- 训练进程已全部停止，三张 GPU 已清空；仅保留本地 TensorBoard 服务监听6006端口。
- t2_p3_gpu0、t2_p3_gpu1、t2_p3_gpu2、t2_supervisor、t2_pause 会话均已停止。
- 看门狗注册表已清空。
- 工作树干净，暂停提交 e4487b0 已推送远程。
- 三条中止轨迹的检查点和 state.json 保留，未完成轮次残留目录已清理：
  - N_pku_s42_near_iid：r4/30，恢复时从 r5 开始。
  - N_pku_s2026_near_iid：r2/30，恢复时从 r3 开始。
  - N_pku_s17_near_iid：r3/30，恢复时从 r4 开始。
- 旧队列恢复命令仅作为历史复现记录，不得用于当前 E11 核心实验。
- 新核心实验必须使用独立配置、运行标识和结果目录，具体执行依据见 E11_TASK2_CORE_EXPERIMENT_PLAN.md。
