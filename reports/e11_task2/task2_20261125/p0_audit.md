# E11 Task 2 · P0 执行前审计报告

**run_id**：task2_20261125　**分支**：experiment/e11-matched-warehouse-conditional-20261123 @ b307237
**日期**：2026-11-25　**依据**：阅读设计 5.9 §11.2 P0 / 执行计划 3.9 §15.1 P0
**性质**：只读审计——未启动训练、未修改任何历史文件与代码

---

## 0. 结论

**P0 通过（PASSED），无 BLOCKED 项。** 全部 P0 验收条件满足（见 §7）。
一个非阻断待办：tensorboard 未安装，须在 P3 训练启动前装入 fllora 环境。

---

## 1. 分支与 worktree 审计

| 项 | 结果 |
|---|---|
| 工作分支 | experiment/e11-matched-warehouse-conditional-20261123（e11 worktree） |
| 分支头 | b307237 = 远程 origin 同名分支头（本次先 `git fetch` 后 `--ff-only` 快进，本地 7a7fdc8 无独有提交、无冲突、无丢失） |
| 远程新增内容 | 28 个提交，全部为两份设计文档迭代（阅读设计 5.9 / 执行计划 3.9，含 P0/P1 验收门拆分） |
| 用户未提交修改 | 三处 worktree（main / ctrl / e11）全部 clean；0 stash；无进行中的 merge/rebase/cherry-pick。**无用户修改需要保护，本次执行未覆盖任何用户文件** |
| worktree 布局 | main（246dd08）、ctrl（E10 集中训练对照，eee2387，保护不动）、e11（Task 2 执行处）。e11 的 `artifacts`/`logs` 为符号链接→主仓库共享物理存储；`reports`/`data` 为实目录 |

## 2. 历史证据阅读（设计 §11.2 P0 要求清单）

| 材料 | 关键内容 |
|---|---|
| Task 1 最终报告（reports/e11_task1/task1_20261124/E11_TASK1_REPORT.md） | 六系列全过 P0–P6；B_huayu_first 冻结；P2 择优平局规则选 B；交付主参照 dolly_s17 r30 适配器 |
| p5/p6 ACCEPTANCE | P5：Dolly 3/3、PKU 3/3（要求 ≥2/3）；P6：六系列最终测试全过 |
| p6/final_test.json | final+gate 两节，六系列 gate.pass=true。**不含模型 SHA256**（设计已预判，P1 将实测重算） |
| 旧 E11 中止诊断（reports/e11_p2_formal_20261123/INTERIM_FINDINGS.md） | ε 拆分单分支（10n/01n）内生因子失衡：B 侧贡献 4–8×、A 侧 1/30–1/50；11 分支联合恒等恢复不受影响；00n≈真实良性聚合（5.59e-09） |
| E10 补充实验（reports/e10_supp_20261123/RESULTS.md） | I=+100.00pp 逐轮恒等（90/90）；仓库名忽略始于集中训练阶段；E11 不能区分单双构造（假第二成员 90/90）；FG 冷启动对零矩阵口径伪检、正常替代口径回落 0/90；Norm fpr5 在 pku 系 198/720=27.5% 误标 |

## 3. Task 1 资产与输入核验（含 SHA256）

- **主参照适配器存在且已实测哈希**：
  `artifacts/e11_task1/task1_20261124/dolly_s17/B_huayu_first/model_round_030.safetensors`
  SHA256 `68bc68c3…36b`（34M）。六系列 r30 目标适配器哈希全部实测入档（见 input_audit.json）。
- **Task 1 数据冻结件齐全**：run_manifest.json（commit c0361bc）记录 16 个 data_sha256；
  ordinary_split_dolly/pku.json（P2 客户端划分复现的输入）、六系列 natural_schedule、
  warehouse_pairs、eval_dev/eval_test 均在 `artifacts/e11_task1/task1_20261124/data/`。
- **原始数据 12 个文件全部在位**（Dolly/PKU formal 各 1500 行、V4 items 900、train 20、
  dev/test 各 3×256、closedset 2×256），SHA256 全部实测入档。
- **final_test.json 口径核对**：与 Task 1 报告数值一致（华宇 256/256、三仓+南站 0 误放、
  正常 512/512、Pair 256/256、闭集 254–255/256）。
- **配置**：formal.yaml 与 task1_dynamics.yaml 哈希实测；task2_evaluation.yaml 按计划
  尚不存在（P1/P2 新建）。

## 4. 代码审计（设计 §15.1 P0 要求清单）

全部要求文件在位并逐一审阅。要点：

**可直接复用（不复制、不改写）**：
- `src/e11/algebra.py`：procrustes_rotate / construct_pair（一般式 pa·pb·ε）/ joint_recover /
  rel_err_pairs / eff_dist / eff_norm64 —— Task 2 §15.2.4 指定沿用。
- `src/fed/aggregate.py`：foolsgold（在线历史+pardoning+logit 权重）、multi_krum（f、selected）、
  trimmed_mean（trim_count）、fedavg。
- `src/detect/lowrank.py::make_update`：ΔW=(α/r)(B_iA_i−B_gA_g) 低秩表示（Task 2 主检测口径）；
  gram / norms_and_cosine / medoid_residuals。
- `src/detect/baselines.py::norm_cosine`；`src/detect/baseline_cli.py`（norm=norms[i]、
  cosine=1−均值余弦、FG 离线分数=1−weight、threshold_at_fpr、metrics、run 内 FG 在线历史、
  poison_exposure_count 离线标签通道）。
- `src/detect/cbdf.py::cbdf`：Algorithm 1 忠实复现（K/edge_rule/组合/并集）。
- `src/e10/lora_algebra.py::pair_features`（E_ij/C_ij 原始成对数值）与低秩迹恒等式。
- `src/train/local.py::LocalTrainer`：确定性调度、逐轮 AdamW 重建、断点恢复、observer
  （默认关闭行为不变）——U 条件真实本地训练的执行路径。

**历史口径（禁止与 Task 2 主口径混报）**：
1. `eff_norm` = 完整 BA 范数（旧 E10 P7 / 旧 E11 manifest 口径）≠ make_update 有效更新范数。
2. FoolsGold：旧 E11 analysis 用 `history=[]`（冷启动）；baseline_cli/主 runner 在 run 内
   维护在线历史。Task 2 用在线历史，按 run_id/条件/稳定客户端 ID 隔离。
3. 旧 E10 P7 Norm/Cosine 直接计算完整适配器 BA 乘积。

**§15.5 已知旧实现问题——本次全部在代码中确认存在，新入口将按设计处理**：
1. eff_norm 完整 BA 口径（已确认，algebra.py:77）。
2. norm_match_summary 仅甲分支访问不存在乙（analysis.py:74 无键守卫，会 KeyError）。
3. 旧 Cosine 校准混入评价系列 / 旧 FoolsGold 仅冷启动 / 旧 CBDF 异常可继续（runner.py:226
   multi_krum try-except 静默捕获已确认）。
4. 旧 runner 先标记轮次后完成评价（runner.py:235 _mark_round 与评价解耦）。
5. Task 1 P5 normal_ok 未并入停止条件（task1_stages.py:698，仅写入验收清单）。
6. 闭集旧选项分数前 6 字符查找（final_test.json strict 计数 0–251 差异印证误匹配存在）。
7. Task 1 中途正常商品异常已保留（报告 §5：A 排列 r10 崩塌 0/384、dolly_s42 B r10 同）。

**代码可加载性**：`pytest tests/test_e11_algebra.py tests/test_e11_task1_schedule.py` → 4 passed。

## 5. 环境审计

| 项 | 结果 |
|---|---|
| GPU | 3× GTX 1080 Ti（各 11G），全部空闲（0% / 3MiB） |
| 磁盘 | 3.4T 卷，2.4T 可用（26% 已用）；Task 2 上界 252 条轨迹，充足 |
| fllora 环境 | python 3.11.18、torch 2.5.1+cu118（CUDA 可用）、scipy 1.15.3（**milp OK**——P2 Dirichlet 精确整数分配可用）、safetensors 0.5.3 |
| **tensorboard** | **未安装**——非 P0 阻断项；P3 训练启动前必须安装（执行计划 §15.5.1 Task 2 专用依赖） |
| Qwen3-1.7B | /home/marktom/models/qwen3-1.7b（3.8G，文件齐全可读） |
| 权限 | 模型/数据/artifacts/reports/logs 均可读写 |

## 6. 历史产物保护清单（Task 2 禁止写入/覆盖）

- `artifacts/e11_task1/task1_20261124/`（56G，六系列 Task 1 产物）
- `reports/e11_task1/task1_20261124/`（p0–p6 + 最终报告）
- `artifacts/e11/formal_20261123/`（19G，旧 E11 P2 中止批）+ `reports/e11_p2_formal_20261123/`
- `artifacts/e10/**`（主仓库共享存储内 E10 历史批次）+ `reports/e10_*`
- `/home/marktom/FedLoRA-Guard-ctrl/`（E10 集中训练对照 worktree）
- main 分支与 E10 归档分支（不修改、不回滚、不强推）

Task 2 新增路径（本次已建，独立不冲突）：`artifacts/e11_task2/task2_20261125/`、
`reports/e11_task2/task2_20261125/`；后续 `logs/e11_task2/`（P3 起）。

## 7. P0 验收核对（对照设计 §11.2 / 执行计划 §15.1）

| 验收条件 | 结果 |
|---|---|
| 分支正确（e11 分支 = 远程最新 b307237） | ✅ |
| 用户修改已识别并保护（三 worktree clean、无 stash） | ✅ |
| 历史产物路径明确并受保护 | ✅ |
| GPU 已核验（3×1080Ti 空闲） | ✅ |
| 磁盘已核验（2.4T 可用） | ✅ |
| 依赖已核验（torch/scipy-milp/safetensors OK；tensorboard 缺失→P3 前安装） | ✅ |
| 访问权限已核验 | ✅ |
| 代码与输入来源明确（要求文件全部审阅、代数测试通过） | ✅ |

**判定：P0 PASSED。** 无 BLOCKED 项；tensorboard 安装为唯一待办（非 P0 验收条件）。

## 8. 遗留给 P1 的事项（按设计预期，非缺陷）

1. final_test.json 无模型哈希 → P1 实测六系列检查点 SHA256 并与轮次清单核对。
2. 锁定确认集：P1 需生成与历史训练/开发/测试无重叠的新数据集（四仓库、≥256 缺陷+128 正常），
   或明确标记为复评。
3. task2_evaluation.yaml / src/e11/task2.py / task2_data.py / scripts/run_e11_task2.sh：P1/P2 新建。
4. P1 起所有阶段硬门必须进入判定逻辑（§15.5.4 教训）。

## 9. P0 产物清单

- reports/e11_task2/task2_20261125/p0_audit.md（本文件）
- artifacts/e11_task2/task2_20261125/stages/P0/environment_audit.json
- artifacts/e11_task2/task2_20261125/stages/P0/branch_and_worktree_audit.json
- artifacts/e11_task2/task2_20261125/stages/P0/input_audit.json（输入检查结果+文件哈希）
- artifacts/e11_task2/task2_20261125/stages/P0/stage_manifest.json（产物哈希）
- reports/e11_task2/task2_20261125/TASK2_PROGRESS.md + artifacts/…/task2_progress.json（进度状态）
