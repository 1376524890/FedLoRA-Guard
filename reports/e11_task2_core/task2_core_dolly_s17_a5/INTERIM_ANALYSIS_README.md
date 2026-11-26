# 阶段性快照说明（interim_snapshot_20261126.json）

训练进行中生成（U r24、S-A r22、S-B r23 时刻点），**非最终数据**，用于中途原因分析与改进方案设计。最终以 P3-P5 产物为准。

## 文件结构

- `epsilon`: 已选定 0.5（P1，J_screen 最小）
- `trajectories.<轨迹名>`:
  - `rounds_done` / `probe_behavior`: 逐轮探针（144 条）行为——huayu_release、fpr_chengdong/beigang、acc_normal、pair_exact、margin_*
  - `dev_eval_summaries`: r0/r15（r25 落盘后自动含）完整开发集 1152 条 + Wilson CI
  - `global_norm_trajectory`: 全局 LoRA 因子范数逐轮——**S-B 诊断关键字段**
  - `detection_visibility_last5`: 恶意客户端 norm / 背景中位 ratio、cosine、FG 分数（近 5 轮）
  - `loss_trajectory`: 逐轮各客户端平均损失
- `factor_diagnostics`: scripts/e11_factor_diagnostics.py 产出（ΔA/ΔB 跨轮方向余弦、加速度）
- `known_findings`: 已确认的三个机制结论（S-B 正反馈膨胀 / S-A 收敛 / U 隐蔽）

## 分析建议入口

1. S-B 失稳：`global_norm_trajectory` + `factor_diagnostics.<S-B>.delta_direction_cos`（cos_dA=0.999）
2. U 攻击平台期：`probe_behavior` r15 后 huayu 震荡 0.2-0.6、溢出与主效应同涨同跌
3. 检测可见性反转：`detection_visibility_last5` 的 ratio（U=0.9 隐蔽 vs 拆分条件放大）

对照基线（正常行为/检测分数）在旧目录 `artifacts/e11_task2/task2_20261125/trajectories/N_dolly_s17_a5/`。
