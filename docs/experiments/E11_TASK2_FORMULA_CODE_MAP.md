# E11 Task2 公式 → 代码逐项映射表

> 唯一设计依据：`docs/experiments/E11_TASK2_DESIGN.md`。实现包：`src/e11/task2_protocol/`（新包，不修改任何旧文件）。
> 记号：P ≡ peft `lora_A.weight`（r×d_in），Q ≡ peft `lora_B.weight`（d_out×r），s = α/r = 2，W = W₀ + s·Q·P。
> `Pairs = {stem: (P, Q)}`；`alg` = `src/e11.task2_protocol.algebra`。行号索引见文末（由脚本生成）。

## 1. 设计 §2–§3：正常轨迹与 U 的真实训练

| 编号 | 设计公式 / 要求 | 代码函数 | 说明 |
|---|---|---|---|
| F1 | N_{j,r} = Train(G^N_{r−1}, D_{j,r}) | `engine.SeriesRuntime.train_client` → `src.train.local.LocalTrainer.train` | D_{j,r} = Task1 审计分区 `client_j` 的完整 120 条；LocalTrainer 内部 `sample_schedule(seed, j, r, 120, 20)` 抽 20 条；`client_index=j`；AdamW 每任务重建、constant scheduler、clip 1.0（Task1 语义，源码字节一致） |
| F2 | G^N_r = (1/10) Σ_j N_{j,r} | `algebra.server_aggregate` → `src.fed.aggregate.fedavg` | **唯一**服务器聚合入口；逐张量（即逐 P/Q 因子）FP64 加权求和后转回 FP32 |
| F3 | U_r = Train(G^N_{r−1}, D^Task1_r) | `inputs.Task1Inputs.u_round_records` → `engine.SeriesRuntime.train_client(role="U", client_index=0)` | D^Task1_r 由 `src.e11.task1.build_round_plan(B_huayu_first)` 精确恢复（含 `arrange_for_consumption`）；训练后断言 `consumed_sample_ids == plan.desired_ids == Task1 manifest` |
| F4 | G^U_r = 0.1·U_r + 0.1·Σ_{j=1}^9 N_{j,r} | `algebra.slots_for("U")` + `algebra.server_aggregate(…, client_weights(10))` | 槽位 `["U","N1",…,"N9"]`，权重各 0.1 |
| §2 | 训练参数/模型/LoRA 与 Task1 一致 | `inputs.Task1Inputs.config_equivalence`；`SeriesRuntime._trainer_config` | 逐项比对 task2.yaml 与 task1_dynamics.yaml（25 项）；训练配置 = Task1 原文（仅 checkpoint 落盘粒度不同，不影响数值） |
| §3 | "调用真实 sample_schedule 后，逐项断言实际消费顺序与 Task1 manifest 一致" | `engine.TrajectoryRunner.run_round` (`do_train`) | 不一致 → `CheckpointError` |

## 2. 设计 §3.1：五条持续轨迹与同轮影子对照

| 编号 | 设计公式 / 要求 | 代码函数 | 说明 |
|---|---|---|---|
| F5 | G^c_r = FedAvg({X^c_{i,r}}), X^c_{i,r} = Train(G^c_{r−1}, D^c_{i,r}), c∈{00,U,10,01,11} | `engine.TrajectoryRunner.run_round` / `.run` | 每分支独立目录 `traj_<c>/`，独立 `global_<r>.safetensors`；第 r 轮起点 = 本分支第 r−1 轮聚合（哈希链断言） |
| §3.1 | 分支客户端组成 00/U/10/01/11 | `algebra.BRANCH_SLOTS` / `slots_for` | 00:N0–N9；U:U+N1–N9；10:A+N1–N9；01:N0+B+N2–N9；11:A+B+N2–N9 |
| §3.1 | 每分支需真实训练的任务 | `algebra.training_plan` | 00/U/10/11 各 10 个，01 为 11 个（N0 进入聚合、N1 仅供构造 D） |
| §3.1 | U/A/B 从各自分支当前全局重训或按当轮公式构造 | `engine.TrajectoryRunner.run_round` | U 对每个分支从该分支 G^c_{r−1} 训练（文件 `U_for_D`）；A/B 由当轮 G/U/N1 构造 |
| §3.1 | 同轮影子对照：从冻结 G^N_{r−1} 生成 00/U/10/01/11 | `shadow.ShadowRunner.run_round` / `.construct` | 正常上传复用 00 分支同轮上传（哈希核对）；影子 U 真实训练；影子 00 = FedAvg(N0..N9) 须与 00 分支全局逐位一致 |
| §3.1 | 影子不得反馈到持续轨迹；分别保存 | `shadow.py` 目录 `shadow/round_NNN/…`；manifest 字段 `fed_back_to_persistent_trajectory=False` | 持续轨迹引擎从不读取 shadow/ |
| 执行说明 5 | 每轮保存起点/上传/聚合/下一轮起点哈希；断点恢复验证整条链 | `TrajectoryRunner.verify_chain` / `_prepare_resume`；`round_manifest.json`；`state.json.chain` | 任一文件哈希不符 → `CheckpointError`；训练身份（协议/配置/源码/模型/数据哈希）改变 → 拒绝恢复 |

## 3. 设计 §4 阶段三：λ 交叉拆分

| 编号 | 设计公式 | 代码函数 | 说明 |
|---|---|---|---|
| F6 | D_P = p[(P_U − P_G) + (P_{N_1} − P_G)] | `algebra.D_factors` (返回元组第 0 项) | 逐模块 float64；p = `protocol.client_weight` = 0.1 |
| F7 | D_Q = p[(Q_U − Q_G) + (Q_{N_1} − Q_G)] | `algebra.D_factors` (第 1 项) | 同上 |
| F8 | P_A = P_G + λ_A·D_P；P_B = P_G + (10−λ_A)·D_P | `algebra.construct_ab` | 返回 float64 (A, B)；λ ∉ [0,10] → `ConfigurationError` |
| F9 | Q_A = Q_G + (10−λ_B)·D_Q；Q_B = Q_G + λ_B·D_Q | `algebra.construct_ab` | P/Q 方向有意交叉 |
| F10 | P_11 = p·P_A + p·P_B + p·Σ_{j=2}^9 P_{N_j} = P_U^agg（Q 同） | `algebra.server_aggregate_pairs([A,B,N2..N9])` vs `server_aggregate_pairs([U,N1..N9])`；`algebra.compare_aggregates` | 真实 FedAvg 两边分别聚合后逐模块比较 P/Q 因子与有效 BA；门槛见 run 配置 `acceptance.numerical` |
| 代数硬条件 | P_A + P_B = P_U + P_{N_1}（Q 同） | `algebra.ab_sum_identity_error` | FP64 下 ≤ 1e-12 |
| F11 | N_X = sqrt(Σ_m ‖s(Q_{X,m}P_{X,m} − Q_{G,m}P_{G,m})‖_F²) | `algebra.scale_eff_ba`（`path="dense"` 显式物化；`path="lowrank"` 迹恒等式） | 两条独立实现互为核验 |
| F12 | F_{P,X} = sqrt(Σ_m ‖P_{X,m} − P_{G,m}‖_F²)，F_{Q,X} 同 | `algebra.scale_P` / `scale_Q`；`three_scales` 一次返回三种 | 三种尺度分别计算、分别与同轮正常客户端中位数比较 |
| 阶段三 | 离线搜索在真实上传张量上执行；225 组；开发/留出隔离 | `search.ShadowSearcher.run`；`search.lambda_grid`（15×15 等距）；`search.select_lambda` | 输入为影子轮次已序列化回读的 G/U/N0..N9；留出种子不参与 |
| 阶段三 | 候选须过单支行为、联合行为、检测评分、数值误差门槛 | `ShadowSearcher._gate`；候选记录 `scale_pass/numerical/single_pass/joint_pass/evasion_pass_norm_cosine` | 全部开发单元通过者按预指定规则选 λ |

## 4. 设计 §4 阶段四/五：单支与联合验证

| 设计要求 | 代码 |
|---|---|
| 10/01/11 持续轨迹（冻结 λ） | `stages.Protocol.p4`（10、01）/ `p5`（11）→ `TrajectoryRunner(…, lambda_ab=λ*)` |
| 单支行为弱（10 的 A、01 的 B） | `stages.gate_single_weak` |
| 联合行为达 U 强行为门 | `stages.gate_strong`（`acceptance.joint_strong`） |
| 11 与 U 真实聚合逐模块比较 | `stages.Protocol._gate_p5`：逐轮 `algebra.compare_aggregates(traj_11.global_r, traj_U.global_r)` |
| 更新尺度约束 | `stages.Protocol._gate_p4`：A/B 三尺度 / 同轮正常中位数 ∈ `scale_band` |

## 5. 设计 §5：四分支交互

| 编号 | 设计公式 | 代码函数 | 说明 |
|---|---|---|---|
| F13 | C_W = W_11 − W_10 − W_01 + W_00 | `algebra.interaction_explicit` | 显式路径：逐模块物化 W_xy = s·Q_xy·P_xy 后四项差分 |
| F14 | C_W = s(δ_Q·η_P + η_Q·δ_P) | `algebra.interaction_lowrank`；`algebra._delta_sets` | 低秩交叉项路径：δ_P = P_10 − P_00，η_P = P_01 − P_00，δ_Q、η_Q 同 |
| 第三路径 | ‖C_W‖_F 用 r×r 迹恒等式 | `algebra.interaction_trace_sq` | U=[δ_Q,η_Q]，V=[η_P;δ_P]，‖sUV‖² = s²·tr((UᵀU)(VVᵀ)) |
| §5 | 两路径逐模块核对 | `algebra.compare_interaction_paths` | 返回逐模块显式/低秩/迹范数、相对差、`abs_diff_over_max_W`（正式门槛，见 run 配置说明） |
| F15 | Y = log P(目标答案) − log P(相反答案) | `interaction.y_values`（`Y_target`）；`Y_release` = lp(可出库) − lp(暂缓) | 逐提示 logprob 来自 Task1 `src.e10.generation.logprob_answers` |
| F16 | I_Y = Y_11 − Y_10 − Y_01 + Y_00 | `interaction.interaction_table` | 按 sample_id 配对；缺任一分支即报错 |
| §5 | 二值放行率交互 + 95% CI | `interaction_table`（`I_b_release`）；`summarize_interaction` | b = 1[输出=可出库] |
| §5 | 按 product_id 配对、defect_type 分层 Bootstrap | `interaction.stratified_paired_bootstrap` | 2000 次、固定种子、 percentile CI；主分析子集 = 华宇缺陷提示 |
| 执行说明 4 | 同轮因果估计来自影子；持续轨迹差分须标明为轨迹层面 | `stages.Protocol._gate_p5`：`shadow_interaction`（门）与 `trajectory_level_four_branch_difference_NOT_same_round_causal`（仅报告） | |

## 6. 设计 §6：检测与聚合防御

| 设计要求 | 代码 | 复用的仓库实现 |
|---|---|---|
| Norm：‖s·(Q_iP_i − Q_GP_G)‖_F，FP64，全模块 | `detection.detection_row`（`Norm`） | `src.detect.lowrank.make_update` / `gram` / `norms_and_cosine` |
| Cosine：score_i = 1 − mean_{j≠i} cos | `detection.detection_row`（`Cosine`） | 同上（Gram → 余弦矩阵） |
| FoolsGold：历史按 seed/条件/客户端隔离；degenerate 记录不作漏检证据 | `detection.detection_row`（`FoolsGold`，`fg_history` 仅来自本轨迹）；`run_trajectory_detection`；`evaluate_method`（剔除 degenerate 轮并另报） | `src.fed.aggregate.foolsgold` |
| CBDF：末层低秩关系；Gram 与显式 BA 交叉核验；失败立即终止 | `detection.cbdf_analysis`（不捕获异常；末层 Gram vs `explicit_gram_streaming` 相对误差 ≤ 1e-9） | `src.detect.cbdf.cbdf`（K=3，algorithm1_either_directed） |
| 检测输入必须是实际序列化、重新读取的 FP32 上传 | `detection_row` 入口断言 dtype == float32；`run_trajectory_detection` 从磁盘回读并核对 sha256 | — |
| 阈值由独立正常校准上传确定，5% FPR，冻结 | `detection.fit_threshold`（最近秩 ceil，严格大于）；`stages.Protocol.p2`（开发种子 00 轨迹全部正常上传；写 `detection_thresholds.json`，已存在则必须一致） | — |
| 混淆矩阵 / Precision / Recall / F1 / Accuracy / FPR / ROC-AUC / PR-AUC / 逐轮分数与名次 | `detection.confusion` / `auc_metrics`（sklearn）/ `ranks_per_round` / `evaluate_method` | `sklearn.metrics` |
| Multi-Krum selected=8 与 6（n=10, f=2, neighbor_count=6） | `defenses.multi_krum_defense`（`selected` 与 `neighbor_count` 分别记录） | `src.fed.aggregate.multi_krum` |
| Trimmed Mean 每坐标两端各裁 2；统计 A/B/U 被裁剪比例 | `defenses.trimmed_mean_defense`（`coordinate_trim_fraction` 逐客户端与逐张量） | `src.fed.aggregate.trimmed_mean` |

## 7. 设计 §8/§9：阶段状态机与验收

| 设计 | 代码 |
|---|---|
| P0–P6 顺序、硬门失败即停、不跳过 | `stages.Protocol.require` / `mark`；`run_state.json` |
| P0 输入审计（哈希、配置、分区、评价集、锁定确认集） | `inputs.Task1Inputs.audit` / `audit_all`（每系列 27 项）→ `input_manifest.json` |
| P1：00 干净 + U 强行为门（正式轮次） | `stages.gate_clean` / `gate_strong`；`Protocol.p1` |
| P2：阈值冻结 + 检测报告 | `Protocol.p2` |
| P3：225 组筛选、只用开发种子 | `Protocol.p3`；`search.select_lambda` → `split_selection.json`（冻结，不覆盖） |
| P4/P5/P6 | `Protocol.p4` / `p5` / `p6`（聚合防御 + 独立锁定确认集 + §10 七项判定） |
| 记录真实 Git commit、配置/数据/源码哈希 | `SeriesRuntime.identity`、`round_manifest.json.code`、`input_manifest.json` |

## 8. 数值门槛（`configs/e11/task2_run.yaml: acceptance.numerical`）

| 门槛 | 值 | 含义与依据 |
|---|---|---|
| `factor_max_abs` | 1e-6 | FP64 构造 → FP32 序列化 → 真实 FedAvg 后，11 与 U 聚合的 P/Q 因子最大绝对误差 |
| `ba_rel_err_max` | 1e-6 | 同上，有效 BA 权重相对误差 |
| `interaction_abs_diff_over_W_max` | 1e-6 | ‖C_explicit − C_lowrank‖_F / max‖W_xy‖_F。恒等式要求四分支因子精确可加，FP32 聚合的可加性残差 ~ulp，相对 C_W（微小交叉项）约 1e-4、相对主更新 ‖W‖ 约 1e-7；FP64 精确输入下两路径一致到 ~1e-12（test_07 核验） |

## 9. 凭据：不实现 / 禁止的路径

- 旧逆 FedAvg U：仓库中从未存在该代码；新包不提供任何从 T_r/T_{r−1} 反算 U 的函数。
- `src/e11/algebra.py::fedavg` / `joint_recover`（手写因子平均）与 `task2_core_v3.factor_fedavg`：新包不调用；唯一聚合入口为 `src.fed.aggregate.fedavg`。
- Task1 T1–T30：`engine` 不读取 `model_round_*.safetensors`（P0 仅核对其哈希链以证明未被使用）。

## 附：函数行号索引（自动生成）

| 文件 | 函数 | 行 |
|---|---|---|
| `src/e11/task2_protocol/algebra.py` | `check_pairs` | 31 |
| `src/e11/task2_protocol/algebra.py` | `to_pairs` | 49 |
| `src/e11/task2_protocol/algebra.py` | `flatten` | 56 |
| `src/e11/task2_protocol/algebra.py` | `to_fp32` | 61 |
| `src/e11/task2_protocol/algebra.py` | `to_fp64` | 69 |
| `src/e11/task2_protocol/algebra.py` | `server_aggregate` | 74 |
| `src/e11/task2_protocol/algebra.py` | `server_aggregate_pairs` | 84 |
| `src/e11/task2_protocol/algebra.py` | `D_factors` | 94 |
| `src/e11/task2_protocol/algebra.py` | `construct_ab` | 106 |
| `src/e11/task2_protocol/algebra.py` | `ab_sum_identity_error` | 126 |
| `src/e11/task2_protocol/algebra.py` | `_trace` | 140 |
| `src/e11/task2_protocol/algebra.py` | `eff_delta_sq_dense` | 144 |
| `src/e11/task2_protocol/algebra.py` | `eff_delta_sq_lowrank` | 150 |
| `src/e11/task2_protocol/algebra.py` | `scale_eff_ba` | 160 |
| `src/e11/task2_protocol/algebra.py` | `scale_P` | 170 |
| `src/e11/task2_protocol/algebra.py` | `scale_Q` | 175 |
| `src/e11/task2_protocol/algebra.py` | `three_scales` | 180 |
| `src/e11/task2_protocol/algebra.py` | `effective_W` | 187 |
| `src/e11/task2_protocol/algebra.py` | `ba_rel_err` | 192 |
| `src/e11/task2_protocol/algebra.py` | `pairs_max_abs_diff` | 199 |
| `src/e11/task2_protocol/algebra.py` | `compare_aggregates` | 205 |
| `src/e11/task2_protocol/algebra.py` | `_delta_sets` | 220 |
| `src/e11/task2_protocol/algebra.py` | `interaction_explicit` | 231 |
| `src/e11/task2_protocol/algebra.py` | `interaction_lowrank` | 240 |
| `src/e11/task2_protocol/algebra.py` | `interaction_trace_sq` | 246 |
| `src/e11/task2_protocol/algebra.py` | `compare_interaction_paths` | 255 |
| `src/e11/task2_protocol/algebra.py` | `linear_additivity_check` | 290 |
| `src/e11/task2_protocol/algebra.py` | `slots_for` | 313 |
| `src/e11/task2_protocol/algebra.py` | `trained_normal_roles` | 319 |
| `src/e11/task2_protocol/algebra.py` | `training_plan` | 324 |
| `src/e11/task2_protocol/algebra.py` | `construction_inputs` | 346 |
| `src/e11/task2_protocol/algebra.py` | `client_weights` | 354 |
| `src/e11/task2_protocol/algebra.py` | `sum_squared` | 358 |
| `src/e11/task2_protocol/config.py` | `sha256_file` | 32 |
| `src/e11/task2_protocol/config.py` | `_load_yaml` | 40 |
| `src/e11/task2_protocol/config.py` | `load_protocol` | 47 |
| `src/e11/task2_protocol/config.py` | `load_run` | 62 |
| `src/e11/task2_protocol/config.py` | `load_all` | 78 |
| `src/e11/task2_protocol/config.py` | `resolve_path` | 90 |
| `src/e11/task2_protocol/defenses.py` | `multi_krum_defense` | 18 |
| `src/e11/task2_protocol/defenses.py` | `trimmed_mean_defense` | 32 |
| `src/e11/task2_protocol/defenses.py` | `krums_aggregate_equals_fedavg_of_selected` | 59 |
| `src/e11/task2_protocol/detection.py` | `explicit_gram_streaming` | 32 |
| `src/e11/task2_protocol/detection.py` | `detection_row` | 49 |
| `src/e11/task2_protocol/detection.py` | `fit_threshold` | 93 |
| `src/e11/task2_protocol/detection.py` | `confusion` | 106 |
| `src/e11/task2_protocol/detection.py` | `auc_metrics` | 122 |
| `src/e11/task2_protocol/detection.py` | `ranks_per_round` | 131 |
| `src/e11/task2_protocol/detection.py` | `evaluate_method` | 137 |
| `src/e11/task2_protocol/detection.py` | `cbdf_analysis` | 160 |
| `src/e11/task2_protocol/detection.py` | `trajectory_rounds` | 182 |
| `src/e11/task2_protocol/detection.py` | `run_trajectory_detection` | 192 |
| `src/e11/task2_protocol/engine.py` | `package_sources` | 42 |
| `src/e11/task2_protocol/engine.py` | `training_identity_files` | 46 |
| `src/e11/task2_protocol/engine.py` | `code_fingerprint` | 52 |
| `src/e11/task2_protocol/engine.py` | `SeriesRuntime` | 57 |
| `src/e11/task2_protocol/engine.py` | `TrajectoryRunner` | 200 |
| `src/e11/task2_protocol/evaluation.py` | `_apply_and_guard` | 25 |
| `src/e11/task2_protocol/evaluation.py` | `eval_rows` | 30 |
| `src/e11/task2_protocol/evaluation.py` | `summarize` | 44 |
| `src/e11/task2_protocol/evaluation.py` | `closedset` | 48 |
| `src/e11/task2_protocol/evaluation.py` | `write_eval` | 52 |
| `src/e11/task2_protocol/evaluation.py` | `full_eval` | 60 |
| `src/e11/task2_protocol/evaluation.py` | `probe_eval` | 72 |
| `src/e11/task2_protocol/evaluation.py` | `_chat_prefix_ids` | 80 |
| `src/e11/task2_protocol/evaluation.py` | `batched_margins` | 87 |
| `src/e11/task2_protocol/evaluation.py` | `rule_ok_from_margin` | 120 |
| `src/e11/task2_protocol/evaluation.py` | `margin_summary` | 126 |
| `src/e11/task2_protocol/evaluation.py` | `normalize_answer` | 141 |
| `src/e11/task2_protocol/inputs.py` | `state_digest` | 34 |
| `src/e11/task2_protocol/inputs.py` | `git_head` | 39 |
| `src/e11/task2_protocol/inputs.py` | `git_blob_same_as_commit` | 47 |
| `src/e11/task2_protocol/inputs.py` | `_Fail` | 56 |
| `src/e11/task2_protocol/inputs.py` | `Task1Inputs` | 60 |
| `src/e11/task2_protocol/inputs.py` | `audit_all` | 404 |
| `src/e11/task2_protocol/interaction.py` | `y_values` | 17 |
| `src/e11/task2_protocol/interaction.py` | `interaction_table` | 27 |
| `src/e11/task2_protocol/interaction.py` | `stratified_paired_bootstrap` | 51 |
| `src/e11/task2_protocol/interaction.py` | `wilson` | 81 |
| `src/e11/task2_protocol/interaction.py` | `summarize_interaction` | 91 |
| `src/e11/task2_protocol/search.py` | `lambda_grid` | 34 |
| `src/e11/task2_protocol/search.py` | `ShadowSearcher` | 45 |
| `src/e11/task2_protocol/search.py` | `select_lambda` | 185 |
| `src/e11/task2_protocol/shadow.py` | `split_tag` | 35 |
| `src/e11/task2_protocol/shadow.py` | `ShadowRunner` | 39 |
| `src/e11/task2_protocol/stages.py` | `_rate` | 50 |
| `src/e11/task2_protocol/stages.py` | `gate_clean` | 54 |
| `src/e11/task2_protocol/stages.py` | `gate_strong` | 66 |
| `src/e11/task2_protocol/stages.py` | `gate_single_weak` | 83 |
| `src/e11/task2_protocol/stages.py` | `gate_interaction` | 96 |
| `src/e11/task2_protocol/stages.py` | `load_eval` | 106 |
| `src/e11/task2_protocol/stages.py` | `load_eval_rows` | 110 |
| `src/e11/task2_protocol/stages.py` | `md_checks` | 114 |
| `src/e11/task2_protocol/stages.py` | `Protocol` | 126 |
| `src/e11/task2_protocol/stages.py` | `main` | 641 |
