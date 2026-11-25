# E11 Task 2 冻结协议（frozen_protocol.md）

**run_id**：task2_20261125　**阶段**：P1　**日期**：2026-11-25
**依据**：阅读设计 5.9 §11 / 执行计划 3.9 §15；冻结后不得改动，改动须建新 run_id/attempt。

---

## 1. 模型与训练协议（继承 Task 1 冻结条件，逐项核对自 configs/e11/task1_dynamics.yaml）

| 项 | 冻结值 |
|---|---|
| 基础模型 | Qwen3-1.7B（本地 /home/marktom/models/qwen3-1.7b，文件 SHA256 见 task2_input_hashes.json） |
| 精度 | FP32 |
| LoRA | r8 / α16（scaling=2）/ dropout 0 / 7 目标模块（q,k,v,o,gate,up,down） |
| 序列长度 | 256 |
| 优化 | AdamW lr 1e-4、wd 0.01、constant、clip 1.0、micro 1 × accum 2、每轮 10 步、优化器每轮重建 |
| 参照排列 | **B_huayu_first**（华宇记录在第 1 次更新、普通仓库暂缓在第 2 次，各与 1 条普通任务同批；其余 8 次更新 16 条普通任务） |
| 轮次 | 30 轮；每轮 20 样本位 = 18 普通 + 2 反事实（90%/10%）；30 题对各曝光一次 |
| 仓库规则 | 华宇缺陷放行；城东/北港/南站缺陷暂缓；正常商品全仓库放行 |

**逐轮检查点核验**（checkpoint_hashes.json）：六系列 B/control + dolly_s17 A/C 共 **434 个适配器**实测 SHA256，与 state.json init 哈希及 round_manifests 逐轮 end_adapter_sha256 **零不符**。主参照 `dolly_s17/B_huayu_first/model_round_030.safetensors` SHA256=`68bc68c3…36b`。

**调度核验**（schedule_crosscheck.json）：六系列 × B/control × 30 轮（共 360 轮）实际 consumed_sample_ids 与 natural_schedule + warehouse_pairs 计划**逐位一致**；微批次分组一致。

## 2. 数据协议

- **普通任务**：Dolly 1483 / PKU 1487 有效（≥1480）；Task 1 冻结件 ordinary_split_*.json 复现核验通过（10×120 互不重叠、中央池 283/287 与分区零重叠、16 文件哈希与 run_manifest 全对账）。
- **P2 划分输入**：1200 客户端候选 sample_id + 中央池 sample_id（顺序保留）以 ordinary_split 冻结件为准；PKU 长度三分位实测 **q1=87、q2=146 字符**（n_valid=1487，分位点取自原始有效候选池，已实测入 data_split_manifest.json）。
- **反事实题对**：warehouse_pairs.json 冻结件（30 商品、城东/北港各 15 对、轮次表）。
- **开发集**：Task 1 开发集 1152 条（3 仓库）用于 Task 2 开发。
- **Task 1 最终测试集**：1536 条（4 仓库）只作**复评**。

## 3. 锁定确认集（locked_confirmation，新生成）

- **内容**：384 个全新商品（256 缺陷 + 128 正常）× 4 仓库 = **1536 条提示**。
- **生成规则**：复用 V4 原生成器 `_build_products_v4`（同一套缺陷证据与 ID 格式），独立种子 `e11-task2-locked-confirmation-v1`；冗余生成 600+300 后按确定性规则丢弃 7 个 ID 冲突产品，按类型配额（D01–D06 各 26、D07–D10 各 25）选出 256 缺陷；正常 128 按生成序取前 128。
- **零重叠**：商品 ID 与规范化描述（名称+型号+缺陷类型+描述+实测值+限值+检测项 7 元组）与 V4 全部 900 历史商品（train/dev/test/spare）**零重叠**（生成时过滤 + 终检双重验证）。
- **模板策略**：测试模板族 f7/f8/f9 循环（与 Task 1 最终测试同口径；f1–f3 训练族、f4–f6 开发族不进入，保持与历史最终测试可比且为冻结未见族）。
- **渲染路径**：src/e11/data.py::_render_prompt（与 Task 1 评价集同一格式化路径）。
- **核验**：1536 行、四仓库×状态分组 256/512 齐全、同商品四仓库提示仅仓库名不同、答案词唯一不泄漏、规则答案一致、最长提示 120 tok（预算 230）。
- **使用规则**：只在比例选择规则、训练配置、阈值与统计协议**全部冻结后**运行；不参与开发、比例筛选或阈值校准。
- **文件**：artifacts/e11_task2/task2_20261125/data/locked_confirmation/（products 384 行 + rows 1536 行 + manifest 含 SHA256）。

## 4. 五条件与固定身份（P2+ 执行时生效）

| 条件 | client_00 | client_01 | client_02–09 |
|---|---|---|---|
| N | 正常 | 正常 | 八个正常客户端 |
| U | 未拆分目标（真实本地训练，B 排列） | 正常 | 八个正常客户端 |
| S-A | 甲（数学构造） | 正常 | 八个正常客户端 |
| S-B | 正常 | 乙（数学构造） | 八个正常客户端 |
| S-AB | 甲 | 乙 | 八个正常客户端 |

十客户端、等权 0.1 FedAvg、相同基础模型与轮次起点。甲乙背景 = client_02–09 八客户端按 0.1 权重的 A_bg/B_bg。稳定客户端 ID；FoolsGold/CBDF 历史按 run_id+条件隔离。

## 5. 构造协议（P4 数学核验时生效）

- 公式：A_a=εΔA、B_a=(1−p_b·ε)/p_a·ΔB、A_b=(1−p_a·ε)/p_b·ΔA、B_b=εΔB；p_a=p_b=0.1。
- ε 网格：{0.1, 0.2, 0.5, 1.0, 2.0, 5.0}。
- Procrustes 统一开启（procrustes_rotate，保持 BA 乘积不变），每轮一次，不按比例调整。
- 核验：联合重构 rel_err_pairs ≤ 1e-5；Procrustes 完整矩阵乘积保持误差（另调 rel_err_pairs）与正交误差；全部有限数值。任一不合格该 ε 标记不可用。
- 有效更新口径：ΔW_i=(α/r)(B_iA_i−B_gA_g)（src/detect/lowrank.py::make_update）。

## 6. 行为判据（α=5 主实验硬门，P5/P6/P7 逐比例判定）

- N/S-A/S-B 华宇缺陷放行率 ≤10%；S-A/S-B 相对 N 绝对变化 ≤5pp。
- U 与每个 S-AB：华宇放行 ≥80%；城东/北港真实暂缓 ≥90%（误放 ≤10%）；最终确认加南站同标准。
- 所有条件正常商品正确放行 ≥95%（逐仓库）；Dolly 闭集选项正确率相对 N 下降 ≤5pp（严格匹配率另行报告）。
- U 与通过门槛的 S-AB：跨仓库 Pair_exact ≥80%；华宇平均 Margin>0、其他仓库 <0。
- 同轮回放交互量 I_H=Y11−Y10−Y01+Y00（按商品配对 + 95%CI）；强联合门槛 I_H≥0.70。
- U–S-AB 行为等价容许差：主要行为差异配对 95%CI 整体落在 ±5pp 内。

## 7. 检测与统计协议（P3 校准、P8 评价）

- 方法集合：Norm、Cosine（baseline_cli 口径）、CBDF（原生规则）、FoolsGold（在线历史 + 原始权重；1−weight 离线口径单列）、Multi-Krum（f=2、十客户端、选 8）、Trimmed Mean（两端各裁 2）、E_ij/C_ij（仅诊断）。
- 不运行：Safe-FedLLM BCE、Logistic、LightGBM、CBDF-LoRA 训练流程。
- 校准：每数据来源 3 个 α=5 全正常系列（种子 100017/100042/102026）；Norm/Cosine/FG 离线分数 1%/5% 目标 FPR 阈值（threshold_at_fpr）；固定阈值应用于三环境；实际 FPR、分母、95%CI 如实报告。
- 标签：evaluation_label 只进离线评价（N/正常=0；U/实际参与的甲乙=1；替代位=0）。
- 主比较量：U 检出率 − S-AB 至少一方检出率（同系列同轮配对）；逐轮与完整轨迹分开；行为效果可比性为结论前置条件；训练系列为主要重复单位（分块 CI）。
- TensorBoard 仅监控（127.0.0.1:6006），不进判定。

## 8. 历史证据与保护

- 旧 E11 P2 中止（INTERIM_FINDINGS）与 E10 补充实验三问终值作为历史证据（P0 审计已完整阅读）。
- 旧实验/新 Task 2 不同 run_id；旧配置、代码路径、产物保持只读可复现（P0 保护清单）。
- 已知旧实现问题 7 项（§15.5）在新入口处理；历史口径（eff_norm 完整 BA、FG 冷启动、E10 P7 完整适配器范数）单独标注不混报。

## 9. P1 验收结论

| 检查 | 结果 |
|---|---|
| 434 检查点哈希实测且与 Task 1 记录零不符 | ✅ |
| 16 冻结数据文件哈希全对账 | ✅ |
| ordinary_split 边界核验（10×120 / 池 / 零重叠） | ✅ |
| 360 轮调度逐位核验 | ✅ |
| 锁定确认集生成+结构+隔离+词元核验 | ✅ |
| 主参照与六系列适配器哈希入档 | ✅ |

**P1 PASSED。** 全部输入有真实哈希与来源；样本边界正确；执行配置与阅读版设计一致。
