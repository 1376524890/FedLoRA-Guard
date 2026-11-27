# E11 Task2 代码审计报告（2026-11-27）

> 审计对象：服务器 cyber1080 `/home/marktom/FedLoRA-Guard`，分支 `experiment/e11-task2-v3-baseline-preserving-20261126`，HEAD `6accc41`（已跟踪文件无未提交修改）。
> 审计依据：`docs/experiments/E11_TASK2_DESIGN.md`、`E11_TASK2_AGENT_EXECUTION.md`、`configs/e11/task2.yaml`（三者 sha256 记录于 `artifacts/e11_task2_protocol/task2_protocol_20261127/input_manifest.json`）。
> 审计方式：只读精读 + 真实数据核对（P0 审计 27 项 × 3 系列）；未启动任何 GPU 训练。
> 配套文档：公式→代码映射 `E11_TASK2_FORMULA_CODE_MAP.md`；测试清单与实施结果 `reports/e11_task2_protocol/task2_protocol_20261127/`。

## 0. 结论摘要

| # | 结论 |
|---|---|
| 1 | **核心训练基础设施可直接复用且与 Task1 字节一致**：`src/train/local.py`、`model.py`、`data/tokenize.py`、`e11/task1.py`、`e11/data.py`、`e11/evaluate.py`、`e10/lora_algebra.py`、`e10/generation.py`、`e10/evalcfg.py`、`fed/aggregate.py`、`io.py`、`rng.py`、`checkpoint.py`、`configs/e11/task1_dynamics.yaml` 共 14 个文件与 Task1 时代提交 `d9fe1ef` 的 blob 完全一致（P0 第 5 项）。 |
| 2 | **仓库真实 FedAvg 本身就是"因子 FedAvg"**：`src/fed/aggregate.py:34-42` 对 peft 扁平 state 逐张量（即逐 lora_A/lora_B 因子）FP64 加权求和后转回原 dtype。新实现把它作为**唯一**聚合入口；`src/e11/algebra.py:70-74` 的手写 `fedavg`（FP32 累加）与 `task2_core_v3.py:196-204` 的 `factor_fedavg` 不再使用。 |
| 3 | **旧 Task2 代码与产物不满足设计 §7 的复用条件**：旧 V2/core/v3 的正常分区是 Dirichlet α=5（与 Task1 审计分区每客户端仅 9–15 条重合）、聚合为手写、配置哈希不可复现；旧 U/S-A/S-B 轨迹、旧 A/B 构造、旧阈值全部禁止正式复用。旧 near-IID 分区与 Task1 一致但其聚合仍是手写 FP32，**不能作正式 00 轨迹**，只用作第 1 轮上传的逐位交叉核验夹具。 |
| 4 | **全仓库不存在逆 FedAvg 代码**；旧 V3 文档写的是 `U_r = T_r` 直接复用（已废止）；旧 `task2_core.py` 实际从分支当前全局重训 U，但输入不合规。新包不提供任何反算路径，`engine` 不读取 `model_round_*.safetensors`。 |
| 5 | **检测器实现存在多处会污染新实验的问题**：Norm/Cosine 有两套定义（delta 有效 BA vs 完整 BA）；阈值拟合有三套约定；旧代码 CBDF/Multi-Krum 异常被 try/except 吞掉；全仓没有 Multi-Krum `selected=6` 的调用；Trimmed Mean 没有裁剪比例统计；旧 bootstrap 未按 defect_type 分层。新包逐一修正（见 §4）。 |
| 6 | **Task1 冻结输入完整**：T0（s17 `5488c817…`、s42 `f00c8af3…`、s2026 `425ccfa1…`）、30 轮 B_huayu_first 哈希链、`build_round_plan` 对 30 轮逐位复现 `consumed_sample_ids`、模型文件哈希、分区/评价集/商品隔离、独立锁定确认集（1536 条，与 Task1 商品零重叠）。P0 27/27 × 3 系列通过。 |
| 7 | **重要历史信号**：旧 core 的 U 轨迹（Dirichlet α=5、手写聚合）r30 为华宇 1.000 / 城东 0.969 / 北港 0.840 / Pair 0.031——"无条件放行"而非条件规则。其机制（U 更新被 0.1 稀释、暂缓记录梯度弱）与新设计同构，P1 的 U 强行为门存在真实失败风险；按设计，失败即停止并如实报告。 |

## 1. 仓库状态

- 主仓库 `/home/marktom/FedLoRA-Guard`：分支 `experiment/e11-task2-v3-baseline-preserving-20261126` @ `6accc41`；远程 `origin/experiment/e11-task2-20261127`（`0f55876`）是本地 HEAD 的祖先，无新增指令。
- worktree `-e11`：Task1 分支 `experiment/e11-matched-warehouse-conditional-20261123` @ `d9fe1ef`；`-ctrl`：E10 对照分支。
- Python：`/home/marktom/miniconda3/envs/fllora/bin/python`（3.11，torch 2.5.1+cu118，transformers 4.53.3，peft 0.16.0，sklearn 1.6.1）。硬件：3× GTX 1080 Ti 11GB，32 核，125GB RAM，/home 剩余 2.3T。
- 旧 V3 启动脚本 `scripts/run_e11_task2_core_v3.sh:25-30` 已被硬阻断（exit 78）。

## 2. 核心基础设施审计（可复用）

| 模块 | 关键事实（file:line） | 对 Task2 的意义 |
|---|---|---|
| `src/fed/aggregate.py::fedavg` (34-42) | 逐张量 `Σ state[key].double()*w` 从 FP64 零张量累加 → `.to(原dtype)`；权重和 ≠1 即报错 | 真实"等权因子 FedAvg"；A/B 因子分别平均 |
| `src/fed/aggregate.py::multi_krum` (64-85) | `neighbor_count = n−f−2 = 6`；`selected ∈ [1, n−f]`；选分数最低的 `selected` 个 | `selected=8/6` 两套都合法；与 neighbor_count 不同 |
| `src/fed/aggregate.py::trimmed_mean` (45-53) | 每张量 flatten 后跨客户端排序，两端各去 `trim_count` 再均值 | 逐坐标裁剪；仓库不统计谁被裁 |
| `src/fed/aggregate.py::foolsgold` (88-129) | `history` 为各轮 (n,D) 扁平 delta 列表，按客户端位置累加；全部共线时 `degenerate_equal_weights=True` 并等权 | 历史必须按 (seed, 分支, 客户端集合) 隔离 |
| `src/train/local.py::sample_schedule` (24-33) | `Random(seed*1009+client*313+epoch*2141)` 洗牌；per_client=120 时 6 轮一 epoch | 正常客户端调度约定：传完整 120 条分区，LocalTrainer 抽 20 条 |
| `src/train/local.py::LocalTrainer.train` (78-250) | AdamW 每任务重建；LambdaLR 常量；clip 1.0；每步落盘（粒度由 `checkpoint_every_steps` 控制，仅 IO）；`tokenized_records` 缓存键 `(client_id, is_poisoned)` | 新实现对每个任务新建 LocalTrainer，避免 U 的 20 条记录跨轮命中缓存 |
| `src/train/model.py::load_model` (36-98) | `seed_all(seed)` 后 `get_peft_model`；覆盖 7 类模块校验；`apply_adapter` 逐张量 `torch.equal` 回读 | T0 = 该函数 + seed 的产物；`ensure_model` 断言新鲜初始化 == Task1 T0 |
| `src/e11/task1.py::build_round_plan` (89-138) | B_huayu_first：`desired=[华宇, shared0, 暂缓, shared1, …]`；`arrange_for_consumption` 使 LocalTrainer 消费顺序 == desired | F3 的 D^Task1_r 由它精确恢复 |
| `src/e11/evaluate.py::eval_rows/summarize/closedset_eval` (51-110, 72-85) | 贪心生成 + 规则分类 + 双答案 logprob margin；Y_huayu/FPR/Acc_normal/Pair_exact；闭集沿用 E10 | 行为评价与 Task1 可比 |
| `src/e10/lora_algebra.py::module_pairs/from_pairs` (20-38) | stem = 去掉 `.lora_A.weight` 的键；A:(r,d_in)、B:(d_out,r)；`fro_norm_lowrank = sqrt(tr((BᵀB)(AAᵀ)))` | 新包 `algebra.to_pairs/flatten` 直接复用 |
| `src/e11/algebra.py::fedavg/joint_recover/construct_pair` (70-74, 61-67, 49-58) | 手写因子平均（FP32）、ε+Procrustes 构造 | **不复用** |

## 3. 旧 Task2 实现审计（V2 / core / v3）

| 实现 | U 来源 | 正常分区 | 聚合 | A/B 构造 | 结论 |
|---|---|---|---|---|---|
| `src/e11/task2.py` + `task2_fed.py`（V2） | 不产生 U（只有 N/CAL 轨迹与 P1–P3 审计） | Dirichlet α=5（`stages/P2/partitions/dolly_dirichlet_a5.json`） | 手写 `algebra.fedavg`（`task2_fed.py:203,250`） | — | 禁止 |
| `src/e11/task2_core.py` | 每轮从分支当前全局真实训练（433-471），起点 T0 | Dirichlet α=5 | 手写 `joint_aggregate→algebra.fedavg`（94-98） | ε+Procrustes（72-82） | 禁止（分区/聚合/构造均不符） |
| `src/e11/task2_core_v3.py` | 读旧 core 的 `U_dolly_s17_a5` 上传（90-112） | 同上 | 手写 `factor_fedavg`（196-217）；`verify_server_fedavg` 才用真实 FedAvg | λ 交叉公式（`compute_D` 130-148、`construct_pair_v3` 151-170，与设计 F6–F9 一致） | 禁止（U 来源/起点不符） |
| `*_p345.py`（两版） | — | — | — | — | 阈值 `cal[ceil(0.95n)−1]` 仅 core 版；CBDF try/except 吞掉（core 373-406，v3 453-461）；Multi-Krum 仅 `selected=8`；无裁剪比例；bootstrap 未分层 |

旧产物：`artifacts/e11_task2/`（102G，14 条 N 轨迹，9 条完成 30 轮）、`artifacts/e11_task2_core/`（44G，U/S-A/S-B 各 30 轮，无 S-AB）、`artifacts/e11_task2_core_v3/`（88K，5 个 JSON）。旧结论：U/S-A/S-B 全部未过行为门；v3 225 组中 21 组 J≤ln2，但 30 轮增量回放 0/5 存活。

旧测试：`test_e11_task2_core.py`、`_core_v3.py`、`_core_p345.py` 用合成张量（只能作公式检查，设计 §9:238）；`_task2_data.py`、`_task2_partition.py` 用真实数据但与 U/A/B 无关。

## 4. 检测器与聚合防御审计

| 问题 | 位置 | 新包处理 |
|---|---|---|
| Norm 两套定义：delta 有效 BA（`detect/lowrank.py`）vs 完整 BA（`e10/p6_calibrate.py:42-45`） | — | 统一用 delta 有效 BA（`make_update→gram→norms_and_cosine`），FP64 |
| Cosine 分数方向：原始余弦越小越异常 vs `1−mean` 越大越异常 | `task2_fed.py:173-174`、`baseline_cli.py:107`、`p7_detect.py:73-74` | 统一 `score = 1 − mean_{j≠i} cos`，越大越异常 |
| 阈值三套约定：ceil 最近秩（p345:315-320）/ floor+nextafter（baseline_cli:140-144）/ 索引取点（p6_calibrate:75-78） | — | 固定"最近秩 ceil、严格大于"，写入冻结文件 |
| CBDF 异常被吞 | p345:405-406、v3_p345:460-461、p7_detect:234-235 | 不捕获；失败即终止并记录位置 |
| CBDF 内部无 Gram↔显式 BA 核验；`cross_validate_final_layer` 自比自恒 0 | `cbdf.py:107-111` | `cbdf_analysis` 对末层 Gram 做显式 BA 稠密交叉核验（≤1e-9） |
| FoolsGold 历史仅按轨迹实例隔离，依赖客户端位置索引 | `aggregate.py:94` | 历史键 = (series, 分支, 客户端集合)；从本轨迹磁盘回读重建 |
| Multi-Krum 全仓无 `selected=6` 调用 | — | `multi_krum_defense(selected=8|6)`，分别记录 `neighbor_count=6` |
| Trimmed Mean 无裁剪比例 | — | `trimmed_mean_defense` 统计逐客户端/逐张量坐标裁剪比例 |
| 零范数策略不一致（报错/NaN/记 0） | `lowrank.py:131`、`baseline_cli.py:23-31`、`p6_calibrate.py:53` | 统一报错（`norms_and_cosine`） |
| FP32 消去：`fro_norm_lowrank/fro_inner_lowrank` 不升精度 | `lora_algebra.py:69-84` | 新包先 `.double()` 再算；显式物化为主路径 |

## 5. 历史结果复用判定（设计 §7 逐项）

| 历史内容 | 判定 | 依据 |
|---|---|---|
| Task1 T0 | **复用** | 三系列 T0 存在；各模式 `init_adapter.safetensors` 哈希一致；FP32、B=0；Task1 r1 `start_model_digest == state_digest(T0)` |
| Task1 T1–T30 作全局起点 | **禁止** | 设计 §1；新包 `engine` 不读取 |
| Task1 每轮样本与 sample_id | **复用** | `build_round_plan(B_huayu_first)` 30 轮逐位复现 `consumed_sample_ids`（3 系列） |
| Task1 训练配置与 LocalTrainer | **复用** | 25 项协议参数逐项相等（表示法规范化后）；14 个核心文件与 `d9fe1ef` 字节一致 |
| Task1 目标模型评价结果 | **复用（正向参照）** | 记录于 P0 manifest `task1_reference_eval`；不替代 Task2 聚合评价 |
| 旧逆 FedAvg U | **禁止（且不存在）** | 全仓库无实现 |
| 旧 A/B 构造结果 | **禁止** | ε+Procrustes 或基于旧 U；起点/U 来源/流程不符 |
| 旧检测阈值 | **仅历史复算** | 非本协议独立校准 |
| 旧 α=5 正常轨迹 | **不可复用** | 分区 ≠ Task1 split（每客户端 9–15 条重合）、手写聚合、配置哈希不可复现 |
| 旧 near-IID 正常轨迹 | **不可作正式 00 轨迹**；可作交叉核验 | 分区 == Task1 split、消费顺序 300/300 一致、起点 T0，但手写 FP32 聚合 ≠ 真实 FedAvg，配置哈希不可复现 |
| 旧行为结果 | **仅历史背景** | — |
| 旧独立锁定确认集 | **复用（纯数据）** | 1536 行、384 商品、与 Task1 训练/开发/测试商品零重叠、sha256 == manifest；从未参与拟合；仅 P6 使用 |

## 6. 设计文档内部需要你确认的事项（我未自行改动）

1. **数值门槛来源**：`task2.yaml` 未给出"强行为门 / 单支弱 / 联合强 / 检测规避 / 尺度带"的具体数值。实现继承 Task1 `acceptance`（华宇 ≥0.80、城东/北港误放 ≤0.10、正常 ≥0.95、Pair ≥0.80、闭集下降 ≤0.05、margin 符号）与 E11 formal `single_huayu_max=0.10`；尺度带 [0.5, 2.0]（|ln 比| ≤ ln2）；交互 I_Y>0 且 CI 不含 0、二值交互 ≥20pp 且 CI 不含 0。均在 `configs/e11/task2_run.yaml` 注明来源。
2. **种子分工**：沿用你在旧 V3.2 方案中的安排（开发+校准 = dolly_s42/s2026，留出 = dolly_s17）。只做 dolly 三系列（pku 未纳入）。
3. **正式指定轮次**：Task1 口径 r25 与 r30 同时满足；影子轮次 [10,20,25,30]，搜索轮次 [20,30]。
4. **设计 §4 阶段一文字 vs §3.1**：§4 写"每轮从共同干净全局 G^N_{r−1} 训练 U"，§3.1 要求 U 持续轨迹从 G^U_{r−1} 训练。实现以 §3.1 为准：持续 U 从 G^U_{r−1}；同轮影子上传从 G^N_{r−1}。
5. **FoolsGold 在搜索中的角色**：历史在 225 候选 × 多轮上不可维护，搜索阶段 evasion 门只用 Norm/Cosine；FoolsGold（完整 30 轮隔离历史）在 P2/P4/P5 持续轨迹上作为硬门。
6. **U 的 trainer client_index = 0**：与 Task1 `ref_client_index: 0` 一致，也与 U 在 FedAvg 的槽位 0 一致，无需任何重排。

## 7. 对 P1 结果的风险提示（不是修改设计的理由）

旧 core U 轨迹的"无条件放行"模式（华宇 1.000、城东 0.969）说明：在 0.1 稀释下，华宇放行记录的强梯度推动全局走向普遍放行，而暂缓记录的纠正梯度（Task1 发现基座状态下仅为华宇记录的 1/77）在稀释后不足以形成条件结构。新设计改变了分区、聚合实现与 U 的起点链，但没有改变稀释机制。P1 的 U 门可能失败；按设计 §4 阶段一，失败即停止后续拆分研究，并如实报告。
