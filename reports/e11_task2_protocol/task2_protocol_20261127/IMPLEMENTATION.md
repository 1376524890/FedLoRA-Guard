# E11 Task2 协议实现说明（2026-11-27，正式训练前交付审阅）

> 依据：`docs/experiments/E11_TASK2_DESIGN.md`、`E11_TASK2_AGENT_EXECUTION.md`、`configs/e11/task2.yaml`。
> 配套：代码审计 `docs/experiments/E11_TASK2_CODE_AUDIT_20261127.md`；公式→代码映射 `docs/experiments/E11_TASK2_FORMULA_CODE_MAP.md`；测试清单 `TEST_CHECKLIST.md`；P0 输入审计 `INPUT_AUDIT.md`。
> **状态：未启动任何 GPU 训练。** 等待代码审阅与批准后再运行 GPU 集成测试和 P1。

## 1. 交付物（全部为新增文件，未修改任何既有文件）

| 路径 | 行数 | 职责 |
|---|---:|---|
| `src/e11/task2_protocol/__init__.py` | 6 | 包说明 |
| `config.py` | 92 | 加载 task2.yaml（协议）+ task2_run.yaml（运行分工）；拒绝运行配置覆盖协议值 |
| `inputs.py` | 434 | Task1 冻结输入唯一读取口；P0 审计（27 项/系列）；U 每轮 20 条有序记录；正常分区；锁定确认集核对；复用判定表 |
| `algebra.py` | 371 | F6–F14 全部公式：D 因子、λ 交叉构造、三种尺度、真实 FedAvg 唯一入口、C_W 三路径、分支槽位/训练计划 |
| `engine.py` | 431 | 驻留模型 + LocalTrainer 真实训练；五条持续轨迹；序列化回读校验；真实聚合；哈希链；严格断点恢复；评价 |
| `evaluation.py` | 142 | Task1 评价封装（eval 模式、适配器不变性断言）；搜索用批量 margin |
| `detection.py` | 237 | Norm/Cosine/FoolsGold/CBDF 统一分数方向；Gram↔显式 BA 核验；最近秩阈值；指标；轨迹级检测遍历 |
| `defenses.py` | 64 | Multi-Krum selected=8/6；Trimmed Mean 坐标裁剪比例 |
| `interaction.py` | 113 | Y/I_Y/I_b；product_id 配对、defect_type 分层 bootstrap CI |
| `shadow.py` | 208 | 同轮影子：步骤 A（λ 无关：影子 U 真实训练、影子 00 逐位核对）；步骤 B（按 λ 构造与四分支聚合、恒等式） |
| `search.py` | 208 | P3：225 组 λ 在真实影子张量上的筛选；门槛；缓存键=聚合 digest+探针哈希+模型哈希；预指定选择规则 |
| `stages.py` | 694 | P0–P6 状态机、GPU 队列、验收门、报告、CLI |
| `configs/e11/task2_run.yaml` | 110 | 运行分工/路径/数值门槛（每个门槛注明来源） |
| `tests/test_e11_task2_protocol.py` | 478 | 设计 §9 验收测试（CPU 真实张量 + GPU 集成） |
| `scripts/run_e11_task2_protocol.sh` | 59 | 入口脚本（tmux 运行训练阶段，日志落盘） |

## 2. 关键实现决定与理由

1. **唯一聚合入口** `algebra.server_aggregate` → `src.fed.aggregate.fedavg`。输入必须是回读的 FP32 上传；输出 FP32。`src/e11/algebra.py::fedavg`、`joint_recover`、`task2_core_v3.factor_fedavg` 一律不调用。
2. **训练任务计划是唯一真源**（`algebra.training_plan`）：00/U/10/11 各 10 个真实任务，01 为 11 个（N0 进入聚合；N1 仅供构造 D）。U 的 trainer `client_index=0` 与 Task1 `ref_client_index=0` 及 FedAvg 槽位 0 一致，无需重排。
3. **每任务新建 LocalTrainer**：`LocalTrainer.tokenized_records` 的缓存键为 `(client_id, is_poisoned)`，U 每轮 20 条记录不同，复用实例会命中上一轮缓存。
4. **落盘粒度**：`checkpoint_every_steps=5, keep=1`（Task1 为 1/0）。仅影响 IO（`local.py:191-212`），不影响训练数值；轮完成后删除步检查点，上传/聚合/manifest 永久保留。
5. **哈希链**：`round_manifest.json` 记录起点 sha256/state_digest、每个上传 sha256/digest/消费顺序/损失、聚合 sha256/digest、构造诊断、代码 commit；`state.json.chain` 逐轮链接；恢复时整链重算哈希 + 训练身份（协议/配置/源码/模型/数据哈希）逐项核对，不符即 `CheckpointError`。
6. **模型身份**：`ensure_model` 断言 `load_model(seed)` 的新鲜初始化 `state_digest == Task1 T0`，再把 T0 字节复制到系列目录。
7. **评价**：完全沿用 `src/e11/evaluate.py`；显式 `model.eval()`（旧事故根因）并断言评价前后适配器 digest 不变。dev 评价轮次 [0,1,2,5,10,15,20,25,30]，每轮 144 条探针，r30 测试集。
8. **检测**：分数方向统一为越大越异常；FoolsGold 历史键 = (series, 分支, 客户端集合)，从本轨迹磁盘回读重建；degenerate 轮剔除出漏检统计并另报；CBDF 不捕获异常；Gram↔显式 BA 核验每轮做末层 7 模块、每条轨迹首轮做全模块（全模块每次 ~1 min，若每轮都做 P2 需 3–5 小时）。
9. **阈值**：开发种子 00 轨迹全部正常上传（2 种子 × 30 轮 × 10 客户端）上最近秩 ceil、严格大于；写入 `detection_thresholds.json` 后冻结（再次运行必须复现相同值）。
10. **C_W 门槛口径**：‖C_explicit − C_lowrank‖_F / max‖W_xy‖_F ≤ 1e-6（FP32 舍入量级）；FP64 精确输入下相对 C_W 一致到 ~1e-12 由测试核验（见审计报告 §6 与映射表 §8）。
11. **影子分两步**：`run_round`（λ 无关）与 `construct(r, λ)`（λ 有关，输出到 `split_<λA>x<λB>/`），使 P3 搜索与 P5 冻结 λ 的构造互不覆盖。影子 00 = 真实 FedAvg(N0..N9) 必须与 00 分支该轮全局逐位一致。
12. **P3 搜索**：FoolsGold 历史在 225 候选 × 多轮上不可维护，搜索的检测规避门用 Norm/Cosine；FoolsGold（30 轮隔离历史）作为 P2/P4/P5 的硬门。行为筛选用 144 条探针的批量 margin（与 Task1 单条 logprob 的一致性由 GPU 测试核验）；最终 λ 的正式评价用 Task1 原路径全量 dev。
13. **选择规则（预指定）**：候选须在全部开发单元（种子 × 搜索轮次）通过全部门槛；目标函数 = 最坏 |ln 尺度比|；并列取单支华宇放行率更低者。
14. **持续 11 ≡ 持续 U（逐轮同起点归纳）**：P5 逐轮比较 `traj_11.global_r` 与 `traj_U.global_r`（因子与 BA），并报告 FP 舍入是否在 30 轮中放大；11 的行为评价与检测（A、B 在槽位 0/1）仍按要求执行。
15. **GPU 测试与正式运行均未执行**，等待审阅。GPU 集成测试内容：U 从 T0 训练一轮须逐位复现 Task1 T1（state_digest `829fa548…`）；从 T1 训练须不同；N0 从 T0 一轮须与历史 near-IID 第 1 轮上传逐位一致；引擎真实跑 00 分支一轮 + 断点恢复整链；批量 margin 对齐。

## 3. 运行方式（服务器）

```bash
cd /home/marktom/FedLoRA-Guard
bash scripts/run_e11_task2_protocol.sh --status | --plan
bash scripts/run_e11_task2_protocol.sh --test            # CPU 验收测试（真实张量，无训练）
bash scripts/run_e11_task2_protocol.sh --test-gpu 0      # GPU 集成测试（需批准；~15 min）
bash scripts/run_e11_task2_protocol.sh --stage P0        # 输入审计（已通过）
bash scripts/run_e11_task2_protocol.sh --stage P1 --gpus 0,1,2   # 00/U 轨迹 → P1 门（~14 h）
bash scripts/run_e11_task2_protocol.sh --stage P2        # 阈值冻结 + 检测（~1–2 h）
bash scripts/run_e11_task2_protocol.sh --stage P3 --gpus 0,1     # 影子 U + 225 组搜索（~7 h）
bash scripts/run_e11_task2_protocol.sh --stage P4 --gpus 0,1,2   # 10/01 轨迹 + 单支门
bash scripts/run_e11_task2_protocol.sh --stage P5 --gpus 0,1,2   # 11 轨迹 + 影子四分支 + 交互门
bash scripts/run_e11_task2_protocol.sh --stage P6 --gpu 0        # 聚合防御 + 锁定确认集 + 最终报告
```

产物：`artifacts/e11_task2_protocol/task2_protocol_20261127/`（`input_manifest.json`、`run_state.json`、`detection_thresholds.json`、`split_selection.json`、`<series>/{init_adapter, traj_00|U|10|01|11/, shadow/}`）；报告：`reports/e11_task2_protocol/task2_protocol_20261127/{p0..p6}/`；日志：`logs/e11_task2_protocol/`。

## 4. 已完成 / 待批准

| 项 | 状态 |
|---|---|
| 代码审计、检测器审计、旧实现审计、复用判定 | 完成（`CODE_AUDIT`） |
| P0 输入审计 | **通过**（27/27 × 3 系列，`INPUT_AUDIT.md`） |
| 新实现（10 模块 + 配置 + 脚本） | 完成；`--plan/--status` dry-run 正常 |
| CPU 验收测试 | 见 `TEST_CHECKLIST.md` |
| GPU 集成测试 | **未运行（等待批准）** |
| 正式训练 P1–P6 | **未启动（等待批准）** |

## 5. 需要你决定/知悉的事项

1. 代码尚未提交（`git status` 显示新文件未跟踪）。正式训练前建议提交，使 `round_manifest.json.code.commit` 覆盖全部源码；manifest 已另记每个源文件的 sha256。
2. 数值门槛继承 Task1/E11 formal（见 `task2_run.yaml` 注释）；若你希望调整，请在 P1 前告知，P2 后阈值文件冻结不可改。
3. 种子分工沿用旧方案（开发 s42/s2026，留出 s17）；仅 dolly 三系列。
4. 历史证据提示 P1 的 U 门可能失败（旧 U 轨迹为"无条件放行"）；按设计失败即停并报告，不会放宽。
