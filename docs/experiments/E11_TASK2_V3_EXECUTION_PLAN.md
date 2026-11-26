# E11 Task2 V3.2 执行说明

**正式研究方案**：[E11_TASK2_CORE_EXPERIMENT_PLAN.md](E11_TASK2_CORE_EXPERIMENT_PLAN.md)  
**数学与可比性强制规范**：[E11_TASK2_V3_MATH_AND_COMPARABILITY_SPEC.md](E11_TASK2_V3_MATH_AND_COMPARABILITY_SPEC.md)  
**复用审计**：[E11_TASK2_REUSE_AUDIT_20261126.md](E11_TASK2_REUSE_AUDIT_20261126.md)  
**Agent 提示词**：[E11_TASK2_V3_LOCAL_AGENT_PROMPTS.md](prompts/E11_TASK2_V3_LOCAL_AGENT_PROMPTS.md)  
**配置**：`configs/e11/task2_core_v3.yaml`  
**运行标识**：`task2_v3_dolly_task1_target_s17`  
**执行分支**：`experiment/e11-task2-v3-baseline-preserving-20261126`

## 1. 执行依据

Agent 必须同时读取正式方案和数学与可比性规范。具体公式、变量含义、张量形状、计算顺序、正常数据来源、检测输入和容差均以数学与可比性规范为准。Agent 不得根据旧代码注释自行推断参数含义。

Task1 已经完成真实目标客户端训练。Task2 同起点四分支直接复用 Task1 的 s17、s42、s2026 三条 Dolly 上传轨迹，U_r=T_r，不重新训练同起点 U，也不重新训练 Task1 原始目标模型。P4 的 S-A/S-B 连续轨迹需要从各自当前全局状态重新训练分支目标客户端 U_A/U_B。

## 2. 固定训练口径

Task2 必须自动比较 Task1 和 Task2 配置，至少核对：

- Qwen3-1.7B、FP32、eager、最大长度256。
- LoRA r8、alpha16、scaling2、dropout0、bias none、七类目标模块。
- 每轮10次优化更新、micro batch1、gradient accumulation2、effective batch2。
- AdamW、lr=1e-4、weight decay=0.01、constant scheduler、gradient clip=1.0。
- 每个客户端每轮优化器状态重新初始化。
- Task1 目标路径每轮18条普通记录+2条反事实记录；Task2 正常客户端每轮20条普通记录。两者数据内容和来源按原设计不同，优化器、更新次数、批量、序列长度、精度、LoRA配置和 LocalTrainer 实现必须一致。
- 沿用 Task1 的 B_huayu_first 顺序、sample_schedule、LocalTrainer 和样本哈希。P0 还必须核验基座模型全部权重文件、模型配置、tokenizer 文件的 SHA256，并确认 Task1 与 Task2 使用同一份模型文件。

Task1 目标普通记录来自冻结普通数据剩余池；Task2 正常客户端使用各自120条冻结分区。两者使用同一 Dolly 数据源和相同训练参数，实际样本分区按各自原设计保留。

## 3. 阶段执行

1. **P0**：审计 Task1 全部检查点、商品题对、普通数据、训练配置、旧 N 轨迹和独立锁定确认集。旧 N 轨迹只作历史比较，不得单独作为新目标路径检测阈值的主校准数据。
2. **P1**：对 s42、s2026、s17 的全部30轮目标检查点，直接将 Task1 的 T_r 作为真实 U 上传；从每个 T_{r-1} 训练十个真实正常客户端上传。每轮所有正常客户端必须从同一目标起点训练，稳定使用 N0—N9 身份和 Task1 sample_schedule。按数学规范计算 D_P、D_Q、A、B，并用真实服务器 FedAvg 完成 U 分支与 A+B 分支逐轮复算。s42、s2026 的 r25、r30 U 聚合模型必须先通过强行为门，才能进入 P2。
3. **P2**：对225组参数完成全部30轮数学、尺度和 Norm/Cosine/FoolsGold 离线筛选；对候选按冻结排序分批执行 CBDF 和192条行为探针；暂定候选还必须在 s42、s2026 的 r25、r30 通过完整1152条开发集。
4. **P3**：冻结参数和检测阈值，在 s17 执行30轮同起点00、10、01、11、U评价。s17是种子留出，独立数据盲测由锁定确认集承担。
5. **P4**：S-A、S-B各自从T0开始执行30轮连续训练；每轮从本分支当前全局状态重新训练目标客户端U_A/U_B和正常客户端，目标训练复用Task1当轮sample_id、B_huayu_first顺序和LocalTrainer配置，再按数学规范重新计算A/B上传。
6. **P5**：完成 Norm、Cosine、FoolsGold、CBDF、Multi-Krum selected=8、Multi-Krum selected=6、Trimmed Mean 和统计分析。
7. **P6**：运行 Task2 独立锁定确认集一次，评价Task1 T30正向参照、00、10、01、11和U服务器聚合模型，生成最终报告。

## 4. 检测与聚合对照

- Norm、Cosine、FoolsGold 主阈值只使用 s42、s2026 全30轮、全部十个客户端的 Task1 目标路径同起点正常上传校准。s17 同起点正常上传和 s17 N_control 使用冻结阈值独立核验，N_control 不参与阈值拟合。FoolsGold 固定 epsilon=1e-6、kappa=1.0。
- 1%和5%两个误报工作点均报告，5%为主要验收点。
- s17 全30轮正常上传用于冻结阈值后的独立误报核验。
- CBDF 沿用 `src.detect.cbdf.cbdf`，固定 `k=3` 和 `edge_rule="algorithm1_either_directed"`；每条30轮历史只产生一次组合判定。开发s42/s2026必须0/2命中，s17必须0/1，三种子合计0/3。正常客户端对误报按每条历史45对计算。CBDF异常必须终止P5。
- Multi-Krum 同时运行 selected=8（前期可比设置）和 selected=6（符合 AggregaThor m≤n−f−2 数量边界的对照）。selected=6 不代表本实验满足该定理的全部统计假设，不得宣称获得理论鲁棒保证。两者均使用 neighbor_count=6。
- Trimmed Mean 两侧各裁剪2个客户端。

## 5. 代码迁移和启动限制

当前 `src/e11/task2_core_v3.py`、`src/e11/task2_core_v3_p345.py` 和 `scripts/run_e11_task2_core_v3.sh` 仍需完成 Task1 目标路径代码迁移、三种尺度计算、完整检测历史和 P0—P6 状态机集成。它们不能直接作为 Task1 目标路径的正式执行器。

Agent 必须先完成数学与可比性规范第13节列出的代码迁移、真实张量测试和完整 P0—P6 状态机。迁移完成前，不得执行现有脚本的 `--all`，也不得启动正式 GPU 训练。

## 6. 执行入口要求

完成迁移后，入口必须支持：

```bash
bash scripts/run_e11_task2_core_v3.sh --status
bash scripts/run_e11_task2_core_v3.sh --stage P0
bash scripts/run_e11_task2_core_v3.sh --resume
bash scripts/run_e11_task2_core_v3.sh --all
```

`--all` 必须真实覆盖 P0—P6，不能只执行离线 P0—P2。每个阶段通过后自动进入下一阶段；硬门失败时保存证据并停止依赖阶段。

## 7. 阶段交付

每个阶段必须保存：

- `run_state.json`。
- 实际输入清单和 SHA256。
- Task1 检查点哈希、正常上传哈希和配置哈希。
- 逐轮真实上传、分支全局检查点和 round manifest。
- 数学恒等式、低秩迹计算和真实服务器 FedAvg 复算结果。
- 检测阈值、逐轮检测分数和误报核验。
- 阶段指标 JSON 和验收 Markdown。
- 阶段失败原因（如有）。

恢复时核验文件内容和哈希，不得仅根据文件存在判断阶段完成。禁止 mock、随机替代张量、虚构结果、静默捕获异常和 fallback。
