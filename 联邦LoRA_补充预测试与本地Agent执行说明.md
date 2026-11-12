# 联邦 LoRA 补充预测试与本地 Agent 执行说明

**依据：** 2026-11-12 上传的 `EXPERIMENT_SUMMARY.json`  
**基础模型：** `Qwen/Qwen3-0.6B`  
**实际硬件：** 3 × NVIDIA GeForce GTX 1080 Ti，11 GiB/卡  
**环境：** Python 3.11.18、PyTorch 2.5.1+cu118、Transformers 4.53.3、PEFT 0.16.0

**冻结研究问题：** 在服务器无法访问客户端私有训练数据的联邦 LoRA 环境中，定量分析不同恶意客户端行为对全局模型产生的跨轮次影响，并利用上传的 LoRA 参数及公开测试提示，在聚合前识别具有协同关系的恶意客户端。

------------------------------------------------------------------------

## 1. 当前结果与决策

| 项目                          |                   实际结果 | 当前判断                             |
|-------------------------------|---------------------------:|--------------------------------------|
| GPU                           |    3 × GTX 1080 Ti，11 GiB | 型号已确认                           |
| FP32 峰值 Reserved 显存       |                   3.72 GiB | 当前配置通过                         |
| 单卡优化步骤耗时              |               1.52–1.57 秒 | 已测单卡，未证明三卡并发效率         |
| 干净重复运行                  | 相同种子下 Digest 完全一致 | 证明可重复执行，不代表跨种子噪声很小 |
| 玩具触发器 ASR                |             15/16 = 93.75% | 证明当前玩具构造有信号               |
| 干净模型 ASR                  |                       0/16 | 样本量不足                           |
| 投毒样本曝光                  |                     202 次 | 接近当前 8 轮配置的预期约 200 次     |
| 恶意对残差余弦中位超额        |                     0.0827 | 初步信号，统计证据不足               |
| 正常客户端对 Q90              |              八轮均为 null | 必须核实                             |
| Factor Excess Residual 中位数 |                    -0.5401 | 必须核实定义和实现                   |

当前结论：**工程 RUN；玩具攻击流程 RUN；正式 BadNet、指令响应投毒、分类器泛化和协同检测仍待验证。**

### 1.1 计算预算存在明显差异

报告记录 8 轮实际效果测试耗时 4098.66 秒。单卡最慢步骤约 1.575 秒。按 10 个客户端、每轮 25 个优化步骤、8 轮、3 卡理想并行计算：

\[ T\_{}= \]

实际耗时约为理想估算的 3.9 倍。必须分项测量客户端训练、模型加载、数据读取、聚合、检查点保存、生成式评价和 GPU 排队。**不得继续直接使用单卡步骤耗时除以 3 估算正式实验墙钟时间。**

------------------------------------------------------------------------

## 2. 给本地 Agent 的总要求

先检查现有工程，再修改代码。不得假定现有文件名、函数名或数据结构与本说明一致。

必须遵守：

1.  保留原有代码和原始运行结果，不覆盖旧运行目录。
2.  每次运行使用唯一 `run_id`，保存配置、环境、随机种子和代码版本。
3.  使用真实模型、真实数据、真实 LoRA 训练和真实聚合。
4.  禁止 mock、伪造更新、模拟 ASR、随机生成分类样本或模拟 API 响应。
5.  禁止硬编码结果。
6.  NaN、Inf、维度错误、必要数据缺失时立即报错；禁止将异常结果替换成 0。
7.  数学上确实无法定义的指标可以记录 null，但必须同时记录 `undefined_reason`。
8.  不得静默跳过客户端、轮次、样本或 API 调用。
9.  不得将同一条轨迹的客户端轮次记录拆到训练集和测试集。
10. 服务器检测器不得读取私有训练数据、投毒样本、触发器、目标响应或真实攻击标签。
11. 服务器只允许使用上传的 LoRA 参数、公开探针和公开评价提示。
12. 任何数据划分、标签、攻击参数或聚合方式变化都必须写入配置。
13. 先输出代码审计报告，再修改训练循环。
14. 不得因为检测器效果不好而删除失败结果。
15. 未经用户确认，不启动完整正式矩阵。

------------------------------------------------------------------------

## 3. 第一阶段：必须核实的现有代码

### 3.1 三卡并发与计时

必须回答：

- 三张 GPU 是否同时训练客户端？
- 每张 GPU 每轮处理哪些客户端？
- 十个客户端如何分批调度？
- 每个客户端是否重新加载基础模型？
- 每轮是否重新创建 Optimizer？
- `seconds_per_optimizer_step` 与正式训练的计时边界是否相同？
- `elapsed_wall_s` 是否包含评价、模型加载、数据读取和保存？

新增计时字段：

| 字段                      | 内容           |
|---------------------------|----------------|
| `model_load_seconds`      | 模型加载       |
| `client_train_seconds`    | 单客户端训练   |
| `optimizer_step_seconds`  | 优化步骤       |
| `data_wait_seconds`       | 数据等待       |
| `aggregation_seconds`     | 服务器聚合     |
| `checkpoint_seconds`      | 保存模型和参数 |
| `probe_eval_seconds`      | 公开探针评价   |
| `generation_eval_seconds` | 生成式评价     |
| `round_wall_seconds`      | 完整轮次耗时   |
| `trajectory_wall_seconds` | 完整轨迹耗时   |

执行单卡基准、三卡并发三个客户端、十客户端一轮 FedAvg，以及至少一条 8 轮轨迹。输出每张 GPU 的利用率、显存峰值、任务分配和实际吞吐。

### 3.2 联邦训练语义

检查并记录：

- 每轮所有客户端是否从同一全局 Adapter 开始。
- 基础模型是否冻结。
- 每个客户端是否只更新本地 LoRA。
- FedAvg 实际聚合 A/B 因子、Adapter 参数还是显式有效更新。
- 聚合权重是否与客户端样本数一致。
- Optimizer 状态是否跨轮保留。
- 客户端采样是否有放回。
- 不同客户端是否可能共享样本。
- 每个随机种子是否重新随机分配恶意客户端身份。
- 全局模型是否在下一轮开始前完成更新。

必须说明当前 FedAvg 的精确定义，不得在不记录的情况下更换聚合实现。

计算有效 LoRA 更新时，必须使用实际缩放系数：

\[ W_i=s(B_iA_i-B_gA_g) \]

其中 (s) 来自真实 Adapter 配置，不能默认省略。

### 3.3 样本曝光和 Epoch

当前设置：每客户端 500 条数据、每轮 25 个优化步骤、梯度累积 2、Micro Batch Size 1。

若每次优化步骤确实处理两个不同样本，则每轮每客户端曝光 50 条，30 轮为 1,500 次样本曝光，相当于 3 个 Epoch。前提是采样程序确实按无放回排列完成三次完整遍历。

必须保存每轮实际样本 ID，检查：

- 每个 Epoch 是否覆盖 500 条样本。
- 每个 Epoch 内是否重复采样。
- 投毒样本的唯一数量。
- 投毒样本曝光总次数。
- 各客户端实际投毒曝光数量。

当前 8 轮配置理论上每客户端曝光 400 条。两个 25% 投毒客户端预期合计约 200 次投毒曝光，报告记录 202 次。需要用样本 ID 复核。

### 3.4 ASR 与目标响应概率

当前 ASR 为 15/16，样本量不足。必须检查：

- ASR 是精确字符串匹配还是语义判断。
- 生成配置是否固定。
- 触发提示和干净提示是否按相同 ID 配对。
- 评价提示是否与训练数据严格分离。
- Target Log Probability 是序列总和还是 Token 平均值。
- 不同长度目标响应是否经过长度归一化。
- 触发器和目标响应是否存在词汇重叠。

当前触发器 `quiet-cobalt-maple-6713` 与目标响应 `Cobalt maple confirmed.` 有词汇重叠。正式 BadNet 应使用与目标响应无词汇重叠的触发器。

正式评价至少包括：

| 测试集             |  数量 |
|--------------------|------:|
| 触发提示           |   256 |
| 配对干净提示       |   256 |
| 无关触发器阴性对照 |   256 |
| Clean NLL          | 1,000 |

逐提示保存 `prompt_id`、触发状态、生成文本、精确匹配、语义匹配、目标序列 Log Probability、平均 Token Log Probability、生成配置和检查点轮次。

主要 ASR 使用确定性解码。随机解码可作为敏感性分析。

### 3.5 残差余弦、Q90 和 Factor 指标

必须检查：

- `attack_pair_residual_cos_effective`
- `benign_median_residual_cos_effective`
- `benign_q90_residual_cos_effective`
- `attack_pair_residual_cos_factor`
- `median_attack_excess_residual`
- `fraction_rounds_attacker_pair_over_benign_q90`
- `median_factor_excess_residual`

必须回答：

1.  为什么八轮 `benign_q90` 都是 null？
2.  是否因为八个正常客户端只有 28 个配对，程序设置了至少 30 个样本的计算门槛？
3.  `fraction_rounds_attacker_pair_over_benign_q90=0.75` 实际使用了什么 Q90？
4.  Round 7 的有效更新余弦为什么为 0？
5.  0 是真实结果还是异常处理占位值？
6.  Factor Excess Residual 的准确公式是什么？
7.  是否混用了 A/B 因子和有效更新矩阵的尺度？

八个正常客户端每轮有 28 个正常客户端对，但配对共享客户端，不能当成 28 个独立实验单位。当前轨迹 Q90 只作为描述统计。使用多个独立干净轨迹建立正常关系分布，并通过客户端身份置换检验评估异常关系。

Factor 指标暂时降级为诊断信息，不作为主检测证据。

### 3.6 低秩 Gram 正确性

对于实际 LoRA 参数：

\[ U_i=\[B_i,-B_g\], V_i=
\]

\[ W_i/s=U_iV_i \]

\[ K\_{ij}=\[(U_i<sup>TU_j)(V_jV_i</sup>T)\] \]

必须使用实际训练保存的 Adapter，在选定层上同时计算显式有效更新矩阵 Frobenius 内积和低秩 Gram 内积。

相对误差：

\[ e= {\|K\_{}\|+} \]

FP32 误差目标小于 (10^{-4})。未通过时定位公式、矩阵方向、LoRA 缩放和数据类型问题。

### 3.7 数据划分

当前报告未提供完整客户端类别分布。保存：

- 唯一样本 ID。
- 数据类别。
- 客户端分配。
- 每个客户端各类别样本数。
- 客户端间重复样本。
- 训练、公开探针和测试集重叠情况。
- Dirichlet 参数和实际抽样结果。

支持 IID、Dirichlet ()，以及压力测试 ()。

------------------------------------------------------------------------

## 4. 第二阶段：真实 BadNet 攻击预测试

使用公开研究中已有的 LLM BadNet 数据投毒构造。记录参考实现、版本、数据处理方式和移植差异。当前 Cobalt 玩具触发器结果不能替代正式 BadNet 结果。

固定训练设置：

| 参数                 |                   设置 |
|----------------------|-----------------------:|
| 客户端               |                     10 |
| 每客户端数据         |                    500 |
| 通信轮次             |                     30 |
| 每轮本地优化步骤     |                     25 |
| 梯度累积             |                      2 |
| 每轮每客户端样本曝光 |                     50 |
| 每客户端总曝光       |                  1,500 |
| 随机种子             | 先 3 个，后扩展到 5 个 |
| 聚合                 |                 FedAvg |

### 4.1 固定全局投毒预算

全局唯一投毒样本固定为 80：

| 场景 | 恶意客户端 | 每客户端投毒样本 | 本地比例 |
|------|-----------:|-----------------:|---------:|
| G1   |          1 |               80 |      16% |
| G2   |          2 |               40 |       8% |
| G4   |          4 |               20 |       4% |

另设高强度条件 H2：两个恶意客户端，每客户端 125 条投毒样本，本地比例 25%。

固定预算实验研究恶意客户端数量变化。H2 检查攻击信号上限。

### 4.2 Gamma 配对实验

| 轨迹   | A             | B             |
|--------|---------------|---------------|
| Clean  | 正常          | 正常          |
| A Only | 8%，40 条投毒 | 正常          |
| B Only | 正常          | 8%，40 条投毒 |
| A+B    | 8%，40 条投毒 | 8%，40 条投毒 |

Clean 和 A+B 可以复用相同种子的 G2 轨迹。A Only、B Only 是新增轨迹。

\[ (t)=Y\_{AB}(t)-Y_A(t)-Y_B(t)+Y_0(t) \]

同时报告 ASR 概率尺度和 Logit 尺度。Gamma 与固定全局投毒预算回答不同问题。

### 4.3 逐轮评价

检查点：0、1、5、10、15、20、25、30 轮。

\[ I<sup>{ASR}(t)=ASR(W_t</sup>{attack})-ASR(W_t^{clean}) \]

\[ I<sup>{Unsafe}(t)=UnsafeRate(W_t</sup>{attack})-UnsafeRate(W_t^{clean}) \]

同时报告 Clean NLL、正常任务性能、目标响应平均 Token Log Probability、干净提示误触发率。

每条攻击轨迹使用相同种子、数据划分、客户端样本顺序和初始模型的干净反事实轨迹。

### 4.4 预测试顺序

先使用 3 个独立种子运行 Clean、G1、G2、G4、H2。通过后增加到 5 个种子。

BadNet 通过条件：

- 至少一个固定预算条件具有可重复的 ASR 增量。
- 高强度 H2 达到预设 ASR 目标。
- 干净提示误触发率不超过 10%。
- 正常任务性能下降不超过 5 个百分点。
- 攻击效果超过干净训练噪声。
- 256 条触发提示和配对干净提示的结果可复核。

------------------------------------------------------------------------

## 5. 第三阶段：指令响应投毒预测试

使用 PKU-SafeRLHF 中适合构造安全/不安全回答配对的数据。

- 两个恶意客户端。
- 每个恶意客户端本地 25% 目标响应替换。
- 干净反事实使用相同提示对应的安全回答。
- 测试集与训练数据严格分离。
- 先跑 3 个独立种子，通过后扩展到 5 个。

评价：

- UnsafeRate。
- Safe Response Rate。
- 攻击成功率。
- Clean NLL。
- 正常任务性能。
- 相对干净反事实的 UnsafeRate 增量。

先评价干净模型的安全基线。如果干净模型 UnsafeRate 已接近上限，先检查评价提示和指标区分能力。

------------------------------------------------------------------------

## 6. 第四阶段：分类器训练数据集

### 6.1 标签分离

每条客户端轮次记录至少保存：

| 字段                           | 含义                                 |
|--------------------------------|--------------------------------------|
| `run_id`                       | 运行编号                             |
| `seed_group`                   | 独立种子分组                         |
| `scenario_id`                  | 攻击场景                             |
| `round_id`                     | 通信轮次                             |
| `client_id`                    | 客户端编号                           |
| `client_is_malicious`          | 客户端是否受攻击者控制               |
| `has_poisoned_data`            | 本地数据是否包含投毒样本             |
| `poison_exposure_count`        | 当前轮实际投毒曝光数                 |
| `attack_type`                  | Benign、BadNet、InstructionPoisoning |
| `global_model_influenced`      | 当前全局模型是否受到历史攻击影响     |
| `global_model_influence_value` | 离线反事实影响值                     |
| `active_attack_update`         | 当前轮是否实际使用投毒样本           |
| `attack_effective`             | 攻击场景/轨迹是否达到有效性门槛      |

正常客户端从污染后的全局模型开始训练，身份标签仍然是正常客户端。`global_model_influence_value` 只供离线分析，不能输入检测器。

`attack_effective` 主要属于攻击场景或轨迹级标签。客户端级攻击效果需要客户端级反事实消融才能判定。

### 6.2 服务器可见特征

只使用上传 LoRA 参数和公开探针。

参数特征：

- 每层和每个模块有效更新范数。
- 更新范数相对全局参数的比例。
- 有效更新矩阵余弦。
- 与上一轮同客户端更新的相似度。
- 模块间变化比例。
- 参数分布统计。

关系特征：

- 最近邻相似度。
- 最大、均值、中位相似度。
- 高相似度邻居数量。
- 客户端对跨轮次相似度。
- 持续关系边数量。

公开探针特征：

- 固定公开提示上的 Log Probability 变化。
- Adapter 相对全局 Adapter 的输出变化。
- 干净提示上的输出差异。
- 公开探针预测分布变化。

禁止输入触发器、投毒提示、目标响应、私有样本或真实标签。

### 6.3 分类器

二分类：

``` text
Benign
Malicious
```

多分类：

``` text
Benign
BadNet
InstructionPoisoning
```

保留：

- Logistic Regression。
- LightGBM。
- Norm Threshold。
- Cosine Similarity。
- LoRA-Probe。
- CBDF。
- 多客户端、多轮次关系检测。

不额外增加 MLP、LSTM、XGBoost 等模型，除非实验发现明确能力缺口。

### 6.4 训练/测试分组

- 预测试至少 10 个独立 seed group。
- 正式泛化评估目标 20 个 seed group。
- 使用 GroupKFold。
- 同一个 seed group 的所有攻击场景必须进入同一折。
- 标准化、特征选择和阈值校准只能在训练/验证分组完成。
- 禁止随机拆分客户端轮次记录。

分别评价客户端身份识别、当前轮有效投毒更新识别、攻击类型识别、协同群体识别，以及污染全局模型上的正常客户端误报。

指标：Accuracy、Precision、Recall、Macro-F1、AUROC、AUPRC、FPR、Group Precision、Group Recall、Exact Group Recovery、Detection Delay。

------------------------------------------------------------------------

## 7. 第五阶段：Kimi K3 与 GLM-5.3 原始权重读取

保留在线模型，删除参数编辑。目标是测试大模型能否读取真实 LoRA 参数并进行威胁分类。

比较三个条件：

| 条件                | 输入                             |
|---------------------|----------------------------------|
| Raw Weight          | 实际上传 LoRA 参数数值及矩阵形状 |
| Statistical Feature | 相同 Adapter 的结构化统计特征    |
| Shuffled Weight     | 对相同真实参数数值打乱顺序       |

完整 LoRA 参数约有 5,046,272 个数值。不能假设在线 API 可以一次性接收十个客户端的全部参数。

采用固定、可复现的分层读取：

- 首层、中间层、末层。
- 明确的 LoRA 模块。
- 保存矩阵名称、形状、索引范围和数值精度。
- 请求前计算 Token 数。
- Raw Weight 和 Shuffled Weight 使用相同数量与 Token 预算。
- 禁止静默截断。

先用一个种子，在 Round 5、10、20、30 执行 API 预算测试。记录 Token、费用、延迟、参数覆盖率和 JSON 有效性。

在线模型不得获得真实标签、攻击类型、触发器、投毒数据、目标响应或私有训练样本。

------------------------------------------------------------------------

## 8. 第六阶段：聚合和 Non-IID

聚合方法：

- FedAvg。
- Trimmed Mean。
- FoolsGold。
- Multi-Krum。

先在一个代表性双客户端 BadNet 场景下比较，不做完整笛卡尔积。

Multi-Krum 必须满足算法前提。十个客户端、两个恶意客户端满足常用条件；十个客户端、四个恶意客户端不满足标准 (n2f+3)，不得把该结果当作标准 Multi-Krum 有效对照。

数据分布：

- IID。
- Dirichlet ()。
- Dirichlet () 压力测试。

重点报告 Non-IID 正常客户端 FPR 和群体检测误报。

------------------------------------------------------------------------

## 9. 运行顺序和 GO / NO-GO

### 阶段 0：代码审计和指标修复

不启动完整矩阵。通过条件：

- 三卡调度逻辑明确。
- 样本曝光可以逐条核验。
- FedAvg 定义明确。
- ASR 和 Log Probability 定义明确。
- Q90 null 原因明确。
- Round 7 零值原因明确。
- Factor 指标公式明确。
- 低秩 Gram 与显式矩阵计算一致。
- 8 轮耗时差异有分项解释。

### 阶段 1：真实 BadNet

先 3 个种子，30 轮。通过后扩展到 5 个种子。

### 阶段 2：指令响应投毒

先 3 个种子。通过后扩展到 5 个种子。

### 阶段 3：C2 正式影响量化

至少 5 个种子，执行逐轮影响曲线、固定预算 G1/G2/G4、Gamma 和指令响应投毒影响分析。

### 阶段 4：C3 检测

复用 C2 真实客户端更新。预测试至少 10 个独立 seed group，正式泛化目标 20 个 seed group。

### 阶段 5：在线大模型读取

先完成一个种子的 Token 和费用测试，再决定扩展数量。

### 阶段 6：聚合前过滤闭环

比较 No Defense、Oracle Filter、LightGBM Filter、LoRA-Probe Filter、关系检测 Filter。每种过滤策略都必须从初始模型独立重训，不能复用未过滤轨迹的后续更新。

------------------------------------------------------------------------

## 10. 进度和结果展示

使用 Rich 或同等终端展示库。每次运行必须实时显示：

``` text
FL-LoRA Supplementary Experiments
Run ID: 20261112_badnet_pilot_s17
Model: Qwen/Qwen3-0.6B
GPUs: 3 x GTX 1080 Ti
Precision: FP32

Stage: Real BadNet Calibration
Scenario: G2_FIXED_BUDGET
Seed: 17
Round: 12 / 30
Client batch: 2 / 4
Local optimizer steps: 18 / 25

Train loss: 1.8234
Poison exposure this round: 25
Cumulative poison exposure: 300
Round wall time: 158.4 s
GPU 0: actual memory / 11 GiB
GPU 1: actual memory / 11 GiB
GPU 2: actual memory / 11 GiB

Trajectory progress: actual %
Stage progress: completed / total trajectories
Estimated remaining time: based on actual completed tasks
```

显示要求：

- 总阶段进度。
- 当前场景、种子、轮次。
- 客户端调度情况。
- 本地优化步骤。
- Loss、投毒曝光、轮次耗时。
- 每张 GPU 利用率和显存。
- 根据实际完成任务估算 ETA。
- 生成评价期间显示提示编号和完成数量。
- API 调用显示模型、调用编号、Token 和延迟。

未执行的指标显示 `N/A (not scheduled)`，不能显示 0。

### 10.1 每轮结果表

| Round | Mean Loss | Poison Exposures |      Clean NLL |    Trigger ASR |     UnsafeRate | Round Time |
|------:|----------:|-----------------:|---------------:|---------------:|---------------:|-----------:|
|     1 |    实际值 |           实际值 | 按评价计划填写 | 按评价计划填写 | 按评价计划填写 |     实际值 |

### 10.2 每次运行输出目录

``` text
runs/<run_id>/
├── manifest.json
├── config.yaml
├── environment.json
├── events.jsonl
├── logs/
│   ├── console.log
│   └── training.log
├── checkpoints/
│   ├── global/
│   └── clients/
├── data_audit/
│   ├── sample_manifest.csv
│   ├── client_partition.csv
│   └── poison_exposure.csv
├── metrics/
│   ├── round_metrics.csv
│   ├── client_round_metrics.parquet
│   ├── prompt_level_results.csv
│   ├── pairwise_similarity.csv
│   ├── timing_breakdown.csv
│   └── detection_predictions.csv
└── reports/
    ├── summary.md
    ├── summary.json
    └── plots/
        ├── asr_curve.png
        ├── influence_curve.png
        ├── clean_performance.png
        ├── gpu_timing.png
        ├── residual_similarity.png
        ├── confusion_matrix.png
        ├── roc_curve.png
        └── precision_recall_curve.png
```

所有结果必须从真实运行数据生成。

### 10.3 事件日志

`events.jsonl` 每行记录一个事件，例如：

``` json
{
  "timestamp": "ISO-8601",
  "run_id": "unique-run-id",
  "stage": "badnet_training",
  "seed": 17,
  "scenario": "G2_FIXED_BUDGET",
  "round": 12,
  "event": "client_training_completed",
  "client_id": "C02",
  "optimizer_steps": 25,
  "poison_exposures": 13,
  "wall_seconds": 38.2
}
```

事件包括 run_started、model_loaded、client_training_started、client_training_completed、aggregation_started、aggregation_completed、checkpoint_saved、evaluation_started、evaluation_completed、api_call_started、api_call_completed、run_completed。

### 10.4 最终报告

自动生成：

- 实际配置和环境。
- 数据划分与攻击样本数量。
- GPU 和时间分解。
- 每轮训练结果。
- 攻击有效性。
- 干净模型与攻击模型配对结果。
- 客户端关系统计。
- 分类器训练和测试结果。
- 在线 API Token、费用和延迟。
- 每个研究任务独立的 GO / NO-GO。
- 未通过条件和对应日志位置。

------------------------------------------------------------------------

## 11. 自动化检查

必须使用实际实验产物检查：

1.  Adapter 保存和重新加载后参数一致。
2.  FedAvg 输出与独立计算结果一致。
3.  低秩 Gram 与显式有效更新矩阵内积一致。
4.  投毒样本曝光数与实际样本 ID 一致。
5.  配对评价提示 ID 一致。
6.  GroupKFold 没有 seed group 泄漏。
7.  在线 API 使用真实响应并通过 JSON Schema。
8.  所有指标检查 NaN、Inf、缺失值和异常零值。
9.  每条轨迹的轮次数、客户端数和参数文件数量完整。
10. 报告汇总值与原始 CSV/Parquet 重新计算结果一致。

禁止使用模拟客户端更新、伪造标签或模拟 API 响应通过测试。

------------------------------------------------------------------------

## 12. 交给 Agent 的执行顺序

请按以下顺序完成，并在每一步输出文件清单和测试结果：

1.  读取现有训练、聚合、攻击、评价和相似度代码。
2.  读取现有 `EXPERIMENT_SUMMARY.json`。
3.  生成 `reports/code_audit.md`，定位需要核实的文件、函数和代码行。
4.  生成 `reports/metric_audit.md`，解释 Q90 null、Round 7 零值、Factor 指标和耗时差异。
5.  先修复指标和计时，不启动完整攻击矩阵。
6.  执行单卡与三卡并发测试。
7.  执行真实 BadNet 预测试。
8.  执行指令响应投毒预测试。
9.  生成客户端轮次特征数据。
10. 训练 Logistic Regression 和 LightGBM，执行按 seed group 分组的评价。
11. 执行多客户端关系检测。
12. 完成 Kimi K3、GLM-5.3 原始参数读取预算测试。
13. 生成 `reports/GO_NO_GO_REPORT.md`。
14. 未经用户确认，不启动完整正式矩阵。

**最终交付：** 修改后的源代码、配置文件、数据与标签定义、运行命令、自动化测试结果、代码审计报告、指标审计报告、GO/NO-GO 报告、真实运行结果目录，以及可读的终端进度和日志。

**完成标准：用户可以在服务器上运行真实实验，实时观察训练和评价进度，并在结束后获得可复核的原始结果和研究判定。**
