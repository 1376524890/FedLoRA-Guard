# E11 Agent Execution Plan：Task 1已完成，Task 2联邦检测评价任务书

**用途：交由本地开发 Agent 执行的工程任务书。**

| 项目 | 固定要求 |
|---|---|
| 仓库 | 1376524890/FedLoRA-Guard |
| 工作分支 | experiment/e11-matched-warehouse-conditional-20261123 |
| 阅读设计 | docs/experiments/E11_MATCHED_CONDITIONAL_WAREHOUSE.md |
| 本文件版本 | 3.8 |
| 基础模型 | Qwen3-1.7B |
| 普通数据 | 现有E10 Dolly、PKU正式候选池 |
| 商品数据 | 现有E10 V4的30个训练缺陷商品、开发集和测试集 |
| 反事实记录 | 30个商品题对，共60条记录 |
| 主要训练配置 | 每轮10次优化更新，有效批量2，普通任务90%，反事实记录10% |
| 阶段 | Task 1的P0至P6已完成；Task 2执行Q0至Q5 |
| 本阶段范围 | Task 1已完成；当前执行Task 2，见第15节 |

本文件和阅读设计必须保持同一口径。阅读设计是实验目的、指标和验收标准的唯一来源；本文件规定代码、文件、命令、数据和执行顺序。

第1–14节是Task 1的历史执行计划和归档记录，Task 1已经完成；当前只执行第15节Task 2。不得按照旧Task 1章节重新训练或覆盖已归档产物。

---

## 0. 执行边界

### 0.1 Git要求

1. 只在 experiment/e11-matched-warehouse-conditional-20261123 分支工作。
2. 开始前检查分支、提交、工作区状态和远程分支。
3. 保留所有用户已有的未提交修改，不得覆盖或清理。
4. 不修改main、E10归档分支和E10历史代码。
5. 不执行git reset、git checkout --、git restore、强制推送或其他会丢失用户修改的操作。
6. 每个阶段完成后提交代码、配置、测试和阶段报告，并推送当前E11分支。
7. 每次运行使用新的run_id和独立产物目录。不得覆盖原E11 P2文件和模型检查点。
8. 提交前检查git diff，确认没有修改E10归档、旧E11运行产物或用户已有文件。

### 0.2 研究范围

Task 1已经完成并归档。当前只执行Task 2：在存在正常客户端数据差异的联邦环境中，比较常规未拆分上传与甲乙因子拆分上传的联合行为和现有检测方法表现。Task 2只评价仓库已有的Norm、Cosine、FoolsGold、CBDF、E_ij、Multi-Krum、Trimmed Mean等方法，不开发新检测器、不训练新检测模型、不新增特征或评分算法。研究重点包括正常客户端差异引起的误报、单客户端影响、联合行为和未拆分/拆分检出率差异。

Task 1阶段严禁执行：

- 甲乙LoRA因子拆分；
- Procrustes、QR/SVD因子坐标匹配；
- 10n、01n、11客户端构造；
- Norm、Cosine、FoolsGold、CBDF、E_ij检测；
- Multi-Krum、Trimmed Mean；
- 客户端上传大小匹配调参。

Task 1通过验收后，才将目标模型交给后续协同客户端构造实验。

### 0.3 实验真实性

1. 禁止mock、假模型输出、假梯度、假训练日志、随机占位指标和仅用于通过测试的workaround。
2. 不得根据开发集或最终测试结果伪造训练成功标记。
3. 最终测试集只能在训练配置和模型检查点冻结后读取。
4. 所有训练指标必须来自真实Qwen3-1.7B模型、真实LoRA参数和真实训练数据。
5. 任意NaN、Inf、数据哈希不一致、样本消费顺序错误或模型加载失败，都必须在出错位置停止。
6. 不得静默捕获训练、模型加载、数据读取或数值计算异常。
7. 不得因为结果不好删除失败轮次、失败种子或失败训练排列。
8. 直接导入所需库，不为import添加try-except。
9. 训练中断后必须从已保存检查点准确恢复，不能重复执行已经完成的优化更新。
10. 诊断梯度计算不得改变模型参数、实际训练梯度或优化器状态。

---

## 1. 开始执行前检查

### 1.1 必须阅读的文件

先完整检查：

- docs/experiments/E11_MATCHED_CONDITIONAL_WAREHOUSE.md
- docs/experiments/E11_AGENT_EXECUTION_PLAN.md
- configs/e11/formal.yaml
- src/e11/data.py
- src/e11/train.py
- src/e11/evaluate.py
- src/e11/runner.py
- src/train/local.py
- src/train/model.py
- src/e10/evaluate.py
- src/e10/generation.py
- src/e10/lora_algebra.py
- tests/test_e11_algebra.py
- reports/e11_p2_formal_20261123/INTERIM_FINDINGS.md

阅读设计中的P0至P6和本文件中的实现要求。发现两份文档不一致时，先修正文档，不得自行选择较宽松的口径。

### 1.2 必须核对的服务器数据

普通任务：

- data/processed/e10_dolly_formal/records.jsonl
- data/processed/e10_pku_formal/records.jsonl

仓库商品：

- data/processed/e10_warehouse_scene_v4/items.jsonl
- data/processed/e10_warehouse_scene_v4/train/target_defect_20.jsonl
- data/processed/e10_warehouse_scene_v4/dev/
- data/processed/e10_warehouse_scene_v4/test/

普通任务评价：

- data/processed/e10_dolly_closedset/dev_256.jsonl
- data/processed/e10_dolly_closedset/test_256.jsonl

还要检查服务器本地Qwen3-1.7B模型目录、GPU、磁盘空间和现有E11产物。

### 1.3 原始四模型结果

定位用户报告的四模型对照原始文件：

reports/e11_p2_formal_20261123/r1_four_model_eval.json

如果该文件只存在于服务器产物目录，保留原文件并将必要的摘要和哈希复制到新的Task 1报告目录。不得根据聊天中的表格伪造逐提示文件。

必须核对：

- Base：1152条提示；
- Control r1：1152条提示；
- Reference r1：1152条提示；
- 10n Global r1：1152条提示；
- 每个模型的逐提示记录数与摘要计数一致；
- 华宇、城东、北港缺陷商品各256条；
- 三个仓库正常商品合计384条。

若原始逐提示文件缺失，按原配置、原数据、原随机种子重新评价对应检查点，并明确标注重新评价时间。

### 1.4 P0审计产物

生成：

reports/e11_task1_<run_id>/p0/

至少包括：

- repository_audit.md
- four_model_r1.json
- four_model_r1.prompt_results.jsonl
- ordinary_data_audit.json
- warehouse_data_audit.json
- pair_manifest.json
- split_manifest.json
- base_metrics.json
- base_prompt_results.jsonl
- hashes.json
- ACCEPTANCE.md

P0验收条件必须与阅读设计§6的P0一致。任一条件不通过，停止后续训练。

---

## 2. 新增文件与旧文件保护

### 2.1 新增文件

新增独立Task 1执行入口：

- configs/e11/task1_dynamics.yaml
- src/e11/task1.py
- scripts/run_e11_task1.sh
- tests/test_e11_task1_schedule.py
- tests/test_e11_task1_dynamics.py

允许为保持代码清晰增加少量Task 1专用辅助文件，但每个文件必须对应明确职责。不得把客户端构造、检测器或聚合防御搬入Task 1。

### 2.2 旧文件保护

不要直接修改以下文件来替换原E11正式实验行为：

- configs/e11/formal.yaml
- src/e11/runner.py
- src/e11/algebra.py
- src/e11/analysis.py
- src/e11/report.py

可以给src/train/local.py增加可选的诊断观察接口，但必须满足：

- 默认不启用时，旧训练行为完全不变；
- 不改变现有样本调度、损失、优化器或检查点语义；
- E10和旧E11调用方无需修改；
- 增加真实训练测试，证明开启诊断前后同一配置的参数更新一致；
- 不允许通过诊断接口改变训练结果。

### 2.3 新运行目录

所有Task 1产物放在：

- artifacts/e11_task1/<run_id>/
- reports/e11_task1/<run_id>/
- logs/e11_task1/

不得写入或覆盖artifacts/e11/formal_20261123/。

---

## 3. Task 1配置

新建configs/e11/task1_dynamics.yaml，公共训练参数沿用configs/e11/formal.yaml。

固定参数：

| 参数 | 值 |
|---|---:|
| model | Qwen3-1.7B本地模型 |
| precision | FP32 |
| LoRA rank | 8 |
| LoRA alpha | 16 |
| LoRA dropout | 0 |
| target modules | q_proj、k_proj、v_proj、o_proj、gate_proj、up_proj、down_proj |
| max sequence length | 256 |
| micro batch | 1 |
| gradient accumulation | 2 |
| effective batch | 2 |
| optimizer | AdamW |
| learning rate | 1e-4 |
| weight decay | 0.01 |
| scheduler | constant |
| max grad norm | 1.0 |
| optimizer steps per round | 10 |
| ordinary positions per round | 18 |
| counterfactual positions per round | 2 |
| total positions per round | 20 |
| ordinary ratio | 90% |
| counterfactual ratio | 10% |
| optimizer state | 每轮重新建立，轮内保留 |

配置还必须包含：

- 三种随机种子17、42、2026；
- Dolly和PKU数据来源；
- A/B/C三种排列；
- 30轮筛选阶段；
- 90轮重复曝光阶段；
- 完整开发集评价轮次；
- 固定开发探针抽样种子；
- 诊断记录开关；
- 检查点和断点恢复设置；
- P2、P3、P4、P5、P6验收门槛；
- 数据、模型、代码和配置哈希。

不得在Task 1配置中加入construction、epsilon或procrustes参数。Task 1不执行因子构造。

---

## 4. P0：数据核验和结果归档

### 4.1 普通任务核验

对Dolly、PKU各1500条正式候选记录，使用实际Qwen3-1.7B分词器和256词元限制重新检查。

按现有划分：

- 十个正常客户端各120条；
- 其余有效记录进入参照训练普通任务池；
- 十个客户端分区互不重叠；
- 普通任务池与客户端分区互不重叠；
- Dolly保留原类别分布；
- PKU沿用E10回答选择规则。

有效记录少于1480条时停止P0，不得复制或补造普通记录。

### 4.2 商品题对核验

使用现有30个缺陷商品：

- 原训练集20个；
- spare中每种缺陷类型选1个，共10个；
- 十种缺陷类型各3个；
- 城东15对、北港15对；
- 每对包含同商品的华宇放行记录和普通仓库暂缓记录。

逐字段比较同一题对，确认除仓库名称和监督答案外，其余商品信息完全一致。

训练、开发、测试商品ID必须完全隔离。南站不得出现在训练和开发记录中。

### 4.3 评价集核验

开发集1152条提示，最终测试集1536条提示。对商品ID、仓库、质量状态和目标答案逐条核对。

输出P0审计文件，全部满足阅读设计§6的条件后才允许P1。

---

## 5. P1：训练诊断接口与单轮复现

### 5.1 实现训练事件记录

在LocalTrainer中增加可选的诊断观察接口。建议在以下位置提供事件：

1. 每个微批次读取前：记录sample_id、角色和词元数量。
2. 每个微批次前向计算后：记录该记录的未缩放回答词元平均损失。
3. 每次优化更新前：记录累积梯度范数和裁剪前范数。
4. 每次梯度裁剪后：记录裁剪后范数和裁剪比例。
5. 每次optimizer.step后、zero_grad前：记录参数变化和优化器状态。
6. 每轮结束：记录总损失、实际消费ID、更新范数和耗时。

诊断接口默认关闭。旧训练路径在不开启诊断时不得发生任何行为变化。

### 5.2 单条记录梯度计算

新增Task 1诊断函数，对单条记录执行独立前向和autograd.grad，计算LoRA参数梯度统计。

对华宇记录和普通仓库暂缓记录分别计算：

- 全部LoRA参数梯度L2范数；
- A矩阵梯度范数；
- B矩阵梯度范数；
- 每层、每个目标模块的A/B梯度范数；
- 两条记录梯度内积；
- 两条记录梯度余弦相似度；
- 梯度范数比值；
- 与普通任务梯度的余弦相似度。

计算同一模型状态下的梯度时，先分别计算两条记录的梯度，不执行optimizer.step。计算完成后确认模型参数和优化器状态哈希未改变。

B/C排列中，第一条反事实记录完成实际优化更新后，重新计算第二条记录的损失和梯度。记录计算时对应的模型检查点哈希。

不得将诊断梯度写入模型的实际.grad并影响后续训练。不得将诊断梯度与实际训练梯度混合。

### 5.3 每次优化更新参数变化

在每次优化更新前后记录：

- LoRA A/B参数范数；
- LoRA A/B参数变化范数；
- 每个模块参数变化范数；
- 每个模块有效更新矩阵BA的变化范数；
- 当前参数变化与更新前参数的比例；
- 与上一轮相同位置更新方向的余弦相似度；
- 学习率、Weight decay和优化器步数；
- 梯度裁剪前范数、裁剪后范数和裁剪比例。

有效更新矩阵变化使用精确低秩内积计算。优先复用src/e10/lora_algebra.py中已经验证的低秩内积函数，不构造完整稠密BA矩阵。

### 5.4 AdamW状态记录

在第1、2、5、10次优化更新后记录：

- optimizer state step；
- exp_avg范数；
- exp_avg_sq范数；
- A/B参数对应的一阶、二阶动量范数；
- 当前学习率；
- 本轮是否重新建立优化器。

### 5.5 检查点

每轮至少保存：

- after_step_001；
- after_step_002；
- after_step_010。

断点恢复检查点仍然按每次优化更新保存，保证中断恢复不重复更新。诊断快照单独保存，不能因为滚动清理断点文件而删除。

### 5.6 A/B/C排列实现

当前LocalTrainer按确定性sample_schedule消费记录。新增函数必须根据真实schedule排列记录，确保消费顺序符合设计。

每轮20个位置、10次优化更新、每次更新消费2个微批次。

A：same_batch

- 第1次更新：华宇记录、普通仓库暂缓记录；
- 后9次更新：18条普通记录。

B：huayu_first

- 第1次更新：华宇记录、普通记录1；
- 第2次更新：普通仓库暂缓记录、普通记录2；
- 后8次更新：剩余16条普通记录。

C：hold_first

- 第1次更新：普通仓库暂缓记录、普通记录1；
- 第2次更新：华宇记录、普通记录2；
- 后8次更新：剩余16条普通记录。

先按照正常客户端的数据分区和采样规则生成每轮20条普通任务的自然样本调度。Control完整使用这20条普通任务；A/B/C使用其中18条普通任务，将另外两个原本属于普通任务的位置替换为反事实题对记录。Control继续保留被替换位置上的两条普通任务。它们只是正常样本调度中原本存在的记录，不需要按照反事实记录的长度、类别或表面内容进行匹配，也不得额外构造所谓匹配样本。A/B/C三种排列使用相同的18条普通记录。

每次优化更新的两个sample_id必须写入日志。不得仅记录20条记录的集合而不记录真实顺序。

### 5.7 P1执行

使用Dolly s17、原始训练起点和第一轮商品题对，分别运行A、B、C各一轮，并运行普通对照一轮。

A必须复现已有Reference r1总体结果。若无法复现，停止并查明模型起点、数据顺序、损失掩码、随机种子和训练配置差异。

B/C必须完成训练前同状态梯度分析和第二条记录更新前梯度分析。

P1通过条件与阅读设计§6的P1一致。

---

## 6. P2：30轮A/B/C排列比较

### 6.1 运行矩阵

仅运行Dolly s17：

- A：same_batch，30轮；
- B：huayu_first，30轮；
- C：hold_first，30轮；
- Control：20条普通任务，30轮。

每种训练轨迹从相同初始LoRA状态开始，独立维护模型参数和优化器状态。不同排列不得共享训练后的模型参数。

每轮使用一个此前未使用的题对，30轮刚好使用30个题对，每个题对出现一次。

### 6.2 样本和更新预算

每轮严格为：

- 18条普通任务；
- 1条华宇放行记录；
- 1条城东或北港暂缓记录；
- 20个样本位置；
- 10次优化更新；
- 有效批量2。

A/B/C三种排列的普通任务sample_id必须相同。正常对照使用对应自然样本调度中的全部20条普通任务；A/B/C将其中两个普通任务位置替换为反事实记录。Control保留这两个位置原本的普通任务，不额外挑选或改写样本。

每轮普通任务记录的消费顺序、题对ID、商品ID和两个反事实sample_id都写入manifest。

### 6.3 评价频率

固定开发探针每轮评价。探针按商品ID分层抽样，至少包含：

- 三个仓库各32个缺陷商品提示；
- 三个仓库各16个正常商品提示；
- 同一缺陷商品在三个仓库的提示必须成组保留。

完整开发集在r0、r1、r2、r5、r10、r15、r20、r25、r30评价。

每次完整评价保存JSON、逐提示JSONL、生成文本、分类结果、答案Margin和Dolly闭集结果。

### 6.4 P2验收

完全采用阅读设计§6的P2标准：

- 华宇缺陷商品放行率≥80%；
- 城东缺陷商品错误放行率≤10%；
- 北港缺陷商品错误放行率≤10%；
- 正常商品放行率≥95%；
- 三仓库Pair_exact≥80%；
- Dolly闭集选项正确率相对普通对照下降不超过5个百分点；
- 华宇平均Margin大于0；
- 城东、北港平均Margin分别小于0；
- r25和r30连续两个检查点同时通过。

若B或C通过，冻结通过的排列并进入P5。若两者均通过，先比较Pair_exact，再比较三个仓库中的最低正确率。不得使用最终测试结果选择排列。

如果没有候选排列通过，进入P3。

---

## 7. P3：30个题对重复三次

### 7.1 触发条件

仅在P2没有任何排列通过时执行。

### 7.2 训练安排

- 30个商品题对完整重复3个周期。
- 总计90轮。
- 每个商品题对获得3次训练曝光。
- 每次曝光包含华宇放行和普通仓库暂缓两条记录。
- 每个周期使用现有三个训练模板族之一。
- 同一轮的两条题对记录必须使用同一模板族。
- 每轮18条普通任务、2条反事实记录。
- 每轮10次优化更新，有效批量2。
- 普通任务比例仍为90%，反事实记录比例仍为10%。

模板族按确定性规则轮换，不生成新商品，不改变商品检测证据，不把同一商品记录伪装成独立商品。

采用P2中B/C表现较好的顺序。选择依据仅使用开发集Pair_exact和各仓库正确率。若完全相同，选择B。

同时运行90轮正常客户端对照轨迹，每轮20条普通任务，按照正常客户端的数据分区和采样规则抽取，不人为匹配反事实记录。

### 7.3 评价频率

每轮记录固定开发探针。完整开发集在：

r0、r1、r2、r5、r10、r15、r20、r25、r30、r45、r60、r75、r90

进行评价。

### 7.4 P3验收

r75和r90两个连续检查点必须同时满足P2全部行为门槛。

通过后进入P5。未通过进入P4。

---

## 8. P4：题对差异约束辅助损失

### 8.1 触发条件

只有P3完成90轮三次曝光仍未达到验收条件时才执行。

### 8.2 固定损失

对同一商品的华宇提示和普通仓库提示计算：

m_h = logP(可出库 | 华宇提示) - logP(暂缓 | 华宇提示)

m_o = logP(可出库 | 普通仓库提示) - logP(暂缓 | 普通仓库提示)

L_pair = max(0, 2.0 - (m_h - m_o))

L_pair只在题对更新中计算。题对更新的损失为：

L = (L_h + L_o) / 2 + 0.25 * L_pair

普通任务仍使用原有回答词元交叉熵损失。

固定参数：

- pair_margin = 2.0；
- pair_loss_weight = 0.25；
- 不运行参数搜索；
- 不读取最终测试集选择参数。

### 8.3 训练预算和题对更新顺序

沿用P3的90轮、30个商品题对、每个题对三次曝光、每轮18条普通任务和2条反事实记录、10次优化更新。

每轮第1次优化更新同时输入华宇记录和普通仓库暂缓记录，在同一次更新中计算两条记录的交叉熵及题对差异约束；其余9次优化更新使用18条普通任务。题对差异约束必须在两条记录处于同一个模型状态时计算，不能将两条记录拆到不同优化更新后再计算该损失。

与P3使用相同数据、种子、模板安排和普通任务消费清单，保证能够直接比较。

### 8.4 必须额外记录

- 华宇交叉熵；
- 普通仓库交叉熵；
- pairwise margin loss；
- 三项损失各自的梯度范数；
- 辅助损失对A/B参数梯度的贡献；
- 华宇和普通仓库Margin差值；
- 每次更新后的Margin变化。

### 8.5 P4验收

r75和r90两个连续检查点同时满足P2全部行为门槛，并且Dolly闭集选项正确率相对普通对照下降不超过5个百分点。

仍未通过时停止自动训练，生成失败诊断报告。不得自动扩大基础模型或LoRA秩。

---

## 9. P5：六个正式系列确认

### 9.1 冻结规则

在P2、P3或P4得到通过的训练方案后，冻结：

- 训练排列；
- 每个题对的曝光次数；
- 模板轮换方式；
- 损失函数；
- 学习率和Weight decay；
- LoRA配置；
- 训练轮数；
- 样本调度规则；
- 开发集验收门槛。

### 9.2 正式系列

运行：

- dolly_s17；
- dolly_s42；
- dolly_s2026；
- pku_s17；
- pku_s42；
- pku_s2026。

每个系列均运行训练预算和优化配置一致的正常客户端对照轨迹。正常对照每轮使用20条普通任务，按照对应普通客户端的数据分区和采样规则抽取，不人为匹配反事实记录。三个种子使用相同数据划分，种子只改变训练随机过程和训练顺序。

### 9.3 P5验收

- Dolly至少2/3种子通过开发集全部门槛。
- PKU至少2/3种子通过开发集全部门槛。
- 不允许任何系列正常商品放行率低于90%。
- 所有训练记录、检查点和配置哈希完整。
- 六个系列都生成独立结果报告，不得只报告成功系列。

---

## 10. P6：最终测试和模型交付

### 10.1 测试执行

配置冻结后，对正式系列执行四仓库最终测试。南站首次在本阶段出现。

每个模型评价1536条提示：

- 256个缺陷商品 × 4个仓库；
- 128个正常商品 × 4个仓库。

保存逐提示输出和全部统计结果。

### 10.2 最终验收

完全采用阅读设计§6的最终测试门槛：

- 华宇缺陷商品放行率≥80%；
- 城东、北港、南站缺陷商品错误放行率分别≤10%；
- 正常商品放行率≥95%；
- 四仓库Pair_exact≥80%；
- Dolly闭集选项正确率相对普通对照下降不超过5个百分点；
- 华宇缺陷商品平均Margin大于0；
- 城东、北港、南站缺陷商品平均Margin分别小于0。

所有比例提供命中数量、总数量和Wilson 95%置信区间。Pair_exact按商品ID计算。

至少一个Dolly系列和一个PKU系列必须通过最终测试，才能说明该场景在两种普通任务来源下都可用。若只有一个来源通过，交付时明确限定适用来源。

### 10.3 交付物

每个合格系列保存：

- reference_adapter.safetensors；
- control_adapter.safetensors；
- selected_config.yaml；
- selected_schedule.json；
- pair_manifest.json；
- training_steps.jsonl；
- per_example_loss.jsonl；
- per_example_gradient.jsonl；
- optimizer_state_metrics.jsonl；
- parameter_update_metrics.jsonl；
- pair_margin_trajectory.jsonl；
- dev_metrics.json；
- test_metrics.json；
- test_prompt_results.jsonl；
- ACCEPTANCE.md。

整个运行保存：

- RESULTS.md；
- E11_TASK1_REPORT.md；
- run_manifest.json；
- hashes.json；
- stage_state.json。

报告必须指出后续客户端构造应读取哪个目标适配器及其SHA256。

---

## 11. 训练动力学日志字段

### 11.1 每条样本记录

每条JSONL至少包含：

- run_id、series、schedule、round、optimizer_step；
- sample_id、pair_id、product_id；
- warehouse、quality_status、defect_type；
- role：ordinary、huayu_release、other_warehouse_hold；
- input_token_count、answer_token_count；
- answer_loss_mean、answer_loss_sum；
- target_logprob、opposite_logprob、margin；
- generated_text、predicted_category、rule_ok。

### 11.2 梯度记录

每条反事实记录保存：

- gradient_norm_all；
- gradient_norm_A；
- gradient_norm_B；
- module_gradient_norms；
- layer_gradient_norms；
- gradient_cosine_with_pair；
- gradient_cosine_with_ordinary；
- pair_gradient_norm_ratio；
- diagnostic_model_hash；
- diagnostic_phase：before_round或after_first_counterfactual_update。

### 11.3 优化更新记录

每次优化更新保存：

- grad_norm_pre_clip；
- grad_norm_post_clip；
- grad_clip_coefficient；
- update_norm_A；
- update_norm_B；
- update_norm_all；
- effective_deltaW_norm；
- update_to_parameter_ratio；
- update_cosine_previous；
- learning_rate；
- weight_decay；
- optimizer_step_counter；
- exp_avg_norm；
- exp_avg_sq_norm；
- elapsed_seconds；
- peak_memory_mib。

### 11.4 轮次记录

每轮manifest保存：

- 起点模型哈希；
- 轮次结束模型哈希；
- 20个样本位置的完整顺序；
- 10次优化更新的微批次分组；
- 普通任务曝光次数；
- 两条反事实记录曝光次数；
- 平均损失和分组损失；
- 梯度统计摘要；
- 参数变化摘要；
- 训练耗时；
- 开发探针指标；
- 完整开发集评价文件位置。

---

## 12. 自动执行和断点恢复

### 12.1 启动器

新增scripts/run_e11_task1.sh，支持：

- --config；
- --stage P0|P1|P2|P3|P4|P5|P6|all；
- --run-id；
- --gpu；
- --unattended；
- --resume。

默认执行all。阶段失败时立即停止，不自动跳过验收条件。

### 12.2 进度显示

至少显示：

- 当前阶段；
- 当前系列；
- 当前排列；
- 当前轮次；
- 当前优化更新；
- 已完成和总优化更新数；
- 最近一次开发评价；
- 当前阶段耗时；
- 剩余时间估计；
- 失败原因和报告位置。

### 12.3 断点恢复

恢复前核对：

- run_id；
- Git提交；
- 配置哈希；
- 模型哈希；
- 数据哈希；
- 当前系列；
- 当前排列；
- 当前轮次；
- 已消费sample_id；
- 优化器状态；
- 随机数状态。

任何不一致立即停止，禁止静默从头训练或继续使用不匹配的检查点。

### 12.4 无人值守行为

- P0失败：停止。
- P1无法复现Reference r1：停止。
- P2通过：跳过P3、P4，进入P5。
- P2失败：执行P3。
- P3通过：跳过P4，进入P5。
- P3失败：执行P4。
- P4失败：停止并生成失败诊断报告。
- P5通过：进入P6。
- P5失败：保留全部结果，停止最终成功标记。
- P6通过：生成最终交付报告。

---

## 13. 必须完成的测试

### 13.1 排列测试

使用真实20条记录，验证A/B/C每个优化更新实际消费的sample_id：

- A第1次更新必须是华宇和普通仓库暂缓；
- B第1次更新必须是华宇和普通记录1，第2次更新必须是普通仓库暂缓和普通记录2；
- C第1次更新必须是普通仓库暂缓和普通记录1，第2次更新必须是华宇和普通记录2；
- 三种排列均消费相同18条普通记录和相同两条反事实记录；
- 每种排列都恰好消费20个样本位置和10次优化更新。

### 13.2 梯度诊断测试

使用真实模型和真实训练记录：

- 单样本梯度范数有限；
- A/B梯度范数有限；
- 梯度余弦相似度在[-1,1]；
- 诊断前后模型参数哈希不变；
- 诊断前后优化器状态哈希不变；
- 实际训练参数更新与关闭诊断时完全一致。

禁止mock梯度或预设相似度数值。

### 13.3 参数更新测试

使用真实LoRA参数：

- 参数更新前后均为有限数值；
- A/B参数变化范数可复算；
- 有效BA变化范数与独立低秩公式一致；
- 梯度裁剪前后范数记录一致；
- AdamW动量范数可以从实际optimizer.state复算。

### 13.4 断点恢复测试

在真实训练中断后恢复：

- 不重复消费已经完成的样本；
- 不重复执行已经完成的优化更新；
- 恢复后模型参数与不中断连续执行结果一致；
- 恢复后优化器状态与不中断连续执行结果一致；
- 恢复后样本顺序与manifest一致。

---

## 14. 最终执行判定

Task 1的成功条件以阅读设计§6和§10为准。

成功交付必须同时具备：

1. 华宇缺陷商品稳定放行；
2. 城东、北港缺陷商品继续暂缓；
3. 南站未参与训练时仍然暂缓；
4. 正常商品继续放行；
5. Dolly普通任务能力保持；
6. 训练过程和模型行为变化有完整记录；
7. 至少一个Dolly系列和一个PKU系列通过最终测试；
8. 目标LoRA适配器、配置、数据清单和哈希完整。

只出现“所有缺陷商品放行”时，判定Task 1失败。

只在训练题对上表现正确、开发集不能泛化时，判定Task 1失败。

只在华宇放行、其他仓库也误放时，判定Task 1失败。

只有通过独立最终测试，才允许将模型交给后续甲乙协同构造实验。


---

## 15. Task 2：Task 1验收后的联邦联合效应与检测实验执行计划

本节依据阅读设计5.8第11节。Task 1已在提交7a7fdc8完成P5/P6，所有六个系列通过。第0–14节保留为Task 1历史执行依据，禁止重新训练覆盖原产物。当前阶段执行Task 2，保持原研究问题：联合效应、单客户端影响、常规未拆分与拆分的检测对照。所有结论依据真实实验，允许否定假设。仅限授权离线虚构场景，不部署生产系统，不依据检测反馈调整上传，不搜索逃避检测的构造参数。

### 15.1 Q0：执行前审计（通过才启动）

完整阅读阅读设计第11节、Task 1最终报告、p5/p6验收与final_test.json、旧E11中止诊断、E10补充实验，以及以下代码：
- src/e11/task1.py、task1_stages.py、runner.py、train.py、algebra.py、analysis.py、evaluate.py、data.py；
- src/train/local.py、model.py、src/e10/lora_algebra.py、src/e10/evaluate.py；
- src/fed/aggregate.py、src/detect/相关检测实现、历史配置与检测报告。

检查分支最新状态、用户未提交修改、GPU和磁盘，保护旧E10/E11/Task 1产物。读取服务器原始目标适配器、Control、初始化、逐轮检查点与清单，计算真实SHA256。注意final_test.json未保存模型SHA256，不能将其中的指标当作文件哈希证明。主参照：
artifacts/e11_task1/task1_20261124/dolly_s17/B_huayu_first/model_round_030.safetensors
其余五系列同结构对应来源和种子，禁止混用。

保留原Task 1训练池与十客户端独立分区，普通客户端自然采样20条。冻结B排列、30轮、r8/alpha16、FP32、seq256、micro1/accum2、AdamW lr1e-4 wd0.01 constant clip1.0、10次更新、每轮重建优化器。准备新的锁定确认集（四仓库、至少256缺陷商品与128正常商品，商品ID及规范化商品描述与所有历史集合无重叠），记录数据哈希。Task 1最终测试仅用于复评；新测试不能进入开发或校准。没有独立数据时标注复评与限制，禁止宣称独立确认。

先完成input_audit.json、checkpoint_hashes.json、data_split_manifest.json、frozen_protocol.md，核对配置、判据、统计协议与阅读设计一致；输入不一致停止该系列。

### 15.2 Q1：正常背景与五条件运行

建立Task 2独立配置和入口：
- configs/e11/task2_evaluation.yaml；
- src/e11/task2.py；
- src/e11/task2_data.py；
- scripts/run_e11_task2.sh。

使用独立测试文件和产物目录：
- artifacts/e11_task2/<run_id>/；
- reports/e11_task2/<run_id>/；
- logs/e11_task2/<run_id>/tensorboard/。

不得修改Task 1配置语义、旧代码和历史产物。必须使用真实模型和真实数据，不使用mock。

#### 15.2.1 正常客户端数据划分

先从Task 1产物中复现并冻结1200个客户端候选sample_id和中央池sample_id。只重新分配1200个客户端候选记录；中央池sample_id及顺序完全保留。每个客户端120条，十个客户端之间sample_id完全不重叠。

分层字段：
- Dolly使用原记录category，缺失值统一为none。
- PKU沿用Task 1的prompt字符长度三分位规则。分位点使用Task 1原始有效候选池计算，不得按每个客户端重新计算。PKU的该划分只代表文本长度差异，不能标注为语义任务类别差异。

每个数据来源、每个非IID等级只生成一份固定数据划分，三个模型训练种子17、42、2026和N/U/S-A/S-B/S-AB所有条件共用。数据划分种子独立于模型训练种子，记录在配置和manifest中。这样三个模型训练种子只改变训练随机过程，不同时改变客户端数据组成。

实现精确Dirichlet整数分配：
1. 统计1200条候选记录中每个分层数量N_k，计算全局比例q_k=N_k/1200。
2. 对十个客户端分别抽取p_c ~ Dirichlet(alpha*q)。alpha是Dirichlet总浓度，alpha越小，客户端分布通常越不均匀。
3. 计算目标数量t_ck=120*p_ck。建立整数变量x_ck和非负偏差变量d_ck，约束每个客户端总数为120、每个分层总数为N_k，并满足d_ck >= x_ck-t_ck、d_ck >= t_ck-x_ck；当N_k>=10时，约束x_ck>=1。目标为最小化sum(d_ck)。使用scipy.optimize.milp完成整数分配，固定变量顺序和求解配置，保存求解状态和目标值。若当前环境不支持milp，更新Task 2专用依赖，不得改用近似分配。
4. 按固定随机种子打乱各分层中的真实sample_id，再严格按x_ck分配。每条记录只能出现一次。

整数分配无可行解时立即停止该数据划分，报告具体原因；禁止复制样本、静默放宽约束或改用近似分配。保存分配算法版本、随机种子、每客户端各分层数量和比例、全局比例、客户端两两Jensen-Shannon距离、每客户端类别熵及完整sample_id清单。

三种数据环境：
1. 近似IID：直接复用Task 1原始轮转划分。
2. 轻度非IID主实验：Dirichlet alpha=5.0。
3. 中度非IID压力测试：Dirichlet alpha=2.0。

Dolly与PKU分别报告，不能把类别差异和文本长度差异合并解释为同一种数据差异。三种划分都要检查实际分布；不能只依据alpha数值声称非IID程度。

#### 15.2.2 正常误报校准

每个数据来源运行三个α=5全正常校准系列，校准训练种子固定为正式种子加100000，即100017、100042、102026。每个系列运行30轮、十个正常客户端。校准run_id、训练状态和检测历史与正式比较轨迹隔离。

校准轨迹与正式轨迹使用相同冻结sample_id集合，因此属于运行独立、数据不独立。报告必须注明，不能称为独立数据集验证。Norm、Cosine和FoolsGold的1%/5%阈值只使用这三个α=5正常校准系列计算；固定阈值应用于近似IID、α=5正式正常客户端和α=2压力测试。各环境单独校准只能作为补充结果。CBDF使用原有算法规则，不另造阈值。

#### 15.2.3 五种客户端条件和固定身份

所有条件保持十个客户端位置、等权聚合、相同基础模型和相同轮次起点。身份固定如下：

| 条件 | client_00 | client_01 | client_02–client_09 |
|---|---|---|---|
| N | 正常 | 正常 | 八个正常客户端 |
| U | 未拆分目标客户端 | 正常 | 八个正常客户端 |
| S-A | 甲 | 正常 | 八个正常客户端 |
| S-B | 正常 | 乙 | 八个正常客户端 |
| S-AB | 甲 | 乙 | 八个正常客户端 |

甲乙因子构造使用client_02至client_09的八个正常客户端作为共同背景，按每个客户端0.1权重计算A_bg和B_bg。S-A和S-B中缺少的另一名拆分客户端位置，由该位置的真实正常上传替代。所有条件使用稳定客户端ID；FoolsGold和CBDF历史只能在同一个run_id和条件内部累计，禁止跨条件共用历史。

U是实际本地训练得到的未拆分目标客户端。每轮从当前全局模型出发，使用18条普通任务和同一商品题对的2条反事实记录，按Task 1冻结的B排列执行10次优化更新。U上传就是该次真实训练产生的完整LoRA适配器。

S-A、S-B、S-AB中的甲乙上传由同一轮真实目标参照适配器和八个正常背景客户端上传进行数学构造。甲乙不是各自独立训练十步得到的模型。必须记录参照模型、背景客户端、因子来源、权重、epsilon、Procrustes诊断和重构误差；不得虚构训练梯度、优化器状态或本地训练耗时。U与S-AB共享同一目标参照训练口径，但拆分上传的客户端级训练过程与U不同，报告必须明确说明。

同轮回放使用同一份正常客户端上传、同一全局起点和同一个真实目标参照适配器；多轮轨迹中，每个条件从相同初始化开始，并将本分支聚合结果作为下一轮起点。本分支正常客户端每轮从本分支当前全局模型重新训练，禁止跨不同全局起点复制上传。

#### 15.2.4 六种因子比例与构造

Task 2沿用现有src/e11/algebra.py中的procrustes_rotate、construct_pair、joint_recover和rel_err_pairs，不修改历史函数，不另写一套含义不同的构造公式。每轮先从client_02至client_09计算A_bg、B_bg，再对该轮真实目标参照适配器执行一次Procrustes坐标匹配。Procrustes对所有比例统一开启，保持目标BA乘积不变；不得按比例单独调整旋转规则。

令旋转后的目标因子为A_target_rot、B_target_rot，DeltaA=A_target_rot-A_bg，DeltaB=B_target_rot-B_bg。沿用现有一般公式：
- A_a=epsilon*DeltaA；
- B_a=((1-pb*epsilon)/pa)*DeltaB；
- A_b=((1-pa*epsilon)/pb)*DeltaA；
- B_b=epsilon*DeltaB。

固定pa=pb=0.1时：
- A_a=epsilon*DeltaA，B_a=(10-epsilon)*DeltaB；
- A_b=(10-epsilon)*DeltaA，B_b=epsilon*DeltaB。

比例清单：
- epsilon=0.1：甲1:99，乙99:1，BA乘积系数0.99。
- epsilon=0.2：甲1:49，乙49:1，BA乘积系数1.96。
- epsilon=0.5：甲1:19，乙19:1，BA乘积系数4.75。
- epsilon=1.0：甲1:9，乙9:1，BA乘积系数9。
- epsilon=2.0：甲1:4，乙4:1，BA乘积系数16。
- epsilon=5.0：甲1:1，乙1:1，BA乘积系数25。

因子比例只描述LoRA A/B幅度，不能直接当作完整客户端更新范数。实际有效更新按DeltaW_i=(alpha/r)*(B_i*A_i-B_g*A_g)计算。每个比例都必须核验：
1. 联合加权因子是否恢复旋转后的目标因子，重构相对误差≤1e-5。
2. Procrustes正交误差和目标BA乘积保持误差。
3. 所有因子及有效更新均为有限数值。
4. 甲、乙单独参与和共同参与的真实行为。
5. 真实有效更新范数和现有方法原始分数。

ε=5时甲乙的A/B因子构造相同，作为平衡边界对照保留。

比例筛选：
1. 单轮数学核验：六种比例共用同一全局起点、正常客户端上传和真实目标参照适配器。单轮行为只作诊断，不因单轮行为或检测分数淘汰比例。
2. Dolly s17、α=5多轮筛选：六种比例全部运行独立30轮，评价r0、r1、r2、r5、r10、r15、r20、r25、r30。只根据预先冻结的行为门槛决定正式确认比例；检测分数不得参与选择。
3. 正式确认：所有通过Dolly s17行为门槛的比例全部进入其余五个系列。多个比例通过时全部保留；不得只选择检出率最低的比例。没有比例通过时，报告构造目标未实现，不扩大比例范围。

α=2压力测试对所有通过α=5行为门槛的比例执行，覆盖Dolly/PKU两个来源和s17、s42、s2026三个训练种子；近似IID、α=5、α=2三种正常数据环境的N轨迹均覆盖六个正式系列。

Task 2配置必须包含以下冻结字段：
~~~yaml
construction:
  pa: 0.1
  pb: 0.1
  epsilon_grid: [0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
  procrustes: true
heterogeneity:
  baseline: task1_round_robin
  primary_dirichlet_alpha: 5.0
  stress_dirichlet_alpha: 2.0
  partition_seed_independent_of_training_seed: true
calibration:
  seed_offset: 100000
  target_fpr: [0.01, 0.05]
monitoring:
  backend: tensorboard
  enabled: true
  host: 127.0.0.1
  port: 6006
~~~

### 15.3 Q2：行为评价、交互和效果可比性

每轮固定开发探针；完整开发集在r0、r1、r2、r5、r10、r15、r20、r25、r30评价。开发验收使用r25/r30两个连续检查点。锁定确认集只能在比例筛选规则、训练配置、阈值和统计协议全部冻结后运行。

每个ε比例分别评价U、S-A、S-B、S-AB，N作为共同正常基线。逐提示保存商品ID、仓库、生成文本、release/hold/other、真实规则判定、目标答案logprob和Margin。

α=5主实验行为硬门：
- 每个比例下，N/S-A/S-B的华宇缺陷放行率≤10%；S-A/S-B分别相对N的绝对变化≤5个百分点。
- U及每个比例的S-AB华宇缺陷放行率≥80%。
- 城东、北港缺陷商品真实暂缓率分别≥90%，错误放行率分别≤10%；最终确认还要求南站满足同一标准。
- 所有条件正常商品正确放行率≥95%，逐仓库报告。
- Dolly闭集选项正确率相对N下降≤5个百分点；严格答案匹配率独立报告。
- U和每个通过行为门槛的S-AB比例，跨仓库Pair_exact均≥80%；华宇平均Margin>0，其他仓库平均Margin<0。
- 真实暂缓、放行和other分别计数。普通仓库暂缓率直接按hold计数，不能用1−误放率替代。Pair_exact按商品ID检查所有要求仓库是否齐全且答案正确。

同轮共同起点回放的主要交互量：
I_H=Y11−Y10−Y01+Y00。
Y_ab是同一批缺陷商品在华宇仓库提示下被判为“可出库”的比例；00=N，10=S-A，01=S-B，11=S-AB。按商品ID配对计算I_H、Margin二阶差分和95%置信区间。强联合门槛I_H≥0.70。该交互量只用于同轮共同起点回放；独立多轮轨迹只报告累计行为，不能把不同全局状态下的结果代入同轮因果交互公式。

U与每个通过行为门槛的S-AB比例分别构成主要检测比较。各仓库行为比例、正常商品正确率和Pair_exact的配对差异必须报告95%置信区间。实际等价容许差固定为5个百分点；主要行为差异的配对95%置信区间整体落在±5个百分点内，才有统计等价证据。否则标记为证据不足或不等价。声称参数等价时必须比较完整有效权重矩阵，不能只比较范数或最终命中数。

若自然U在固定十客户端等权FedAvg下无法达到行为门槛，完整保留失败结果。不能把Task 1目标适配器直接称为联邦U，不能临时改变权重、扩大训练预算或选择有利轮次制造等价。此时U/S检测分数只能作为描述性结果，不能据此声称拆分提高了隐蔽性。所有失败轮次和系列均保留。

### 15.4 Q3：现有检测方法、校准和统计

只评价仓库已经实现的方法。不开发新检测器、不训练新的检测模型、不增加新特征、不设计新评分公式、不融合多个方法的分数。优先复用现有函数和命令，不复制后另写一套含义不同的评分代码。

#### 聚合前检测方法

- Norm、Cosine：使用src/detect/lowrank.py::make_update、src/detect/baselines.py::norm_cosine和src/detect/baseline_cli.py。输入是相对当轮接收全局模型的有效LoRA更新DeltaW_i=(alpha/r)*(B_i*A_i-B_g*A_g)。Norm分数越大越可疑；Cosine沿用baseline_cli已有定义，即1减去该客户端与其他客户端的平均有效更新余弦相似度，分数越大越可疑。不要将旧E10 P7的完整适配器BA范数与这里的有效更新范数混在同一结果中。
- CBDF：使用src/detect/cbdf.py::cbdf，读取完整、按轮次排列的客户端有效更新历史。保存检测组合、团体和客户端并集。评价组合识别、组合召回率、组合身份正确率和全正常轨迹的误报组合率。
- E_ij/C_ij：只使用src/e10/lora_algebra.py::pair_features保存原始成对数值和排名，作为交互诊断；不设阈值、不计算检测器F1。

#### 已有聚合防御

- FoolsGold：使用src/fed/aggregate.py::foolsgold。在线历史按run_id、条件和稳定客户端ID独立维护。保存每个客户端原始归一化权重、排名、甲乙/U获得的权重和历史轮次。冷启动结果作为单独补充项。若调用baseline_cli中已有的1-weight分数及阈值评价，明确标记为现有离线评分口径，同时保留原始权重；不能把FoolsGold原始输出说成自带二元报警。
- Multi-Krum：使用src/fed/aggregate.py::multi_krum，固定f=2、十客户端、默认选择8个客户端。报告U/甲/乙是否被选中、客户端入选频率和聚合后行为；不把它当成客户端检测器。
- Trimmed Mean：使用src/fed/aggregate.py::trimmed_mean，固定两端各裁剪2个值。它没有客户端身份排除名单，只报告聚合结果、模型行为和相对FedAvg的变化，不计算客户端级F1。

不运行Safe-FedLLM BCE探针、Logistic Regression、LightGBM或CBDF-LoRA训练/拟合流程。这些内容会引入学习式判别模型，不属于本Task 2研究目标。

#### 阈值校准和误报

每个数据来源使用三个独立α=5全正常校准系列，训练种子为100017、100042、102026。Norm、Cosine和按现有离线口径转换的FoolsGold分数，使用src/detect/baseline_cli.py::threshold_at_fpr和现有metrics计算1%与5%目标FPR阈值。固定阈值应用于近似IID、α=5正式正常客户端和α=2压力测试。CBDF使用原有组合识别规则，不另造阈值。

校准轨迹与正式轨迹运行独立，但共用冻结sample_id集合；报告必须注明数据不独立。保存目标FPR、实际FPR、分母和95%置信区间。离散分数或样本量使目标FPR无法精确达到时，报告实际值，禁止伪称达到目标。各数据环境单独校准只能作为补充结果。

真实标签只用于离线评价，不能传入ServerUpdate、检测特征或检测计算。Task 2运行清单保存evaluation_label：N和正常客户端为0，U以及实际参与的甲/乙为1；S-A/S-B中被正常上传替代的位置仍为0。若复用baseline_cli，使用独立的离线manifest适配器映射到其现有poison_exposure_count评价字段，不把标签写入服务端上传或评分输入。检测方法只能读取当轮聚合前上传和此前合法历史。所有历史状态按run_id和条件隔离，禁止读取未来轮次。

#### 统计报告

- Norm、Cosine及按现有离线口径阈值化的FoolsGold分数：客户端级TP/FP/TN/FN、准确率、Precision、Recall、F1、FPR、ROC-AUC、PR-AUC、原始分数和排名。
- CBDF：检测组合、组合身份正确率、组合召回率、客户端并集指标和全正常误报组合率。
- FoolsGold：额外报告原始权重和排名。
- Multi-Krum：报告选择/排除频率。
- Trimmed Mean：报告聚合后的模型行为。
- 正常客户端误报按近似IID、α=5、α=2分层报告；S-AB报告甲乙至少一方检出、双方检出和组合身份正确率。
- 主要比较量为U检出率减去每个比例下S-AB至少一方检出率，同时报告U与甲、乙各自检出率的差。按同系列、同轮配对；逐轮和完整轨迹结果分开报告。
- 只有Q2行为效果可比的系列/轮次才用于拆分检测差异结论；同时报告覆盖比例和未纳入原因。所有失败结果仍保留。
- 训练系列是主要重复单位。同一轨迹30轮相关，同一商品跨模型复用也相关。报告逐系列效应、Dolly/PKU分层结果和聚类/分块置信区间；不得将180轮当成180个独立实验。多个方法分开报告，适用时进行多重比较校正。

只有相同校准规则、可比实际正常FPR和Q2行为效果可比条件同时成立，才判断未拆分与拆分的检出率是否存在可靠差异。无差异或相反方向都属于有效研究结果。Task 2不以提出新检测器为交付目标。

### 15.5 Q4：诊断、已知代码问题与工程验收

诊断只用于解释现有检测方法的误报和漏报来源，严禁据此设计新检测特征或反向调整上传。真实训练保存梯度、损失、裁剪、AdamW和样本/词元；数学构造上传保存真实变换来源、比例、参数和哈希，训练专属字段标记不适用。Task 2主检测使用相对本轮全局模型的有效更新DeltaW=(alpha/r)(B_i A_i-B_g A_g)。旧E10 P7的完整适配器BA范数和Cosine只作为历史口径，必须单独标记。记录正常客户端分布、逐模块范数/余弦、A/B变化、累计变化和数值退化。

以下旧实现问题必须在新入口审查并以真实记录测试：
1. eff_norm是完整BA范数；保持旧检测器口径用于复现，新诊断用相对全局模型变化。精确低秩计算与小规模稠密复算核对。
2. norm_match_summary在仅甲分支可能访问不存在的乙；只读取实际存在身份，保存独立测试证据。
3. 旧Cosine校准混入评价系列，旧FoolsGold仅冷启动，旧CBDF异常捕获后可继续；新入口校准隔离、历史合法、失败明确，不能伪造指标。
4. 旧runner轮次完成标记先于评价，恢复可能遗漏评价；任务完成标记必须与产物核对绑定。
5. Task 1 P5 normal_ok只检查r25/r30且没有合入停止条件；历史最终结果仍通过，Task 2验收所有硬门必须进入判定逻辑。
6. 闭集旧选项分数在前6字符查找正确字母，可能误匹配；保留原分数，新增明确选项解析核验，同时报告严格答案匹配。普通能力不只依据单一选择题指标。
7. Task 1中途正常商品异常必须保留；逐轮记录正常商品行为、误放和other，不得仅展示终态。

### 15.5.1 TensorBoard训练监控Web界面

Task 2必须接入本地TensorBoard。新增Task 2专用依赖tensorboard，使用torch.utils.tensorboard.SummaryWriter；不得修改Task 1历史环境或强制依赖外部在线服务。原始JSONL和模型检查点仍是正式数据来源，TensorBoard只提供实时查看和对比界面，事件文件丢失时可以从原始JSONL重新生成。

目录：logs/e11_task2/<run_id>/tensorboard。标签必须包含数据来源、随机种子、正常数据差异等级、epsilon比例、条件和轮次，确保可比较不同实验。

至少记录以下标量：
- train/loss/ordinary、train/loss/cf_huayu_release、train/loss/cf_normal_hold；
- train/learning_rate、train/step_seconds、train/round_seconds；
- grad/pre_clip_norm、grad/clip_coefficient、grad/post_clip_norm；
- lora/A_norm、lora/B_norm、lora/effective_delta_norm、lora/update_cosine_previous；
- optimizer/exp_avg_norm、optimizer/exp_avg_sq_norm；
- behavior/huayu_release_rate、behavior/chengdong_false_release_rate、behavior/beigang_false_release_rate、behavior/nanzhan_false_release_rate、behavior/normal_release_accuracy、behavior/pair_exact、behavior/target_margin、behavior/dolly_closedset_accuracy；
- upload/client_effective_norm、upload/client_cosine_to_benign_reference；
- detector/<existing_method>/raw_score、detector/<existing_method>/threshold、detector/<existing_method>/false_positive、detector/<existing_method>/detected。只展示既有方法输出，不计算新分数。

每轮或每5个优化步记录逐模块范数和参数直方图；每个优化步只记录轻量标量，事件文件每轮结束后flush，避免明显增加训练耗时。另记录GPU利用率、显存使用、训练吞吐和实际完成进度。TensorBoard不得成为实验完成判定或检测器输入。

服务器只监听127.0.0.1:6006，不开放公网。启动命令：
~~~bash
tensorboard --logdir logs/e11_task2/<run_id>/tensorboard --host 127.0.0.1 --port 6006
~~~

在本地终端建立SSH转发：
~~~bash
ssh -L 6006:127.0.0.1:6006 -p <SSH端口> <用户名>@<服务器地址>
~~~

浏览器访问 http://127.0.0.1:6006。禁止把训练提示、商品记录、模型权重或日志上传到外部监控服务。

### 15.6 Q5：阶段交付和停止条件

Q0输入通过后执行Q1–Q4；实际研究假设失败时记录结果与原因，禁止自动调整已冻结方法。缺数据、数值错误或检测器失败时明确停止受影响任务。每轮/阶段检查实际文件和哈希后写入完成状态。断点恢复不重复已完成训练、不遗漏评价。脚本显示真实进度、阶段结果和剩余时间估计，支持run_id、GPU、resume、unattended。不得覆盖用户修改，不使用Git回滚或强制推送。完成代码、测试和报告后提交当前E11分支，保留main及历史归档。

交付：Q0审计、冻结协议、三种正常数据划分及实际差异统计、六种拆分比例对照、五条件逐轮模型/上传清单与哈希、行为评价及交互统计、效果可比性报告、现有检测方法及校准清单、客户端/组合逐轮评分、混淆矩阵、TensorBoard事件文件、逐系列配对统计、诊断记录、RESULTS.md和明确的假设支持/否定/证据不足结论。报告注明服务器实际训练与仅离线回放的区别、未完成项和原始数据位置。Task 1历史报告保留，Task 2没有实验数据前不得标注成功。
