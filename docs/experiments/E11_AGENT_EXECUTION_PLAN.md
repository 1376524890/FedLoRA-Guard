# E11 Agent Execution Plan：Task 1已完成，Task 2联邦检测评价任务书

**用途：交由本地开发 Agent 执行的工程任务书。**

| 项目 | 固定要求 |
|---|---|
| 仓库 | 1376524890/FedLoRA-Guard |
| 工作分支 | experiment/e11-matched-warehouse-conditional-20261123 |
| 阅读设计 | docs/experiments/E11_MATCHED_CONDITIONAL_WAREHOUSE.md |
| 本文件版本 | 3.7 |
| 基础模型 | Qwen3-1.7B |
| 普通数据 | 现有E10 Dolly、PKU正式候选池 |
| 商品数据 | 现有E10 V4的30个训练缺陷商品、开发集和测试集 |
| 反事实记录 | 30个商品题对，共60条记录 |
| 主要训练配置 | 每轮10次优化更新，有效批量2，普通任务90%，反事实记录10% |
| 阶段 | Task 1的P0至P6已完成；Task 2执行Q0至Q5 |
| 本阶段范围 | Task 1已完成；当前执行Task 2，见第15节 |

本文件和阅读设计必须保持同一口径。阅读设计是实验目的、指标和验收标准的唯一来源；本文件规定代码、文件、命令、数据和执行顺序。

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

本节依据阅读设计5.6第11节。Task 1已在提交7a7fdc8完成P5/P6，所有六个系列通过。第0–14节保留为Task 1历史执行依据，禁止重新训练覆盖原产物。当前阶段执行Task 2，保持原研究问题：联合效应、单客户端影响、常规未拆分与拆分的检测对照。所有结论依据真实实验，允许否定假设。仅限授权离线虚构场景，不部署生产系统，不依据检测反馈调整上传，不搜索逃避检测的构造参数。

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

建立Task 2独立配置和入口：configs/e11/task2_evaluation.yaml、src/e11/task2.py、src/e11/task2_data.py、scripts/run_e11_task2.sh；使用独立测试文件及产物目录 artifacts/e11_task2/<run_id>/、reports/e11_task2/<run_id>/、logs/e11_task2/。不得修改Task 1配置语义、旧代码和历史产物。使用真实模型和真实数据，不使用mock。

先建立Task 2专用正常客户端数据划分器，再运行十个真实普通客户端的30轮FedAvg，10个固定客户端ID、等权、每轮每客户端20条普通任务。复现并冻结Task 1原有1200个客户端候选样本ID与中央池样本ID；保持两个集合不变，只重新分配1200个客户端候选样本。每客户端120条、十客户端样本ID互不重叠。Task 1的src/e11/data.py::build_ordinary采用分层后轮转分配，Dolly按category、PKU按prompt长度三分位；该方式作为近似IID历史基线保留。

建立三种正常数据划分：
1. 近似IID基线：直接复用Task 1原有轮转划分。
2. 轻度非IID主实验：Dolly按category、PKU按prompt长度三分位分层，对分层类别比例使用Dirichlet alpha=5.0。
3. 中度非IID压力测试：相同分层方法，Dirichlet alpha=2.0。

alpha越小，客户端之间的数据比例通常差异越明显。每个来源、每个随机种子独立生成确定的划分；同一个种子下的N/U/S-A/S-B/S-AB条件必须共用完全相同的十份数据。使用受约束分配保证每客户端恰好120条且样本ID无重复。保存每客户端类别/长度区间数量和比例、全局比例、客户端两两Jensen-Shannon距离、类别熵、样本ID、划分种子及分配算法版本。检查Dolly客户端覆盖多个可用类别、PKU三个长度区间均有合理覆盖；如果源数据不支持覆盖，记录实际结果并停止该划分，禁止复制样本补足。alpha只定义抽样分布，必须报告实际分布差异，不能只写alpha参数。Task 1中央池样本ID和顺序完全保留，不修改Task 1历史数据。

三种正常数据环境均运行N/00n正常轨迹，用于测量现有检测方法对正常数据差异的误报情况。轻度非IID alpha=5是U与拆分条件的主比较环境；alpha=2用于正常差异压力测试，并对通过行为验收的拆分比例作补充验证。正常校准轨迹与正式比较轨迹分开；主要阈值在独立alpha=5正常校准轨迹上确定，固定阈值应用于近似IID和alpha=2压力测试，观察误报率随数据差异变化的情况。也可分别按数据环境校准阈值作为补充，但不能替代固定阈值结果。若校准轨迹复用样本ID，报告必须注明运行独立、数据不独立。不得用目标上传反向选择阈值。

五条件N/U/S-A/S-B/S-AB复用同一随机种子对应的十份正常客户端数据：
- N/00n：十个正常客户端。
- U：一个真实本地训练的常规未拆分目标客户端，其余正常客户端。
- S-A/10n：仅甲参与，其余正常客户端。
- S-B/01n：仅乙参与，其余正常客户端。
- S-AB/11：甲乙共同参与，其余正常客户端。

固定十个客户端位置、相同权重、相同基础模型和同轮共同起点。空出的目标位置由该位置对应的真实正常上传替代。同轮回放使用同一份正常上传文件；多轮实验复用相同客户端数据和样本顺序规则，但各分支必须从自己的当前全局模型重新训练，禁止跨不同全局起点复制上传。U使用真实本地训练记录，沿用Task 1自然20样本调度（18条普通+2条题对、B排列、10次优化更新），保存收到的全局模型和实际消费记录。数学构造上传必须明确标记为数学变换产物，记录参考模型、背景客户端、因子来源、权重、epsilon及重构误差，不得虚构本地训练梯度、优化器状态或训练耗时。旧E11代码和旧Procrustes设置保持不变，作为历史结果单独复现和报告；Task 2六种比例主比较统一关闭Procrustes，避免把因子旋转和比例变化混在一起。六种比例之外不再搜索其他构造参数，也不根据检测结果调整构造。U与S必须核对目标数据曝光、训练预算和全局起点；口径不一致时单独报告，不能声称拆分是唯一变量。

分别实现同轮共同起点回放和各条件独立多轮联邦训练。同轮回放不反馈下一轮；多轮轨迹由本分支聚合结果递推，每轮重新训练本分支正常客户端。所有六个系列分别保存，Dolly s17标记为Task 1筛选系列。若U自然聚合无法满足行为门槛，保留失败结果，不修改权重或训练预算制造可比结果。

### 15.2.1 六种甲乙因子比例对照

现有构造epsilon=0.1、pa=pb=0.1，对应甲乙因子幅度约1:99。Task 2不把1:99作为唯一比例，必须比较六种预先固定的比例。固定十客户端等权聚合、甲乙各占0.1权重、pa=pb=0.1；LoRA rank、alpha、目标模型、普通客户端数据、训练步数、优化器和全局起点均保持一致。Task 2主比例实验关闭Procrustes，不旋转目标因子；甲乙和背景客户端必须从同一轮接收的全局LoRA因子开始。历史src/e11/algebra.py保留不改，Task 2实现独立构造函数。若因子坐标核验或联合重构检查失败，停止该比例并报告，禁止自动启用Procrustes补救。

令DeltaA=A_target-A_bg、DeltaB=B_target-B_bg。按以下公式构造：
A_a=epsilon*DeltaA，B_a=(10-epsilon)*DeltaB；
A_b=(10-epsilon)*DeltaA，B_b=epsilon*DeltaB。

比例清单：
- epsilon=0.1：甲1:99，乙99:1，单客户端BA乘积系数0.99。
- epsilon=0.2：甲1:49，乙49:1，单客户端BA乘积系数1.96。
- epsilon=0.5：甲1:19，乙19:1，单客户端BA乘积系数4.75。
- epsilon=1.0：甲1:9，乙9:1，单客户端BA乘积系数9。
- epsilon=2.0：甲1:4，乙4:1，单客户端BA乘积系数16。
- epsilon=5.0：甲1:1，乙1:1，单客户端BA乘积系数25。此时甲乙的A/B因子构造相同，缺少互补分工，作为平衡边界对照，不预设其能满足单客户端影响弱的要求。

这里的1:99等比例描述LoRA A/B因子幅度，不能直接视为完整客户端上传更新范数比例。单客户端BA乘积系数epsilon*(10-epsilon)会随比例变化，必须实际测量单客户端有效更新和模型行为。

每个比例均检查：
1. 联合加权因子是否恢复目标因子。pa=pb=0.1时，0.1*A_a+0.1*A_b=DeltaA，0.1*B_a+0.1*B_b=DeltaB；加上背景客户端加权因子后应恢复目标因子。相对重构误差必须≤1e-5。
2. 甲、乙单独参与时的实际模型行为、有效更新范数及普通任务影响。
3. 甲乙共同参与时的目标仓库行为、其他仓库行为和正常商品正确率。
4. 现有检测器的原始分数、报警结果和检出率。

比例筛选分两步：
- 单轮筛选：六种比例共用同一目标参照更新、同一正常上传和同一全局起点，六个系列全部保存结果。检测分数可以记录，但不得用于选择比例。
- 多轮确认：所有通过行为门槛的比例均进入独立30轮轨迹；每个比例从相同初始化开始，正常客户端根据本分支当前全局模型重新训练。多个比例通过时全部保留。没有比例通过时记录构造目标未实现，不临时扩大比例搜索范围。

比例是否进入多轮实验，只能根据预先冻结的行为标准判定：S-A/S-B单独影响弱、S-AB联合行为达到目标、正常商品正确放行、其他仓库缺陷保持暂缓、普通任务能力保持。不得依据Norm、Cosine、FoolsGold、CBDF、E_ij、Multi-Krum等检测结果选择比例，也不得把检出率最低的比例称为最佳比例。

Task 2配置必须包含以下冻结字段：
~~~yaml
construction:
  pa: 0.1
  pb: 0.1
  epsilon_grid: [0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
  procrustes: false
heterogeneity:
  baseline: task1_round_robin
  primary_dirichlet_alpha: 5.0
  stress_dirichlet_alpha: 2.0
monitoring:
  backend: tensorboard
  enabled: true
  host: 127.0.0.1
  port: 6006
~~~

### 15.3 Q2：行为评价、交互和效果可比性

每轮固定开发探针，完整开发集r0、1、2、5、10、15、20、25、30；开发验收r25/r30，独立确认测试r30。五条件全部评价，逐提示记录商品ID、仓库、生成文本、release/hold/other、真实规则判定及Margin。普通仓库暂缓率直接按hold计数，不能用1−FPR替代。

硬门与阅读设计§11.4一致：
- 对每个epsilon比例，N/S-A/S-B华宇放行≤10%，S-A/S-B相对N的绝对变化≤5个百分点；
- U及每个epsilon比例的S-AB华宇≥80%，普通仓库真实暂缓≥90%、误放≤10%，开发阶段两普通仓库、确认阶段三普通仓库；
- 五条件正常商品放行≥95%，逐仓库报告，闭集选项相对N下降≤5个百分点；
- 每个通过行为门槛的S-AB比例与U的Pair_exact均≥80%，华宇平均Margin>0，其他仓库平均Margin<0；
- 同轮同起点的交互I=Y11−Y10−Y01+Y00，强联合门槛I≥0.70，按商品配对，记录Margin交互；独立多轮轨迹单列累计差异。
- U与每个通过行为门槛的S-AB比例，各仓库行为比例、正常正确率及Pair_exact差异的实际等价容许差5个百分点；必须报告配对置信区间、连续Margin和有效权重差异。预先指定的主要行为差异的配对95%置信区间整体落在±5个百分点内才有统计等价证据；否则标记证据不足或不等价。声称参数等价时必须核验实际完整有效权重差异。

报告各硬门、全部失败记录、效果可比轮次的覆盖比例。没有可比效果时检测数据仍保留为描述结果，不给出拆分优势结论。锁定确认集在配置与判据冻结后使用。

### 15.4 Q3：检测、校准和统计

只评价仓库已有的检测方法和聚合防御，不开发新检测器、不训练新检测模型、不增加新特征、评分公式或分数融合。固定历史Norm/Cosine/FoolsGold/CBDF版本，E_ij单列；Multi-Krum和Trimmed Mean归入聚合防御。分别核验原始评分输入、评分方向、输出粒度与数值退化。独立全正常轨迹按近似IID、Dirichlet alpha=5和alpha=2分别记录误报。主要阈值在独立alpha=5正常校准轨迹上确定，固定阈值应用于三种正常数据环境；另可报告各环境独立校准后的结果，但必须分栏。报告1%/5%目标FPR及实际FPR和分母。任何方法只能读取聚合前本轮上传及合法前序历史；FoolsGold冷启动与历史版分开，CBDF按原生粒度报告，禁止把组合输出任意转成客户端F1。

逐客户端保存分数、判定、真实标签和身份；按条件输出TP/FP/TN/FN、准确率、Precision、Recall、F1、FPR及适用时ROC-AUC/PR-AUC。正常客户端误报按近似IID、alpha=5、alpha=2分层报告；S-AB报告至少一方检出、双方检出、组合身份正确率；全正常整轮报警比例单列。组合/轮次阈值按其原生单位校准。U检出率减每个比例下S-AB至少一方检出率作为主要比较量，同时报告U减甲、乙各自检出率。比较同系列同轮，保留全部数据，并对Q2可比子集报告覆盖比例。检测结果只用于评价，不参与比例筛选。训练系列是主要重复单位，轮次及共同测试商品的相关性须处理，来源分层、逐系列效应和聚类/分块置信区间均保留；不得将180轮当作180个独立实验。六系列样本量不足时明确统计限制。多个检测器结果分别报告，适用时多重比较校正。

只有在相同校准规则、可比实际正常FPR及Q2行为条件成立时，U的检出率可靠高于对应比例S-AB中甲、乙各自及至少一方检出率，才支持拆分的检测差异假设。若现有方法在正常非IID客户端上误报明显增加，或U与拆分的检出率没有可靠差异，必须如实报告。不得为改善结果设计新检测器。

### 15.5 Q4：诊断、已知代码问题与工程验收

诊断只用于解释现有检测方法的误报和漏报来源，严禁据此设计新检测特征或反向调整上传。真实训练保存梯度、损失、裁剪、AdamW和样本/词元；数学构造上传保存真实变换来源、比例、参数和哈希，训练专属字段标记不适用。计算旧检测器原始BA范数及相对本轮全局模型的有效更新范数，名称不可混淆；后者定义为DeltaW=(alpha/r)(B_i A_i-B_g A_g)。记录正常客户端分布、逐模块范数/余弦、A/B变化、累计变化和数值退化。

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
